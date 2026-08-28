import numpy as np
import torch
from torch.fft import rfft, irfft
from torch.autograd import grad
from scipy.special import spherical_jn
from .lj_eos import lj_eos
from .solvers import *

kB = 1.380649e-23
NA = 6.02214076e23

pi = np.pi
torch.set_default_dtype(torch.float64)


def lancsoz(k, M):
    return np.sinc(k/M)


def yukawa_ft(k, sigma, epsilon, l):

    u_hat = -epsilon*\
        np.piecewise(k,[k==0.0,k>0.0],
                     [4*pi*sigma**3*(l+1.0)/l**2,
                      lambda k:
                      (2*sigma**2*(2*k*pi*sigma*np.cos(2*k*pi*sigma)+l*np.sin(2*k*pi*sigma)))/(k*(l**2+(2*k*pi*sigma)**2))])

    return u_hat


class dft_core():

    def __init__(self, parameters, temperature, system_size, points, device):

        self.parameters = parameters
        self.sigma = self.parameters['sigma']
        self.epsilon = self.parameters['epsilon']
        self.T = temperature
        self.Tstar = self.T/self.epsilon
        self.system_size = system_size
        self.points = points
        self.device = device

        # Shared with the solvers, which work on flat arrays.
        self.npoints = int(points)
        self.shape = (self.npoints,)
        self.sqrt_npoints = np.sqrt(self.npoints)

        self.kB = 1.380649e-23
        self.NA = 6.02214076e23

        self.d = self.sigma*(1.0+0.2977*self.Tstar)/(1.0+0.33163*self.Tstar+0.0010477*self.Tstar**2)
        self.R = 0.5*self.d
        self.R_sq = self.R**2
        self.R_cu = self.R**3
        self.four_pi_R_sq = 4.0*pi*self.R_sq
        self.four_pi_R = 4.0*pi*self.R

        self.cell_size = system_size/points
        self.z = torch.linspace(0.5*self.cell_size, system_size-0.5*self.cell_size,
                                self.npoints, device=device)

        # rho is real, so only the non-redundant half of the spectrum is needed.
        kz = np.fft.rfftfreq(self.npoints, d=self.cell_size)
        kcut = (self.npoints//2+1)/self.system_size
        k = np.abs(kz)

        # Precompute common terms
        two_pi_R_K = 2.0*pi*self.R*k
        four_pi_R_K = 2.0*two_pi_R_K
        lanczos_term = lancsoz(kz, kcut)

        # w2, w3, watt and ulj are purely real, so they are stored as real
        # tensors: half the memory and a cheaper complex*real product.
        w2_hat = self.four_pi_R_sq*spherical_jn(0, two_pi_R_K)*lanczos_term
        w3_hat = (4./3.)*pi*self.R_cu*(spherical_jn(0, two_pi_R_K)+spherical_jn(2, two_pi_R_K)) \
            *lanczos_term
        watt_hat = (spherical_jn(0, four_pi_R_K)+spherical_jn(2, four_pi_R_K))*lanczos_term

        l = np.array([2.544944560171334,15.464088962136243])
        eps = 1.857708161877173*self.epsilon*np.array([1,-1])
        ulj_hat = (yukawa_ft(k,self.sigma,eps[0],l[0])+yukawa_ft(k,self.sigma,eps[1],l[1]))*lanczos_term

        # w2vec_hat = -i 2 pi kz w3_hat is purely imaginary. Only the real
        # vector 2 pi kz is kept; the -i and w3_hat are applied on the fly.
        kvec = 2.0*pi*kz.copy()

        # The gradient kernel is odd in k, but for an even number of points the
        # Nyquist bin is its own conjugate partner and carries no consistent
        # sign, so the convolution stops being Hermitian there. Zeroing the
        # derivative on that bin is the standard treatment and also makes the
        # rfft result identical to the full complex transform.
        if self.npoints % 2 == 0:
            kvec[-1] = 0.0

        self.w2_hat = torch.tensor(w2_hat, device=device)
        self.w3_hat = torch.tensor(w3_hat, device=device)
        self.watt_hat = torch.tensor(watt_hat, device=device)
        self.ulj_hat = torch.tensor(ulj_hat, device=device)
        self.kvec = torch.tensor(kvec, device=device)

        # Clear temporary arrays to free memory
        del kz,k,two_pi_R_K,four_pi_R_K,lanczos_term,kvec
        del w2_hat,w3_hat,watt_hat,ulj_hat

    def weighted_densities(self):

        self.rho.requires_grad = True

        self.rho_hat = rfft(self.rho)

        self.n2 = irfft(self.rho_hat*self.w2_hat, n=self.npoints)
        self.n0 = self.n2/self.four_pi_R_sq

        # n3 and n2vec share the same product rho_hat*w3_hat.
        rho_w3 = self.rho_hat*self.w3_hat
        self.n3 = irfft(rho_w3, n=self.npoints).clamp(max=1.0-1e-16)
        self.n2vec = irfft(-1j*(self.kvec*rho_w3), n=self.npoints)

        self.rhobar = irfft(self.rho_hat*self.watt_hat, n=self.npoints)
        self.ulj = irfft(self.rho_hat*self.ulj_hat, n=self.npoints)

    def functional(self,fmt):

        self.weighted_densities()

        # Hard-Sphere Contribution
        one_minus_n3 = 1.0-self.n3
        f1 = -torch.log(one_minus_n3)
        f2 = one_minus_n3.reciprocal()

        n3s = self.n3.clamp(min=1e-4)
        one_minus_n3s = 1.0-n3s
        one_minus_n3s_sq = one_minus_n3s*one_minus_n3s
        f4 = torch.where(self.n3 > 1e-4,
                         (n3s+one_minus_n3s_sq*torch.log(one_minus_n3s))
                         /(36*pi*n3s*n3s*one_minus_n3s_sq),
                         1/(24*pi) + 2/(27*pi)*self.n3 + 5/(48*pi)*self.n3**2)

        # n1 = n2/(4 pi R) and n1vec = n2vec/(4 pi R), so
        #   n1*n2 - n1vec*n2vec = (n2^2 - n2vec^2)/(4 pi R),
        # and clamping n1vec*n2vec <= n1*n2 is the same as clamping
        # n2vec^2 <= n2^2. This drops n1 and n1vec entirely.
        n2_sq = self.n2*self.n2
        n2vec_sq = (self.n2vec*self.n2vec).clamp(max=n2_sq)
        vec_term = (n2_sq-n2vec_sq)/self.four_pi_R

        if fmt == 'WB':

            self.Phi_hs = f1*self.n0+f2*vec_term+f4*(n2_sq*self.n2-3.0*self.n2*n2vec_sq)

        elif fmt == 'ASWB':

            xi = (n2vec_sq/n2_sq).clamp(max=1.0-1e-16)
            self.Phi_hs = f1*self.n0+f2*vec_term+f4*(self.n2*n2_sq)*(1.0-xi)**3

        else:
            raise ValueError("fmt must be 'WB' or 'ASWB'")

        self.Fhs = self.Phi_hs.sum()*self.cell_size

        # Attractive Contribution
        eta = (self.rhobar*(pi*self.d**3/6)).clamp(max=1.0-1e-16)
        one_minus_eta = 1.0-eta
        eos_term = self.eos.helmholtz_energy(self.rhobar)
        correction_term_hs = (4.0*eta-3.0*eta*eta)/(one_minus_eta*one_minus_eta)
        correction_term_mfa = (16./9.)*pi*(self.epsilon/self.T)*self.sigma**3*self.rhobar

        self.Phi_cor = eos_term-correction_term_hs+correction_term_mfa
        self.Phi_mfa = 0.5*self.rho*self.ulj/self.T
        self.Phi_att = self.rhobar*self.Phi_cor+self.Phi_mfa
        self.Fatt = self.Phi_att.sum()*self.cell_size

        self.Fres = self.Fhs+self.Fatt

    def functional_derivative(self, fmt):

        self.functional(fmt)
        self.dFres = grad(self.Fres, self.rho)[0]
        self.dFres = self.dFres.detach()/self.cell_size

        self.rho.requires_grad=False

    def euler_lagrange(self, lnrho, fmt='WB'):

        self.functional_derivative(fmt)
        # Multiplying by a 0/1 mask instead of boolean indexing: advanced
        # indexing calls nonzero() internally, which forces a device sync on
        # every use. The excluded cells get exactly zero residual, so the
        # solvers can then work on the full array.
        self.res = (self.mu-self.dFres-self.Vext-lnrho)*self.valid

    def loss(self):
        return torch.linalg.vector_norm(self.res)/self.sqrt_npoints

    def initial_condition(self, bulk_density, Vext, potential_cutoff=50.0, model='bulk'):

        self.rhob = bulk_density
        self.eos = lj_eos(self.parameters, self.T, device=self.device)
        self.mu = (self.eos.chemical_potential(bulk_density)
                   +torch.log(self.rhob)).to(device=self.device)

        self.Vext = (Vext/self.T).to(device=self.device)
        self.excluded = self.Vext >= potential_cutoff
        self.valid = self.Vext < potential_cutoff
        self.Vext[self.excluded] = potential_cutoff

        self.rho = torch.empty(self.shape, device=self.device)
        if model == 'bulk':
            self.rho[:] = self.rhob
        elif model == 'ideal':
            self.rho = self.rhob*torch.exp(-self.Vext)

    def equilibrium_density_profile(self, bulk_density, fmt='WB', solver='anderson',
                                    alpha0=0.2, dt=0.1, anderson_mmax=10, anderson_damping=0.1,
                                    gmres_tol=1e-4, gmres_max_iter=30,
                                    tol=1e-6, max_it=1000, logoutput=False):

        self.rhob = bulk_density
        self.mu = (self.eos.chemical_potential(bulk_density)
                   +torch.log(self.rhob)).to(device=self.device)
        self.fmt = fmt
        self.rho = self.rho.detach().clone()
        self.rho[self.excluded] = 1e-15

        if solver == 'picard':
            picard(self,alpha0,tol,max_it,logoutput)

        elif solver == 'picard_ls':
            picard_line_search(self,alpha0,tol,max_it,logoutput)

        elif solver == 'anderson':
            anderson(self,anderson_mmax,anderson_damping,tol,max_it,logoutput)

        elif solver == 'fire':
            fire(self,alpha0,dt,tol,max_it,logoutput)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.error = self.error.cpu()

        self.total_molecules = (self.rho*self.valid).sum().cpu()*self.cell_size
        Phi = self.rho*(torch.log(self.rho)-1.0)+self.rho*(self.Vext-self.mu)
        self.Omega = Phi.sum()*self.cell_size+self.Fres.detach()