import numpy as np
import scipy
import torch
from .lj_eos import lj_eos
from .solvers import *

torch.set_default_dtype(torch.float64)
pi = np.pi


def lancsoz(kx, ky, kz, M):
    return np.sinc(kx/M[0])*np.sinc(ky/M[1])*np.sinc(kz/M[2])


def yukawa_ft(k, sigma, epsilon, l):
    u_hat = -epsilon*\
        np.piecewise(k,[k==0.0,k>0.0],
                     [4*pi*sigma**3*(l+1.0)/l**2,
                      lambda k:
                      (2*sigma**2*(2*k*pi*sigma*np.cos(2*k*pi*sigma)+l*np.sin(2*k*pi*sigma)))/(k*(l**2+(2*k*pi*sigma)**2))])
    return u_hat


class dft_core():

    def __init__(self, parameters, temperature, system_size, angles, points, device):

        self.parameters = parameters
        self.sigma = self.parameters['sigma']
        self.epsilon = self.parameters['epsilon']
        self.T = temperature
        self.Tstar = self.T/self.epsilon
        self.system_size = system_size
        self.points = points
        self.device = device

        # Real-space grid shape, used by every irfftn call.
        self.shape = tuple(int(p) for p in points)
        self.npoints = int(np.prod(self.shape))
        self.sqrt_npoints = np.sqrt(self.npoints)

        self.kB = 1.380649e-23
        self.NA = 6.02214076e23

        if angles is not None:
            self.alpha, self.beta, self.gamma = angles
            self.orthogonal = False

            cos_alpha = np.cos(self.alpha)
            cos_beta = np.cos(self.beta)
            cos_gamma = np.cos(self.gamma)
            sin_gamma = np.sin(self.gamma)

            zeta = (cos_alpha-cos_beta*cos_gamma)/sin_gamma

            self.H = torch.tensor([
                [1.0, cos_gamma, cos_beta],
                [0.0, sin_gamma, zeta],
                [0.0, 0.0, np.sqrt(1.0-cos_beta**2-zeta**2)]
            ], device=device)

            self.H_T = self.H.T
            self.H_inv_T = torch.linalg.inv(self.H_T)
            self.det_H = sin_gamma*np.sqrt(1.0-cos_beta**2-zeta**2)
        else:
            self.orthogonal = True
            self.H = torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], device=device)
            self.det_H = 1.0

        self.d = self.sigma*(1.0+0.2977*self.Tstar)/(1.0+0.33163*self.Tstar+0.0010477*self.Tstar**2)
        self.R = 0.5*self.d
        self.R_sq = self.R**2
        self.R_cu = self.R**3
        self.four_pi_R_sq = 4.0*pi*self.R_sq
        self.four_pi_R = 4.0*pi*self.R

        self.system_volume = self.system_size.prod()*self.det_H
        self.cell_size = system_size/points
        self.cell_volume = self.cell_size.prod()*self.det_H

        # Spatial grid in skewed coordinates
        u = torch.linspace(0.5*self.cell_size[0], system_size[0]-0.5*self.cell_size[0], self.shape[0], device=device)
        v = torch.linspace(0.5*self.cell_size[1], system_size[1]-0.5*self.cell_size[1], self.shape[1], device=device)
        w = torch.linspace(0.5*self.cell_size[2], system_size[2]-0.5*self.cell_size[2], self.shape[2], device=device)
        self.U, self.V, self.W = torch.meshgrid(u, v, w, indexing='ij')

        # Transform to cartesian coordinates
        s = torch.stack([self.U, self.V, self.W], dim=0)
        r = torch.einsum('ij,j...->i...', self.H, s)
        self.X, self.Y, self.Z = r[0], r[1], r[2]

        ku = np.fft.fftfreq(self.shape[0], d=self.cell_size[0])
        kv = np.fft.fftfreq(self.shape[1], d=self.cell_size[1])
        kw = np.fft.rfftfreq(self.shape[2], d=self.cell_size[2])

        # Transform to cartesian frequency space
        Ku, Kv, Kw = np.meshgrid(ku, kv, kw, indexing='ij')

        if self.orthogonal:
            Kx = Ku
            Ky = Kv
            Kz = Kw
            del self.U, self.V, self.W
        else:
            Kx = Ku
            Ky = (Kv-Ku*cos_gamma)/sin_gamma
            Kz = (Ku*(zeta*cos_gamma/sin_gamma-cos_beta)\
                  -Kv*zeta/sin_gamma+Kw)/np.sqrt(1.0-cos_beta**2-zeta**2)

        K = np.sqrt(Kx**2+Ky**2+Kz**2)
        kcut = (np.asarray(self.shape)//2+1)/self.system_size

        # Precompute common terms
        two_pi_R_K = 2.0*pi*self.R*K
        four_pi_R_K = 2.0*two_pi_R_K
        lanczos_term = lancsoz(Ku, Kv, Kw, kcut)

        w2_hat = self.four_pi_R_sq*scipy.special.spherical_jn(0,two_pi_R_K)*lanczos_term
        w3_hat = (4./3.)*pi*self.R_cu*(scipy.special.spherical_jn(0, two_pi_R_K)+scipy.special.spherical_jn(2,two_pi_R_K))*lanczos_term
        watt_hat = (scipy.special.spherical_jn(0, four_pi_R_K)+scipy.special.spherical_jn(2,four_pi_R_K))*lanczos_term

        l = np.array([2.544944560171334,15.464088962136243])
        eps = 1.857708161877173*self.epsilon*np.array([1,-1])
        ulj_hat = (yukawa_ft(K,self.sigma,eps[0],l[0])+yukawa_ft(K,self.sigma,eps[1],l[1]))*lanczos_term

        self.w2_hat = torch.tensor(w2_hat, device=device)
        self.w3_hat = torch.tensor(w3_hat, device=device)
        self.watt_hat = torch.tensor(watt_hat, device=device)
        self.ulj_hat = torch.tensor(ulj_hat, device=device)

        kvec = 2.0*pi*np.stack([Kx, Ky, Kz])

        if self.shape[0] % 2 == 0:
            kvec[:, self.shape[0]//2, :, :] = 0.0
        if self.shape[1] % 2 == 0:
            kvec[:, :, self.shape[1]//2, :] = 0.0
        if self.shape[2] % 2 == 0:
            kvec[:, :, :, -1] = 0.0

        self.kvec = torch.tensor(kvec, device=device)

        del u,v,w,s,r,ku,kv,kw,Ku,Kv,Kw,Kx,Ky,Kz,K,two_pi_R_K,four_pi_R_K,lanczos_term
        del w2_hat,w3_hat,watt_hat,ulj_hat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def weighted_densities(self):

        self.rho.requires_grad = True

        self.rho_hat = torch.fft.rfftn(self.rho)

        self.n2 = torch.fft.irfftn(self.rho_hat*self.w2_hat, s=self.shape)
        self.n0 = self.n2/self.four_pi_R_sq

        rho_w3 = self.rho_hat*self.w3_hat
        self.n3 = torch.fft.irfftn(rho_w3, s=self.shape).clamp(max=1.0-1e-16)
        self.n2vec = torch.fft.irfftn(-1j*(self.kvec*rho_w3), dim=(1,2,3), s=self.shape)

        self.rhobar = torch.fft.irfftn(self.rho_hat*self.watt_hat, s=self.shape)
        self.ulj = torch.fft.irfftn(self.rho_hat*self.ulj_hat, s=self.shape)

    def helmholtz_functional(self,fmt):

        self.weighted_densities()

        # Hard-Sphere Contribution
        one_minus_n3 = 1.0-self.n3
        f1 = -torch.log(one_minus_n3)
        f2 = one_minus_n3.reciprocal()

        n3_safe = self.n3.clamp(min=1e-4)
        one_minus_n3_safe = 1.0-n3_safe
        one_minus_n3_safe_sq = one_minus_n3_safe*one_minus_n3_safe
        f4 = torch.where(
            self.n3 > 1e-4,
            (n3_safe+one_minus_n3_safe_sq*torch.log(one_minus_n3_safe))
            /(36.0*pi*n3_safe*n3_safe*one_minus_n3_safe_sq),
            1.0/(24.0*pi)+(2.0/(27.0*pi))*self.n3+(5.0/(48.0*pi))*self.n3**2)

        n2_sq = self.n2*self.n2
        n2vec_sq = (self.n2vec*self.n2vec).sum(dim=0).clamp(max=n2_sq)
        vec_term = (n2_sq-n2vec_sq)/self.four_pi_R

        if fmt == 'WB':

            self.Phi_hs = f1*self.n0+f2*vec_term+f4*(n2_sq*self.n2-3.0*self.n2*n2vec_sq)

        elif fmt == 'ASWB':

            xi = (n2vec_sq/n2_sq).clamp(max=1.0-1e-16)
            self.Phi_hs = f1*self.n0+f2*vec_term+f4*(self.n2*n2_sq)*(1.0-xi)**3

        else:
            raise ValueError("fmt must be 'WB' or 'ASWB'")

        self.Fhs = self.Phi_hs.sum()*self.cell_volume

        # Attractive Contribution
        self.Phi_mfa = 0.5*self.rho*self.ulj/self.T

        eta = (self.rhobar*(pi*self.d**3/6.0)).clamp(max=1.0-1e-16)
        one_minus_eta = 1.0-eta
        eos_term = self.eos.helmholtz_energy(self.rhobar)
        correction_term_hs = (4.0*eta-3.0*eta*eta)/(one_minus_eta*one_minus_eta)
        correction_term_mfa = -(16./9.)*pi*(self.epsilon/self.T)*self.sigma**3*self.rhobar
        self.Phi_cor = eos_term-correction_term_hs-correction_term_mfa

        self.Phi_att = self.Phi_mfa+self.rhobar*self.Phi_cor

        self.Fatt = self.Phi_att.sum()*self.cell_volume

        self.Fex = self.Fhs+self.Fatt

    def helmholtz_functional_derivative(self, fmt):

        self.helmholtz_functional(fmt)
        self.dFex = torch.autograd.grad(self.Fex, self.rho)[0]
        self.dFex = self.dFex.detach()/self.cell_volume

        self.rho.requires_grad=False

    def euler_lagrange(self, lnrho, fmt):

        self.helmholtz_functional_derivative(fmt)
        self.res = (self.mu-lnrho-self.dFex-self.Vext)*self.valid

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

    def equilibrium_density_profile(self, bulk_density, fmt='ASWB', solver='anderson',
                                    alpha0=0.2, dt=0.1, anderson_mmax=10, anderson_damping=0.1,
                                    tol=1e-6, max_it=1000, logoutput=False):

        self.rhob = bulk_density
        self.fmt = fmt
        self.mu = (self.eos.chemical_potential(bulk_density)
                   +torch.log(self.rhob)).to(device=self.device)
        self.rho = self.rho.detach().clone()
        self.rho[self.excluded] = 1e-16

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

        self.total_molecules = (self.rho*self.valid).sum().cpu()*self.cell_volume
        Phi = self.rho*(torch.log(self.rho)-1.0)+self.rho*(self.Vext-self.mu)
        self.Omega = Phi.sum()*self.cell_volume+self.Fex.detach()
