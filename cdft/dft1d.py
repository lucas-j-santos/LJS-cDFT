import numpy as np
import torch
from scipy.special import spherical_jn, sici
from .lj_eos import lj_eos
from .solvers import *

torch.set_default_dtype(torch.float64)
pi = np.pi

def lancsoz(k, M):
    return np.sinc(k/M)

def yukawa_ft(k, sigma, epsilon, l):

    x = 2.0*pi*k*sigma
    u_hat = -4.0*pi*epsilon*sigma**3*((1.0+l)*spherical_jn(0, x)-x*spherical_jn(1, x))/(x**2+l**2)

    return u_hat

def lj_att_ft(k, sigma, epsilon):

    q = 2.0*pi*np.abs(np.asarray(k, dtype=float))

    out = np.empty_like(q)
    zero = (q == 0.0)
    out[zero] = 16.0*pi*epsilon*(sigma**12/(9.0*sigma**9)-sigma**6/(3.0*sigma**3))
    qb = q[~zero]
    
    if qb.size:
        qd = qb*sigma
        si, ci = sici(qd)
        S = {1: pi/2-si}
        C = {1: -ci}
        sin_qd, cos_qd = np.sin(qd), np.cos(qd)
        for m in range(1, 11):
            S[m+1] = (qb/m)*(C[m]+sin_qd/(qb*sigma**m))
            C[m+1] = (qb/m)*(cos_qd/(qb*sigma**m)-S[m])
        out[~zero] = (16.0*pi*epsilon/qb)*(sigma**12*S[11]-sigma**6*S[5])

    return out

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

        kz = np.fft.rfftfreq(self.npoints, d=self.cell_size)
        kcut = (self.npoints//2+1)/self.system_size
        k = np.abs(kz)

        # Precompute common terms
        two_pi_R_K = 2.0*pi*self.R*k
        four_pi_R_K = 2.0*two_pi_R_K
        lanczos_term = lancsoz(kz, kcut)

        w2_hat = self.four_pi_R_sq*spherical_jn(0, two_pi_R_K)*lanczos_term
        w3_hat = (4./3.)*pi*self.R_cu*(spherical_jn(0, two_pi_R_K)+spherical_jn(2, two_pi_R_K)) \
            *lanczos_term
        watt_hat = (spherical_jn(0, four_pi_R_K)+spherical_jn(2, four_pi_R_K))*lanczos_term

        l = np.array([2.544944560171334,15.464088962136243])
        eps = 1.857708161877173*self.epsilon*np.array([1,-1])
        ulj_hat = (yukawa_ft(k,self.sigma,eps[0],l[0])+yukawa_ft(k,self.sigma,eps[1],l[1]))*lanczos_term
        # ulj_hat = lj_att_ft(k, self.sigma, self.epsilon)*lanczos_term

        kvec = 2.0*pi*kz.copy()

        if self.npoints % 2 == 0:
            kvec[-1] = 0.0

        self.w2_hat = torch.tensor(w2_hat, device=device)
        self.w3_hat = torch.tensor(w3_hat, device=device)
        self.watt_hat = torch.tensor(watt_hat, device=device)
        self.ulj_hat = torch.tensor(ulj_hat, device=device)
        self.kvec = torch.tensor(kvec, device=device)

        del kz,k,two_pi_R_K,four_pi_R_K,lanczos_term,kvec
        del w2_hat,w3_hat,watt_hat,ulj_hat

    def weighted_densities(self):

        self.rho.requires_grad = True

        self.rho_hat = torch.fft.rfft(self.rho)

        self.n2 = torch.fft.irfft(self.rho_hat*self.w2_hat, n=self.npoints)
        self.n0 = self.n2/self.four_pi_R_sq

        # n3 and n2vec share the same product rho_hat*w3_hat.
        rho_w3 = self.rho_hat*self.w3_hat
        self.n3 = torch.fft.irfft(rho_w3, n=self.npoints).clamp(max=1.0-1e-16)
        self.n2vec = torch.fft.irfft(-1j*(self.kvec*rho_w3), n=self.npoints)

        self.rhobar = torch.fft.irfft(self.rho_hat*self.watt_hat, n=self.npoints).clamp(max=1.2/self.sigma**3)
        self.ulj = torch.fft.irfft(self.rho_hat*self.ulj_hat, n=self.npoints)

    def helmholtz_functional(self,fmt):

        self.weighted_densities()

        # Hard-Sphere Contribution
        one_minus_n3 = 1.0-self.n3
        f1 = -torch.log(one_minus_n3)
        f2 = one_minus_n3.reciprocal()

        n3s = self.n3.clamp(min=1e-4)
        one_minus_n3s = 1.0-n3s
        one_minus_n3s_sq = one_minus_n3s*one_minus_n3s
        f4 = torch.where(self.n3 > 1e-4,
                         (n3s+one_minus_n3s_sq*torch.log(one_minus_n3s))/(36*pi*n3s*n3s*one_minus_n3s_sq),
                         1/(24*pi) + 2/(27*pi)*self.n3 + 5/(48*pi)*self.n3**2)

        n2_sq = self.n2*self.n2
        n2vec_sq = (self.n2vec*self.n2vec).clamp(max=n2_sq)
        vec_term = (n2_sq-n2vec_sq)/self.four_pi_R

        if fmt == 'WB':

            Phi_hs = f1*self.n0+f2*vec_term+f4*(n2_sq*self.n2-3.0*self.n2*n2vec_sq)

        elif fmt == 'ASWB':

            xi = (n2vec_sq/n2_sq).clamp(max=1.0-1e-16)
            Phi_hs = f1*self.n0+f2*vec_term+f4*(self.n2*n2_sq)*(1.0-xi)**3

        else:
            raise ValueError("fmt must be 'WB' or 'ASWB'")

        self.F_hs = Phi_hs.sum()*self.cell_size

        del Phi_hs

        # Attractive Contribution
        Phi_mfa = 0.5*self.rho*self.ulj/self.T
        self.F_mfa = Phi_mfa.sum()*self.cell_size

        eta = (self.rhobar*(pi*self.d**3/6.0))
        one_minus_eta = 1.0-eta
        eos_term = self.eos.helmholtz_energy(self.rhobar)
        correction_term_hs = (4.0*eta-3.0*eta*eta)/(one_minus_eta*one_minus_eta)
        correction_term_mfa = -(16./9.)*pi*(self.epsilon/self.T)*self.sigma**3*self.rhobar
        Phi_corr = self.rhobar*(eos_term-correction_term_hs-correction_term_mfa)
        self.F_corr = Phi_corr.sum()*self.cell_size 

        del Phi_mfa, Phi_corr

        self.F_att = self.F_mfa+self.F_corr
        
        self.F_ex = self.F_hs+self.F_att

    def helmholtz_functional_derivative(self, fmt):

        self.helmholtz_functional(fmt)
        self.dF_ex = torch.autograd.grad(self.F_ex, self.rho)[0]
        self.dF_ex = self.dF_ex.detach()/self.cell_size

        self.rho.requires_grad=False

    def euler_lagrange(self, lnrho, fmt='WB'):

        self.helmholtz_functional_derivative(fmt)

        if self.N_target is None:
            # grand canonical
            self.res = (self.mu-lnrho-self.dF_ex-self.Vext)*self.valid
        else:
            # canonical
            g = -(self.dF_ex+self.Vext)
            g_valid = torch.where(self.valid, g, self._neg_inf)
            self.mu = (np.log(self.N_target)-np.log(self.cell_size)-torch.logsumexp(g_valid, dim=0))
            self.res = (self.mu+g-lnrho)*self.valid

    def loss(self):
        return torch.linalg.vector_norm(self.res)/self.sqrt_npoints

    def initial_condition(self, bulk_density, Vext, potential_cutoff=50.0, model='bulk'):

        self.eos = lj_eos(self.parameters, self.T, device=self.device)
        self.mu_id = torch.log(bulk_density)
        self.mu_ex = self.eos.chemical_potential(bulk_density) 
        self.mu = (self.mu_id+self.mu_ex).to(device=self.device)
        self.rhob = torch.as_tensor(bulk_density).to(device=self.device)

        self.Vext = (Vext/self.T).to(device=self.device)
        self.excluded = self.Vext >= potential_cutoff
        self.valid = self.Vext < potential_cutoff
        self.Vext[self.excluded] = potential_cutoff

        self.N_target = None
        self._neg_inf = torch.tensor(float('-inf'), device=self.device)

        self.rho = torch.empty(self.shape, device=self.device)
        if model == 'bulk':
            self.rho[:] = self.rhob
        elif model == 'ideal':
            self.rho = self.rhob*torch.exp(-self.Vext)

    def equilibrium_density_profile(self, bulk_density=None, fmt='WB', solver='anderson',
                                    alpha0=0.2, dt=0.1, anderson_mmax=10, anderson_damping=0.1,
                                    tol=1e-6, max_it=1000, logoutput=False, N_target=None):

        self.fmt = fmt
        self.N_target = None if N_target is None else float(N_target)

        if self.N_target is None:
            if bulk_density is None:
                raise ValueError("bulk density is mandatory without N_target!")
            self.mu_id = torch.log(bulk_density)
            self.mu_ex = self.eos.chemical_potential(bulk_density) 
            self.mu = (self.mu_id+self.mu_ex).to(device=self.device) 
            self.rhob = torch.as_tensor(bulk_density).to(device=self.device)

        self.rho = self.rho.detach().clone()
        self.rho[self.excluded] = 1e-16

        if solver == 'picard':
            picard(self,alpha0,tol,max_it,logoutput)

        elif solver == 'picard_ls':
            picard_line_search(self,alpha0,tol,max_it,logoutput)

        elif solver == 'fire':
            fire(self,alpha0,dt,tol,max_it,logoutput)

        elif solver == 'anderson':
            anderson(self,anderson_mmax,anderson_damping,tol,max_it,logoutput)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.error = self.error.cpu()

        self.total_molecules = (self.rho*self.valid).sum().cpu()*self.cell_size
        Phi_id = self.rho*(torch.log(self.rho)-1.0)
        self.F_id = Phi_id.sum()*self.cell_size
        self.F_ext = (self.rho*self.Vext).sum()*self.cell_size
 
        self.F_intr = self.F_id+self.F_ex.detach()
        self.F = self.F_intr+self.F_ext
        self.Omega = self.F-self.mu*self.total_molecules.to(self.F.device)
 
        del Phi_id
        self.N_target = None