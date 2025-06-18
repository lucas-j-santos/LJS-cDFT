import numpy as np
import torch
from torch.fft import fft, ifft
from torch.autograd import grad
from scipy.special import spherical_jn
from .lj_eos import lj_eos
from .solver import *

kB = 1.380649e-23
NA = 6.02214076e23

pi = np.pi
torch.set_default_dtype(torch.float64)

def lancsoz(k,M):
    return np.sinc(k/M)

def yukawa_ft(k,sigma,epsilon,l):
    
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

        self.kB = 1.380649e-23
        self.NA = 6.02214076e23
        
        self.d = self.sigma*(1.0+0.2977*self.Tstar)/(1.0+0.33163*self.Tstar+0.0010477*self.Tstar**2)
        self.R = 0.5*self.d
        
        self.cell_size = system_size/points
        self.z = torch.linspace(0.5*self.cell_size, system_size-0.5*self.cell_size, points, device=device)

        kz = np.fft.fftfreq(points, d=self.cell_size)
        kcut = kz.max()
        k = np.abs(kz) 

        # Precompute common terms
        two_pi_R_K = 2.0*pi*self.R*k
        four_pi_R_K = 2.0*two_pi_R_K
        lanczos_term = lancsoz(kz, kcut)

        w2_hat = np.empty(points,dtype=np.complex128)
        w3_hat = np.empty_like(w2_hat)
        w2vec_hat = np.empty_like(w2_hat)
        watt_hat = np.empty_like(w2_hat)
        ulj_hat = np.empty_like(w2_hat)
        
        w2_hat = 4.0*pi*self.R**2*spherical_jn(0, two_pi_R_K)*lanczos_term
        w3_hat = (4./3.)*pi*self.R**3*(spherical_jn(0, two_pi_R_K)+spherical_jn(2, two_pi_R_K)) \
            *lanczos_term
        w2vec_hat = -1j*2.0*pi*kz*w3_hat
        watt_hat = (spherical_jn(0, four_pi_R_K)+spherical_jn(2, four_pi_R_K))*lanczos_term

        l = np.array([2.544944560171334,15.464088962136243])
        eps = 1.857708161877173*self.epsilon*np.array([1,-1])
        ulj_hat = (yukawa_ft(k,self.sigma,eps[0],l[0])+yukawa_ft(k,self.sigma,eps[1],l[1]))*lanczos_term

        self.w2_hat = torch.tensor(w2_hat,device=device)
        self.w3_hat = torch.tensor(w3_hat,device=device)
        self.w2vec_hat = torch.tensor(w2vec_hat,device=device)
        self.watt_hat = torch.tensor(watt_hat,device=device)
        self.ulj_hat = torch.tensor(ulj_hat,device=device) 

        # Clear temporary arrays to free memory
        del kz,k,two_pi_R_K,four_pi_R_K,lanczos_term

    def weighted_densities(self):

        self.rho.requires_grad = True

        self.rho_hat = fft(self.rho)
        self.n2 = ifft(self.rho_hat*self.w2_hat).real
        self.n0 = self.n2/(4.*pi*self.R**2)
        self.n1 = self.n2/(4.*pi*self.R)
        self.n3 = ifft(self.rho_hat*self.w3_hat).real.clamp_(max=1.0-1e-16)
        self.n2vec = ifft(self.rho_hat*self.w2vec_hat).real
        self.n1vec = self.n2vec/(4.*pi*self.R)
        self.rhobar = ifft(self.rho_hat*self.watt_hat).real
        self.ulj = ifft(self.rho_hat*self.ulj_hat).real

    def functional(self,fmt):

        self.weighted_densities()

        # Hard-Sphere Contribution 
        one_minus_n3 = 1.0-self.n3
        one_minus_n3_sq = one_minus_n3**2
        f1 = -torch.log(one_minus_n3)
        f2 = one_minus_n3.pow(-1)
        f4 = (self.n3+one_minus_n3_sq*torch.log(one_minus_n3))/(36.0*pi*self.n3**2*one_minus_n3_sq)

        del one_minus_n3, one_minus_n3_sq

        # Small n3 approximation
        mask = self.n3 <= 1e-4
        f4[mask] = 1/(24*pi) + 2/(27*pi)*self.n3[mask]+(5/48*pi)*self.n3[mask]**2

        del mask

        if fmt == 'WB':

            n1_n2 = self.n1*self.n2
            n1vec_n2vec = self.n1vec*self.n2vec
            n2_sq = self.n2**2
            n2vec_sq = self.n2vec*self.n2vec

            n1vec_n2vec.clamp(max=n1_n2)
            n2vec_sq.clamp(max=n2_sq)
            
            self.Phi_hs = f1*self.n0+f2*(n1_n2-n1vec_n2vec)+f4*(n2_sq*self.n2-3.0*self.n2*n2vec_sq) 

            del f1, f2, f4, n1_n2, n1vec_n2vec, n2_sq, n2vec_sq
            
        elif fmt == 'ASWB':

            n1_n2 = self.n1*self.n2
            n1vec_n2vec = self.n1vec*self.n2vec
            n2_sq = self.n2**2
            n2vec_sq = self.n2vec*self.n2vec
            xi = n2vec_sq/n2_sq

            n1vec_n2vec.clamp(max=n1_n2)
            n2vec_sq.clamp(max=n2_sq)
            xi.clamp_(max=1.0-1e-16)
            
            self.Phi_hs = f1*self.n0+f2*(n1_n2-n1vec_n2vec)+f4*self.n2**3*(1.0-xi)**3

            del f1, f2, f4, n1_n2, n1vec_n2vec, n2_sq, n2vec_sq, xi

        self.Fhs = self.Phi_hs.sum()*self.cell_size

        # Attractive Contribution
        eta = self.rhobar*pi*self.d**3/6
        eos_term = self.eos.helmholtz_energy(self.rhobar)
        correction_term_hs = (4.0*eta-3.0*eta**2)/((1.0-eta)**2)
        correction_term_mfa = (16./9.)*pi*(self.epsilon/self.T)*self.sigma**3*self.rhobar

        self.Phi_cor = eos_term-correction_term_hs+correction_term_mfa
        self.Phi_mfa = 0.5*self.rho*self.ulj/self.T
        self.Phi_att = self.rhobar*self.Phi_cor+self.Phi_mfa
        self.Fatt = self.Phi_att.sum()*self.cell_size 

        del eta, eos_term, correction_term_hs, correction_term_mfa 

        self.Fres = self.Fhs+self.Fatt

    def functional_derivative(self, fmt):

        self.functional(fmt)
        self.dFres = grad(self.Fres, self.rho)[0]
        self.dFres = self.dFres.detach()/self.cell_size

        self.rho.requires_grad=False

    def euler_lagrange(self, lnrho, fmt='WB'):
        
        self.functional_derivative(fmt)
        self.res = torch.empty_like(self.rho)
        self.res[self.valid] = self.mu-self.dFres[self.valid]-self.Vext[self.valid]-lnrho[self.valid]

    def loss(self):
        return torch.norm(self.res[self.valid])/np.sqrt(self.points)
    
    def initial_condition(self, bulk_density, Vext, potential_cutoff=50.0, model='bulk'):
        
        self.rhob = bulk_density
        self.eos = lj_eos(self.parameters, self.T)
        self.mu = self.eos.chemical_potential(bulk_density)+torch.log(self.rhob)

        self.Vext = Vext/self.T
        self.excluded = self.Vext >= potential_cutoff
        self.valid = self.Vext < potential_cutoff
        self.Vext[self.excluded] = potential_cutoff

        self.rho = torch.empty(self.points,device=self.device)
        if model == 'bulk':
            self.rho[:] = self.rhob
        elif model == 'ideal':
            self.rho = self.rhob*torch.exp(-self.Vext)  

    def equilibrium_density_profile(self, bulk_density, fmt='WB', solver='anderson',
                                    alpha0=0.2, dt=0.1, anderson_mmax=10, anderson_damping=0.1, 
                                    tol=1e-6, max_it=1000, logoutput=False):
        
        self.rhob = bulk_density
        self.mu = self.eos.chemical_potential(bulk_density)+torch.log(self.rhob)
        self.fmt = fmt
        self.rho[self.excluded] = 1e-15

        if solver == 'picard': 
            picard(self,alpha0,tol,max_it,logoutput)

        elif solver == 'anderson':
            anderson(self,anderson_mmax,anderson_damping,tol,max_it,logoutput)

        elif solver == 'fire':
            fire(self,alpha0,dt,tol,max_it,logoutput)

        torch.cuda.empty_cache()
        self.error = self.error.cpu()

        self.total_molecules = self.rho[self.valid].cpu().sum()*self.cell_size
        Phi = self.rho*(torch.log(self.rho)-1.0)+self.rho*(self.Vext-self.mu)
        self.Omega = Phi.sum()*self.cell_size+self.Fres.detach()