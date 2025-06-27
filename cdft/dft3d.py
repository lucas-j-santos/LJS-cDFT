import numpy as np
import torch
from torch.fft import fftn, ifftn
from torch.linalg import inv
from torch.autograd import grad
from scipy.special import spherical_jn
from .lj_eos import lj_eos
from .solver import *

torch.set_default_dtype(torch.float64)
pi = np.pi

def lancsoz(kx,ky,kz,M):
    return np.sinc(kx/M[0])*np.sinc(ky/M[1])*np.sinc(kz/M[2])

def yukawa_ft(k,sigma,epsilon,l):
    
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
            self.H_inv_T = inv(self.H_T)
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
        u = torch.linspace(0.5*self.cell_size[0], system_size[0]-0.5*self.cell_size[0], points[0], device=device)
        v = torch.linspace(0.5*self.cell_size[1], system_size[1]-0.5*self.cell_size[1], points[1], device=device)
        w = torch.linspace(0.5*self.cell_size[2], system_size[2]-0.5*self.cell_size[2], points[2], device=device)
        self.U, self.V, self.W = torch.meshgrid(u, v, w, indexing='ij')

        # Transform to cartesian coordinates
        s = torch.stack([self.U, self.V, self.W], dim=0)
        r = r = torch.einsum('ij,j...->i...', self.H, s)
        self.X, self.Y, self.Z = r[0], r[1], r[2]

        # Frequency grid in skewed coordinates
        ku = np.fft.fftfreq(points[0], d=self.cell_size[0])
        kv = np.fft.fftfreq(points[1], d=self.cell_size[1])
        kw = np.fft.fftfreq(points[2], d=self.cell_size[2])

        # Transform to cartesian frequency space
        Ku, Kv, Kw = np.meshgrid(ku, kv, kw, indexing='ij')
        
        if self.orthogonal:
            Kx = Ku
            Ky = Kv
            Kz = Kw
            del self.U, self.V, self.W 
        else:
            Kx = Ku
            Ky = (Kv-Ku*np.cos(self.gamma))/np.sin(self.gamma)
            Kz = (Ku*(zeta*np.cos(self.gamma)/np.sin(self.gamma)-np.cos(self.beta))\
                  -Kv*zeta/np.sin(self.gamma)+Kw)/np.sqrt(1.0-np.cos(self.beta)**2-zeta**2)
        
        K = np.sqrt(Kx**2+Ky**2+Kz**2)
        # kcut = np.array([Kx.max(), Ky.max(), Kz.max()])
        kcut = (0.5*self.points+1)/self.system_size

        # Precompute common terms
        two_pi_R_K = 2.0*pi*self.R*K
        four_pi_R_K = 2.0*two_pi_R_K
        lanczos_term = lancsoz(ku, kv, kw, kcut)

        w2_hat = np.empty((points[0],points[1],points[2]), dtype=np.complex128)
        w3_hat = np.empty_like(w2_hat)
        w2vec_hat = np.empty((3,points[0],points[1],points[2]), dtype=np.complex128)
        watt_hat = np.empty_like(w2_hat)
        ulj_hat = np.empty_like(w2_hat)
        
        # Weight functions in Fourier space
        w2_hat = self.four_pi_R_sq*spherical_jn(0,two_pi_R_K)*lanczos_term
        w3_hat = (4./3.)*pi*self.R_cu*(spherical_jn(0, two_pi_R_K)+spherical_jn(2,two_pi_R_K))*lanczos_term
        watt_hat = (spherical_jn(0, four_pi_R_K)+spherical_jn(2,four_pi_R_K))*lanczos_term
        w2vec_hat[0] = -1j*2.0*pi*Kx*w3_hat
        w2vec_hat[1] = -1j*2.0*pi*Ky*w3_hat
        w2vec_hat[2] = -1j*2.0*pi*Kz*w3_hat

        l = np.array([2.544944560171334,15.464088962136243])
        eps = 1.857708161877173*self.epsilon*np.array([1,-1])
        ulj_hat = (yukawa_ft(K,self.sigma,eps[0],l[0])+yukawa_ft(K,self.sigma,eps[1],l[1]))*lanczos_term

        self.w2_hat = torch.tensor(w2_hat,device=device)
        self.w3_hat = torch.tensor(w3_hat,device=device)
        self.w2vec_hat = torch.tensor(w2vec_hat,device=device)
        self.watt_hat = torch.tensor(watt_hat,device=device)
        self.ulj_hat = torch.tensor(ulj_hat,device=device) 

        # Clear temporary arrays to free memory
        del u,v,w,s,r,ku,kv,kw,Ku,Kv,Kw,Kx,Ky,Kz,K,two_pi_R_K,four_pi_R_K,lanczos_term
        torch.cuda.empty_cache()

    def weighted_densities(self):

        self.rho.requires_grad = True

        self.rho_hat = fftn(self.rho)
        self.n2 = ifftn(self.rho_hat*self.w2_hat).real
        self.n0 = self.n2/(4.*pi*self.R**2)
        self.n1 = self.n2/(4.*pi*self.R)
        self.n3 = ifftn(self.rho_hat*self.w3_hat).real.clamp_(max=1.0-1e-16)
        self.n2vec = ifftn(self.rho_hat*self.w2vec_hat, dim=(1,2,3)).real
        self.n1vec = self.n2vec/(4.*pi*self.R)
        self.rhobar = ifftn(self.rho_hat*self.watt_hat).real
        self.ulj = ifftn(self.rho_hat*self.ulj_hat).real

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
            n1vec_n2vec = (self.n1vec*self.n2vec).sum(dim=0)
            n2_sq = self.n2**2
            n2vec_sq = (self.n2vec*self.n2vec).sum(dim=0)

            n1vec_n2vec.clamp(max=n1_n2)
            n2vec_sq.clamp(max=n2_sq)
            
            self.Phi_hs = f1*self.n0+f2*(n1_n2-n1vec_n2vec)+f4*(n2_sq*self.n2-3.0*self.n2*n2vec_sq) 

            del f1, f2, f4, n1_n2, n1vec_n2vec, n2_sq, n2vec_sq
            
        elif fmt == 'ASWB':

            n1_n2 = self.n1*self.n2
            n1vec_n2vec = (self.n1vec*self.n2vec).sum(dim=0)
            n2_sq = self.n2**2
            n2vec_sq = (self.n2vec*self.n2vec).sum(dim=0)
            xi = n2vec_sq/n2_sq

            n1vec_n2vec.clamp(max=n1_n2)
            n2vec_sq.clamp(max=n2_sq)
            xi.clamp_(max=1.0-1e-16)
            
            self.Phi_hs = f1*self.n0+f2*(n1_n2-n1vec_n2vec)+f4*self.n2**3*(1.0-xi)**3

            del f1, f2, f4, n1_n2, n1vec_n2vec, n2_sq, n2vec_sq, xi

        self.Fhs = self.Phi_hs.sum()*self.cell_volume

        # Attractive Contribution
        eta = self.rhobar*pi*self.d**3/6
        eta.clamp_(max=1.0-1e-16)
        eos_term = self.eos.helmholtz_energy(self.rhobar)
        correction_term_hs = (4.0*eta-3.0*eta**2)/((1.0-eta)**2)
        correction_term_mfa = (16./9.)*pi*(self.epsilon/self.T)*self.sigma**3*self.rhobar

        self.Phi_cor = eos_term-correction_term_hs+correction_term_mfa
        self.Phi_mfa = 0.5*self.rho*self.ulj/self.T
        self.Phi_att = self.rhobar*self.Phi_cor+self.Phi_mfa
        self.Fatt = self.Phi_att.sum()*self.cell_volume

        del eta, eos_term, correction_term_hs, correction_term_mfa 

        self.Fres = self.Fhs+self.Fatt

    def functional_derivative(self, fmt):

        self.functional(fmt)
        self.dFres = grad(self.Fres, self.rho)[0]
        self.dFres = self.dFres.detach()/self.cell_volume

        self.rho.requires_grad=False

    def euler_lagrange(self, lnrho, fmt='ASWB'):
        
        self.functional_derivative(fmt)
        self.res = torch.empty_like(self.rho)
        self.res[self.valid] = self.mu-self.dFres[self.valid]-self.Vext[self.valid]-lnrho[self.valid]

    def loss(self):
        return torch.norm(self.res[self.valid])/np.sqrt(self.points.prod())
    
    def initial_condition(self, bulk_density, Vext, potential_cutoff=50.0, model='bulk'):
        
        self.rhob = bulk_density
        self.eos = lj_eos(self.parameters, self.T)
        self.mu = self.eos.chemical_potential(bulk_density)+torch.log(self.rhob)

        self.Vext = Vext/self.T
        self.excluded = self.Vext >= potential_cutoff
        self.valid = self.Vext < potential_cutoff
        self.Vext[self.excluded] = potential_cutoff

        self.rho = torch.empty((self.points[0],self.points[1],self.points[2]),device=self.device)
        if model == 'bulk':
            self.rho[:] = self.rhob
        elif model == 'ideal':
            self.rho = self.rhob*torch.exp(-self.Vext)   
    
    def equilibrium_density_profile(self, bulk_density, fmt='ASWB', solver='anderson',
                                    alpha0=0.2, dt=0.1, anderson_mmax=10, anderson_damping=0.1, 
                                    tol=1e-6, max_it=1000, logoutput=False):
        
        self.rhob = bulk_density
        self.mu = self.eos.chemical_potential(bulk_density)+torch.log(self.rhob)
        self.fmt = fmt
        self.rho[self.excluded] = 1e-15

        if solver == 'picard': 
            picard(self,alpha0,tol,max_it,logoutput)

        elif solver == 'picard_ls': 
            picard_line_search(self,alpha0,tol,max_it,logoutput)

        elif solver == 'anderson':
            anderson(self,anderson_mmax,anderson_damping,tol,max_it,logoutput)

        elif solver == 'fire':
            fire(self,alpha0,dt,tol,max_it,logoutput)

        torch.cuda.empty_cache()
        self.error = self.error.cpu()

        self.total_molecules = self.rho[self.valid].cpu().sum()*self.cell_volume
        Phi = self.rho*(torch.log(self.rho)-1.0)+self.rho*(self.Vext-self.mu)
        self.Omega = Phi.sum()*self.cell_volume+self.Fres.detach()