import numpy as np
import time
from torch import tensor,pi,float64,complex128,log,exp,isnan
from torch import empty,empty_like,zeros,zeros_like,linspace,stack,einsum,norm,meshgrid,cuda
from torch.fft import fftn, ifftn
from torch.linalg import solve, inv
from torch.autograd import grad
from scipy.special import spherical_jn
from .lj_eos import lj_eos

kB = 1.380649e-23
NA = 6.02214076e23

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

        if angles is not None:
            self.alpha, self.beta, self.gamma = angles
            self.orthogonal = False
            
            cos_alpha = np.cos(self.alpha)
            cos_beta = np.cos(self.beta)
            cos_gamma = np.cos(self.gamma)
            sin_gamma = np.sin(self.gamma)
            
            zeta = (cos_alpha-cos_beta*cos_gamma)/sin_gamma
            
            self.H = tensor([
                [1.0, cos_gamma, cos_beta],
                [0.0, sin_gamma, zeta],
                [0.0, 0.0, np.sqrt(1.0-cos_beta**2-zeta**2)]
            ], device=device, dtype=float64)
            
            self.H_T = self.H.T
            self.H_inv_T = inv(self.H_T)
            self.det_H = sin_gamma*np.sqrt(1.0-cos_beta**2-zeta**2)
        else:
            self.orthogonal = True
            self.H = tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], device=device, dtype=float64)
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
        u = linspace(0.5*self.cell_size[0], system_size[0]-0.5*self.cell_size[0], points[0],device=device,dtype=float64)
        v = linspace(0.5*self.cell_size[1], system_size[1]-0.5*self.cell_size[1], points[1],device=device,dtype=float64)
        w = linspace(0.5*self.cell_size[2], system_size[2]-0.5*self.cell_size[2], points[2],device=device,dtype=float64)
        self.U, self.V, self.W = meshgrid(u, v, w, indexing='ij')

        # Transform to cartesian coordinates
        s = stack([self.U, self.V, self.W], dim=0)
        r = r = einsum('ij,j...->i...', self.H, s)
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
        kcut = np.array([Kx.max(), Ky.max(), Kz.max()])

        # Precompute common terms
        two_pi_R_K = 2.0*pi*self.R*K
        four_pi_R_K = 4.0 * pi * self.R * K
        lanczos_term = lancsoz(ku, kv, kw, kcut)

        w2_hat = np.empty((points[0],points[1],points[2]),dtype=np.complex128)
        w3_hat = np.empty_like(w2_hat)
        w2vec_hat = np.empty((3,points[0],points[1],points[2]),dtype=np.complex128)
        watt_hat = np.empty_like(w2_hat)
        ulj_hat = np.empty_like(w2_hat)
        
        # Weight functions in Fourier space
        w2_hat = self.four_pi_R_sq*spherical_jn(0,two_pi_R_K)*lanczos_term
        w3_hat = (4./3.)*pi*self.R_cu*(spherical_jn(0, two_pi_R_K)+spherical_jn(2,two_pi_R_K))*lanczos_term
        watt_hat = (spherical_jn(0, four_pi_R_K)+spherical_jn(2,four_pi_R_K))*lanczos_term
        w2vec_hat[0] = -1j*2.0*pi*Kx*w3_hat
        w2vec_hat[1] = -1j*2.0*pi*Ky*w3_hat
        w2vec_hat[2] = -1j*2.0*pi*Kz*w3_hat
        watt_hat = (spherical_jn(0, 4.0*pi*self.R*K)+spherical_jn(2, 4.0*pi*self.R*K))*lanczos_term

        l = np.array([2.544944560171334,15.464088962136243])
        eps = 1.857708161877173*self.epsilon*np.array([1,-1])
        ulj_hat = (yukawa_ft(K,self.sigma,eps[0],l[0])+yukawa_ft(K,self.sigma,eps[1],l[1]))*lanczos_term

        self.w2_hat = tensor(w2_hat,device=device,dtype=complex128)
        self.w3_hat = tensor(w3_hat,device=device,dtype=complex128)
        self.w2vec_hat = tensor(w2vec_hat,device=device,dtype=complex128)
        self.watt_hat = tensor(watt_hat,device=device,dtype=complex128)
        self.ulj_hat = tensor(ulj_hat,device=device,dtype=complex128) 

        # Clear temporary arrays to free memory
        del u,v,w,s,r,ku,kv,kw,Ku,Kv,Kw,Kx,Ky,Kz,K,two_pi_R_K,four_pi_R_K,lanczos_term
        cuda.empty_cache()

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
        f1 = -log(one_minus_n3)
        f2 = one_minus_n3.pow(-1)
        f4 = (self.n3+one_minus_n3_sq*log(one_minus_n3))/(36.0*pi*self.n3**2*one_minus_n3_sq)

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
            
            self.Phi_hs = f1*self.n0+f2*(n1_n2-n1vec_n2vec)+f4*(n2_sq*self.n2-3.0*self.n2*n2vec_sq) 

            del n1_n2, n1vec_n2vec, n2_sq, n2vec_sq
            
        elif fmt == 'ASWB':

            n1_n2 = self.n1*self.n2
            n1vec_n2vec = (self.n1vec*self.n2vec).sum(dim=0)
            n2_sq = self.n2**2
            n2vec_sq = (self.n2vec*self.n2vec).sum(dim=0)

            xi = n2vec_sq/n2_sq
            xi.clamp_(max=1.0)
            
            self.Phi_hs = f1*self.n0+f2*(n1_n2-n1vec_n2vec)+f4*self.n2**3*(1.0-xi)**3

            del n1_n2, n1vec_n2vec, n2_sq, n2vec_sq, xi

        self.Fhs = self.Phi_hs.sum() 

        # Attractive Contribution
        eta = self.rhobar*pi*self.d**3/6
        eos_term = self.eos.helmholtz_energy(self.rhobar)
        correction_term_hs = (4.0*eta-3.0*eta**2)/((1.0-eta)**2)
        correction_term_mfa = (16./9.)*pi*(self.epsilon/self.T)*self.sigma**3*self.rhobar

        self.Phi_cor = eos_term-correction_term_hs+correction_term_mfa
        self.Phi_mfa = 0.5*self.rho*self.ulj/self.T
        self.Phi_att = self.rhobar*self.Phi_cor+self.Phi_mfa
        self.Fatt = self.Phi_att.sum() 

        del eta, eos_term, correction_term_hs, correction_term_mfa 

        self.Fres = self.Fhs+self.Fatt

    def functional_derivative(self, fmt):

        self.functional(fmt)
        self.dFres = grad(self.Fres, self.rho)[0]
        self.dFres = self.dFres.detach()

        self.rho.requires_grad=False
    
    def initial_condition(self, bulk_density, Vext, potential_cutoff=50.0):
        
        self.rhob = bulk_density
        self.eos = lj_eos(self.parameters, self.T)
        self.mu = self.eos.chemical_potential(bulk_density)+log(self.rhob)

        self.Vext = Vext/self.T
        self.excluded = self.Vext >= potential_cutoff
        self.valid = self.Vext < potential_cutoff
        self.Vext[self.excluded] = potential_cutoff

        self.rho = empty((self.points[0],self.points[1],self.points[2]),device=self.device,dtype=float64)
        # self.rho = self.rhob*exp(-0.01*self.Vext)
        self.rho[:] = self.rhob

    def equilibrium_density_profile(self, bulk_density, fmt='ASWB', solver='fire',
                                    alpha0=0.2, dt=0.1, anderson_mmax=5, anderson_damping=0.1, 
                                    tol=1e-6, max_it=1000, logoutput=False):
        
        self.rhob = bulk_density
        self.mu = self.eos.chemical_potential(bulk_density)+log(self.rhob)

        self.rho[self.excluded] = 1e-15
        lnrho = log(self.rho)
        
        F = empty_like(self.rho)
        self.functional_derivative(fmt)
        F[self.valid] = self.mu-self.dFres[self.valid]-self.Vext[self.valid]-lnrho[self.valid]
        self.points_sqrt = np.sqrt(self.points.prod())
        error = norm(F[self.valid])/self.points_sqrt

        if solver == 'picard':

            alpha = alpha0
            self.it = 0
            tic = time.process_time()
            for i in range(max_it):
                lnrho[self.valid] += alpha*F[self.valid]
                self.rho[self.valid] = exp(lnrho[self.valid])
                self.functional_derivative(fmt) 
                F[self.valid] = self.mu-self.dFres[self.valid]-self.Vext[self.valid]-lnrho[self.valid]
                error = norm(F[self.valid])/self.points_sqrt
                self.it += 1
                if error < tol: break
                if isnan(error): break
                if logoutput: print(self.it, error)
            toc = time.process_time()
            self.process_time = toc-tic

        elif solver == 'fire':

            alpha = alpha0
            Ndelay = 20
            Nnegmax = 2000
            dtmax = 10*dt
            dtmin = 0.02*dt
            Npos = 0
            Nneg = 0
            finc = 1.1
            fdec = 0.5
            fa = 0.99
            V = zeros_like(self.rho)
            self.it = 0
            tic = time.process_time()

            for i in range(max_it):

                P = (F[self.valid]*V[self.valid]).sum() 
                if (P > 0):
                    Npos = Npos+1
                    if Npos > Ndelay:
                        dt = min(dt*finc,dtmax)
                        alpha = alpha*fa
                else:
                    Npos = 0
                    Nneg = Nneg+1
                    if Nneg > Nnegmax: break
                    if i > Ndelay:
                        if dt*fdec >= dtmin:
                            dt *= fdec
                        alpha = alpha0
                    lnrho[self.valid] -= 0.5*dt*V[self.valid]
                    V[self.valid] = 0.0
                    self.rho[self.valid] = exp(lnrho[self.valid])
                    self.functional_derivative(fmt) 

                V[self.valid] += 0.5*dt*F[self.valid]
                V[self.valid] = (1.0-alpha)*V[self.valid]+alpha*F[self.valid]*norm(V[self.valid])/norm(F[self.valid])
                lnrho[self.valid] += dt*V[self.valid]
                self.rho[self.valid] = exp(lnrho[self.valid])
                self.functional_derivative(fmt)
                F[self.valid] = self.mu-self.dFres[self.valid]-self.Vext[self.valid]-lnrho[self.valid]
                V[self.valid] += 0.5*dt*F[self.valid]

                error = norm(F[self.valid])/self.points_sqrt
                self.it += 1
                if error < tol: break
                if isnan(error): break
                if logoutput: print(self.it, error)

            toc = time.process_time()
            self.process_time = toc-tic
            
            del V

        elif solver == 'anderson':

            # Anderson Mixing parameters
            mmax = anderson_mmax  # Number of previous iterations to store
            damping = anderson_damping  # Damping coefficient

            # Initialize history buffers
            resm = []  # Residual history
            rhom = []  # Solution history

            self.it = 0
            tic = time.process_time()

            for i in range(max_it):

                # Calculate residual
                self.functional_derivative(fmt)
                F[self.valid] = self.mu-self.dFres[self.valid]-self.Vext[self.valid]-lnrho[self.valid]
                error = norm(F[self.valid])/self.points_sqrt

                # Check for convergence
                if error < tol or isnan(error):
                    break

                # Store residual and solution
                resm.append(F[self.valid].clone())
                rhom.append(lnrho[self.valid].clone())

                # Drop old values if history is full
                if len(resm) > mmax:
                    resm.pop(0)
                    rhom.pop(0)

                m = len(resm)  # Current history size

                # Build the Anderson matrix and vector
                R = zeros((m+1, m+1), device=self.device, dtype=float64)
                anderson_alpha = zeros(m+1, device=self.device, dtype=float64)
                
                if m > 0:
                    resm_tensor = stack(resm)  # Shape: (m, *points)
                    R[:m, :m] = einsum('ik,jk->ij', resm_tensor.view(m,-1), resm_tensor.view(m,-1))
                    R[:m, m] = 1.0
                    R[m, :m] = 1.0
                R[m, m] = 0.0

                anderson_alpha[m] = 1.0

                # Solve for alpha coefficients
                try:
                    anderson_alpha = solve(R, anderson_alpha)
                except:
                    # Fallback to Picard if matrix is singular
                    anderson_alpha = zeros(m+1, device=self.device, dtype=float64)
                    anderson_alpha[m] = 1.0

                # Update solution using Anderson mixing
                lnrho[self.valid] = einsum('i,i...->...', anderson_alpha[:m], (stack(rhom[:m])+damping*stack(resm[:m])))
                self.rho[self.valid] = exp(lnrho[self.valid])
                self.it += 1

                if logoutput:
                    print(self.it, error)

            toc = time.process_time()
            self.process_time = toc-tic

        del F

        cuda.empty_cache()
        self.error = error.cpu()
        Phi = zeros_like(self.Phi_att)

        self.total_molecules = self.rho[self.valid].cpu().sum()*self.cell_volume
        Phi = self.rho*(log(self.rho)-1.0)+self.rho*(self.Vext-(log(self.rhob)+self.mu))
        self.Omega = (Phi.sum()+self.Fres.detach())*self.cell_volume