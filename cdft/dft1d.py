import numpy as np
import time
from torch import tensor,pi,float64,complex128,log,exp,isnan
from torch import empty,empty_like,zeros,zeros_like,linspace,stack,einsum,norm,cuda
from torch.fft import fft, ifft
from torch.linalg import solve
from torch.autograd import grad
from scipy.special import spherical_jn
from .lj_eos import lj_eos

kB = 1.380649e-23
NA = 6.02214076e23

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
        
        self.d = self.sigma*(1.0+0.2977*self.Tstar)/(1.0+0.33163*self.Tstar+0.0010477*self.Tstar**2)
        self.R = 0.5*self.d

        self.cell_size = system_size/points

        self.z = linspace(0.5*self.cell_size, system_size-0.5*self.cell_size, points, device=device, dtype=float64)

        kz = np.fft.fftfreq(points, d=self.cell_size)
        kcut = kz.max()
        k = np.abs(kz) 

        w2_hat = np.empty(points,dtype=np.complex128)
        w3_hat = np.empty_like(w2_hat)
        w2vec_hat = np.empty_like(w2_hat)
        watt_hat = np.empty_like(w2_hat)
        ulj_hat = np.empty_like(w2_hat)
        
        w2_hat = 4.0*pi*self.R**2*spherical_jn(0, 2.0*pi*self.R*k)*lancsoz(kz,kcut)
        w3_hat = (4./3.)*pi*self.R**3*(spherical_jn(0, 2.0*pi*self.R*k)+spherical_jn(2, 2.0*pi*self.R*k)) \
            *lancsoz(kz,kcut)
        w2vec_hat = -1j*2.0*pi*kz*w3_hat
        watt_hat = (spherical_jn(0, 4.0*pi*self.R*k)+spherical_jn(2, 4.0*pi*self.R*k))*lancsoz(kz,kcut)

        l = np.array([2.544944560171334,15.464088962136243])
        eps = 1.857708161877173*self.epsilon*np.array([1,-1])
        ulj_hat = (yukawa_ft(k,self.sigma,eps[0],l[0])+yukawa_ft(k,self.sigma,eps[1],l[1]))*lancsoz(kz,kcut)

        self.w2_hat = tensor(w2_hat,device=device,dtype=complex128)
        self.w3_hat = tensor(w3_hat,device=device,dtype=complex128)
        self.w2vec_hat = tensor(w2vec_hat,device=device,dtype=complex128)
        self.watt_hat = tensor(watt_hat,device=device,dtype=complex128)
        self.ulj_hat = tensor(ulj_hat,device=device,dtype=complex128) 

        del kz, k

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

    def line_search(self, lnrho, delta_lnrho, res0, fmt):
        alpha = 2.0 
        # Reduce step until a feasible solution is found
        for _ in range(8):
            alpha *= 0.5
  
            # Calculate full step
            lnrho_new = lnrho.clone()
            lnrho_new[self.valid] += alpha * delta_lnrho[self.valid]
            rho_new = exp(lnrho_new)
            
            try:
                # Calculate new residual
                self.rho = rho_new
                self.functional_derivative(fmt)
                EL_new = self.mu-self.dFres[self.valid]-self.Vext[self.valid]-lnrho_new[self.valid]
                res2 = norm(EL_new)/self.points_sqrt
                
                if res2 > res0:
                    continue
                    
                # Calculate intermediate step
                lnrho_new = lnrho.clone()
                lnrho_new[self.valid] += 0.5 * alpha * delta_lnrho[self.valid]
                rho_new = exp(lnrho_new)
                
                self.rho = rho_new
                self.functional_derivative(fmt)
                EL_new = self.mu-self.dFres[self.valid]-self.Vext[self.valid]-lnrho_new[self.valid]
                res1 = norm(EL_new)/self.points_sqrt
                
                # Estimate optimal alpha
                denominator = res2 - 2.0 * res1 + res0
                if abs(denominator) > 1e-16:
                    alpha_opt = alpha * 0.25 * (res2 - 4.0 * res1 + 3.0 * res0) / denominator
                else:
                    continue
                    
                # Prohibit negative steps
                if alpha_opt <= 0.0:
                    alpha_opt = 0.5 * alpha if res1 < res2 else alpha
                    
                # Prohibit too large steps
                if alpha_opt > alpha:
                    alpha_opt = alpha
                    
                alpha = alpha_opt
                break
                
            except:
                continue
                
        return alpha
    
    def initial_condition(self, bulk_density, Vext, potential_cutoff=50.0):
        
        self.rhob = bulk_density
        self.eos = lj_eos(self.parameters, self.T)
        self.mu = self.eos.chemical_potential(bulk_density)+log(self.rhob)

        self.Vext = Vext/self.T
        self.excluded = self.Vext >= potential_cutoff
        self.valid = self.Vext < potential_cutoff
        self.Vext[self.excluded] = potential_cutoff

        self.rho = empty(self.points,device=self.device,dtype=float64)
        # self.rho = self.rhob*exp(-0.01*self.Vext)
        self.rho[:] = self.rhob

    def equilibrium_density_profile(self, bulk_density, fmt='ASWB', solver='anderson',
                                    alpha0=0.2, dt=0.1, anderson_mmax=10, anderson_damping=0.1, 
                                    tol=1e-6, max_it=1000, logoutput=False):
        
        self.rhob = bulk_density
        self.mu = self.eos.chemical_potential(bulk_density)+log(self.rhob)

        self.rho[self.excluded] = 1e-15
        lnrho = log(self.rho)
        
        EL = empty_like(self.rho)
        self.functional_derivative(fmt)
        EL[self.valid] = self.mu-self.dFres[self.valid]-self.Vext[self.valid]-lnrho[self.valid]
        self.points_sqrt = np.sqrt(self.points)
        error = norm(EL[self.valid])/self.points_sqrt

        if solver == 'picard':

            alpha = alpha0
            self.it = 0
            tic = time.process_time()
            for i in range(max_it):
                lnrho[self.valid] += alpha*EL[self.valid]
                self.rho[self.valid] = exp(lnrho[self.valid])
                self.functional_derivative(fmt) 
                EL[self.valid] = self.mu-self.dFres[self.valid]-self.Vext[self.valid]-lnrho[self.valid]
                error = norm(EL[self.valid])/self.points_sqrt
                self.it += 1
                if error < tol: break
                if isnan(error): break
                if logoutput: print(self.it, error)
            toc = time.process_time()
            self.process_time = toc-tic

        elif solver == 'picard_ls': 
            
            self.it = 0
            tic = time.process_time()
            
            for k in range(max_it):
                # Calculate residual
                self.functional_derivative(fmt)
                EL[self.valid] = self.mu-self.dFres[self.valid]-self.Vext[self.valid]-lnrho[self.valid]
                error = norm(EL[self.valid])/self.points_sqrt   
                if logoutput:
                    print(f"Picard (line search) | {k:>4} | {time.process_time()-tic:7.3f} | {error:.6e}")       
                if error < tol: break
                if isnan(error): break    
                # Line search
                alpha = self.line_search(lnrho, EL, error, fmt)
                # Update solution
                lnrho[self.valid] += alpha * EL[self.valid]
                self.rho[self.valid] = exp(lnrho[self.valid])
                self.it += 1
                
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

                P = (EL[self.valid]*V[self.valid]).sum() 
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

                V[self.valid] += 0.5*dt*EL[self.valid]
                V[self.valid] = (1.0-alpha)*V[self.valid]+alpha*EL[self.valid]*norm(V[self.valid])/norm(EL[self.valid])
                lnrho[self.valid] += dt*V[self.valid]
                self.rho[self.valid] = exp(lnrho[self.valid])
                self.functional_derivative(fmt)
                EL[self.valid] = self.mu-self.dFres[self.valid]-self.Vext[self.valid]-lnrho[self.valid]
                V[self.valid] += 0.5*dt*EL[self.valid]

                error = norm(EL[self.valid])/self.points_sqrt
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
                EL[self.valid] = self.mu-self.dFres[self.valid]-self.Vext[self.valid]-lnrho[self.valid]
                error = norm(EL[self.valid])/self.points_sqrt

                # Check for convergence
                if error < tol or isnan(error):
                    break

                # Store residual and solution
                resm.append(EL[self.valid].clone())
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

        del EL

        cuda.empty_cache()
        self.error = error.cpu()

        self.total_molecules = self.rho[self.valid].cpu().sum()*self.cell_size
        Phi = self.rho*(log(self.rho)-1.0)+self.rho*(self.Vext-self.mu)
        self.Omega = (Phi.sum()+self.Fres.detach())*self.cell_size