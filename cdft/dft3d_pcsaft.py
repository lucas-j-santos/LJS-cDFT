import numpy as np
import time
from torch import tensor,pi,float64,complex128,log,exp,isnan
from torch import empty,empty_like,zeros,zeros_like,stack,linspace,meshgrid,einsum,norm,cuda
from torch.fft import fftn, ifftn
from torch.linalg import solve
from torch.autograd import grad
from scipy.special import spherical_jn
from .pcsaft_eos import pcsaft

kB = 1.380649e-23
NA = 6.02214076e23
psi = 1.3862

a = tensor([
    [0.91056314451539, -0.30840169182720, -0.09061483509767],
    [0.63612814494991, 0.18605311591713, 0.45278428063920],
    [2.68613478913903, -2.50300472586548, 0.59627007280101],
    [-26.5473624914884, 21.4197936296668, -1.72418291311787],
    [97.7592087835073, -65.2558853303492, -4.13021125311661],
    [-159.591540865600, 83.3186804808856, 13.7766318697211],
    [91.2977740839123, -33.7469229297323, -8.67284703679646],
], dtype=float64)

b = tensor([
    [0.72409469413165, -0.57554980753450, 0.09768831158356],
    [2.23827918609380, 0.69950955214436, -0.25575749816100],
    [-4.00258494846342, 3.89256733895307, -9.15585615297321],
    [-21.00357681484648, -17.21547164777212, 20.64207597439724],
    [26.8556413626615, 192.6722644652495, -38.80443005206285],
    [206.5513384066188, -161.8264616487648, 93.6267740770146],
    [-355.60235612207947, -165.2076934555607, -29.66690558514725],
], dtype=float64)

aq = tensor([
    [1.237830788, 1.285410878, 1.794295401],
    [2.435503144, -11.46561451, 0.769510293],
    [1.633090469, 22.08689285, 7.264792255],
    [-1.611815241, 7.46913832, 94.48669892],
    [6.977118504, -17.19777208, -77.1484579]
], dtype=float64)

bq = tensor([
    [0.454271755, -0.813734006, 6.868267516],
    [-4.501626435, 10.06402986, -5.173223765],
    [3.585886783, -10.87663092, -17.2402066],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
], dtype=float64)

cq = tensor([
    [-0.500043713, 2.000209381, 3.135827145],
    [6.531869153, -6.78386584, 7.247588801],
    [-16.01477983, 20.38324603, 3.075947834],
    [14.42597018, -10.89598394, 0.0],
    [0.0, 0.0, 0.0]
], dtype=float64)

def lancsoz(kx,ky,kz,M):
    return np.sinc(kx/M[0])*np.sinc(ky/M[1])*np.sinc(kz/M[2])

class dft_core():

    def __init__(self, pcsaft_parameters, temperature, system_size, points, device):

        self.pcsaft_parameters = pcsaft_parameters
        self.m = self.pcsaft_parameters['m']
        self.sigma = self.pcsaft_parameters['sigma']
        self.epsilon = self.pcsaft_parameters['epsilon']
        try: self.q = pcsaft_parameters['q']
        except: self.q = None
        self.Nc = len(self.m)
        self.T = temperature
        self.system_size = system_size
        self.points = points 
        self.device = device
        
        self.d = self.sigma*(1.0-0.12*np.exp(-3.0*self.epsilon/T))
        if self.q is not None:
            self.q2 = 1e-19*self.q**2/(self.m*self.epsilon*kB*self.sigma**5)
        self.R = 0.5*self.d

        self.m_ij = empty((self.Nc,self.Nc),device=self.device,dtype=float64) 
        self.m_ijk = empty((self.Nc,self.Nc,self.Nc),device=self.device,dtype=float64) 
        self.sigma_ij = empty_like(self.m_ij)
        self.epsilon_ij = empty_like(self.m_ij)

        for i in range(self.Nc):
            for j in range(self.Nc):
                self.m_ij[i,j] = min(2.0,(self.m[i]*self.m[j]).sqrt())
                self.sigma_ij[i,j] = 0.5*(self.sigma[i]+self.sigma[j])
                self.epsilon_ij[i,j] = (self.epsilon[i]*self.epsilon[j]).sqrt()
                for k in range(self.Nc):
                    self.m_ijk[i,j,k] = min(2.0,(self.m[i]*self.m[j]*self.m[k])**(1/3))

        self.anij = empty((5,self.Nc,self.Nc),device=self.device,dtype=float64)
        self.bnij = empty_like(self.anij)
        self.cnijk = empty((5,self.Nc,self.Nc,self.Nc),device=self.device,dtype=float64)
        for i in range(5):
            self.anij[i] = ((self.m_ij-1.0)/self.m_ij*((self.m_ij-2.0)/self.m_ij*aq[i,2]+aq[i,1])+aq[i,0]) 
            self.bnij[i] = ((self.m_ij-1.0)/self.m_ij*((self.m_ij-2.0)/self.m_ij*bq[i,2]+bq[i,1])+bq[i,0])
            self.cnijk[i] = ((self.m_ijk-1.0)/self.m_ijk*((self.m_ijk-2.0)/self.m_ijk*cq[i,2]+cq[i,1])+cq[i,0])

        self.cell_size = system_size/points
        self.cell_volume = self.cell_size[0]*self.cell_size[1]*self.cell_size[2] 

        self.x = linspace(0.5*self.cell_size[0], system_size[0]-0.5*self.cell_size[0], points[0],device=device,dtype=float64)
        self.y = linspace(0.5*self.cell_size[1], system_size[1]-0.5*self.cell_size[1], points[1],device=device, dtype=float64)
        self.z = linspace(0.5*self.cell_size[2], system_size[2]-0.5*self.cell_size[2], points[2],device=device, dtype=float64)
        self.X, self.Y, self.Z = meshgrid(self.x, self.y, self.z, indexing='ij')

        kx = np.fft.fftfreq(points[0], d=self.cell_size[0])
        ky = np.fft.fftfreq(points[1], d=self.cell_size[1])
        kz = np.fft.fftfreq(points[2], d=self.cell_size[2])
        kcut = np.array([kx.max(), ky.max(), kz.max()])
        Kx, Ky, Kz = np.meshgrid(kx,ky,kz, indexing ='ij')
        K = np.sqrt(Kx**2+Ky**2+Kz**2)

        w2_hat = np.empty((self.Nc,points[0],points[1],points[2]),dtype=np.float64)
        w3_hat = np.empty_like(w2_hat)
        w2vec_hat = np.empty((self.Nc,3,points[0],points[1],points[2]),dtype=np.complex128)
        w2hc_hat = np.empty_like(w2_hat)
        w3hc_hat = np.empty_like(w2_hat)
        wdisp_hat = np.empty_like(w2_hat)

        for i in range(self.Nc):
            w2_hat[i] = 4.0*pi*self.R[i]**2*spherical_jn(0, 2.*pi*self.R[i]*K)*lancsoz(kx,ky,kz,kcut)
            w3_hat[i] = (4./3.)*pi*self.R[i]**3*(spherical_jn(0, 2.*pi*self.R[i]*K)+spherical_jn(2, 2.*pi*self.R[i]*K)) \
                *lancsoz(kx,ky,kz,kcut)
            w2vec_hat[i,0] = -1j*2.0*pi*Kx*w3_hat[i]
            w2vec_hat[i,1] = -1j*2.0*pi*Ky*w3_hat[i]
            w2vec_hat[i,2] = -1j*2.0*pi*Kz*w3_hat[i]
            w2hc_hat[i] = spherical_jn(0, 4.*pi*self.R[i]*K)*lancsoz(kx,ky,kz,kcut)
            w3hc_hat[i] = (spherical_jn(0, 4.*pi*self.R[i]*K)+spherical_jn(2, 4.*pi*self.R[i]*K))*lancsoz(kx,ky,kz,kcut) 
            wdisp_hat[i] = (spherical_jn(0, 4.*pi*psi*self.R[i]*K)+spherical_jn(2, 4.*pi*psi*self.R[i]*K))*lancsoz(kx,ky,kz,kcut)

        self.m = self.m.to(device)
        self.sigma = self.sigma.to(device)
        self.epsilon = self.epsilon.to(device)
        self.d = self.d.to(device)
        self.R = self.R.to(device) 
        if self.q is not None: self.q2 = self.q2.to(device)
        self.w2_hat = tensor(w2_hat,device=device,dtype=float64)
        self.w3_hat = tensor(w3_hat,device=device,dtype=float64)
        self.w2vec_hat = tensor(w2vec_hat,device=device,dtype=complex128)
        self.w2hc_hat = tensor(w2hc_hat,device=self.device,dtype=float64)
        self.w3hc_hat = tensor(w3hc_hat,device=self.device,dtype=float64) 
        self.wdisp_hat = tensor(wdisp_hat,device=device,dtype=float64)

        del kx, ky, kz, K

    def weighted_densities(self):

        self.rho.requires_grad = True

        self.rho_hat = fftn(self.rho, dim=(1,2,3))
        ni = self.m[:,None,None,None]*ifftn(self.rho_hat*self.w2_hat, dim=(1,2,3)).real
        self.n0 = (ni/(4.*pi*self.R[:,None,None,None]**2)).sum(axis=0)
        self.n1 = (ni/(4.*pi*self.R[:,None,None,None])).sum(axis=0)
        self.n2 = ni.sum(axis=0)
        self.n3 = (self.m[:,None,None,None]*ifftn(self.rho_hat*self.w3_hat, dim=(1,2,3)).real).sum(axis=0)
        nivec = self.m[:,None,None,None,None]*ifftn(self.rho_hat[:,None,:]*self.w2vec_hat, dim=(2,3,4)).real
        self.n1vec = (nivec/(4.*pi*self.R[:,None,None,None,None])).sum(axis=0)
        self.n2vec= nivec.sum(axis=0)
        self.n2_hc = ifftn(self.rho_hat*self.w2hc_hat, dim=(1,2,3)).real
        self.n3_hc = ifftn(self.rho_hat*self.w3hc_hat, dim=(1,2,3)).real
        self.ni_disp = ifftn(self.rho_hat*self.wdisp_hat, dim=(1,2,3)).real

        del ni, nivec

        self.n3[self.n3>=1.0] = 1.0-1e-15

    def functional(self, fmt):

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

            del f1, f2, f4, n1_n2, n1vec_n2vec, n2_sq, n2vec_sq
            
        elif fmt == 'ASWB':

            n1_n2 = self.n1*self.n2
            n1vec_n2vec = (self.n1vec*self.n2vec).sum(dim=0)
            n2_sq = self.n2**2
            n2vec_sq = (self.n2vec*self.n2vec).sum(dim=0)

            xi = n2vec_sq/n2_sq
            xi.clamp_(max=1.0)
            
            self.Phi_hs = f1*self.n0+f2*(n1_n2-n1vec_n2vec)+f4*self.n2**3*(1.0-xi)**3

            del f1, f2, f4, n1_n2, n1vec_n2vec, n2_sq, n2vec_sq, xi

        self.F_hs = self.Phi_hs.sum() 

        # Hard-Chain Contribution
        zeta2 = (pi/6.)*einsum('i...,i,i->...', self.n3_hc, self.m, self.d**2)
        zeta3 = (pi/6.)*einsum('i...,i,i->...', self.n3_hc, self.m, self.d**3)
        zeta3[zeta3>=1.0] = 1.0-1e-15

        temp = (1.0-zeta3)
        ydd = 1.0/temp+(1.5*self.d[:,None,None,None]*zeta2)/temp**2+(0.5*(self.d[:,None,None,None]*zeta2)**2)/temp**3
        
        self.Phi_hc = ((self.m[:,None,None,None]-1.0)*self.rho*((log(self.rho)-1.0)-(log(ydd*self.n2_hc)-1.0))).sum(axis=0)
                
        self.F_hc = self.Phi_hc.sum()

        del zeta2, zeta3, ydd, temp 

        # Dispersive Contribution
        n_disp = self.ni_disp.sum(axis=0) 
        xbar = self.ni_disp/n_disp 
        mbar = einsum('i...,i->...', xbar, self.m)
        etabar = (pi/6.0)*einsum('i...,i,i->...', self.ni_disp, self.m, self.d**3) 
        etabar[etabar>=1.0] = 1.0-1e-15

        I1 = zeros_like(n_disp)
        I2 = zeros_like(n_disp)   
        for i in range(7):
            etabar_i = etabar**i
            I1 += ((mbar-1.0)/mbar*((mbar-2.0)/mbar*a[i,2]+a[i,1])+a[i,0])*etabar_i
            I2 += ((mbar-1.0)/mbar*((mbar-2.0)/mbar*b[i,2]+b[i,1])+b[i,0])*etabar_i
        
        C1 = (1.0+mbar*(8.0*etabar-2.0*etabar**2)/(1.0-etabar)**4 \
            +(1.0-mbar)*(20.0*etabar-27.0*etabar**2+12.0*etabar**3-2.0*etabar**4)/((1.0-etabar)*(2.0-etabar))**2).pow(-1)
        
        mix1 = einsum('i...,j...,i,j,ij,ij->...',
                            xbar, xbar, self.m, self.m,
                            self.epsilon_ij/self.T, self.sigma_ij**3) 
        mix2 = einsum('i...,j...,i,j,ij,ij->...',
                            xbar, xbar, self.m, self.m,
                            (self.epsilon_ij/self.T)**2, self.sigma_ij**3)

        a_disp = (-2.0*I1*mix1-mbar*C1*I2*mix2)*pi*n_disp
        self.Phi_disp = n_disp*a_disp 
        self.F_disp = self.Phi_disp.sum()

        del I1, I2, mbar, C1, mix1, mix2

        # Quadrupolar Contribution
        if self.q is None:
            self.Phi_qq = zeros_like(self.Phi_hs)
        else:
            poweta = empty((5,self.points),device=self.device,dtype=float64)
            for i in range(5):
                poweta[i] = etabar**i
        
            J2 = einsum('nij,n...->ij...', self.anij, poweta) \
                +einsum('nij,ij,n...->ij...', self.bnij, self.epsilon_ij/self.T, poweta)
            J3 = einsum('nijk,n...->ijk...', self.cnijk, poweta)

            f_q2 = einsum('i...,j...,i,j,i,j,ij,i,j,ij...->...',
                                xbar, xbar, self.epsilon/self.T, self.epsilon/self.T, 
                                self.sigma**5, self.sigma**5,
                                self.sigma_ij**(-7), self.q2, self.q2, J2)
            
            f_q3 = einsum('i...,j...,k...,i,j,k,i,j,k,ij,ik,jk,i,j,k,ijk...->...', xbar, xbar, xbar,
                                self.epsilon/self.T, self.epsilon/self.T, self.epsilon/self.T,
                                self.sigma**5, self.sigma**5, self.sigma**5,
                                self.sigma_ij**(-3), self.sigma_ij**(-3), self.sigma_ij**(-3),
                                self.q2, self.q2, self.q2, J3)
            
            f_q2 = -f_q2*pi*0.5625*n_disp
            f_q3 = f_q3*pi**2*0.5625*n_disp**2

            a_qq = f_q2/(1.0-f_q3/f_q2)
            self.Phi_qq = n_disp*a_qq

            del n_disp, xbar, poweta, J2, J3, f_q2, f_q3, a_qq
        
        self.F_qq = self.Phi_qq.sum() 

        self.Fres = self.F_hs+self.F_hc+self.F_disp+self.F_qq

    def functional_derivative(self, fmt):

        self.functional(fmt)
        self.dFres = grad(self.Fres, self.rho)[0]
        self.dFres = self.dFres.detach()

        self.rho.requires_grad=False
    
    def initial_condition(self, bulk_density, composition, Vext, potential_cutoff=50.0):
        
        self.rhob = bulk_density*composition
        eos = pcsaft(self.pcsaft_parameters, self.T)
        self.mu = eos.chemical_potential(bulk_density, composition)+log(self.rhob)

        self.Vext = tensor(Vext/self.T,device=self.device,dtype=float64)
        self.excluded = self.Vext >= potential_cutoff
        self.valid = self.Vext < potential_cutoff
        self.Vext[self.excluded] = potential_cutoff

        self.rho = empty((self.Nc,self.points[0],self.points[1],self.points[2]),device=self.device,dtype=float64)
        for i in range(self.Nc):
            # self.rho[i] = self.rhob[i]*exp(-0.01*self.Vext[i])
            self.rho[i] = self.rhob[i] 
    
    def equilibrium_density_profile(self, bulk_density, composition, fmt='WB', solver='fire',
                                    alpha0=0.2, dt=0.1,  anderson_mmax=10, anderson_damping=0.1,
                                      tol=1e-8, max_it=1000, logoutput=False):
        
        self.rhob = bulk_density*composition 
        eos = pcsaft(self.pcsaft_parameters, self.T)
        self.mu = eos.chemical_potential(bulk_density, composition)+log(self.rhob)

        self.rhob = self.rhob.to(self.device)
        self.mu = self.mu.to(self.device)
        self.rho[self.excluded] = 1e-15
        lnrho = log(self.rho)

        self.functional_derivative(fmt)
        F = self.mu[:,None,None,None]-self.dFres-self.Vext-lnrho
        
        self.Nc_dot_points = self.Nc*self.points[0]*self.points[1]*self.points[2]
        error = norm(F[self.valid])/np.sqrt(self.Nc_dot_points)

        if solver == 'picard':

            alpha = alpha0
            self.it = 0
            tic = time.process_time()
            for i in range(max_it):
                lnrho[self.valid] += alpha*F[self.valid]
                self.rho[self.valid] = exp(lnrho[self.valid])
                self.functional_derivative(fmt) 
                F = self.mu[:,None,None,None]-self.dFres-self.Vext-lnrho
                error = norm(F[self.valid])/np.sqrt(self.Nc_dot_points)
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
                F = self.mu[:,None,None,None]-self.dFres-self.Vext-lnrho
                V[self.valid] += 0.5*dt*F[self.valid]

                error = norm(F[self.valid])/np.sqrt(self.Nc_dot_points)
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
                F = self.mu[:,None,None,None]-self.dFres-self.Vext-lnrho
                error = norm(F[self.valid])/np.sqrt(self.Nc_dot_points)

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
        self.total_molecules = empty(self.Nc,dtype=float64)
        Phi = zeros_like(self.Phi_disp)
        for i in range(self.Nc):
            self.total_molecules[i] = self.rho[i].cpu().sum()*self.cell_volume
            Phi += self.rho[i]*(log(self.rho[i])-1.0)+self.rho[i]*(self.Vext[i]-self.mu[i])

        self.Omega = (Phi.sum()+self.Fres.detach())*self.cell_volume
