from torch import pi,tensor,float64,zeros,zeros_like,empty,empty_like,clone,einsum,exp
from torch.autograd import grad

kB = 1.380649e-23
NA = 6.02214076e23

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

class pcsaft():

    def __init__(self, parameters, T):

        self.parameters = parameters
        self.m = self.parameters['m']
        self.sigma = self.parameters['sigma']
        self.epsilon = self.parameters['epsilon']
        try: self.q = self.parameters['q']
        except: self.q = None
        self.T = T
        self.Nc = len(self.m)

        self.m_ij = empty((self.Nc,self.Nc), dtype=float64) 
        self.m_ijk = empty((self.Nc,self.Nc,self.Nc), dtype=float64) 
        self.sigma_ij = empty_like(self.m_ij)
        self.epsilon_ij = empty_like(self.m_ij)

        for i in range(self.Nc):
            for j in range(self.Nc):
                self.m_ij[i,j] = min(2.0,(self.m[i]*self.m[j]).sqrt())
                self.sigma_ij[i,j] = 0.5*(self.sigma[i]+self.sigma[j])
                self.epsilon_ij[i,j] = (self.epsilon[i]*self.epsilon[j]).sqrt()
                for k in range(self.Nc):
                    self.m_ijk[i,j,k] = min(2.0,(self.m[i]*self.m[j]*self.m[k])**(1/3))

        self.anij = empty((5,self.Nc,self.Nc),dtype=float64)
        self.bnij = empty_like(self.anij)
        self.cnijk = empty((5,self.Nc,self.Nc,self.Nc),dtype=float64)
        for i in range(5):
            self.anij[i] = ((self.m_ij-1.0)/self.m_ij*((self.m_ij-2.0)/self.m_ij*aq[i,2]+aq[i,1])+aq[i,0]) 
            self.bnij[i] = ((self.m_ij-1.0)/self.m_ij*((self.m_ij-2.0)/self.m_ij*bq[i,2]+bq[i,1])+bq[i,0])
            self.cnijk[i] = ((self.m_ijk-1.0)/self.m_ijk*((self.m_ijk-2.0)/self.m_ijk*cq[i,2]+cq[i,1])+cq[i,0])

        self.d = self.sigma*(1.0-0.12*(-3.0*self.epsilon/(self.T)).exp())
        if self.q is not None:
            self.q2 = 1e-19*self.q**2/(self.m*self.epsilon*kB*self.sigma**5)

    def helmholtz_energy(self, rho, x):

        rho.requires_grad=True
        x.requires_grad=True

        m_bar = (x*self.m).sum()
        
        # Hard-Sphere Contribution

        zeta0 = (pi/6.)*rho*(x*self.m).sum()
        zeta1 = (pi/6.)*rho*(x*self.m*self.d).sum()
        zeta2 = (pi/6.)*rho*(x*self.m*self.d**2).sum()
        zeta3 = (pi/6.)*rho*(x*self.m*self.d**3).sum()
        eta = clone(zeta3)

        omzeta3 = 1.0-zeta3
        f_hs = 1./zeta0 \
        *(3.*zeta1*zeta2/(omzeta3)+zeta2**3/(zeta3*(omzeta3)**2)+(zeta2**3/zeta3**2-zeta0)*(omzeta3).log())
        f_hs = m_bar*f_hs

        # Hard-Chain Contribution

        dii = self.d**2/(2.0*self.d)
        g_hs = 1.0/omzeta3+dii*3.0*zeta2/(omzeta3**2)+dii**2*2.0*zeta2**2/(omzeta3**3)
        f_hc = -(x*(self.m-1.0)*g_hs.log()).sum()

        # Dispersive Contribution

        C1 = 1.0+m_bar*(8.0*eta-2.0*eta**2)/(1.0-eta)**4 \
        + (1.0-m_bar)*(20.0*eta-27.0*eta**2+12.0*eta**3-2.0*eta**4)/((1.0-eta)*(2.0-eta))**2
        C1 = 1.0/C1

        I1 = zeros(1, dtype=float64)
        I2 = zeros_like(I1)
        for i in range(7):
            I1 += ((m_bar-1.0)/m_bar*((m_bar-2.0)/m_bar*a[i,2]+a[i,1])+a[i,0])*eta**i
            I2 += ((m_bar-1.0)/m_bar*((m_bar-2.0)/m_bar*b[i,2]+b[i,1])+b[i,0])*eta**i
        
        mix1 = einsum('i,j,i,j,ij,ij->', x, x, self.m, self.m, self.epsilon_ij/self.T, self.sigma_ij**3)
        mix2 = einsum('i,j,i,j,ij,ij->', x, x, self.m, self.m, (self.epsilon_ij/self.T)**2, self.sigma_ij**3)

        f_disp = (-2.0*I1*mix1-m_bar*C1*I2*mix2)*pi*rho

        # Quadrupolar Contribution

        if self.q is None:
            f_qq = 0.0 
        else:
            poweta = empty(5,dtype=float64)
            for i in range(5):
                poweta[i] = eta**i

            J2 = einsum('nij,n->ij',self.anij+self.bnij*self.epsilon_ij/self.T,poweta)
            J3 = einsum('nijk,n->ijk',self.cnijk,poweta) 

            f_q2 = einsum('i,j,i,j,i,j,ij,i,j,ij->',
                                x, x, self.epsilon/self.T, self.epsilon/self.T, self.sigma**5, self.sigma**5,
                                self.sigma_ij**(-7), self.q2, self.q2, J2)
            
            f_q3 = einsum('i,j,k,i,j,k,i,j,k,ij,ik,jk,i,j,k,ijk->', x, x, x,
                                self.epsilon/self.T, self.epsilon/self.T, self.epsilon/self.T,
                                self.sigma**5, self.sigma**5, self.sigma**5,
                                self.sigma_ij**(-3), self.sigma_ij**(-3), self.sigma_ij**(-3),
                                self.q2, self.q2, self.q2, J3)
            
            f_q2 = -f_q2*pi*0.5625*rho
            f_q3 = f_q3*pi**2*0.5625*rho**2
            f_qq = f_q2/(1.0-f_q3/f_q2)

        f_res = f_hs+f_hc+f_disp+f_qq

        return f_res
    
    def calc_pressure(self, rho, x):
        
        f_res = self.helmholtz_energy(rho, x)
        df_drho = grad(f_res, rho, create_graph=True)[0]
        Z = 1.0+rho*df_drho
        P = Z*kB*self.T*rho # J A⁻³
    
        return P
    
    def pressure(self, rho, x):

        P = self.calc_pressure(rho, x)*1e30 # Pa
        rho.requires_grad=False
        x.requires_grad=False
    
        return P.detach()

    def residue(self, rho, x, Psys):
        Pcalc = self.calc_pressure(rho, x)
        res = (Pcalc-Psys)/Psys
        return res

    def diff_residue(self, rho, x, Psys):
        res = self.residue(rho, x, Psys)
        dres = grad(res, rho)[0]
        return dres
        
    def density(self, P, x, phase):
        
        if phase == 'vap':
            eta = 1e-10
            rho0 = eta/((pi/6.)*(x*self.m*self.d**3).sum())
        elif phase == 'liq':
            eta = 0.5
            rho0 = eta/((pi/6.)*(x*self.m*self.d**3).sum())
        else:
            rho0 = phase

        for i in range(1000):
            res = self.residue(rho0.detach(),x.detach(),P*1e-30).detach() 
            rho = rho0.detach()-res/self.diff_residue(rho0.detach(),x.detach(),P*1e-30).detach()
            rho0 = clone(rho)
            if abs(res) < 1e-10:
                break

        return rho
    
    def chemical_potential(self, rho, x):

        f_res = self.helmholtz_energy(rho, x)
        df_drho = grad(f_res, rho, retain_graph=True)[0]
        df_dx = grad(f_res, x, retain_graph=False)[0]
        Z = 1.0+rho*df_drho
        mu_res = f_res+(Z-1.0)+df_dx-(x*df_dx).sum()

        rho.requires_grad=False
        x.requires_grad=False

        return mu_res.detach()
    
    def fugacity_coefficient(self, rho, x):

        f_res = self.helmholtz_energy(rho, x)
        df_drho = grad(f_res, rho, retain_graph=True)[0]
        df_dx = grad(f_res, x, retain_graph=False)[0]
        Z = 1.0+rho*df_drho
        mu_res = f_res+(Z-1.0)+df_dx-(x*df_dx).sum()

        rho.requires_grad=False
        x.requires_grad=False

        return exp(mu_res.detach())/Z.detach() 
    
    def vapor_pressure(self, P0):

        x = tensor([1.0],dtype=float64) 

        for i in range(1000):

            rhoV = self.density(P0, x, 'vap')  
            rhoL = self.density(P0, x, 'liq')  
            phiV = self.fugacity_coefficient(rhoV, x)
            phiL = self.fugacity_coefficient(rhoL, x)

            res = abs(phiL/phiV-1.0) 
            P = P0*phiL/phiV
            P0 = clone(P)

            if res < 1e-10:
                break

        return P