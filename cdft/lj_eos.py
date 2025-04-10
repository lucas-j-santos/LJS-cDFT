import numpy as np
from torch import pi,tensor,float64,clone,exp
from torch.autograd import grad

kB = 1.380649e-23
NA = 6.02214076e23

xlj = tensor([0.8623085097507421,2.976218765822098,-8.402230115796038,0.105413662920355,
              -0.8564583828174598,1.582759470107601,0.763942948305453,1.753173414312048,
              2.798291772190376e3,-4.8394220260857657e-2,0.9963265197721935,-3.698000291272493e1,
              2.084012299434647e1,8.305402124717285e1,-9.574799715203068e2,-1.477746229234994e2,
              6.398607852471505e1,1.603993673294834e1,6.805916615864377e1,-2.791293578795945e3,
              -6.245128304568454,-8.116836104958410e3,1.488735559561229e1,-1.059346754655084e4,
              -1.31607632802822e2,-8.867771540418822e3,-3.986982844450543e1,-4.689270299917261e3,
              2.593535277438717e2,-2.694523589434903e3,-7.218487631550215e2,1.721802063863269e2],
              dtype=float64)

def acoef(Tstar):
    a = tensor([xlj[0]*Tstar+xlj[1]*np.sqrt(Tstar)+xlj[2]+xlj[3]/Tstar+xlj[4]/Tstar**2, 
                xlj[5]*Tstar+xlj[6]+xlj[7]/Tstar+xlj[8]/Tstar**2, xlj[9]*Tstar+xlj[10]+xlj[11]/Tstar, 
                xlj[12], xlj[13]/Tstar+xlj[14]/Tstar**2, xlj[15]/Tstar, xlj[16]/Tstar+xlj[17]/Tstar**2,
                xlj[18]/Tstar**2], dtype=float64) 
    return a 

def bcoef(Tstar):
    b = tensor([xlj[19]/Tstar**2+xlj[20]/Tstar**3, xlj[21]/Tstar**2+xlj[22]/Tstar**4,
                xlj[23]/Tstar**2+xlj[24]/Tstar**3, xlj[25]/Tstar**2+xlj[26]/Tstar**4,
                xlj[27]/Tstar**2+xlj[28]/Tstar**3, xlj[29]/Tstar**2+xlj[30]/Tstar**3+xlj[31]/Tstar**4],
                dtype=float64)
    return b

class lj_eos():

    def __init__(self, parameters, temperature):

        self.parameters = parameters
        self.sigma = self.parameters['sigma']
        self.epsilon = self.parameters['epsilon']
        self.T = temperature
        self.Tstar = self.T/self.epsilon
        self.d = self.sigma*(1+0.2977*self.Tstar)/(1+0.33163*self.Tstar+1.0477e-3*self.Tstar**2)

    def helmholtz_energy(self, rho):
        
        rhostar = rho*self.sigma**3
        a = acoef(self.Tstar)
        b = bcoef(self.Tstar)
        gamma = 3.0
        F = exp(-gamma*rhostar**2)
        G = [None] * 6
        G[0] = (1 - F) / (2 * gamma)
        for i in range(1, 6):
            G[i] = -(F * rhostar**(2*i) - 2*i*G[i-1]) / (2 * gamma)
        fres = sum(a[i] * rhostar**(i+1) / (i+1) for i in range(8))
        fres += sum(b[i] * G[i] for i in range(6))

        return fres/self.Tstar # fres/N kB T
    
    def compressibility_factor(self, rho):

        rho.requires_grad=True
        
        f_res = self.helmholtz_energy(rho)
        df_drho = grad(f_res, rho, create_graph=True)[0]
        Z = 1.0+rho*df_drho
        
        return Z
    
    def pressure(self, rho):

        Z = self.compressibility_factor(rho)
        P = Z*kB*self.T*rho*1e30

        rho.requires_grad = False

        return P.detach()
    
    def chemical_potential(self, rho):

        f_res = self.helmholtz_energy(rho)
        Z = self.compressibility_factor(rho)
        mu_res = f_res+(Z-1.0)

        rho.requires_grad=False

        return mu_res.detach()
    
    def fugacity_coefficient(self, rho):
        
        Z = self.compressibility_factor(rho)
        mu_res = self.chemical_potential(rho)

        rho.requires_grad=False

        return exp(mu_res.detach())/Z.detach() 

    def residue(self, rho, Psys):

        Z = self.compressibility_factor(rho)
        Pcalc = Z*kB*self.T*rho 
        res = (Pcalc-Psys)/Psys
        return res

    def diff_residue(self, rho, Psys):
        res = self.residue(rho, Psys)
        dres = grad(res, rho)[0]
        return dres
        
    def density(self, P, phase):
        
        if phase == 'vap':
            eta = 1e-10
            rho0 = eta/((pi/6.)*self.d**3)
            rho0 = tensor([rho0],dtype=float64)
        elif phase == 'liq':
            eta = 0.5
            rho0 = eta/((pi/6.)*self.d**3)
            rho0 = tensor([rho0],dtype=float64)
        else:
            rho0 = phase

        for i in range(1000):
            res = self.residue(rho0.detach(),P*1e-30).detach() 
            rho = rho0.detach()-res/self.diff_residue(rho0.detach(),P*1e-30).detach()
            rho0 = clone(rho)
            if abs(res) < 1e-10:
                break

        return rho
    
    def vapor_pressure(self, P0):

        for i in range(1000):

            rhoV = self.density(P0, 'vap')  
            rhoL = self.density(P0, 'liq')  
            phiV = self.fugacity_coefficient(rhoV)
            phiL = self.fugacity_coefficient(rhoL)

            res = abs(phiL/phiV-1.0) 
            P = P0*phiL/phiV
            P0 = clone(P)

            if res < 1e-10:
                break

        return P