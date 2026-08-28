import numpy as np
import torch

torch.set_default_dtype(torch.float64)

pi = np.pi
kB = 1.380649e-23
NA = 6.02214076e23

xlj = torch.tensor([0.8623085097507421,2.976218765822098,-8.402230115796038,0.105413662920355,
              -0.8564583828174598,1.582759470107601,0.763942948305453,1.753173414312048,
              2.798291772190376e3,-4.8394220260857657e-2,0.9963265197721935,-3.698000291272493e1,
              2.084012299434647e1,8.305402124717285e1,-9.574799715203068e2,-1.477746229234994e2,
              6.398607852471505e1,1.603993673294834e1,6.805916615864377e1,-2.791293578795945e3,
              -6.245128304568454,-8.116836104958410e3,1.488735559561229e1,-1.059346754655084e4,
              -1.31607632802822e2,-8.867771540418822e3,-3.986982844450543e1,-4.689270299917261e3,
              2.593535277438717e2,-2.694523589434903e3,-7.218487631550215e2,1.721802063863269e2])


def acoef(Tstar):
    a = torch.stack([xlj[0]*Tstar+xlj[1]*np.sqrt(Tstar)+xlj[2]+xlj[3]/Tstar+xlj[4]/Tstar**2,
                xlj[5]*Tstar+xlj[6]+xlj[7]/Tstar+xlj[8]/Tstar**2, xlj[9]*Tstar+xlj[10]+xlj[11]/Tstar,
                xlj[12], xlj[13]/Tstar+xlj[14]/Tstar**2, xlj[15]/Tstar, xlj[16]/Tstar+xlj[17]/Tstar**2,
                xlj[18]/Tstar**2])
    return a


def bcoef(Tstar):
    b = torch.stack([xlj[19]/Tstar**2+xlj[20]/Tstar**3, xlj[21]/Tstar**2+xlj[22]/Tstar**4,
                xlj[23]/Tstar**2+xlj[24]/Tstar**3, xlj[25]/Tstar**2+xlj[26]/Tstar**4,
                xlj[27]/Tstar**2+xlj[28]/Tstar**3, xlj[29]/Tstar**2+xlj[30]/Tstar**3+xlj[31]/Tstar**4])
    return b


class lj_eos():

    def __init__(self, parameters, temperature, device=None, dtype=torch.float64):

        self.parameters = parameters
        self.sigma = self.parameters['sigma']
        self.epsilon = self.parameters['epsilon']
        self.T = temperature
        self.Tstar = self.T/self.epsilon
        self.d = self.sigma*(1+0.2977*self.Tstar)/(1+0.33163*self.Tstar+1.0477e-3*self.Tstar**2)

        a = acoef(self.Tstar).to(dtype)
        self._c = (a/torch.arange(1.0, 9.0, dtype=dtype)).contiguous()
        self._b = bcoef(self.Tstar).to(dtype).contiguous()

        self._cache = {}
        if device is not None:
            self._coeffs(torch.device(device), dtype)

    def _coeffs(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cache:
            self._cache[key] = (self._c.to(device=device, dtype=dtype),
                                self._b.to(device=device, dtype=dtype))
        return self._cache[key]

    def helmholtz_energy(self, rho):

        c, b = self._coeffs(rho.device, rho.dtype)

        rhostar = rho*self.sigma**3
        rhostar_sq = rhostar*rhostar
        gamma = 3.0
        F = torch.exp(-gamma*rhostar_sq)

        fex = c[7]*rhostar+c[6]
        for i in range(5, -1, -1):
            fex = fex*rhostar+c[i]
        fex = fex*rhostar

        two_gamma = 2.0*gamma
        G = (1.0-F)/two_gamma
        fex = fex+b[0]*G
        rhostar_pow = rhostar_sq
        for i in range(1, 6):
            G = -(F*rhostar_pow-2*i*G)/two_gamma
            fex = fex+b[i]*G
            rhostar_pow = rhostar_pow*rhostar_sq

        return fex/self.Tstar  # fex/N kB T

    def compressibility_factor(self, rho):

        rho.requires_grad=True

        fex = self.helmholtz_energy(rho)
        df_drho = torch.autograd.grad(fex, rho, create_graph=True)[0]
        Z = 1.0+rho*df_drho

        return Z

    def pressure(self, rho):

        Z = self.compressibility_factor(rho)
        P = Z*kB*self.T*rho*1e30

        rho.requires_grad = False

        return P.detach()

    def chemical_potential(self, rho):

        rho.requires_grad = True
        fex = self.helmholtz_energy(rho)
        df_drho = torch.autograd.grad(fex, rho)[0]
        mu_ex = fex+rho*df_drho

        rho.requires_grad = False

        return mu_ex.detach()

    def fugacity_coefficient(self, rho):

        Z = self.compressibility_factor(rho)
        mu_ex = self.chemical_potential(rho)

        rho.requires_grad=False

        return torch.exp(mu_ex.detach())/Z.detach()

    def residue(self, rho, Psys):

        Z = self.compressibility_factor(rho)
        Pcalc = Z*kB*self.T*rho
        res = (Pcalc-Psys)/Psys
        return res

    def diff_residue(self, rho, Psys):
        res = self.residue(rho, Psys)
        dres = torch.autograd.grad(res, rho)[0]
        return dres

    def density(self, P, phase):

        if phase == 'vap':
            eta = 1e-10
            rho0 = eta/((pi/6.)*self.d**3)
            rho0 = torch.tensor([rho0], dtype=self._c.dtype)
        elif phase == 'liq':
            eta = 0.5
            rho0 = eta/((pi/6.)*self.d**3)
            rho0 = torch.tensor([rho0], dtype=self._c.dtype)
        else:
            rho0 = phase

        for i in range(1000):
            res = self.residue(rho0.detach(),P*1e-30).detach()
            rho = rho0.detach()-res/self.diff_residue(rho0.detach(),P*1e-30).detach()
            rho0 = torch.clone(rho)
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
            P0 = torch.clone(P)

            if res < 1e-10:
                break

        return P
