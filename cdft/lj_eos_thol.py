import numpy as np
import torch

torch.set_default_dtype(torch.float64)

pi = np.pi
kB = 1.380649e-23
NA = 6.02214076e23

TC_STAR = 1.32
RHOC_STAR = 0.31

_n = np.array([
    0.52080730e-2,  0.21862520e+1, -0.21610160e+1,  0.14527000e+1,
   -0.20417920e+1,  0.18695286e+0, -0.90988445e-1, -0.49745610e+0,
    0.10901431e+0, -0.80055922e+0, -0.56883900e+0, -0.62086250e+0,
   -0.14667177e+1,  0.18914690e+1, -0.13837010e+0, -0.38696450e+0,
    0.12657020e+0,  0.60578100e+0,  0.11791890e+1, -0.47732679e+0,
   -0.99218575e+1, -0.57479320e+0,  0.37729230e-2])

_t = np.array([1.000, 0.320, 0.505, 0.672, 0.843, 0.898, 1.294, 2.590,
               1.786, 2.770, 1.786, 1.205, 2.830, 2.548, 4.650, 1.385,
               1.460, 1.351, 0.660, 1.496, 1.830, 1.616, 4.970])

_d = np.array([4, 1, 1, 2, 2, 3, 5, 2, 2, 3, 1, 1,
               1, 1, 2, 3, 3, 2, 1, 2, 3, 1, 1])

_l = np.array([1, 2, 1, 2, 2, 1])                       # termos 7..12

_eta = np.array([2.067, 1.522, 8.820, 1.722, 0.679, 1.883,
                 3.925, 2.461, 28.20, 0.753, 0.820])     # termos 13..23
_beta = np.array([0.625, 0.638, 3.910, 0.156, 0.157, 0.153,
                  1.160, 1.730, 383.0, 0.112, 0.119])
_gamma = np.array([0.710, 0.860, 1.940, 1.480, 1.490, 1.945,
                   3.020, 1.110, 1.170, 1.330, 0.240])
_epsg = np.array([0.2053, 0.4090, 0.6000, 1.2030, 1.8290, 1.3970,
                  1.3900, 0.5390, 0.9340, 2.3690, 2.4300])


class lj_eos():
    """EOS de Thol et al. (2016) com a mesma interface da versao Johnson.

    Troca de uma linha:  from .lj_eos_thol import lj_eos
    """

    def __init__(self, parameters, temperature, device=None):

        self.parameters = parameters
        self.sigma = self.parameters['sigma']
        self.epsilon = self.parameters['epsilon']
        self.T = temperature
        self.Tstar = self.T/self.epsilon
        self.d = self.sigma*(1+0.2977*self.Tstar)/(1+0.33163*self.Tstar+1.0477e-3*self.Tstar**2)

        tau = TC_STAR/self.Tstar
        self.tau = tau

        # --- tudo que depende so de tau sai do caminho quente ---------
        N1 = _n[0:6]*tau**_t[0:6]
        N2 = _n[6:12]*tau**_t[6:12]
        N3 = _n[12:23]*tau**_t[12:23]*np.exp(-_beta*(tau-_gamma)**2)

        # grupo 1: polinomio em delta (graus 1..4)
        c1 = np.zeros(5)
        for i in range(6):
            c1[_d[i]] += N1[i]

        # grupo 2: dois polinomios, um multiplicado por exp(-delta) e
        # outro por exp(-delta^2), conforme l_i
        cA = np.zeros(6)
        cB = np.zeros(6)
        for j in range(6):
            (cA if _l[j] == 1 else cB)[_d[6+j]] += N2[j]

        self._c1 = torch.tensor(c1)
        self._cA = torch.tensor(cA)
        self._cB = torch.tensor(cB)
        self._N3 = torch.tensor(N3)
        self._eta = torch.tensor(_eta)
        self._epsg = torch.tensor(_epsg)
        self._d3 = [int(x) for x in _d[12:23]]

        self._cache = {}
        if device is not None:
            self._coeffs(torch.device(device))

    def _coeffs(self, device):
        key = str(device)
        if key not in self._cache:
            self._cache[key] = (self._c1.to(device), self._cA.to(device),
                                self._cB.to(device), self._N3.to(device),
                                self._eta.to(device), self._epsg.to(device))
        return self._cache[key]

    def helmholtz_energy(self, rho):
        """alpha^r = a_residual/(kB T), mesma convencao do lj_eos.py."""

        c1, cA, cB, N3, eta, epsg = self._coeffs(rho.device)

        delta = rho*(self.sigma**3/RHOC_STAR)

        # potencias inteiras por multiplicacao: 5 mults em vez de 23 pow()
        D1 = delta
        D2 = D1*D1
        D3 = D2*D1
        D4 = D2*D2
        D5 = D4*D1
        D = (None, D1, D2, D3, D4, D5)

        # grupo 1, por Horner
        alpha = (((c1[4]*D1+c1[3])*D1+c1[2])*D1+c1[1])*D1

        # grupo 2: um exp por valor de l, em vez de um por termo
        polyA = ((((cA[5]*D1+cA[4])*D1+cA[3])*D1+cA[2])*D1+cA[1])*D1
        polyB = ((((cB[5]*D1+cB[4])*D1+cB[3])*D1+cB[2])*D1+cB[1])*D1
        alpha = alpha+polyA*torch.exp(-D1)+polyB*torch.exp(-D2)

        # grupo 3: 11 gaussianas, cada uma com o proprio centro
        for j in range(11):
            dj = delta-epsg[j]
            alpha = alpha+N3[j]*D[self._d3[j]]*torch.exp(-eta[j]*dj*dj)

        return alpha

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
            rho0 = torch.tensor([rho0])
        elif phase == 'liq':
            eta = 0.5
            rho0 = eta/((pi/6.)*self.d**3)
            rho0 = torch.tensor([rho0])
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
