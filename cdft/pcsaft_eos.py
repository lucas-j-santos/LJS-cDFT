import torch

torch.set_default_dtype(torch.float64)

kB = 1.380649e-23
NA = 6.02214076e23
pi = torch.pi

a = torch.tensor([
    [0.91056314451539, -0.30840169182720, -0.09061483509767],
    [0.63612814494991, 0.18605311591713, 0.45278428063920],
    [2.68613478913903, -2.50300472586548, 0.59627007280101],
    [-26.5473624914884, 21.4197936296668, -1.72418291311787],
    [97.7592087835073, -65.2558853303492, -4.13021125311661],
    [-159.591540865600, 83.3186804808856, 13.7766318697211],
    [91.2977740839123, -33.7469229297323, -8.67284703679646]
    ])

b = torch.tensor([
    [0.72409469413165, -0.57554980753450, 0.09768831158356],
    [2.23827918609380, 0.69950955214436, -0.25575749816100],
    [-4.00258494846342, 3.89256733895307, -9.15585615297321],
    [-21.00357681484648, -17.21547164777212, 20.64207597439724],
    [26.8556413626615, 192.6722644652495, -38.80443005206285],
    [206.5513384066188, -161.8264616487648, 93.6267740770146],
    [-355.60235612207947, -165.2076934555607, -29.66690558514725]
    ])

aq = torch.tensor([
    [1.237830788, 1.285410878, 1.794295401],
    [2.435503144, -11.46561451, 0.769510293],
    [1.633090469, 22.08689285, 7.264792255],
    [-1.611815241, 7.46913832, 94.48669892],
    [6.977118504, -17.19777208, -77.1484579]])

bq = torch.tensor([
    [0.454271755, -0.813734006, 6.868267516],
    [-4.501626435, 10.06402986, -5.173223765],
    [3.585886783, -10.87663092, -17.2402066],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
    ])

cq = torch.tensor([
    [-0.500043713, 2.000209381, 3.135827145],
    [6.531869153, -6.78386584, 7.247588801],
    [-16.01477983, 20.38324603, 3.075947834],
    [14.42597018, -10.89598394, 0.0],
    [0.0, 0.0, 0.0]
    ])


def mixing_tensors(m, sigma, epsilon, T, q2=None, device=None):
    """Combining rules and every constant that depends only on (parameters, T).

    Shared by pcsaft and by the DFT modules so the two cannot drift apart.
    Returns a dict of tensors already on `device`.
    """
    m = m.to(device=device)
    sigma = sigma.to(device=device)
    epsilon = epsilon.to(device=device)
    Nc = len(m)

    # vectorised combining rules (the original triple Python loop was O(Nc^3)
    # with one kernel launch per element)
    m_ij = torch.minimum(torch.sqrt(m[:, None]*m[None, :]),
                         torch.full((Nc, Nc), 2.0, device=device))
    sigma_ij = 0.5*(sigma[:, None]+sigma[None, :])
    epsilon_ij = torch.sqrt(epsilon[:, None]*epsilon[None, :])
    m_ijk = torch.minimum((m[:, None, None]*m[None, :, None]*m[None, None, :])**(1.0/3.0),
                          torch.full((Nc, Nc, Nc), 2.0, device=device))
    eps_ij_T = epsilon_ij/T

    r_ij = (m_ij-1.0)/m_ij
    s_ij = (m_ij-2.0)/m_ij
    r_ijk = (m_ijk-1.0)/m_ijk
    s_ijk = (m_ijk-2.0)/m_ijk

    aq_d = aq.to(device=device)
    bq_d = bq.to(device=device)
    cq_d = cq.to(device=device)

    anij = (r_ij[None]*(s_ij[None]*aq_d[:, 2, None, None]+aq_d[:, 1, None, None])
            + aq_d[:, 0, None, None])
    bnij = (r_ij[None]*(s_ij[None]*bq_d[:, 2, None, None]+bq_d[:, 1, None, None])
            + bq_d[:, 0, None, None])
    cnijk = (r_ijk[None]*(s_ijk[None]*cq_d[:, 2, None, None, None]
                          + cq_d[:, 1, None, None, None])
             + cq_d[:, 0, None, None, None])

    out = {'m': m, 'sigma': sigma, 'epsilon': epsilon,
           'm_ij': m_ij, 'm_ijk': m_ijk,
           'sigma_ij': sigma_ij, 'epsilon_ij': epsilon_ij, 'eps_ij_T': eps_ij_T,
           'anij': anij, 'bnij': bnij, 'cnijk': cnijk,
           # dispersive quadratic forms: mix1 = x^T A1 x, mix2 = x^T A2 x
           'A1': m[:, None]*m[None, :]*eps_ij_T*sigma_ij**3,
           'A2': m[:, None]*m[None, :]*eps_ij_T**2*sigma_ij**3,
           # a,b coefficients of the I1/I2 series, on device
           'a': a.to(device=device),
           'b': b.to(device=device)}

    if q2 is not None:
        q2 = q2.to(device=device)
        eps_T = epsilon/T
        u = eps_T*sigma**5*q2                       # per-component factor
        Q2 = u[:, None]*u[None, :]*sigma_ij**(-7)
        Q3 = (u[:, None, None]*u[None, :, None]*u[None, None, :]
              * sigma_ij[:, :, None]**(-3)
              * sigma_ij[:, None, :]**(-3)
              * sigma_ij[None, :, :]**(-3))
        # fold every constant into the eta-series coefficients: the whole
        # quadrupolar term collapses to two polynomial contractions
        out['AB2'] = (anij+bnij*eps_ij_T)*Q2          # (5, Nc, Nc)
        out['C3'] = cnijk*Q3                          # (5, Nc, Nc, Nc)
        out['q2'] = q2

    return out


def horner(coeffs, x):
    """sum_i coeffs[i] * x**i, evaluated without pow()."""
    out = coeffs[-1]*x+coeffs[-2]
    for i in range(len(coeffs)-3, -1, -1):
        out = out*x+coeffs[i]
    return out


class pcsaft():

    def __init__(self, parameters, temperature, device=None):

        self.parameters = parameters
        self.m = parameters['m']
        self.sigma = parameters['sigma']
        self.epsilon = parameters['epsilon']
        self.q = parameters.get('q', None)
        self.T = temperature
        self.Nc = len(self.m)
        self.device = device

        self.d = self.sigma*(1.0-0.12*(-3.0*self.epsilon/self.T).exp())
        if self.q is not None:
            self.q2 = 1e-19*self.q**2/(self.m*self.epsilon*kB*self.sigma**5)
        else:
            self.q2 = None

        C = mixing_tensors(self.m, self.sigma, self.epsilon, self.T, q2=self.q2, device=device)
        self.C = C
        self.m_ij = C['m_ij']
        self.m_ijk = C['m_ijk']
        self.sigma_ij = C['sigma_ij']
        self.epsilon_ij = C['epsilon_ij']
        self.anij = C['anij']
        self.bnij = C['bnij']
        self.cnijk = C['cnijk']

        self.d = self.d.to(device=device)
        self.m = C['m']
        self.sigma = C['sigma']
        self.epsilon = C['epsilon']
        self.half_d = 0.5*self.d

    # -----------------------------------------------------------------
    def helmholtz_energy(self, rho, x):

        C = self.C
        m, d = self.m, self.d

        m_bar = (x*m).sum()

        # Hard-Sphere Contribution
        c = (pi/6.)*rho
        xm = x*m
        zeta0 = c*xm.sum()
        zeta1 = c*(xm*d).sum()
        zeta2 = c*(xm*d**2).sum()
        zeta3 = c*(xm*d**3).sum()
        eta = torch.clone(zeta3)

        omzeta3 = 1.0-zeta3
        log_omzeta3 = omzeta3.log()
        zeta2_cu = zeta2**3
        f_hs = m_bar/zeta0*(3.*zeta1*zeta2/omzeta3
                            + zeta2_cu/(zeta3*omzeta3**2)
                            + (zeta2_cu/zeta3**2-zeta0)*log_omzeta3)

        # Hard-Chain Contribution
        dz = self.half_d*zeta2
        g_hs = 1.0/omzeta3+3.0*dz/omzeta3**2+2.0*dz**2/omzeta3**3
        f_hc = -(x*(m-1.0)*g_hs.log()).sum()

        # Dispersive Contribution
        C1 = 1.0+m_bar*(8.0*eta-2.0*eta**2)/(1.0-eta)**4 \
            + (1.0-m_bar)*(20.0*eta-27.0*eta**2+12.0*eta**3-2.0*eta**4)/((1.0-eta)*(2.0-eta))**2
        C1 = 1.0/C1

        # I1 = sum_i (a0_i + r*a1_i + r*s*a2_i) eta^i, with r,s independent of i
        r = (m_bar-1.0)/m_bar
        s = (m_bar-2.0)/m_bar
        ac, bc = C['a'], C['b']
        I1 = horner(ac[:, 0], eta)+r*horner(ac[:, 1], eta)+r*s*horner(ac[:, 2], eta)
        I2 = horner(bc[:, 0], eta)+r*horner(bc[:, 1], eta)+r*s*horner(bc[:, 2], eta)

        mix1 = torch.einsum('i,ij,j->', x, C['A1'], x)
        mix2 = torch.einsum('i,ij,j->', x, C['A2'], x)

        f_disp = (-2.0*I1*mix1-m_bar*C1*I2*mix2)*pi*rho

        # Quadrupolar Contribution
        if self.q is None:
            f_qq = 0.0
        else:
            poweta = torch.stack([eta**i for i in range(5)])
            f_q2 = torch.einsum('nij,i,j,n->', C['AB2'], x, x, poweta)
            f_q3 = torch.einsum('nijk,i,j,k,n->', C['C3'], x, x, x, poweta)
            f_q2 = -f_q2*pi*0.5625*rho
            f_q3 = f_q3*pi**2*0.5625*rho**2
            f_qq = f_q2/(1.0-f_q3/f_q2)

        return f_hs+f_hc+f_disp+f_qq

    # -----------------------------------------------------------------
    def _energy_and_derivatives(self, rho, x, need_x=False):
        """One forward pass, both derivatives. The original code evaluated
        helmholtz_energy up to three times per Newton step."""
        rho.requires_grad = True
        x.requires_grad = True
        f_res = self.helmholtz_energy(rho, x)
        if need_x:
            df_drho, df_dx = torch.autograd.grad(f_res, (rho, x), create_graph=True)
        else:
            df_drho = torch.autograd.grad(f_res, rho, create_graph=True)[0]
            df_dx = None
        return f_res, df_drho, df_dx

    def compressibility_factor(self, rho, x):
        _, df_drho, _ = self._energy_and_derivatives(rho, x)
        return 1.0+rho*df_drho

    def pressure(self, rho, x):
        Z = self.compressibility_factor(rho, x)
        P = Z*kB*self.T*rho*1e30  # Pa
        rho.requires_grad = False
        x.requires_grad = False
        return P.detach()

    def chemical_potential(self, rho, x):
        f_res, df_drho, df_dx = self._energy_and_derivatives(rho, x, need_x=True)
        Z = 1.0+rho*df_drho
        mu_res = f_res+(Z-1.0)+df_dx-(x*df_dx).sum()
        rho.requires_grad = False
        x.requires_grad = False
        return mu_res.detach()

    def fugacity_coefficient(self, rho, x):
        f_res, df_drho, df_dx = self._energy_and_derivatives(rho, x, need_x=True)
        Z = 1.0+rho*df_drho
        mu_res = f_res+(Z-1.0)+df_dx-(x*df_dx).sum()
        rho.requires_grad = False
        x.requires_grad = False
        return torch.exp(mu_res.detach())/Z.detach()

    # -----------------------------------------------------------------
    def residue(self, rho, x, Psys):
        Z = self.compressibility_factor(rho, x)
        return (Z*kB*self.T*rho-Psys)/Psys

    def diff_residue(self, rho, x, Psys):
        return torch.autograd.grad(self.residue(rho, x, Psys), rho)[0]

    def _residue_and_diff(self, rho, x, Psys):
        """res and dres/drho from a single forward pass."""
        rho.requires_grad = True
        x.requires_grad = True
        f_res = self.helmholtz_energy(rho, x)
        df_drho = torch.autograd.grad(f_res, rho, create_graph=True)[0]
        Z = 1.0+rho*df_drho
        res = (Z*kB*self.T*rho-Psys)/Psys
        dres = torch.autograd.grad(res, rho)[0]
        rho.requires_grad = False
        x.requires_grad = False
        return res.detach(), dres.detach()

    def density(self, P, x, phase, tol=1e-10, max_it=1000):

        if phase == 'vap':
            eta = 1e-10
            rho0 = eta/((pi/6.)*(x*self.m*self.d**3).sum())
        elif phase == 'liq':
            eta = 0.5
            rho0 = eta/((pi/6.)*(x*self.m*self.d**3).sum())
        else:
            rho0 = phase

        rho0 = torch.clone(rho0.detach())
        Psys = P*1e-30
        for _ in range(max_it):
            res, dres = self._residue_and_diff(rho0, x.detach(), Psys)
            rho0 = torch.clone(rho0-res/dres)
            if abs(res) < tol:
                break

        return rho0

    def vapor_pressure(self, P0, tol=1e-10, max_it=1000):

        x = torch.tensor([1.0], device=self.device)

        for _ in range(max_it):
            rhoV = self.density(P0, x, 'vap')
            rhoL = self.density(P0, x, 'liq')
            phiV = self.fugacity_coefficient(rhoV, x)
            phiL = self.fugacity_coefficient(rhoL, x)

            res = abs(phiL/phiV-1.0)
            P0 = torch.clone(P0*phiL/phiV)
            if res < tol:
                break

        return P0