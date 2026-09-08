import numpy as np
import torch
from torch import pi, float64
from torch.fft import rfft, irfft
from torch.autograd import grad
from scipy.special import spherical_jn
from .pcsaft_eos import pcsaft, mixing_tensors, horner
from .solvers import *

torch.set_default_dtype(torch.float64)

kB = 1.380649e-23
NA = 6.02214076e23
psi = 1.3862


def lancsoz(k, M):
    return np.sinc(k/M)


class dft_core():

    def __init__(self, pcsaft_parameters, temperature, system_size, points, device):

        self.pcsaft_parameters = pcsaft_parameters
        self.T = temperature
        self.system_size = system_size
        self.points = points
        self.device = device

        m0 = pcsaft_parameters['m']
        sigma0 = pcsaft_parameters['sigma']
        epsilon0 = pcsaft_parameters['epsilon']
        self.q = pcsaft_parameters.get('q', None)
        self.Nc = len(m0)

        d0 = sigma0*(1.0-0.12*np.exp(-3.0*epsilon0/self.T))
        q2_0 = None
        if self.q is not None:
            q2_0 = 1e-19*self.q**2/(m0*epsilon0*kB*sigma0**5)

        C = mixing_tensors(m0, sigma0, epsilon0, self.T, q2=q2_0, device=device)
        self.C = C
        self.m = C['m']
        self.sigma = C['sigma']
        self.epsilon = C['epsilon']
        self.sigma_ij = C['sigma_ij']
        self.epsilon_ij = C['epsilon_ij']
        self.d = d0.to(device=device)
        self.R = 0.5*self.d
        if q2_0 is not None:
            self.q2 = C['q2']

        self.spherical = (self.Nc == 1 and float(self.m[0]) == 1.0)

        # --- grid ------------------------------------------------------
        self.npz = int(points)                    # grid points along z
        self.shape = (self.Nc, self.npz)          # shape of rho
        self.npoints = self.Nc*self.npz           # used by the shared solvers
        self.sqrt_npoints = np.sqrt(self.npoints)

        self.cell_size = system_size/points
        self.z = torch.linspace(0.5*self.cell_size, system_size-0.5*self.cell_size, self.npz, device=device)

        # rho is real: half spectrum only
        kz = np.fft.rfftfreq(self.npz, d=self.cell_size)
        k = np.abs(kz)

        # M = number of non-redundant k-values. The original used kz.max(),
        # which for even N is (N/2-1)/L instead of (N/2+1)/L.
        M = self.npz//2+1
        kcut = M/self.system_size
        lanczos_term = lancsoz(kz, kcut)**self.lanczos_power

        Rn = np.asarray(0.5*d0)

        # every weight below is purely REAL -- stored as real tensors
        nk = len(k)
        w2_hat = np.empty((self.Nc, nk))
        w3_hat = np.empty_like(w2_hat)
        w2hc_hat = np.empty_like(w2_hat)
        w3hc_hat = np.empty_like(w2_hat)
        wdisp_hat = np.empty_like(w2_hat)

        for i in range(self.Nc):
            Ri = float(Rn[i])
            j0_2 = spherical_jn(0, 2.*np.pi*Ri*k)
            j2_2 = spherical_jn(2, 2.*np.pi*Ri*k)
            j0_4 = spherical_jn(0, 4.*np.pi*Ri*k)
            j2_4 = spherical_jn(2, 4.*np.pi*Ri*k)
            j0_4p = spherical_jn(0, 4.*np.pi*psi*Ri*k)
            j2_4p = spherical_jn(2, 4.*np.pi*psi*Ri*k)
            w2_hat[i] = 4.0*np.pi*Ri**2*j0_2*lanczos_term
            w3_hat[i] = (4./3.)*np.pi*Ri**3*(j0_2+j2_2)*lanczos_term
            w2hc_hat[i] = j0_4*lanczos_term
            w3hc_hat[i] = (j0_4+j2_4)*lanczos_term
            wdisp_hat[i] = (j0_4p+j2_4p)*lanczos_term

        kvec = 2.0*np.pi*kz.copy()
        if self.npz % 2 == 0:
            kvec[-1] = 0.0        # Nyquist bin: odd kernel has no fixed sign

        self.w2_hat = torch.tensor(w2_hat, device=device)
        self.w3_hat = torch.tensor(w3_hat, device=device)
        self.w2hc_hat = torch.tensor(w2hc_hat, device=device)
        self.w3hc_hat = torch.tensor(w3hc_hat, device=device)
        self.wdisp_hat = torch.tensor(wdisp_hat, device=device)
        self.kvec = torch.tensor(kvec, device=device)

        del kz, k, lanczos_term, kvec
        del w2_hat, w3_hat, w2hc_hat, w3hc_hat, wdisp_hat

        self._m2 = self.m[:, None]
        self._d2 = self.d[:, None]
        self._R2 = self.R[:, None]

    # -----------------------------------------------------------------
    def weighted_densities(self):

        self.rho.requires_grad = True

        self.rho_hat = rfft(self.rho, dim=1)
        n = self.npz

        ni = self._m2*irfft(self.rho_hat*self.w2_hat, n=n, dim=1)
        self.n0 = (ni/(4.*np.pi*self._R2**2)).sum(dim=0)
        self.n1 = (ni/(4.*np.pi*self._R2)).sum(dim=0)
        self.n2 = ni.sum(dim=0)

        rho_w3 = self.rho_hat*self.w3_hat
        self.n3 = (self._m2*irfft(rho_w3, n=n, dim=1)).sum(dim=0).clamp(max=1.0-1e-16)
        nivec = self._m2*irfft(-1j*(self.kvec[None]*rho_w3), n=n, dim=1)
        self.n1vec = (nivec/(4.*np.pi*self._R2)).sum(dim=0)
        self.n2vec = nivec.sum(dim=0)

        self.n2_hc = irfft(self.rho_hat*self.w2hc_hat, n=n, dim=1)
        self.n3_hc = irfft(self.rho_hat*self.w3hc_hat, n=n, dim=1)
        self.ni_disp = irfft(self.rho_hat*self.wdisp_hat, n=n, dim=1)

    # -----------------------------------------------------------------
    def functional(self, fmt):

        self.weighted_densities()
        C = self.C

        # ---- Hard-Sphere -------------------------------------------
        one_minus_n3 = 1.0-self.n3
        f1 = -torch.log(one_minus_n3)
        f2 = one_minus_n3.reciprocal()

        n3s = self.n3.clamp(min=1e-4)
        omn3s = 1.0-n3s
        omn3s_sq = omn3s*omn3s
        f4 = torch.where(self.n3 > 1e-4,
                         (n3s+omn3s_sq*torch.log(omn3s))/(36.0*np.pi*n3s*n3s*omn3s_sq),
                         1/(24*np.pi)+2/(27*np.pi)*self.n3+5/(48*np.pi)*self.n3**2)

        # in a mixture n1 and n2 carry different per-component weights, so the
        # (n2^2 - n2vec^2)/(4 pi R) identity of the LJ code does NOT apply
        n1_n2 = self.n1*self.n2
        n2_sq = self.n2*self.n2
        n2vec_sq = (self.n2vec*self.n2vec).clamp(max=n2_sq)
        n1vec_n2vec = (self.n1vec*self.n2vec).clamp(max=n1_n2)

        if fmt == 'WB':
            self.Phi_hs = f1*self.n0+f2*(n1_n2-n1vec_n2vec) \
                + f4*(n2_sq*self.n2-3.0*self.n2*n2vec_sq)
        elif fmt == 'ASWB':
            xi = (n2vec_sq/n2_sq).clamp(max=1.0-1e-16)
            self.Phi_hs = f1*self.n0+f2*(n1_n2-n1vec_n2vec) \
                + f4*(self.n2*n2_sq)*(1.0-xi)**3
        else:
            raise ValueError("fmt must be 'WB' or 'ASWB'")

        self.F_hs = self.Phi_hs.sum()*self.cell_size

        # ---- Hard-Chain --------------------------------------------
        if self.spherical:
            self.Phi_hc = torch.zeros_like(self.Phi_hs)
        else:
            zeta2 = (np.pi/6.)*torch.einsum('i...,i->...', self.n3_hc, self.m*self.d**2)
            zeta3 = ((np.pi/6.)*torch.einsum('i...,i->...', self.n3_hc, self.m*self.d**3)) \
                .clamp(max=1.0-1e-16)
            omz = 1.0-zeta3
            dz = self._d2*zeta2
            ydd = 1.0/omz+1.5*dz/omz**2+0.5*dz*dz/omz**3
            # n2_hc can ring slightly negative near a wall; one NaN from the
            # log would poison the whole sum
            ydd_n2 = (ydd*self.n2_hc).clamp(min=1e-300)
            self.Phi_hc = ((self._m2-1.0)*self.rho
                           * (torch.log(self.rho)-torch.log(ydd_n2))).sum(dim=0)

        self.F_hc = self.Phi_hc.sum()*self.cell_size

        # ---- Dispersive --------------------------------------------
        n_disp = self.ni_disp.sum(dim=0)
        xbar = self.ni_disp/n_disp.clamp(min=1e-300)
        mbar = torch.einsum('i...,i->...', xbar, self.m)
        etabar = ((np.pi/6.0)*torch.einsum('i...,i->...', self.ni_disp, self.m*self.d**3)) \
            .clamp(max=1.0-1e-16)

        r = (mbar-1.0)/mbar
        s_ = (mbar-2.0)/mbar
        ac, bc = C['a'], C['b']
        I1 = horner(ac[:, 0], etabar)+r*horner(ac[:, 1], etabar)+r*s_*horner(ac[:, 2], etabar)
        I2 = horner(bc[:, 0], etabar)+r*horner(bc[:, 1], etabar)+r*s_*horner(bc[:, 2], etabar)

        om_eta = 1.0-etabar
        C1 = (1.0+mbar*(8.0*etabar-2.0*etabar**2)/om_eta**4
              + (1.0-mbar)*(20.0*etabar-27.0*etabar**2+12.0*etabar**3-2.0*etabar**4)
              / (om_eta*(2.0-etabar))**2).reciprocal()

        mix1 = torch.einsum('i...,ij,j...->...', xbar, C['A1'], xbar)
        mix2 = torch.einsum('i...,ij,j...->...', xbar, C['A2'], xbar)

        a_disp = (-2.0*I1*mix1-mbar*C1*I2*mix2)*np.pi*n_disp
        self.Phi_disp = n_disp*a_disp
        self.F_disp = self.Phi_disp.sum()*self.cell_size

        # ---- Quadrupolar -------------------------------------------
        if self.q is None:
            self.Phi_qq = torch.zeros_like(self.Phi_hs)
        else:
            f_q2 = torch.zeros_like(n_disp)
            f_q3 = torch.zeros_like(n_disp)
            p = torch.ones_like(etabar)
            for nn in range(5):
                f_q2 = f_q2+p*torch.einsum('ij,i...,j...->...', C['AB2'][nn], xbar, xbar)
                f_q3 = f_q3+p*torch.einsum('ijk,i...,j...,k...->...', C['C3'][nn], xbar, xbar, xbar)
                if nn < 4:
                    p = p*etabar

            f_q2 = -f_q2*np.pi*0.5625*n_disp
            f_q3 = f_q3*np.pi**2*0.5625*n_disp**2
            self.Phi_qq = n_disp*(f_q2/(1.0-f_q3/f_q2))

        self.F_qq = self.Phi_qq.sum()*self.cell_size

        self.Fres = self.F_hs+self.F_hc+self.F_disp+self.F_qq

    # -----------------------------------------------------------------
# -----------------------------------------------------------------
    def functional_derivative(self, fmt):

        self.functional(fmt)
        self.dFres = torch.autograd.grad(self.Fres, self.rho)[0]
        self.dFres = self.dFres.detach()/self.cell_size
        self.rho.requires_grad = False

    def euler_lagrange(self, lnrho, fmt='ASWB'):

        self.functional_derivative(fmt)
        self.res = (self.mu[:, None]-self.dFres-self.Vext-lnrho)*self.valid

    def loss(self):
        return torch.linalg.vector_norm(self.res)/self.sqrt_npoints

    # -----------------------------------------------------------------
    def initial_condition(self, bulk_density, composition, Vext, potential_cutoff=50.0):

        self.rhob = bulk_density*composition
        self.eos = pcsaft(self.pcsaft_parameters, self.T, device=self.device)
        self.mu = (self.eos.chemical_potential(bulk_density, composition)
                   + torch.log(self.rhob)).to(device=self.device)
        self.rhob = self.rhob.to(device=self.device)

        self.Vext = (Vext/self.T).to(device=self.device)
        self.excluded = self.Vext >= potential_cutoff
        self.valid = self.Vext < potential_cutoff
        self.Vext[self.excluded] = potential_cutoff

        self.rho = torch.empty(self.shape, device=self.device)
        for i in range(self.Nc):
            self.rho[i] = self.rhob[i]

    def equilibrium_density_profile(self, bulk_density, composition, fmt='ASWB',
                                    solver='anderson', alpha0=0.2, dt=0.1,
                                    anderson_mmax=10, anderson_damping=0.1,
                                    tol=1e-6, max_it=1000, logoutput=False):

        self.rhob = (bulk_density*composition).to(device=self.device)
        self.mu = (self.eos.chemical_potential(bulk_density, composition)
                   + torch.log(bulk_density*composition)).to(device=self.device)
        self.fmt = fmt

        self.rho = self.rho.detach().clone()
        self.rho[self.excluded] = 1e-15

        if solver == 'picard':
            picard(self, alpha0, tol, max_it, logoutput)
        elif solver == 'picard_ls':
            picard_line_search(self, alpha0, tol, max_it, logoutput)
        elif solver == 'anderson':
            anderson(self, anderson_mmax, anderson_damping, tol, max_it, logoutput)
        elif solver == 'fire':
            fire(self, alpha0, dt, tol, max_it, logoutput)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.error = self.error.cpu()

        self.total_molecules = torch.empty(self.Nc)
        Phi = torch.zeros_like(self.Phi_disp)
        for i in range(self.Nc):
            self.total_molecules[i] = (self.rho[i]*self.valid[i]).sum().cpu()*self.cell_size
            Phi += self.rho[i]*(torch.log(self.rho[i])-1.0) \
                + self.rho[i]*(self.Vext[i]-self.mu[i])

        self.Omega = Phi.sum()*self.cell_size+self.Fres.detach()