import numpy as np
import torch
from scipy.special import spherical_jn
from .pcsaft_eos import pcsaft, mixing_tensors, horner
from .solvers import *

torch.set_default_dtype(torch.float64)
psi = 1.3862
pi = np.pi


def lancsoz(kx, ky, kz, M):
    return np.sinc(kx/M[0])*np.sinc(ky/M[1])*np.sinc(kz/M[2])


class dft_core():

    def __init__(self, pcsaft_parameters, temperature, system_size, points, device):

        self.pcsaft_parameters = pcsaft_parameters
        self.T = temperature
        self.system_size = system_size
        self.points = points
        self.device = device

        self.kB = 1.380649e-23
        self.NA = 6.02214076e23

        m0 = pcsaft_parameters['m']
        sigma0 = pcsaft_parameters['sigma']
        epsilon0 = pcsaft_parameters['epsilon']
        self.q = pcsaft_parameters.get('q', None)
        self.Nc = len(m0)

        d0 = sigma0*(1.0-0.12*np.exp(-3.0*epsilon0/self.T))
        q2_0 = None
        if self.q is not None:
            q2_0 = 1e-19*self.q**2/(m0*epsilon0*self.kB*sigma0**5)

        # every combining rule and every constant that depends only on
        # (parameters, T) is built once, on device -- shared with pcsaft_eos
        C = mixing_tensors(m0, sigma0, epsilon0, self.T, q2=q2_0,device=device)
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

        # is this a pure spherical fluid? then the chain term vanishes
        self.spherical = (self.Nc == 1 and float(self.m[0]) == 1.0)

        # --- grid ------------------------------------------------------
        self.grid_shape = tuple(int(p) for p in points)
        self.shape = (self.Nc,)+self.grid_shape          # shape of rho
        self.npoints = self.Nc*int(np.prod(self.grid_shape))
        self.sqrt_npoints = np.sqrt(self.npoints)

        self.system_volume = self.system_size.prod()
        self.cell_size = system_size/points
        self.cell_volume = self.cell_size.prod()

        n = self.grid_shape
        self.x = torch.linspace(0.5*self.cell_size[0], system_size[0]-0.5*self.cell_size[0], n[0], device=device)
        self.y = torch.linspace(0.5*self.cell_size[1], system_size[1]-0.5*self.cell_size[1], n[1], device=device)
        self.z = torch.linspace(0.5*self.cell_size[2], system_size[2]-0.5*self.cell_size[2], n[2], device=device)
        self.X, self.Y, self.Z = torch.meshgrid(self.x, self.y, self.z, indexing='ij')

        # rho is real: only the non-redundant half of the last axis is needed
        kx = np.fft.fftfreq(n[0], d=self.cell_size[0])
        ky = np.fft.fftfreq(n[1], d=self.cell_size[1])
        kz = np.fft.rfftfreq(n[2], d=self.cell_size[2])
        Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
        K = np.sqrt(Kx**2+Ky**2+Kz**2)

        # M = number of non-redundant k-values per axis (Lanczos sigma-factor).
        # N//2+1 is exact for both parities.
        M = np.asarray(self.grid_shape)//2+1
        kcut = M/self.system_size
        # must be evaluated on the 3D meshgrid, not on the 1D frequency axes
        lanczos_term = lancsoz(Kx, Ky, Kz, kcut)
        Rn = d0.numpy()*0.5 if hasattr(d0, 'numpy') else np.asarray(0.5*d0)

        # all of these are purely REAL: stored as real tensors, which halves
        # the memory and turns complex*complex into complex*real
        shape_k = K.shape
        w2_hat = np.empty((self.Nc,)+shape_k)
        w3_hat = np.empty_like(w2_hat)
        w2hc_hat = np.empty_like(w2_hat)
        w3hc_hat = np.empty_like(w2_hat)
        wdisp_hat = np.empty_like(w2_hat)

        for i in range(self.Nc):
            Ri = float(Rn[i])
            j0_2 = spherical_jn(0, 2.*pi*Ri*K)
            j2_2 = spherical_jn(2, 2.*pi*Ri*K)
            j0_4 = spherical_jn(0, 4.*pi*Ri*K)
            j2_4 = spherical_jn(2, 4.*pi*Ri*K)
            j0_4p = spherical_jn(0, 4.*pi*psi*Ri*K)
            j2_4p = spherical_jn(2, 4.*pi*psi*Ri*K)
            w2_hat[i] = 4.0*pi*Ri**2*j0_2*lanczos_term
            w3_hat[i] = (4./3.)*pi*Ri**3*(j0_2+j2_2)*lanczos_term
            w2hc_hat[i] = j0_4*lanczos_term
            w3hc_hat[i] = (j0_4+j2_4)*lanczos_term
            wdisp_hat[i] = (j0_4p+j2_4p)*lanczos_term

        # w2vec_hat = -i 2 pi K w3_hat is purely imaginary; only 2 pi K is
        # stored (3 real arrays instead of 3*Nc complex ones)
        kvec = 2.0*pi*np.stack([Kx, Ky, Kz])
        # on an even axis the Nyquist bin is its own conjugate partner, so the
        # odd gradient kernel has no consistent sign there; zeroing it is the
        # standard treatment and makes rfftn match the full complex transform
        if n[0] % 2 == 0:
            kvec[:, n[0]//2, :, :] = 0.0
        if n[1] % 2 == 0:
            kvec[:, :, n[1]//2, :] = 0.0
        if n[2] % 2 == 0:
            kvec[:, :, :, -1] = 0.0

        self.w2_hat = torch.tensor(w2_hat, device=device)
        self.w3_hat = torch.tensor(w3_hat, device=device)
        self.w2hc_hat = torch.tensor(w2hc_hat, device=device)
        self.w3hc_hat = torch.tensor(w3hc_hat, device=device)
        self.wdisp_hat = torch.tensor(wdisp_hat, device=device)
        self.kvec = torch.tensor(kvec, device=device)

        del kx, ky, kz, Kx, Ky, Kz, K, lanczos_term, kvec
        del w2_hat, w3_hat, w2hc_hat, w3hc_hat, wdisp_hat

        # broadcast helpers
        self._m4 = self.m[:, None, None, None]
        self._d4 = self.d[:, None, None, None]
        self._R4 = self.R[:, None, None, None]

    # -----------------------------------------------------------------
    def weighted_densities(self):

        self.rho.requires_grad = True

        self.rho_hat = torch.fft.rfftn(self.rho, dim=(1, 2, 3))
        s = self.grid_shape

        ni = self._m4*torch.fft.irfftn(self.rho_hat*self.w2_hat, dim=(1, 2, 3), s=s)
        self.n0 = (ni/(4.*pi*self._R4**2)).sum(dim=0)
        self.n1 = (ni/(4.*pi*self._R4)).sum(dim=0)
        self.n2 = ni.sum(dim=0)

        # n3 and n2vec share the same product rho_hat*w3_hat
        rho_w3 = self.rho_hat*self.w3_hat
        self.n3 = (self._m4*torch.fft.irfftn(rho_w3, dim=(1, 2, 3), s=s)).sum(dim=0) \
            .clamp(max=1.0-1e-15)
        nivec = self.m[:, None, None, None, None] \
            * torch.fft.irfftn(-1j*(self.kvec[None]*rho_w3[:, None]), dim=(2, 3, 4), s=s)
        self.n1vec = (nivec/(4.*pi*self.R[:, None, None, None, None])).sum(dim=0)
        self.n2vec = nivec.sum(dim=0)

        self.n2_hc = torch.fft.irfftn(self.rho_hat*self.w2hc_hat, dim=(1, 2, 3), s=s)
        self.n3_hc = torch.fft.irfftn(self.rho_hat*self.w3hc_hat, dim=(1, 2, 3), s=s)
        self.ni_disp = torch.fft.irfftn(self.rho_hat*self.wdisp_hat, dim=(1, 2, 3), s=s)

    # -----------------------------------------------------------------
    def functional(self, fmt):

        self.weighted_densities()
        C = self.C

        # ---- Hard-Sphere -------------------------------------------
        one_minus_n3 = 1.0-self.n3
        f1 = -torch.log(one_minus_n3)
        f2 = one_minus_n3.reciprocal()

        # f4 loses all precision as n3 -> 0; both branches of the where are
        # evaluated on a clamped n3 so neither value nor gradient can be NaN
        n3s = self.n3.clamp(min=1e-4)
        omn3s = 1.0-n3s
        omn3s_sq = omn3s*omn3s
        f4 = torch.where(self.n3 > 1e-4,
                         (n3s+omn3s_sq*torch.log(omn3s))/(36.0*pi*n3s*n3s*omn3s_sq),
                         1/(24*pi)+2/(27*pi)*self.n3+5/(48*pi)*self.n3**2)

        # NOTE: in a mixture n1 and n2 carry different per-component weights,
        # so the (n2^2 - n2vec^2)/(4 pi R) identity used in the LJ code does
        # NOT hold here. n1 and n1vec are kept.
        n1_n2 = self.n1*self.n2
        n2_sq = self.n2*self.n2
        n2vec_sq = (self.n2vec*self.n2vec).sum(dim=0).clamp(max=n2_sq)
        n1vec_n2vec = (self.n1vec*self.n2vec).sum(dim=0).clamp(max=n1_n2)

        if fmt == 'WB':
            self.Phi_hs = f1*self.n0+f2*(n1_n2-n1vec_n2vec) \
                + f4*(n2_sq*self.n2-3.0*self.n2*n2vec_sq)
        elif fmt == 'ASWB':
            xi = (n2vec_sq/n2_sq).clamp(max=1.0-1e-16)
            self.Phi_hs = f1*self.n0+f2*(n1_n2-n1vec_n2vec) \
                + f4*(self.n2*n2_sq)*(1.0-xi)**3
        else:
            raise ValueError("fmt must be 'WB' or 'ASWB'")

        self.F_hs = self.Phi_hs.sum()*self.cell_volume

        # ---- Hard-Chain --------------------------------------------
        if self.spherical:
            self.Phi_hc = torch.zeros_like(self.Phi_hs)
        else:
            zeta2 = (pi/6.)*torch.einsum('i...,i->...', self.n3_hc, self.m*self.d**2)
            zeta3 = ((pi/6.)*torch.einsum('i...,i->...', self.n3_hc, self.m*self.d**3)) \
                .clamp(max=1.0-1e-15)
            omz = 1.0-zeta3
            dz = self._d4*zeta2
            ydd = 1.0/omz+1.5*dz/omz**2+0.5*dz*dz/omz**3
            # n2_hc can ring slightly negative near a wall; log of that is NaN
            # and one NaN poisons the whole sum
            ydd_n2 = (ydd*self.n2_hc).clamp(min=1e-300)
            self.Phi_hc = ((self._m4-1.0)*self.rho
                           * (torch.log(self.rho)-torch.log(ydd_n2))).sum(dim=0)

        self.F_hc = self.Phi_hc.sum()*self.cell_volume

        # ---- Dispersive --------------------------------------------
        n_disp = self.ni_disp.sum(dim=0)
        xbar = self.ni_disp/n_disp.clamp(min=1e-300)
        mbar = torch.einsum('i...,i->...', xbar, self.m)
        etabar = ((pi/6.0)*torch.einsum('i...,i->...', self.ni_disp, self.m*self.d**3)) \
            .clamp(max=1.0-1e-15)

        # I1 = sum_i (a0_i + r a1_i + r s a2_i) eta^i, with r and s independent
        # of i: three Horner polynomials instead of 7 pow() kernels each
        r = (mbar-1.0)/mbar
        s_ = (mbar-2.0)/mbar
        ac, bc = C['a'], C['b']
        I1 = horner(ac[:, 0], etabar)+r*horner(ac[:, 1], etabar)+r*s_*horner(ac[:, 2], etabar)
        I2 = horner(bc[:, 0], etabar)+r*horner(bc[:, 1], etabar)+r*s_*horner(bc[:, 2], etabar)

        om_eta = 1.0-etabar
        C1 = (1.0+mbar*(8.0*etabar-2.0*etabar**2)/om_eta**4
              + (1.0-mbar)*(20.0*etabar-27.0*etabar**2+12.0*etabar**3-2.0*etabar**4)
              / (om_eta*(2.0-etabar))**2).reciprocal()

        # A1 and A2 fold m_i m_j (eps_ij/T)^n sigma_ij^3, all constant
        mix1 = torch.einsum('i...,ij,j...->...', xbar, C['A1'], xbar)
        mix2 = torch.einsum('i...,ij,j...->...', xbar, C['A2'], xbar)

        a_disp = (-2.0*I1*mix1-mbar*C1*I2*mix2)*pi*n_disp
        self.Phi_disp = n_disp*a_disp
        self.F_disp = self.Phi_disp.sum()*self.cell_volume

        # ---- Quadrupolar -------------------------------------------
        if self.q is None:
            self.Phi_qq = torch.zeros_like(self.Phi_hs)
        else:
            # every per-component constant is folded into AB2 and C3 at init,
            # so the 10- and 16-operand einsums collapse to one quadratic and
            # one cubic form per power of eta
            f_q2 = torch.zeros_like(n_disp)
            f_q3 = torch.zeros_like(n_disp)
            p = torch.ones_like(etabar)
            for nn in range(5):
                f_q2 = f_q2+p*torch.einsum('ij,i...,j...->...', C['AB2'][nn], xbar, xbar)
                f_q3 = f_q3+p*torch.einsum('ijk,i...,j...,k...->...', C['C3'][nn], xbar, xbar, xbar)
                if nn < 4:
                    p = p*etabar

            f_q2 = -f_q2*pi*0.5625*n_disp
            f_q3 = f_q3*pi**2*0.5625*n_disp**2
            self.Phi_qq = n_disp*(f_q2/(1.0-f_q3/f_q2))

        self.F_qq = self.Phi_qq.sum()*self.cell_volume

        self.Fres = self.F_hs+self.F_hc+self.F_disp+self.F_qq

    # -----------------------------------------------------------------
    def functional_derivative(self, fmt):

        self.functional(fmt)
        self.dFres = torch.autograd.grad(self.Fres, self.rho)[0]
        self.dFres = self.dFres.detach()/self.cell_volume
        self.rho.requires_grad = False

    def euler_lagrange(self, lnrho, fmt='ASWB'):

        self.functional_derivative(fmt)
        # 0/1 mask instead of boolean indexing: advanced indexing calls
        # nonzero() internally and forces a device sync on every use
        self.res = (self.mu[:, None, None, None]-self.dFres-self.Vext-lnrho)*self.valid

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
            self.total_molecules[i] = (self.rho[i]*self.valid[i]).sum().cpu()*self.cell_volume
            Phi += self.rho[i]*(torch.log(self.rho[i])-1.0) \
                + self.rho[i]*(self.Vext[i]-self.mu[i])

        self.Omega = Phi.sum()*self.cell_volume+self.Fres.detach()