import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import torch

CSV_COLUMNS = ['N_target', 'N_real', 'mu', 'P_Pa', 'Omega',
               'F_intr', 'F_ext', 'F', 'error', 'iterations']


def bulk_density_from_mu(eos, mu, rho_guess, tol=1e-12, max_it=200):

    mu = float(mu)

    def f(rho):
        r = torch.tensor([float(rho)])
        return float(torch.log(r)+eos.chemical_potential(r))-mu

    r0 = float(rho_guess)
    r1 = r0*1.01
    f0, f1 = f(r0), f(r1)
    for _ in range(max_it):
        if abs(f1-f0) < 1e-300:
            break
        r2 = r1-f1*(r1-r0)/(f1-f0)
        if r2 <= 0.0:
            r2 = 0.5*r1
        r0, f0, r1 = r1, f1, r2
        f1 = f(r1)
        if abs(f1) < tol:
            break
    return r1


def check_canonical(dft, bulk_density, fmt='ASWB', solver='anderson', **kw):

    dft.equilibrium_density_profile(bulk_density, fmt=fmt, solver=solver, **kw)
    N = float(dft.total_molecules)
    mu_gc = float(dft.mu)

    dft.equilibrium_density_profile(fmt=fmt, solver=solver, N_target=N, **kw)
    mu_c = float(dft.mu)
    N_back = float(dft.total_molecules)

    print("grand canonical : mu = %.10f   N = %.10f" % (mu_gc, N))
    print("canonical (fixed N): mu = %.10f   N = %.10f" % (mu_c, N_back))
    print("mu error = %.3e   erro em N = %.3e"
          % (abs(mu_c-mu_gc), abs(N_back-N)))
    return abs(mu_c-mu_gc), abs(N_back-N)


def sweep_N(dft, N_values, rho_start=None, fmt='ASWB', solver='anderson',
            tol=1e-8, rho_bulk_guess=1e-8, logoutput=False,
            csv_path='sweep_N.csv', overwrite=True, **solver_kw):
    """Canonical sweep over N.

    Each converged point is written to csv_path as soon as it converges (one
    row per point), so an interrupted run keeps everything already computed.
    Columns: see CSV_COLUMNS.

    csv_path : output file; None disables writing.
    overwrite: True recreates the file (header only) at the start;
               False appends to an existing file (to resume a run).

    Points that raise or do not converge are only printed, never written.
    The return value is unchanged: (mu, P, Om, N_real, ok).
    """
    n = len(N_values)
    mu = np.full(n, np.nan)
    P = np.full(n, np.nan)
    Om = np.full(n, np.nan)
    N_real = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)

    if csv_path is not None and (overwrite or not os.path.exists(csv_path)):
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(csv_path, index=False)

    if rho_start is not None:
        dft.rho = rho_start.clone()
    guess = dft.rho.detach().clone()
    rb = rho_bulk_guess

    for i, N in enumerate(N_values):
        dft.rho = guess.clone()
        try:
            dft.equilibrium_density_profile(fmt=fmt, solver=solver, tol=tol,
                                            logoutput=logoutput, N_target=N,
                                            **solver_kw)
        except Exception as exc:
            print("  N=%.6f raised an exception: %s" % (N, exc))
            continue

        err = float(dft.error)
        if not (np.isfinite(err) and err <= tol):
            print("  N=%.6f did not converge (err=%.2e)" % (N, err))
            continue

        mu[i] = float(dft.mu)
        Om[i] = float(dft.Omega)
        N_real[i] = float(dft.total_molecules)
        rb = bulk_density_from_mu(dft.eos, mu[i], rb)
        P[i] = float(dft.eos.pressure(torch.tensor([rb])))
        ok[i] = True
        guess = dft.rho.detach().clone()

        if csv_path is not None:
            row = {'N_target': float(N), 'N_real': N_real[i], 'mu': mu[i],
                   'P_Pa': P[i], 'Omega': Om[i],
                   'F_intr': float(dft.F_intr), 'F_ext': float(dft.F_ext),
                   'F': float(dft.F), 'error': err,
                   'iterations': int(getattr(dft, 'it', -1))}
            # mode 'a' opens, writes and closes: the row is on disk on return
            pd.DataFrame([row], columns=CSV_COLUMNS).to_csv(
                csv_path, mode='a', header=False, index=False,
                float_format='%.12e')

        if logoutput:
            print("N=%.6f  mu=%.6f  P=%.6e Pa  |N-N_alvo|=%.2e"
                  % (N, mu[i], P[i], abs(N_real[i]-N)))

    return mu, P, Om, N_real, ok


def read_sweep(csv_path):
    """Read one or more sweep_N CSV files back, sorted by N_real, with
    duplicate N_target rows (e.g. a resumed run) reduced to the last one.

    csv_path: a path or a list of paths (e.g. an adsorption and a desorption
    sweep written to separate files).
    """
    paths = [csv_path] if isinstance(csv_path, (str, os.PathLike)) else csv_path
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates(subset='N_target', keep='last')
    return df.sort_values('N_real').reset_index(drop=True)


def maxwell_construction(N, mu):
    """Equilibrium transition by the equal-area construction in the mu(N) loop.

    Condition:   integral_{N1}^{N2} [ mu(N) - mu_eq ] dN = 0

    where N1 and N2 are the points at which mu(N) = mu_eq on the TWO STABLE BRANCHES.
    The integral is taken between N1 and N2, not over the entire array: outside this
    interval, mu - mu_eq does not change sign, and the two external contributions
    only cancel if the loop is symmetric. For an asymmetric loop -- which is the
    realistic case -- integrating over the entire array shifts mu_eq.

    This is equivalent to the DOUBLE-TANGENT construction in F(N): the line that
    touches F at N1 and N2 has slope mu_eq, and the physical F in the interval is
    given by this line (the convex envelope of F).

    Independent verification: with the correct mu_eq, the two minima of
    W(N) = F(N) - mu_eq N are at the SAME HEIGHT.
    """

    N = np.asarray(N, dtype=float)
    mu = np.asarray(mu, dtype=float)
    good = np.isfinite(mu) & np.isfinite(N)
    N, mu = N[good], mu[good]
    o = np.argsort(N)
    N, mu = N[o], mu[o]

    d = np.diff(mu)
    turns = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]+1
    if len(turns) < 2:
        return {'hysteresis': False}
    i1, i2 = turns[0], turns[-1]          # espinodais
    mu_sp_ads, mu_sp_des = mu[i1], mu[i2]

    def area(m):
        N1 = brentq(lambda z: np.interp(z, N[:i1+1], mu[:i1+1])-m, N[0], N[i1])
        N2 = brentq(lambda z: np.interp(z, N[i2:], mu[i2:])-m, N[i2], N[-1])
        s = (N >= N1) & (N <= N2)
        Ns = np.r_[N1, N[s], N2]
        ms = np.r_[m, mu[s], m]
        return np.trapezoid(ms-m, Ns), N1, N2

    # mu_eq must be reachable on BOTH stable branches: the low-N branch spans
    # [mu[0], mu[i1]] and the high-N branch [mu[i2], mu[-1]]. Using only the
    # spinodals as the bracket fails when the sweep stops before a branch
    # reaches the other spinodal's mu.
    lo = max(min(mu[0], mu[i1]), min(mu[i2], mu[-1]))
    hi = min(max(mu[0], mu[i1]), max(mu[i2], mu[-1]))
    eps = 1e-12*(hi-lo)
    if not hi > lo or area(lo+eps)[0]*area(hi-eps)[0] > 0.0:
        raise ValueError(
            "the sweep does not extend far enough along the stable branches "
            "to close the equal-area construction: extend N_values below "
            "N=%.6g and/or above N=%.6g" % (N[0], N[-1]))
    mu_eq = brentq(lambda m: area(m)[0], lo+eps, hi-eps, xtol=1e-14)
    _, N1, N2 = area(mu_eq)

    return {'hysteresis': True, 'mu_eq': mu_eq,
            'N_coex_vap': N1, 'N_coex_liq': N2,
            'mu_spinodal_ads': mu_sp_ads, 'N_spinodal_ads': N[i1],
            'mu_spinodal_des': mu_sp_des, 'N_spinodal_des': N[i2]}


def check_maxwell(N, mu, Om, mu_eq, F=None):
    """Independent check of the Maxwell construction, plus the barrier.

    With the correct mu_eq the two minima of W = F - mu_eq N sit at the
    SAME height, so dW should be ~0. The maximum between them is the
    nucleation barrier, in kT -- the quantity the grand canonical route
    cannot provide.

    F: optional total Helmholtz energy (the 'F' column of sweep_N's CSV).
       If given it is used directly; otherwise F = Omega + mu N, which is
       the same quantity reconstructed from Omega.
    """
    N = np.asarray(N, float)
    if F is None:
        F = np.asarray(Om, float)+np.asarray(mu, float)*N
    W = np.asarray(F, float)-mu_eq*N
    d = np.diff(W)
    t = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]+1
    # Classify by the SIGN of the derivative, not by position in the list:
    # a numerical plateau at the top yields two indices for the same
    # maximum, and then t[0], t[1], t[2] stops being (min, max, min).
    mins = [i for i in t if d[i] > 0]
    maxs = [i for i in t if d[i] < 0]
    if len(mins) < 2 or len(maxs) < 1:
        return None
    i1, i2 = mins[0], mins[-1]
    im = max(maxs, key=lambda i: W[i])
    return dict(dW=W[i2]-W[i1],
                barrier=W[im]-0.5*(W[i1]+W[i2]),
                N_min1=N[i1], N_max=N[im], N_min2=N[i2])