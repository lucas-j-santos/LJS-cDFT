import numpy as np
import torch


# =====================================================================
# Varredura em N com o dft3d no modo canonico.
#
# Percorre a isoterma inteira -- ramo instavel incluido -- e devolve
# mu(N), de onde sai P(N) invertendo a EOS bulk. Os extremos de mu(N)
# sao as espinodais; a construcao de Maxwell da a transicao de
# equilibrio.
#
# NAO TESTADO: valide primeiro com o check_canonical() abaixo.
# =====================================================================


def bulk_density_from_mu(eos, mu, rho_guess, tol=1e-12, max_it=200):
    """Inverte  mu = ln(rho_b) + mu_ex(rho_b)  por secante.

    mu(rho) e monotono crescente no gas estavel, entao a secante basta.
    """
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
            r2 = 0.5*r1                      # mantem positivo
        r0, f0, r1 = r1, f1, r2
        f1 = f(r1)
        if abs(f1) < tol:
            break
    return r1


def check_canonical(dft, bulk_density, fmt='ASWB', solver='anderson', **kw):
    """Teste de sanidade: resolve no grande canonico, pega o N que saiu,
    reimpoe esse N no canonico e confere que mu volta ao mesmo valor.

    Se isto nao fechar, nao confie na varredura.
    """
    dft.equilibrium_density_profile(bulk_density, fmt=fmt, solver=solver, **kw)
    N = float(dft.total_molecules)
    mu_gc = float(dft.mu)

    dft.equilibrium_density_profile(fmt=fmt, solver=solver, N_target=N, **kw)
    mu_c = float(dft.mu)
    N_back = float(dft.total_molecules)

    print("grande canonico : mu = %.10f   N = %.10f" % (mu_gc, N))
    print("canonico (N fixo): mu = %.10f   N = %.10f" % (mu_c, N_back))
    print("erro em mu = %.3e   erro em N = %.3e"
          % (abs(mu_c-mu_gc), abs(N_back-N)))
    return abs(mu_c-mu_gc), abs(N_back-N)


def sweep_N(dft, N_values, rho_start=None, fmt='ASWB', solver='anderson',
            tol=1e-8, rho_bulk_guess=1e-8, logoutput=False, **solver_kw):
    """Percorre N_values. Devolve (mu, P, Omega, N_real, ok).

    N_values deve ir do poro vazio ao cheio em passos pequenos: passos
    grandes perto da dobra pulam justamente o ramo que voce quer.
    """
    n = len(N_values)
    mu = np.full(n, np.nan)
    P = np.full(n, np.nan)
    Om = np.full(n, np.nan)
    N_real = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)

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
            print("  N=%.6f levantou: %s" % (N, exc))
            continue

        err = float(dft.error)
        if not (np.isfinite(err) and err <= tol):
            print("  N=%.6f nao convergiu (err=%.2e)" % (N, err))
            continue

        mu[i] = float(dft.mu)
        Om[i] = float(dft.Omega)
        N_real[i] = float(dft.total_molecules)
        rb = bulk_density_from_mu(dft.eos, mu[i], rb)
        P[i] = float(dft.eos.pressure(torch.tensor([rb])))
        ok[i] = True
        guess = dft.rho.detach().clone()

        if logoutput:
            print("N=%.6f  mu=%.6f  P=%.6e Pa  |N-N_alvo|=%.2e"
                  % (N, mu[i], P[i], abs(N_real[i]-N)))

    return mu, P, Om, N_real, ok


def maxwell_construction(N, mu):
    """Transicao de equilibrio por areas iguais no laco mu(N).

    Usa a curva inteira em vez de comparar Omega de dois ramos
    convergidos em separado, e da as espinodais de brinde.
    """
    N = np.asarray(N, dtype=float)
    mu = np.asarray(mu, dtype=float)
    good = np.isfinite(mu)
    N, mu = N[good], mu[good]

    dmu = np.diff(mu)
    turns = np.where(np.sign(dmu[:-1]) != np.sign(dmu[1:]))[0]+1
    if len(turns) < 2:
        return {'hysteresis': False}

    i_ads, i_des = turns[0], turns[-1]

    def area(mu_try):
        return np.trapezoid(mu-mu_try, N)

    lo, hi = min(mu[i_des], mu[i_ads]), max(mu[i_des], mu[i_ads])
    for _ in range(200):
        mid = 0.5*(lo+hi)
        if area(mid) > 0:
            lo = mid
        else:
            hi = mid

    return {'hysteresis': True,
            'mu_eq': 0.5*(lo+hi),
            'mu_spinodal_ads': mu[i_ads], 'N_spinodal_ads': N[i_ads],
            'mu_spinodal_des': mu[i_des], 'N_spinodal_des': N[i_des]}
