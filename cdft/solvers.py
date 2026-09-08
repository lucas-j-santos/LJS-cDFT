import numpy as np
import torch
import time


class picard():

    def __init__(self, dft, alpha, tol, max_it, logoutput):

        lnrho = torch.log(dft.rho)
        dft.it = 0
        stopped = False
        tic = time.process_time()
        for i in range(max_it):
            # Calculate residual
            dft.euler_lagrange(lnrho, dft.fmt)
            dft.error = dft.loss()
            if dft.error < tol or not torch.isfinite(dft.error):
                stopped = True
                break
            if logoutput: print(dft.it, dft.error.item())
            lnrho.add_(dft.res, alpha=alpha)
            dft.rho = torch.exp(lnrho)
            dft.it += 1
        if not stopped:
            dft.euler_lagrange(lnrho, dft.fmt)
            dft.error = dft.loss()
        toc = time.process_time()
        dft.process_time = toc-tic


class picard_line_search():

    def __init__(self, dft, alpha0, tol, max_it, logoutput):

        lnrho = torch.log(dft.rho)
        dft.it = 0
        stopped = False
        tic = time.process_time()
        for i in range(max_it):
            # Calculate residual
            dft.euler_lagrange(lnrho, dft.fmt)
            dft.error = dft.loss()
            if dft.error < tol or not torch.isfinite(dft.error):
                stopped = True
                break
            if logoutput: print(dft.it, dft.error.item())
            direction = dft.res.clone()
            # Perform line search for optimal step size
            alpha = self.line_search(dft, lnrho, direction, alpha0, dft.error)
            # Update solution
            lnrho.add_(direction, alpha=alpha)
            dft.rho = torch.exp(lnrho)
            dft.it += 1
        if not stopped:
            dft.euler_lagrange(lnrho, dft.fmt)
            dft.error = dft.loss()
        toc = time.process_time()
        dft.process_time = toc-tic

    def line_search(self, dft, lnrho, direction, alpha0, res0):

        alpha = alpha0
        # Try different step sizes to find the best one
        for _ in range(8):
            alpha *= 0.5
            # Calculate full step
            lnrho_new = lnrho+alpha*direction
            # Calculate residual for full step
            try:
                dft.rho = torch.exp(lnrho_new)
                dft.euler_lagrange(lnrho_new, dft.fmt)
                res2 = dft.loss()
            except Exception:
                continue
            if res2 > res0:
                continue
            # Calculate intermediate step
            lnrho_half = lnrho+(0.5*alpha)*direction
            # Calculate residual for half step
            dft.rho = torch.exp(lnrho_half)
            dft.euler_lagrange(lnrho_half, dft.fmt)
            res1 = dft.loss()
            # Estimate optimal step size using quadratic approximation
            denominator = res2-2*res1+res0
            if abs(denominator) > 1e-10:
                alpha_opt = alpha*0.25*(res2-4*res1+3*res0)/denominator
            else:
                continue
            # Ensure step size is positive and reasonable
            if alpha_opt <= 0:
                alpha_opt = 0.5*alpha if res1 < res2 else alpha
            if alpha_opt > alpha:
                alpha_opt = alpha
            alpha = alpha_opt
            break

        return float(alpha)


class anderson():

    def __init__(self, dft, anderson_mmax, anderson_damping, tol, max_it, logoutput):

        # Anderson Mixing parameters
        mmax = anderson_mmax  # Number of previous iterations to store
        damping = anderson_damping  # Damping coefficient

        N = dft.npoints
        resm = torch.zeros((mmax, N), device=dft.device, dtype=dft.rho.dtype)
        rhom = torch.zeros((mmax, N), device=dft.device, dtype=dft.rho.dtype)
        gram = torch.zeros((mmax, mmax), device=dft.device, dtype=dft.rho.dtype)

        m = 0
        lnrho = torch.log(dft.rho)
        dft.it = 0
        stopped = False
        tic = time.process_time()
        for i in range(max_it):
            # Calculate residual
            dft.euler_lagrange(lnrho, dft.fmt)
            dft.error = dft.loss()
            if dft.error < tol or not torch.isfinite(dft.error):
                stopped = True
                break
            if logoutput: print(dft.it, dft.error.item())

            slot = dft.it % mmax
            r = dft.res.reshape(-1)
            resm[slot] = r
            rhom[slot] = lnrho.reshape(-1)
            m = min(m+1, mmax)

            new_row = resm[:m].mv(r)
            gram[slot, :m] = new_row
            gram[:m, slot] = new_row

            R = np.zeros((m+1, m+1))
            R[:m, :m] = gram[:m, :m].cpu().numpy()
            R[:m, m] = 1.0
            R[m, :m] = 1.0
            rhs = np.zeros(m+1)
            rhs[m] = 1.0
            try:
                anderson_alpha = np.linalg.solve(R, rhs)[:m]
            except np.linalg.LinAlgError:
                anderson_alpha = np.full(m, np.nan)
            if not np.isfinite(anderson_alpha).all():
                anderson_alpha = np.zeros(m)
                anderson_alpha[slot] = 1.0

            a = torch.as_tensor(anderson_alpha, device=dft.device, dtype=rhom.dtype)
            lnrho = (a.matmul(rhom[:m])+damping*a.matmul(resm[:m])).view(dft.shape)
            dft.rho = torch.exp(lnrho)
            dft.it += 1
        if not stopped:
            dft.euler_lagrange(lnrho, dft.fmt)
            dft.error = dft.loss()
        toc = time.process_time()
        dft.process_time = toc-tic


class fire():

    def __init__(self, dft, alpha0, dt, tol, max_it, logoutput):

        # Fire parameters
        alpha = alpha0
        Ndelay = 20
        Nnegmax = 2000
        dtmax = 10*dt
        dtmin = 0.02*dt
        Npos = 1
        Nneg = 0
        finc = 1.1
        fdec = 0.5
        fa = 0.99
 
        V = torch.zeros_like(dft.rho)

        lnrho = torch.log(dft.rho)
        dft.euler_lagrange(lnrho, dft.fmt)
        dft.error = dft.loss()

        dft.it = 0
        tic = time.process_time()

        for i in range(max_it):

            P = torch.dot(dft.res.reshape(-1), V.reshape(-1))
            if (P > 0):
                Npos = Npos+1
                if Npos > Ndelay:
                    dt = min(dt*finc,dtmax)
                    alpha = max(1e-10,alpha*fa)
            else:
                Npos = 1
                Nneg = Nneg+1
                if Nneg > Nnegmax: break
                if i > Ndelay:
                    dt = max(dt*fdec,dtmin)
                    alpha = alpha0
                lnrho.add_(V, alpha=-0.5*dt)
                V.zero_()
                dft.rho = torch.exp(lnrho)
                dft.euler_lagrange(lnrho, dft.fmt)

            V.add_(dft.res, alpha=0.5*dt)
            vnorm = torch.linalg.vector_norm(V)
            rnorm = torch.linalg.vector_norm(dft.res).clamp(min=1e-300)
            V = (1.0-alpha)*V+(alpha*vnorm/rnorm)*dft.res
            # V *= (1.0/(1.0-(1.0-alpha)**Npos))
            lnrho.add_(V, alpha=dt)
            dft.rho = torch.exp(lnrho)
            dft.euler_lagrange(lnrho, dft.fmt)
            V.add_(dft.res, alpha=0.5*dt)

            dft.error = dft.loss()
            dft.it += 1
            if dft.error < tol or not torch.isfinite(dft.error): break
            if logoutput: print(dft.it, dft.error.item())

        toc = time.process_time()
        dft.process_time = toc-tic