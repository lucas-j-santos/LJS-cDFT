import numpy as np
import torch
import time


class picard():

    def __init__(self, dft, alpha, tol, max_it, logoutput):

        lnrho = torch.log(dft.rho)
        dft.it = 0
        tic = time.process_time()
        for i in range(max_it):
            # Calculate residual
            dft.euler_lagrange(lnrho, dft.fmt)
            dft.error = dft.loss()
            if dft.error < tol or torch.isnan(dft.error): break
            if logoutput: print(dft.it, dft.error.item())
            # Update solution. res is exactly zero on the excluded cells, so
            # the update is a no-op there and no masking is needed.
            lnrho.add_(dft.res, alpha=alpha)
            dft.rho = torch.exp(lnrho)
            dft.it += 1
        toc = time.process_time()
        dft.process_time = toc-tic


class picard_line_search():

    def __init__(self, dft, alpha0, tol, max_it, logoutput):

        lnrho = torch.log(dft.rho)
        dft.it = 0
        tic = time.process_time()
        for i in range(max_it):
            # Calculate residual
            dft.euler_lagrange(lnrho, dft.fmt)
            dft.error = dft.loss()
            if dft.error < tol or torch.isnan(dft.error): break
            if logoutput: print(dft.it, dft.error.item())
            # The line search overwrites dft.res while probing trial steps, so
            # the search direction has to be kept aside before it starts.
            direction = dft.res.clone()
            # Perform line search for optimal step size
            alpha = self.line_search(dft, lnrho, direction, alpha0, dft.error)
            # Update solution
            lnrho.add_(direction, alpha=alpha)
            dft.rho = torch.exp(lnrho)
            dft.it += 1
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

        # Pre-allocated circular history buffers. The Anderson condition
        # (minimise ||sum_i a_i r_i|| subject to sum_i a_i = 1) does not depend
        # on the ordering of the history, so old entries can simply be
        # overwritten instead of shifting the whole buffer every iteration.
        N = dft.npoints
        resm = torch.zeros((mmax, N), device=dft.device, dtype=dft.rho.dtype)
        rhom = torch.zeros((mmax, N), device=dft.device, dtype=dft.rho.dtype)
        # Gram matrix of the stored residuals, updated one row/column per
        # iteration instead of being rebuilt from m^2 inner products.
        gram = torch.zeros((mmax, mmax), device=dft.device, dtype=dft.rho.dtype)

        m = 0
        lnrho = torch.log(dft.rho)
        dft.it = 0
        tic = time.process_time()
        for i in range(max_it):
            # Calculate residual
            dft.euler_lagrange(lnrho, dft.fmt)
            dft.error = dft.loss()
            if dft.error < tol or torch.isnan(dft.error): break
            if logoutput: print(dft.it, dft.error.item())

            # Store residual and solution in the next slot
            slot = dft.it % mmax
            r = dft.res.reshape(-1)
            resm[slot] = r
            rhom[slot] = lnrho.reshape(-1)
            m = min(m+1, mmax)

            # Only the new row/column of the Gram matrix has to be computed
            new_row = resm[:m].mv(r)
            gram[slot, :m] = new_row
            gram[:m, slot] = new_row

            # Solve the small bordered system on the CPU: it is (m+1)x(m+1)
            # with m <= 10, so a GPU solve is pure launch latency.
            R = np.zeros((m+1, m+1))
            R[:m, :m] = gram[:m, :m].cpu().numpy()
            R[:m, m] = 1.0
            R[m, :m] = 1.0
            rhs = np.zeros(m+1)
            rhs[m] = 1.0
            try:
                anderson_alpha = np.linalg.solve(R, rhs)[:m]
            except np.linalg.LinAlgError:
                # Fallback to Picard if the matrix is singular: put all the
                # weight on the most recent iterate.
                anderson_alpha = np.zeros(m)
                anderson_alpha[slot] = 1.0

            a = torch.as_tensor(anderson_alpha, device=dft.device, dtype=rhom.dtype)
            # Two matrix-vector products instead of building the (m, N)
            # combination rhom + damping*resm as a temporary.
            lnrho = (a.matmul(rhom[:m])+damping*a.matmul(resm[:m])).view(dft.shape)
            dft.rho = torch.exp(lnrho)
            dft.it += 1
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
        # Velocity. Stays exactly zero on the excluded cells because the
        # residual is zero there, so no masking is needed.
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
            rnorm = torch.linalg.vector_norm(dft.res)
            V = (1.0-alpha)*V+(alpha*vnorm/rnorm)*dft.res
            # V *= (1.0/(1.0-(1.0-alpha)**Npos))
            lnrho.add_(V, alpha=dt)
            dft.rho = torch.exp(lnrho)
            dft.euler_lagrange(lnrho, dft.fmt)
            V.add_(dft.res, alpha=0.5*dt)

            dft.error = dft.loss()
            dft.it += 1
            if dft.error < tol or torch.isnan(dft.error): break
            if logoutput: print(dft.it, dft.error.item())

        toc = time.process_time()
        dft.process_time = toc-tic
