import torch
import time
from torch.linalg import solve

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
            if logoutput: print(dft.it, dft.error)
            # Update solution
            lnrho[dft.valid] += alpha*dft.res[dft.valid]
            dft.rho[dft.valid] = torch.exp(lnrho[dft.valid])
            dft.it += 1
        toc = time.process_time()
        dft.process_time = toc-tic

class anderson():

    def __init__(self, dft, anderson_mmax, anderson_damping, tol, max_it, logoutput):
        
        # Anderson Mixing parameters
        mmax = anderson_mmax  # Number of previous iterations to store
        damping = anderson_damping  # Damping coefficient
        # Initialize history buffers
        resm = []  # Residual history
        rhom = []  # Solution history

        lnrho = torch.log(dft.rho)
        dft.it = 0
        tic = time.process_time()
        for i in range(max_it):
            # Calculate residual
            dft.euler_lagrange(lnrho, dft.fmt)
            dft.error = dft.loss()
            if dft.error < tol or torch.isnan(dft.error): break
            if logoutput: print(dft.it, dft.error)
            # Store residual and solution
            resm.append(dft.res[dft.valid].clone())
            rhom.append(lnrho[dft.valid].clone())
            # Drop old values if history is full
            if len(resm) > mmax:
                resm.pop(0)
                rhom.pop(0)
            m = len(resm)  # Current history size
            # Build the Anderson matrix and vector
            R = torch.zeros((m+1, m+1), device=dft.device)
            anderson_alpha = torch.zeros(m+1, device=dft.device)  
            if m > 0:
                resm_tensor = torch.stack(resm)  # Shape: (m, *points)
                R[:m, :m] = torch.einsum('ik,jk->ij', resm_tensor.view(m,-1), resm_tensor.view(m,-1))
                R[:m, m] = 1.0
                R[m, :m] = 1.0
            R[m, m] = 0.0
            anderson_alpha[m] = 1.0
            # Solve for alpha coefficients
            try:
                anderson_alpha = solve(R, anderson_alpha)
            except:
                # Fallback to Picard if matrix is singular
                anderson_alpha = torch.zeros(m+1, device=dft.device)
                anderson_alpha[m] = 1.0
            # Update solution using Anderson mixing
            lnrho[dft.valid] = torch.einsum('i,i...->...', anderson_alpha[:m], (torch.stack(rhom[:m])+damping*torch.stack(resm[:m])))
            dft.rho[dft.valid] = torch.exp(lnrho[dft.valid])
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
        Npos = 0
        Nneg = 0
        finc = 1.1
        fdec = 0.5
        fa = 0.99
        # Velocity
        V = torch.zeros_like(dft.rho)
        
        lnrho = torch.log(dft.rho) 
        dft.euler_lagrange(lnrho, dft.fmt)
        dft.error = dft.loss()

        dft.it = 0
        tic = time.process_time()

        for i in range(max_it):

            P = (dft.res[dft.valid]*V[dft.valid]).sum() 
            if (P > 0):
                Npos = Npos+1
                if Npos > Ndelay:
                    dt = min(dt*finc,dtmax)
                    alpha = alpha*fa
            else:
                Npos = 0
                Nneg = Nneg+1
                if Nneg > Nnegmax: break
                if i > Ndelay:
                    if dt*fdec >= dtmin:
                        dt *= fdec
                    alpha = alpha0
                lnrho[dft.valid] -= 0.5*dt*V[dft.valid]
                V[dft.valid] = 0.0
                dft.rho[dft.valid] = torch.exp(lnrho[dft.valid])
                dft.euler_lagrange(lnrho, dft.fmt) 

            V[dft.valid] += 0.5*dt*dft.res[dft.valid]
            V[dft.valid] = (1.0-alpha)*V[dft.valid]+alpha*dft.res[dft.valid]*torch.norm(V[dft.valid])/torch.norm(dft.res[dft.valid])
            lnrho[dft.valid] += dt*V[dft.valid]
            dft.rho[dft.valid] = torch.exp(lnrho[dft.valid])
            dft.euler_lagrange(lnrho, dft.fmt)
            V[dft.valid] += 0.5*dt*dft.res[dft.valid]

            dft.error = dft.loss()
            dft.it += 1
            if dft.error < tol or torch.isnan(dft.error): break
            if logoutput: print(dft.it, dft.error)

        toc = time.process_time()
        dft.process_time = toc-tic