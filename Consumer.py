from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from scipy import optimize

class ConsumerClass:
    """ a consumer with nested CES preferences over three goods

    Good 1 is food. Goods 2 and 3 are bus trips and train trips, and they sit
    together in a nest.

    The problem is written in *nested* budget shares, in the same two steps as
    the nests themselves:

        s1 = the share of income spent on food
        w  = the share of the remaining (travel) budget spent on the bus

    so that s2 = (1-s1)*w and s3 = (1-s1)*(1-w). Any (s1,w) in the unit square
    is a possible choice, and every possible choice is in the unit square, so
    the constraint set is exactly the box that L-BFGS-B takes as `bounds`.

    """

    def __init__(self,par=None):

        # a. setup
        self.setup()

        # b. update parameters
        if not par is None:
            for k,v in par.items():
                self.par.__dict__[k] = v

    def setup(self):
        """ set the baseline parameters """

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. preference weights
        par.alpha = 0.60 # weight on food
        par.beta = 0.50 # weight on the bus

        # b. substitution
        par.sigma_A = 0.80 
        par.sigma_B = 0.40 

        # c. prices and income
        par.p1 = 1.0 # price of food
        par.p2 = 1.0 # price of a bus trip
        par.p3 = 1.5 # price of a train trip
        par.I = 10.0 # income

        # d. numerical settings
        par.s_min = 1e-12 # smallest quantity allowed in .ces(), see the note there

    def __str__(self):
        """ print the parameters """

        par = self.par

        lines = ['ConsumerClass']
        lines.append(f'  alpha = {par.alpha:.4f}, beta = {par.beta:.4f}')
        lines.append(f'  sigma_A = {par.sigma_A:.4f}, sigma_B = {par.sigma_B:.4f}')
        lines.append(f'  p1 = {par.p1:.4f}, p2 = {par.p2:.4f}, p3 = {par.p3:.4f}')
        lines.append(f'  I = {par.I:.4f}')

        return '\n'.join(lines)

    ###################
    # 1. the CES nest #
    ###################

    def ces(self,z1,z2,w,sigma):
        """ the CES aggregate of two inputs

        Computes (w*z1**rho + (1-w)*z2**rho)**(1/rho) with rho = 1-1/sigma.

        The inputs are floored at par.s_min, because z**rho is not defined for
        z = 0 when rho < 0, and the corners of the unit square do give zeros.

        Note on the size of par.s_min. It must be *far* below the step that
        L-BFGS-B uses to estimate the gradient, which is `eps` and about 1e-8 by
        default. Otherwise both z and z+eps get floored to the same number near a
        bound, utility comes out exactly equal in the two points, the estimated
        gradient is zero, and the solver stops on the bound and stays there. With
        s_min = 1e-8 that really happens: the answers in section 4 for high tax
        rates come out as zero revenue. 1e-12 is small enough to be invisible and
        large enough to keep z**rho from overflowing.

        Args:

            z1 (float or ndarray): first input
            z2 (float or ndarray): second input
            w (float): weight on the first input
            sigma (float): substitution parameter, must not be 1

        Returns:

            (float or ndarray): the CES aggregate

        """

        par = self.par

        assert not np.isclose(sigma,1.0), 'sigma = 1 gives rho = 0 and a division by zero'

        z1 = np.maximum(z1,par.s_min)
        z2 = np.maximum(z2,par.s_min)

        rho = 1-1/sigma

        return (w*z1**rho + (1-w)*z2**rho)**(1/rho)

    def utility(self,x1,x2,x3):
        """ nested CES utility of a bundle of quantities

        Two steps: first combine goods 2 and 3 into the travel composite, then
        combine good 1 and the composite into utility. Use .ces() for both.

        Args:

            x1 (float or ndarray): quantity of good 1
            x2 (float or ndarray): quantity of good 2
            x3 (float or ndarray): quantity of good 3

        Returns:

            (float or ndarray): utility

        """

        par = self.par

        # a. travel composition of bus and train
        travel = self.ces(x2,x3,par.beta,par.sigma_B)

        # b. utility of food and travel composition
        u = self.ces(x1,travel,par.alpha,par.sigma_A)

        return u
    

    ###############################
    # 2. the nested budget shares #
    ###############################

    def shares(self,s1,w):
        """ the three budget shares implied by the nested shares

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:

            (tuple): the three budget shares, which always sum to one

        """

        return s1,(1-s1)*w,(1-s1)*(1-w)

    def quantities(self,s1,w):
        """ the quantities implied by the nested shares

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:

            (tuple): the three quantities

        """

        par = self.par

        s1,s2,s3 = self.shares(s1,w)

        return s1*par.I/par.p1, s2*par.I/par.p2, s3*par.I/par.p3

    def value_of_choice(self,s1,w):
        """ utility of the bundle implied by the nested shares
        x1,x2,x3 = self.quantities(s1,w)
        u = self.utility(x1,x2,x3)
        return u

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:

            (float or ndarray): utility

        """

        x1,x2,x3 = self.quantities(s1,w)
        u = self.utility(x1,x2,x3)

        return u

    def objective(self,s):
        """ minus utility, for a minimizer

        Nothing else is needed: the bounds are the whole constraint.

        Args:

            s (ndarray): array of length 2 with (s1,w)

        Returns:

            (float): minus utility

        """

        return -self.value_of_choice(s[0],s[1])

    #################
    # 3. solving it #
    #################

    def solve_grid(self,N=200,do_print=True):
        """ solve by a 2-dimensional grid search over the nested shares """

        par = self.par
        opt = SimpleNamespace()

        # a. define the grid
        s1_vec = np.linspace(0,1,N)
        w_vec = np.linspace(0,1,N)
        s1_grid,w_grid = np.meshgrid(s1_vec,w_vec,indexing='ij')

        # b. utility in each grid point
        u_grid = self.value_of_choice(s1_grid,w_grid)

        # c. the best point
        index_best = np.argmax(u_grid)
        i,j = np.unravel_index(index_best,u_grid.shape)

        # d. results
        opt.s1 = s1_grid[i,j]
        opt.w = w_grid[i,j]
        opt.s1,opt.s2,opt.s3 = self.shares(opt.s1,opt.w)
        opt.u = u_grid[i,j]

        opt.s1_grid = s1_grid
        opt.w_grid = w_grid
        opt.u_grid = u_grid

        if do_print:
           print(f's1 = {opt.s1:.4f}, w = {opt.w:.4f}, u = {opt.u:.4f}')

        return opt

    def solve(self,s0=None,do_print=True,**kwargs):
        """ solve with L-BFGS-B

        The bounds are ((0,1),(0,1)) -- the whole constraint set.

        Args:

            s0 (ndarray): starting guess for (s1,w)
            do_print (bool): print the solution
            kwargs: passed on to optimize.minimize, e.g. options={'ftol':1e-10}

        Returns:

            (SimpleNamespace): the solution and the convergence path

        """

        par = self.par
        opt = SimpleNamespace()

        # a. starting guess
        if s0 is None: s0 = np.array([0.5,0.5])
        s0 = np.asarray(s0,dtype=float)

        # b. registrér the path with a callback
        path = [s0.copy()]

        # c. minimér
        res = optimize.minimize(self.objective,s0,method='L-BFGS-B',
            bounds=((0,1),(0,1)),
            callback=lambda sk: path.append(sk.copy()),
            **kwargs)

        # d. results
        opt.s1,opt.w = res.x
        opt.s1,opt.s2,opt.s3 = self.shares(opt.s1,opt.w)
        opt.u = -res.fun
        opt.path = np.array(path)
        opt.res = res

        if do_print:
            print(f's1 = {opt.s1:.4f}, w = {opt.w:.4f}, u = {opt.u:.4f}')

        return opt

# 2.1.1 grid search

# %%
c = ConsumerClass()
opt = c.solve_grid(N=200)

# %%
fig = plt.figure(figsize=(12,5))

ax1 = fig.add_subplot(1,2,1,projection='3d')
ax1.plot_surface(opt.s1_grid, opt.w_grid, opt.u_grid, cmap='viridis')
ax1.scatter([opt.s1], [opt.w], [opt.u], color='red', s=50, label='Løsning')
ax1.set_xlabel(r'$s_1$'); ax1.set_ylabel(r'$w$'); ax1.set_zlabel(r'$u$')
ax1.set_title('Nytte over de nestede andele (3D)')

ax2 = fig.add_subplot(1,2,2)
cont = ax2.contourf(opt.s1_grid, opt.w_grid, opt.u_grid, levels=30, cmap='viridis')
ax2.scatter(opt.s1, opt.w, color='red', marker='*', s=150, label='Løsning')
ax2.set_xlabel(r'$s_1$'); ax2.set_ylabel(r'$w$')
ax2.set_title('Nytte over de nestede andele (contour)')
fig.colorbar(cont, ax=ax2, label=r'$u$')
ax2.legend()

fig.tight_layout()
plt.show()

# --- 2.1.3: compare N = 50, 100, 500, 1000 ---
# %%
N_liste = [50, 100, 500, 1000]

for N in N_liste:
    opt = c.solve_grid(N=N, do_print=False)
    antal_evalueringer = N*N
    print(f'N = {N:5d}:  s1 = {opt.s1:.6f}, w = {opt.w:.6f}, u = {opt.u:.6f}, evalueringer = {antal_evalueringer}')


# %%
import time

# a. L-BFGS-B
t0 = time.time()
opt_lbfgsb = c.solve()
t1 = time.time()

print(f'L-BFGS-B:    s1 = {opt_lbfgsb.s1:.6f}, w = {opt_lbfgsb.w:.6f}, u = {opt_lbfgsb.u:.6f}')
print(f'  funktionsevalueringer: {opt_lbfgsb.res.nfev}')
print(f'  tid: {(t1-t0)*1000:.3f} ms')

# b. grid search (N=200, til comparison)
t0 = time.time()
opt_grid = c.solve_grid(N=200, do_print=False)
t1 = time.time()

print(f'Grid search: s1 = {opt_grid.s1:.6f}, w = {opt_grid.w:.6f}, u = {opt_grid.u:.6f}')
print(f'  funktionsevalueringer: {200*200}')
print(f'  tid: {(t1-t0)*1000:.3f} ms')

# %%
opt_lbfgsb = c.solve()

# a. path on top of the contour plot
fig, ax = plt.subplots(figsize=(6,5))
cont = ax.contourf(opt_grid.s1_grid, opt_grid.w_grid, opt_grid.u_grid, levels=30, cmap='viridis')
fig.colorbar(cont, ax=ax, label=r'$u$')

ax.plot(opt_lbfgsb.path[:,0], opt_lbfgsb.path[:,1], 'o-', color='red', markersize=4, label='Convergence path')
ax.scatter(opt_lbfgsb.path[0,0], opt_lbfgsb.path[0,1], color='white', edgecolor='black', s=80, zorder=5, label='Start')
ax.scatter(opt_lbfgsb.s1, opt_lbfgsb.w, color='red', marker='*', s=150, zorder=5, label='Solution')

ax.set_xlabel(r'$s_1$'); ax.set_ylabel(r'$w$')
ax.set_title('L-BFGS-B convergence path')
ax.legend()
fig.tight_layout()
plt.show()

# b. distance to the endpoint, per iteration, on a log scale
slutpunkt = opt_lbfgsb.path[-1]
afstande = np.linalg.norm(opt_lbfgsb.path - slutpunkt, axis=1)

fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.semilogy(afstande, 'o-')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Afstand til slutpunkt (log-skala)')
ax2.set_title('Konvergenshastighed')
fig2.tight_layout()
plt.show()

# %%
start_points = {
    'center':          [0.5, 0.5],
    'corner (0,0)':    [0.0, 0.0],
    'corner (0,1)':    [0.0, 1.0],
    'corner (1,0)':    [1.0, 0.0],
    'corner (1,1)':    [1.0, 1.0],
    'extra':           [0.25, 0.75],
}

for name, s0 in start_points.items():
    opt_s = c.solve(s0=s0, do_print=False)
    n_iter = opt_s.res.nit
    print(f'{name:16s}: s1 = {opt_s.s1:.6f}, w = {opt_s.w:.6f}, u = {opt_s.u:.6f}, '
          f'iterations = {n_iter}, success = {opt_s.res.success}')

#2.2.4 the tolerance parameters ftol and gtol
# %%
ftol_values = [1e-4, 1e-8, 1e-12, 1e-16]
gtol_values = [1e-2, 1e-5, 1e-8, 1e-12]

results = []

# a. vary ftol (gtol at default)
for ftol in ftol_values:
    opt_s = c.solve(do_print=False, options={'ftol': ftol})
    results.append(('ftol', ftol, opt_s.s1, opt_s.w, opt_s.u, opt_s.res.nfev))

# b. vary gtol (ftol at default)
for gtol in gtol_values:
    opt_s = c.solve(do_print=False, options={'gtol': gtol})
    results.append(('gtol', gtol, opt_s.s1, opt_s.w, opt_s.u, opt_s.res.nfev))

print(f'{"setting":8s} {"value":10s} {"s1":>10s} {"w":>10s} {"u":>10s} {"nfev":>6s}')
for setting, value, s1, w, u, nfev in results:
    print(f'{setting:8s} {value:<10.0e} {s1:>10.6f} {w:>10.6f} {u:>10.6f} {nfev:>6d}')




# %%

# 2.2.5 Which settings to use
#question 1: ftol = 1e-08 (or gtol = 1e-05). Both reach the highest utility (u = 3.401680) with the fewest evaluations (18) needed to get there.
#question 2: past that point, tightening further (gtol = 1e-08, 1e-12) needs more evaluations (21) but gives the same utility — no improvement, just extra cost.