""" Defintion of the HotcentAllElectron class (and 
supporting methods) for calculations with Hotcent's
own atomic DFT calculator.

The code below draws heavily from the Hotbit code 
written by Pekka Koskinen (https://github.com/pekkosk/
hotbit/blob/master/hotbit/parametrization/atom.py).
"""
from __future__ import division, print_function
import os
import sys
import pickle
import numpy as np
from math import pi, sqrt
from scipy.interpolate import splrep, splev
from ase.data import atomic_numbers, covalent_radii
from ase.units import Bohr
from hotcent.interpolation import Function, SplineFunction
from hotcent.atom import AllElectron
from hotcent.confinement import ZeroConfinement
try:
    import pylab as pl
except:
    pl = None


class HotcentAE(AllElectron):
    def __init__(self,
                 symbol,
                 convergence={'density':1e-7, 'energies':1e-7},
                 restart=None,
                 write=None,
                 **kwargs):
        """
        Run Kohn-Sham all-electron calculations for a given atom.

        Examples:
        ---------
        atom = HotcentAllElectron('C')
        from hotcent.confinement import PowerConfinement
        atom = HotcentAllElectron('C', confinement=PowerConfinement(r0=3., s=2))
        atom.run()

        Parameters:
        -----------
        convergence:    convergence criterion dictionary
                        * density: max change for integrated |n_old-n_new|
                        * energies: max change in single-particle energy (Hartree)
        write:          filename: save rgrid, effective potential and
                        density to a file for further calculations.
        restart:        filename: make an initial guess for effective
                        potential and density from another calculation.
        """
        AllElectron.__init__(self, symbol, **kwargs)

        self.convergence = convergence
        self.write = write
        self.restart = restart
        
        self.set_output(self.txt)

        if self.xcname in ['PW92', 'LDA']:
            self.xcf = XC_PW92()
        else:
            raise NotImplementedError('XC not implemented: %s' % xcname)

        if self.scalarrel:
            print('Using scalar relativistic corrections.', file=self.txt)


        maxnodes = max( [n - l - 1 for n, l, nl in self.list_states()] )
        self.rmin = 1e-2 / self.Z
        self.N = (maxnodes + 1) * self.nodegpts
        print('max %i nodes, %i grid points' % (maxnodes, self.N), file=self.txt)

        self.xgrid = np.linspace(0, np.log(self.rmax / self.rmin), self.N)
        self.rgrid = self.rmin * np.exp(self.xgrid)
        self.grid = RadialGrid(self.rgrid)

        self.timer.stop('init')

    def set_output(self, txt):
        """ Set output channel and give greetings. """
        if txt == '-':
            self.txt = sys.stdout
        elif txt == None:
            self.txt = open(os.devnull,'w')
        else:
            self.txt = open(txt, 'a')
        print('*******************************************', file=self.txt)
        print('Kohn-Sham all-electron calculation for %2s ' % self.symbol, file=self.txt)
        print('*******************************************', file=self.txt)

    def calculate_energies(self, echo=False):
        """
        Calculate energy contributions.
        """
        self.timer.start('energies')
        self.bs_energy = 0.0
        for n, l, nl in self.list_states():
            self.bs_energy += self.configuration[nl] * self.enl[nl]

        self.exc = np.array([self.xcf.exc(self.dens[i]) for i in range(self.N)])
        self.Hartree_energy = self.grid.integrate(self.Hartree * self.dens, use_dV=True) / 2
        self.vxc_energy = self.grid.integrate(self.vxc * self.dens, use_dV=True)
        self.exc_energy = self.grid.integrate(self.exc * self.dens, use_dV=True)
        self.confinement_energy = self.grid.integrate(self.conf * self.dens, use_dV=True)
        self.total_energy = self.bs_energy - self.Hartree_energy 
        self.total_energy += - self.vxc_energy + self.exc_energy

        if echo:
            print('\n\nEnergetics:', file=self.txt)
            print('-------------', file=self.txt)
            print('\nsingle-particle energies', file=self.txt)
            print('------------------------', file=self.txt)
            for n, l, nl in self.list_states():
                print('%s, energy %.15f' % (nl, self.enl[nl]), file=self.txt)

            print('\nvalence orbital energies', file=self.txt)
            print('--------------------------', file=self.txt)
            for nl in self.configuration:
                print('%s, energy %.15f' % (nl, self.enl[nl]), file=self.txt)

            print('\n', file=self.txt)
            print('total energies:', file=self.txt)
            print('---------------', file=self.txt)
            print('sum of eigenvalues:     %.15f' % self.bs_energy, file=self.txt)
            print('Hartree energy:         %.15f' % self.Hartree_energy, file=self.txt)
            print('vxc correction:         %.15f' % self.vxc_energy, file=self.txt)
            print('exchange + corr energy: %.15f' % self.exc_energy, file=self.txt)
            print('----------------------------', file=self.txt)
            print('total energy:           %.15f\n\n' % self.total_energy, file=self.txt)

        self.timer.stop('energies')

    def calculate_density(self):
        """ Calculate the radial electron density.; sum_nl |Rnl(r)|**2/(4*pi) """
        self.timer.start('density')
        dens = np.zeros_like(self.rgrid)
        for n,l,nl in self.list_states():
            dens += self.configuration[nl] * (self.unlg[nl] ** 2)

        nel = self.grid.integrate(dens)

        if abs(nel - self.nel) > 1e-10:
            raise RuntimeError('Integrated density %.3g, number of electrons %.3g' % (nel, self.nel))

        dens = dens / (4 * np.pi * self.rgrid **2)

        self.timer.stop('density')
        return dens

    def calculate_Hartree_potential(self):
        """
        Calculate Hartree potential.

        Everything is very sensitive to the way this is calculated.
        If you can think of how to improve this, please tell me!
        """
        self.timer.start('Hartree')
        dV = self.grid.get_dvolumes()
        r, r0 = self.rgrid, self.grid.get_r0grid()
        N = self.N
        n0 = 0.5 * (self.dens[1:] + self.dens[:-1])
        n0 *= self.nel / sum(n0 * dV)

        lo, hi, Hartree = np.zeros(N), np.zeros(N), np.zeros(N)
        lo[0] = 0.0
        for i in range(1, N):
            lo[i] = lo[i-1] + dV[i-1] * n0[i-1]

        hi[-1] = 0.0
        for i in range(N - 2, -1, -1):
            hi[i] = hi[i + 1] + n0[i] * dV[i] / r0[i]

        for i in range(N):
            Hartree[i] = lo[i] / r[i] + hi[i]
        self.Hartree = Hartree
        self.timer.stop('Hartree')

    def calculate_veff(self):
        """ Calculate effective potential. """
        self.timer.start('veff')
        self.vxc = self.xcf.vxc(self.dens)
        self.timer.stop('veff')
        return self.nucl + self.Hartree + self.vxc + self.conf

    def guess_density(self):
        """ Guess initial density. """
        r2 = 0.02 * self.Z # radius at which density has dropped to half; improve this!
        dens = np.exp(-self.rgrid / (r2 / np.log(2)))
        dens = dens / self.grid.integrate(dens, use_dV=True) * self.nel
        return dens

    def get_veff_and_dens(self):
        """ Construct effective potential and electron density. If restart
            file is given, try to read from there, otherwise make a guess.
        """
        done = False
        if self.restart is not None:
            # use density and effective potential from another calculation
            try:
                with open(self.restart) as f:
                    rgrid = pickle.load(f)
                    veff = pickle.load(f)
                    dens = pickle.load(f)
                v = splrep(rgrid, veff)
                d = splrep(rgrid, dens)
                self.veff = np.array([splev(r,v) for r in self.rgrid])
                self.dens = np.array([splev(r,d) for r in self.rgrid])
                done = True
            except IOError:
                print("Could not open restart file, " \
                      "starting from scratch.", file=self.txt)

        if not done:
            self.veff = self.nucl + self.conf
            self.dens = self.guess_density()

    def run(self):
        val = self.get_valence_orbitals()
        enl = {}
        Rnlg = {}
        unlg = {}
        bar = '=' * 50
        confinement = self.confinement

        for nl, wf_confinement in self.wf_confinement.items():
            assert nl in val, 'Confinement %s not in %s' % (nl, str(val))
            self.confinement = wf_confinement
            if self.confinement is None:
                self.confinement = ZeroConfinement()

            print(bar, file=self.txt)
            print('Applying %s' % self.confinement, file=self.txt)
            print('to get a confined %s orbital' % nl, file=self.txt)
            print(bar, file=self.txt)

            self._run()

            Rnlg[nl] = self.Rnlg[nl].copy()
            unlg[nl] = self.unlg[nl].copy()
            enl[nl] = self.enl[nl]

        self.confinement = confinement
        if self.confinement is None:
            self.confinement = ZeroConfinement()

        print(bar, file=self.txt)
        print('Applying %s' % self.confinement, file=self.txt)
        print('to get the confined electron density', file=self.txt)
        nl_0 = [nl for nl in val if nl not in self.wf_confinement]
        if len(nl_0) > 0:
            print('as well as the confined %s orbital%s' % \
                  (' and '.join(nl_0), 's' if len(nl_0) > 1 else ''),
                  file=self.txt)
        print(bar, file=self.txt)

        self._run()

        self.Rnlg.update(Rnlg)
        self.unlg.update(unlg)
        self.enl.update(enl)
        for nl in val:
            self.Rnl_fct[nl] = Function('spline', self.rgrid, self.Rnlg[nl])
            self.unl_fct[nl] = Function('spline', self.rgrid, self.unlg[nl])

        self.veff = self.calculate_veff()

        if self.write != None:
            with open(self.write, 'w') as f:
                pickle.dump(self.rgrid, f)
                pickle.dump(self.veff, f)
                pickle.dump(self.dens, f)

        self.solved = True
        self.timer.summary()
        self.txt.flush()

    def _run(self):
        """
        Solve the self-consistent potential.
        """
        self.timer.start('solve ground state')
        print('\nStart iteration...', file=self.txt)
        self.enl = {}
        self.d_enl = {}
        for n, l, nl in self.list_states():
            self.enl[nl] = 0.0
            self.d_enl[nl] = 0.0

        N = self.grid.get_N()

        # make confinement and nuclear potentials; intitial guess for veff
        self.conf = np.array([self.confinement(r) for r in self.rgrid])
        self.nucl = np.array([self.V_nuclear(r) for r in self.rgrid])
        self.get_veff_and_dens()
        self.calculate_Hartree_potential()

        for it in range(self.maxiter):
            self.veff = self.mix * self.calculate_veff() + (1 - self.mix) * self.veff
            if self.scalarrel:
                veff = SplineFunction(self.rgrid, self.veff)
                self.dveff = np.array([veff(r, der=1) for r in self.rgrid])
            d_enl_max, itmax = self.solve_eigenstates(it)

            dens0 = self.dens.copy()
            self.dens = self.calculate_density()
            diff = self.grid.integrate(np.abs(self.dens - dens0), use_dV=True)

            if diff < self.convergence['density'] and d_enl_max < self.convergence['energies'] and it > 5:
                break
            self.calculate_Hartree_potential()

            if np.mod(it, 10) == 0:
                line = 'iter %3i, dn=%.1e>%.1e, max %i sp-iter' % \
                       (it, diff, self.convergence['density'], itmax)
                
                print(line, file=self.txt)

            if it == self.maxiter - 1:
                if self.timing:
                    self.timer.summary()
                raise RuntimeError('Density not converged in %i iterations' % (it + 1))
            self.txt.flush()

        self.calculate_energies(echo=True)
        print('converged in %i iterations' % it, file=self.txt)
        line = '%9.4f electrons, should be %9.4f' % \
               (self.grid.integrate(self.dens, use_dV=True), self.nel)
        print(line, file=self.txt)

        self.timer.stop('solve ground state')
        
    def solve_eigenstates(self, iteration, itmax=100):
        """
        Solve the eigenstates for given effective potential.

        u''(r) - 2*(v_eff(r)+l*(l+1)/(2r**2)-e)*u(r)=0
        ( u''(r) + c0(r)*u(r) = 0 )

        r=r0*exp(x) --> (to get equally spaced integration mesh)

        u''(x) - u'(x) + c0(x(r))*u(r) = 0
        """
        self.timer.start('eigenstates')

        rgrid = self.rgrid
        xgrid = self.xgrid
        dx = xgrid[1] - xgrid[0]
        N = self.N
        c2 = np.ones(N)
        c1 = -np.ones(N)
        d_enl_max = 0.0
        itmax = 0

        for n, l, nl in self.list_states():
            nodes_nl = n - l - 1

            if iteration == 0:
                eps = -1.0 * self.Z **2 / n**2
            else:
                eps = self.enl[nl]

            if iteration <= 3:
                delta = 0.5 * self.Z **2 / n ** 2  # previous!!!!!!!!!!
            else:
                delta = self.d_enl[nl]

            direction = 'none'
            epsmax = self.veff[-1] - l * (l + 1) / (2 * self.rgrid[-1] ** 2)
            it = 0
            u = np.zeros(N)
            hist = []

            while True:
                eps0 = eps
                c0, c1, c2 = self.construct_coefficients(l, eps)

                # boundary conditions for integration from analytic behaviour (unscaled)
                # u(r)~r**(l+1)   r->0
                # u(r)~exp( -sqrt(c0(r)) ) (set u[-1]=1 and use expansion to avoid overflows)
                u[0:2] = rgrid[0:2] ** (l + 1)

                if not(c0[-2] < 0 and c0[-1] < 0):
                    pl.plot(c0)
                    pl.show()

                assert c0[-2] < 0 and c0[-1] < 0

                u, nodes, A, ctp = shoot(u, dx, c2, c1, c0, N)
                it += 1
                norm = self.grid.integrate(u ** 2)
                u = u / sqrt(norm)

                if nodes > nodes_nl:
                    # decrease energy
                    if direction == 'up': 
                        delta /= 2
                    eps -= delta
                    direction = 'down'
                elif nodes < nodes_nl:
                    # increase energy
                    if direction == 'down': 
                        delta /= 2
                    eps += delta
                    direction = 'up'
                elif nodes == nodes_nl:
                    shift = -0.5 * A / (rgrid[ctp] * norm)
                    if abs(shift) < 1e-8:  # convergence
                        break
                    if shift > 0:
                        direction = 'up'
                    elif shift < 0:
                        direction = 'down'
                    eps += shift

                if eps > epsmax:
                    eps = 0.5 * (epsmax + eps0)
                hist.append(eps)

                if it > 100:
                    print('Epsilon history for %s' % nl, file=self.txt)
                    for h in hist:
                        print(h)
                    print('nl=%s, eps=%f' % (nl,eps), file=self.txt)
                    print('max epsilon', epsmax, file=self.txt)
                    raise RuntimeError('Eigensolver out of iterations. Atom not stable?')

            itmax = max(it, itmax)
            self.unlg[nl] = u
            self.Rnlg[nl] = self.unlg[nl] / self.rgrid
            self.d_enl[nl] = abs(eps - self.enl[nl])
            d_enl_max = max(d_enl_max, self.d_enl[nl])
            self.enl[nl] = eps

            if self.verbose:
                line = '-- state %s, %i eigensolver iterations, e=%9.5f, de=%9.5f' % \
                       (nl, it, self.enl[nl], self.d_enl[nl])
                print(line, file=self.txt)

            assert nodes == nodes_nl
            assert u[1] > 0.0

        self.timer.stop('eigenstates')
        return d_enl_max, itmax

    def construct_coefficients(self, l, eps):
        c = 137.036
        c2 = np.ones(self.N)
        if self.scalarrel == False:
            c0 = -2 * (0.5 * l * (l + 1) + self.rgrid ** 2 * (self.veff - eps))
            c1 = -1. * np.ones(self.N)
        else:
            # from Paolo Giannozzi: Notes on pseudopotential generation
            ScR_mass = 1 + 0.5 * (eps - self.veff) / c ** 2
            c0 = -l * (l + 1) - 2 * ScR_mass * self.rgrid ** 2 * (self.veff - eps)
            c0 -= self.dveff * self.rgrid / (2 * ScR_mass * c ** 2)
            c1 = self.rgrid * self.dveff / (2 * ScR_mass * c ** 2) - 1
        return c0, c1, c2


def shoot(u, dx, c2, c1, c0, N):
    """
    Integrate diff equation

           2
         d u      du
         --- c  + -- c  + u c  = 0
           2  2   dx  1      0
         dx

    in equispaced grid (spacing dx) using simple finite difference formulas

    u'(i) = (u(i+1) - u(i-1)) / (2*dx) and
    u''(i) = (u(i+1) - 2*u(i) + u(i-1)) / dx**2

    u[0:2] *has already been set* according to boundary conditions.

    return u, number of nodes, the discontinuity of derivative at
    classical turning point (ctp), and ctp
    c0(r) is negative with large r, and turns positive at ctp.
    """
    fp = c2 / dx ** 2 + 0.5 * c1 / dx
    fm = c2 / dx ** 2 - 0.5 * c1 / dx
    f0 = c0 - 2 * c2 / dx ** 2
 
    # backward integration down to classical turning point ctp
    # (or one point beyond to get derivative)
    # If no ctp, integrate half-way
    u[-1] = 1.0
    u[-2] = u[-1] * f0[-1] / fm[-1]
    all_negative = np.all(c0 < 0)
    for i in range(N - 2 , 0, -1):
        u[i - 1] = (-fp[i] * u[i + 1] - f0[i] * u[i]) / fm[i]
        if abs(u[i - 1]) > 1e10: 
            u[i - 1:] *= 1e-10  # numerical stability
        if c0[i] > 0:
            ctp = i
            break
        if all_negative and i == N // 2:
            ctp = N // 2
            break

    utp = u[ctp]
    utp1 = u[ctp + 1]
    dright = (u[ctp + 1] - u[ctp - 1]) / (2 * dx)

    for i in range(1, ctp + 1):
        u[i + 1] = (-f0[i] * u[i] - fm[i] * u[i - 1]) / fp[i]

    dleft = (u[ctp + 1] - u[ctp - 1]) / (2 * dx)
    scale = utp / u[ctp]
    u[:ctp + 1] *= scale
    u[ctp + 1] = utp1  # above overrode
    dleft *= scale
    u = u * np.sign(u[1])

    nodes = sum((u[0:ctp - 1] * u[1:ctp]) < 0)
    A = (dright - dleft) * utp
    return u, nodes, A, ctp


class RadialGrid:
    def __init__(self,grid):
        """
        mode
        ----

        rmin                                                        rmax
        r[0]     r[1]      r[2]            ...                     r[N-1] grid
        I----'----I----'----I----'----I----'----I----'----I----'----I
           r0[0]     r0[1]     r0[2]       ...              r0[N-2]       r0grid
           dV[0]     dV[1]     dV[2]       ...              dV[N-2]       dV

           dV[i] is volume element of shell between r[i] and r[i+1]
        """

        rmin, rmax = grid[0], grid[-1]
        N = len(grid)
        self.N = N
        self.grid = grid
        self.dr = self.grid[1:N] - self.grid[0:N - 1]
        self.r0 = self.grid[0:N - 1] + self.dr / 2
        # first dV is sphere (treat separately), others are shells
        self.dV = 4 * np.pi * self.r0 ** 2 * self.dr
        self.dV *= (4 * np.pi * rmax ** 3 / 3) / sum(self.dV)

    def get_grid(self):
        """ Return the whole radial grid. """
        return self.grid

    def get_N(self):
        """ Return the number of grid points. """
        return self.N

    def get_drgrid(self):
        """ Return the grid spacings (array of length N-1). """
        return self.dr

    def get_r0grid(self):
        """ Return the mid-points between grid spacings (array of length N-1). """
        return self.r0

    def get_dvolumes(self):
        """ Return dV(r)'s=4*pi*r**2*dr. """
        return self.dV

    def plot(self, screen=True):
        rgrid = self.get_grid()
        pl.scatter(list(range(len(rgrid))), rgrid)
        if screen: 
            pl.show()

    def integrate(self, f, use_dV=False):
        """
        Integrate function f (given with N grid points).
        int_rmin^rmax f*dr (use_dv=False) or int_rmin^rmax*f dV (use_dV=True)
        """
        if use_dV:
            return ((f[0:self.N - 1] + f[1:self.N]) * self.dV).sum() * 0.5
        else:
            return ((f[0:self.N - 1] + f[1:self.N]) * self.dr).sum() * 0.5


class XC_PW92:
    def __init__(self):
        """ The Perdew-Wang 1992 LDA exchange-correlation functional. """
        self.small = 1e-90
        self.a1 = 0.21370
        self.c0 = 0.031091
        self.c1 = 0.046644
        self.b1 = 1.0 / 2.0 / self.c0 * np.exp(-self.c1 / 2.0 / self.c0)
        self.b2 = 2 * self.c0 * self.b1 ** 2
        self.b3 = 1.6382
        self.b4 = 0.49294

    def exc(self, n, der=0):
        """ Exchange-correlation with electron density n. """
        e = self.e_x(n, der=der) + self.e_corr(n, der=der)
        if type(e) != np.float64:
            e[n < self.small] = 0.
        elif n < self.small:
            e = 0.
        return e

    def e_x(self, n, der=0):
        """ Exchange. """
        if der == 0:
            return -3. / 4 * (3 * n / pi) ** (1. / 3)
        elif der == 1:
            return -3. / (4 * pi) * (3 * n / pi) ** (-2. / 3)

    def e_corr(self, n, der=0):
        """ Correlation energy. """
        rs = (3. / (4 * pi * n)) ** (1. / 3)
        aux = 2 * self.c0 
        aux *= self.b1 * np.sqrt(rs) + self.b2 * rs + self.b3 * rs ** (3. / 2) + self.b4 * rs ** 2
        if der == 0:
            return -2 * self.c0 * (1 + self.a1 * rs) * np.log(1 + aux ** -1)
        elif der == 1:
            return (-2 * self.c0 * self.a1 * np.log(1 + aux ** -1) \
                    -2 * self.c0 * (1 + self.a1 * rs) * (1 + aux ** -1) ** -1 * (-aux ** -2) \
                   * 2 * self.c0 * (self.b1 / (2 * np.sqrt(rs)) + self.b2 + 3 * self.b3 * np.sqrt(rs) / 2 \
                   + 2 * self.b4 * rs)) * (-(4 * pi * n **2 *rs **2) ** -1)

    def vxc(self, n):
        """ Exchange-correlation potential (functional derivative of exc). """
        v = self.exc(n) + n * self.exc(n, der=1)
        v[n < self.small] = 0.
        return v
