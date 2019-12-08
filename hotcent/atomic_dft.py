""" Definition of the AtomicDFT class for atomic
DFT calculations.

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
from hotcent.atomic_base import AtomicBase, nl2tuple
from hotcent.confinement import ZeroConfinement
from hotcent.xc import XC_PW92, LibXC
try:
    import matplotlib.pyplot as plt
except:
    plt = None
warning_extension = 'Warning: C-extension "%s" not available'
try:
    from _hotcent import shoot
except ModuleNotFoundError:
    print(warning_extension % 'shoot')
    from hotcent.shoot import shoot
try:
    from _hotcent import hartree
except ModuleNotFoundError:
    print(warning_extension % 'hartree')
    hartree = None


class AtomicDFT(AtomicBase):
    def __init__(self,
                 symbol,
                 xcname='LDA',
                 convergence={'density':1e-7, 'energies':1e-7},
                 restart=None,
                 write=None,
                 **kwargs):
        """ Run Kohn-Sham all-electron calculations for a given atom.

        Example:
        ---------
        from hotcent.confinement import PowerConfinement
        atom = AtomicDFT('C',
                         xcname='GGA_C_PBE+GGA_X_PBE',
                         confinement=PowerConfinement(r0=3.0, s=2))
        atom.run()

        Parameters:
        -----------
        xcname:         Name of the XC functional. If 'LDA' or 'PW92' are
                        provided, then Hotcent's native LDA implementation
                        will be used. For all other functionals, the PyLibXC
                        module is required, which is bundled with LibXC.
                        The names of the implemented functionals can be found
                        on https://www.tddft.org/programs/libxc/functionals/
                        Often one needs to combine different LibXC functionals,
                        for example:
                          xcname='GGA_X_PBE+GGA_C_PBE'  # for PBE XC

        convergence:    convergence criterion dictionary
                        * density: max change for integrated |n_old-n_new|
                        * energies: max change in single-particle energy (Ha)

        write:          filename: save rgrid, effective potential and
                        density to a file for further calculations.

        restart:        filename: make an initial guess for effective
                        potential and density from another calculation.
        """
        AtomicBase.__init__(self, symbol, **kwargs)

        self.xcname = xcname
        self.convergence = convergence
        self.write = write
        self.restart = restart
        
        self.set_output(self.txt)

        if self.xcname in ['PW92', 'LDA']:
            self.xc = XC_PW92()
        else:
            self.xc = LibXC(self.xcname)

        if self.scalarrel:
            print('Using scalar relativistic corrections.', file=self.txt)

        maxnodes = max([n - l - 1 for n, l, nl in self.list_states()])
        self.rmin = 1e-2 / self.Z
        self.N = (maxnodes + 1) * self.nodegpts
        print('max %i nodes, %i grid points' % (maxnodes, self.N),
              file=self.txt)

        self.xgrid = np.linspace(0, np.log(self.rmax / self.rmin), self.N)
        self.rgrid = self.rmin * np.exp(self.xgrid)
        self.grid = RadialGrid(self.rgrid)
        self.timer.stop('init')

    def set_output(self, txt):
        """ Set output channel and give greetings. """
        if txt == '-':
            self.txt = sys.stdout
        elif txt is None:
            self.txt = open(os.devnull,'w')
        else:
            self.txt = open(txt, 'a')
        print('*******************************************', file=self.txt)
        print('Kohn-Sham all-electron calculation for %s' % self.symbol,
              file=self.txt)
        print('*******************************************', file=self.txt)

    def calculate_energies(self, echo='valence'):
        """ Calculate energy contributions. """
        self.timer.start('energies')
        assert echo in [None, 'valence', 'all']

        self.bs_energy = 0.0
        for n, l, nl in self.list_states():
            self.bs_energy += self.configuration[nl] * self.enl[nl]

        self.exc, self.vxc = self.xc.evaluate(self.dens, self.grid)
        self.vhar_energy = self.grid.integrate(self.vhar * self.dens,
                                               use_dV=True) / 2
        self.vxc_energy = self.grid.integrate(self.vxc * self.dens, use_dV=True)
        self.exc_energy = self.grid.integrate(self.exc * self.dens, use_dV=True)
        self.confinement_energy = self.grid.integrate(self.conf * self.dens,
                                                      use_dV=True)
        self.total_energy = self.bs_energy - self.vhar_energy
        self.total_energy += - self.vxc_energy + self.exc_energy

        if echo is not None:
            line = '%s orbital eigenvalues:' % echo
            print('\n'+line, file=self.txt)
            print('-' * len(line), file=self.txt)
            for n, l, nl in self.list_states():
                if echo == 'all' or nl in self.valence:
                    print('  %s:   %.12f' % (nl, self.enl[nl]), file=self.txt)

            print('\nenergy contributions:', file=self.txt)
            print('----------------------------------------', file=self.txt)
            print('sum of eigenvalues:     %.12f' % self.bs_energy,
                  file=self.txt)
            print('Hartree energy:         %.12f' % self.vhar_energy,
                  file=self.txt)
            print('vxc correction:         %.12f' % self.vxc_energy,
                  file=self.txt)
            print('exchange + corr energy: %.12f' % self.exc_energy,
                  file=self.txt)
            print('----------------------------------------', file=self.txt)
            print('total energy:           %.12f\n' % self.total_energy,
                  file=self.txt)

        self.timer.stop('energies')

    def calculate_density(self):
        """ Calculate the radial electron density:
        sum_nl occ_nl |Rnl(r)|**2 / (4*pi)
        """
        self.timer.start('density')
        dens = np.zeros_like(self.rgrid)
        for n,l,nl in self.list_states():
            dens += self.configuration[nl] * (self.unlg[nl] ** 2)

        nel1 = self.grid.integrate(dens)
        nel2 = self.get_number_of_electrons()

        if abs(nel1 - nel2) > 1e-10:
            err = 'Integrated density %.3g' % nel1
            err += ', number of electrons %.3g' % nel2
            raise RuntimeError(err)

        dens = dens / (4 * np.pi * self.rgrid **2)

        self.timer.stop('density')
        return dens

    def calculate_hartree_potential(self):
        """ Calculate the Hartree potential. """
        self.timer.start('Hartree')
        dV = self.grid.get_dvolumes()
        r, r0 = self.rgrid, self.grid.get_r0grid()
        N = self.N
        n0 = 0.5 * (self.dens[1:] + self.dens[:-1])
        nel = self.get_number_of_electrons()
        n0 *= nel / np.sum(n0 * dV)

        if hartree is not None:
            self.vhar = hartree(n0, dV, r, r0, N)
        else:
            lo, hi, self.vhar = np.zeros(N), np.zeros(N), np.zeros(N)
            lo[0] = 0.0
            for i in range(1, N):
                lo[i] = lo[i-1] + dV[i-1] * n0[i-1]

            hi[-1] = 0.0
            for i in range(N - 2, -1, -1):
                hi[i] = hi[i + 1] + n0[i] * dV[i] / r0[i]

            for i in range(N):
                self.vhar[i] = lo[i] / r[i] + hi[i]

        self.timer.stop('Hartree')

    def calculate_veff(self):
        """ Calculate effective potential. """
        self.timer.start('veff')
        exc, self.vxc = self.xc.evaluate(self.dens, self.grid)
        self.timer.stop('veff')
        return self.vnuc + self.vhar + self.vxc + self.conf

    def guess_density(self):
        """ Guess initial density. """
        r2 = 0.02 * self.Z  # radius at which density has dropped to half;
                            # can this be improved?
        dens = np.exp(-self.rgrid / (r2 / np.log(2)))
        nel = self.get_number_of_electrons()
        dens = dens / self.grid.integrate(dens, use_dV=True) * nel
        return dens

    def get_veff_and_dens(self):
        """ Construct effective potential and electron density. If restart
        a file is given, try to read from there, otherwise make a guess.
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
                self.veff = splev(self.rgrid, v)
                self.dens = splev(self.rgrid, d)
                done = True
            except IOError:
                print("Could not open restart file, " \
                      "starting from scratch.", file=self.txt)

        if not done:
            self.veff = self.vnuc + self.conf
            self.dens = self.guess_density()

    def run(self, wf_confinement_scheme='standard'):
        assert wf_confinement_scheme in ['standard', 'perturbative']

        val = self.get_valence_orbitals()
        enl = {}
        Rnlg = {}
        unlg = {}
        bar = '=' * 50
        confinement = self.confinement

        if wf_confinement_scheme == 'perturbative':
            print(bar, file=self.txt)
            print('Initial run without any confinement', file=self.txt)
            print('for pre-converging orbitals and eigenvalues', file=self.txt)
            print(bar, file=self.txt)
            self.confinement = ZeroConfinement()
            self._run()
            veff = self.veff.copy()

        for nl, wf_confinement in self.wf_confinement.items():
            assert nl in val, 'Confinement %s not in %s' % (nl, str(val))
            self.confinement = wf_confinement
            if self.confinement is None:
                self.confinement = ZeroConfinement()

            print(bar, file=self.txt)
            print('Applying %s' % self.confinement, file=self.txt)
            print('to get a confined %s orbital' % nl, file=self.txt)
            print(bar, file=self.txt)

            if wf_confinement_scheme == 'standard':
                self._run()
            elif wf_confinement_scheme == 'perturbative':
                self.veff = veff + self.confinement(self.rgrid)
                self.solve_single_eigenstate(nl)
                print('Confined %s eigenvalue: %.6f' % (nl, self.enl[nl]),
                      file=self.txt)

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

        # restore overwritten attributes
        self.Rnlg.update(Rnlg)
        self.unlg.update(unlg)
        self.enl.update(enl)
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
        """ Solve the self-consistent potential. """
        self.timer.start('solve ground state')
        print('\nStart iteration...', file=self.txt)
        self.enl = {}
        self.d_enl = {}
        for n, l, nl in self.list_states():
            self.enl[nl] = 0.0
            self.d_enl[nl] = 0.0

        N = self.grid.get_N()

        # make confinement and nuclear potentials; intitial guess for veff
        self.conf = self.confinement(self.rgrid)
        self.vnuc = self.nuclear_potential(self.rgrid)
        self.get_veff_and_dens()
        self.calculate_hartree_potential()

        for it in range(self.maxiter):
            self.veff *= 1. - self.mix
            self.veff += self.mix * self.calculate_veff()
            if self.scalarrel:
                veff = SplineFunction(self.rgrid, self.veff)
                self.dveff = veff(self.rgrid, der=1)
            d_enl_max, itmax = self.solve_eigenstates(it)

            dens0 = self.dens.copy()
            self.dens = self.calculate_density()
            diff = self.grid.integrate(np.abs(self.dens - dens0), use_dV=True)

            if diff < self.convergence['density'] and it > 5:
                if d_enl_max < self.convergence['energies']:
                    break
            self.calculate_hartree_potential()

            if np.mod(it, 10) == 0:
                line = 'iter %3i, dn=%.1e>%.1e, max %i sp-iter' % \
                       (it, diff, self.convergence['density'], itmax)
                
                print(line, file=self.txt)

            if it == self.maxiter - 1:
                if self.timing:
                    self.timer.summary()
                err = 'Density not converged in %i iterations' % (it + 1)
                raise RuntimeError(err)
            self.txt.flush()

        self.calculate_energies(echo='valence')
        print('converged in %i iterations' % it, file=self.txt)
        nel = self.get_number_of_electrons()
        line = '%9.4f electrons, should be %9.4f' % \
               (self.grid.integrate(self.dens, use_dV=True), nel)
        print(line, file=self.txt)

        self.timer.stop('solve ground state')

    def solve_eigenstates(self, iteration, itmax=100):
        """ Solve the eigenstates for given effective potential.

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

                # boundary conditions for integration from analytic behaviour
                # (unscaled)
                # u(r)~r**(l+1)   r->0
                # u(r)~exp( -sqrt(c0(r)) ) (set u[-1]=1
                # and use expansion to avoid overflows)
                u[0:2] = rgrid[0:2] ** (l + 1)

                if not(c0[-2] < 0 and c0[-1] < 0):
                    plt.plot(c0)
                    plt.show()

                assert c0[-2] < 0 and c0[-1] < 0
                self.timer.start('shoot')
                u, nodes, A, ctp = shoot(u, dx, c2, c1, c0, N)
                self.timer.stop('shoot')
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
                    err = 'Eigensolver out of iterations. Atom not stable?'
                    raise RuntimeError(err)

            itmax = max(it, itmax)
            self.unlg[nl] = u
            self.Rnlg[nl] = self.unlg[nl] / self.rgrid
            self.d_enl[nl] = abs(eps - self.enl[nl])
            d_enl_max = max(d_enl_max, self.d_enl[nl])
            self.enl[nl] = eps

            if self.verbose:
                line = '-- state %s, %i eigensolver iterations' % (nl, it)
                line += ', e=%9.5f, de=%9.5f' % (self.enl[nl], self.d_enl[nl])
                print(line, file=self.txt)

            assert nodes == nodes_nl
            assert u[1] > 0.0

        self.timer.stop('eigenstates')
        return d_enl_max, itmax

    def solve_single_eigenstate(self, nl, iteration=0):
        """
        Solve a single eigenstate nl for given effective potential.

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

        n, l = nl2tuple(nl)
        nodes_nl = n - l - 1
        eps = self.enl[nl]
        delta = self.d_enl[nl]

        direction = 'none'
        epsmax = self.veff[-1] - l * (l + 1) / (2 * self.rgrid[-1] ** 2)
        it = 0
        u = np.zeros(N)
        hist = []

        if self.scalarrel:
            veff = SplineFunction(self.rgrid, self.veff)
            self.dveff = veff(self.rgrid, der=1)

        while True:
            eps0 = eps
            c0, c1, c2 = self.construct_coefficients(l, eps)

            # boundary conditions for integration from analytic behaviour
            # (unscaled)
            # u(r)~r**(l+1)   r->0
            # u(r)~exp( -sqrt(c0(r)) ) (set u[-1]=1
            # and use expansion to avoid overflows)
            u[0:2] = rgrid[0:2] ** (l + 1)

            if not(c0[-2] < 0 and c0[-1] < 0):
                plt.plot(c0)
                plt.show()

            assert c0[-2] < 0 and c0[-1] < 0

            self.timer.start('shoot')
            u, nodes, A, ctp = shoot(u, dx, c2, c1, c0, N)
            self.timer.stop('shoot')
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
                err = 'Eigensolver out of iterations. Atom not stable?'
                raise RuntimeError(err)

            itmax = max(it, itmax)
            self.unlg[nl] = u
            self.Rnlg[nl] = self.unlg[nl] / self.rgrid
            self.d_enl[nl] = abs(eps - self.enl[nl])
            d_enl_max = max(d_enl_max, self.d_enl[nl])
            self.enl[nl] = eps

            if self.verbose:
                line = '-- state %s, %i eigensolver iterations' % (nl, it)
                line += ', e=%9.5f, de=%9.5f' % (self.enl[nl], self.d_enl[nl])
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
            c0 = -l * (l + 1)
            c0 -= 2 * ScR_mass * self.rgrid ** 2 * (self.veff - eps)
            c0 -= self.dveff * self.rgrid / (2 * ScR_mass * c ** 2)
            c1 = self.rgrid * self.dveff / (2 * ScR_mass * c ** 2) - 1
        return c0, c1, c2


class RadialGrid:
    def __init__(self,grid):
        """
        mode
        ----

        rmin                                                        rmax
        r[0]     r[1]      r[2]            ...                     r[N-1] grid
        I----'----I----'----I----'----I----'----I----'----I----'----I
           r0[0]     r0[1]     r0[2]       ...              r0[N-2]     r0grid
           dV[0]     dV[1]     dV[2]       ...              dV[N-2]         dV

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
        """ Return the mid-points between grid spacings
        (array of length N-1).
        """
        return self.r0

    def get_dvolumes(self):
        """ Return dV(r)'s=4*pi*r**2*dr. """
        return self.dV

    def plot(self, screen=True):
        rgrid = self.get_grid()
        plt.scatter(list(range(len(rgrid))), rgrid)
        if screen: 
            plt.show()

    def integrate(self, f, use_dV=False):
        """ Integrate function f (given with N grid points).
        int_rmin^rmax f*dr (use_dv=False) or int_rmin^rmax*f dV (use_dV=True)
        """
        if use_dV:
            return ((f[0:self.N - 1] + f[1:self.N]) * self.dV).sum() * 0.5
        else:
            return ((f[0:self.N - 1] + f[1:self.N]) * self.dr).sum() * 0.5

    def gradient(self, f):
        return np.gradient(f, self.grid)

    def divergence(self, f):
        return (1. / self.grid ** 2) * self.gradient(self.grid ** 2 * f)
