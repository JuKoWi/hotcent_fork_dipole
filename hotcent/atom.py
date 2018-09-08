""" Defintion of the KSAllElectron class (and 
supporting methods) for atomic DFT calculations.

The code below draws heavily from the Hotbit code 
written by Pekka Koskinen (https://github.com/pekkosk/
hotbit/blob/master/hotbit/parametrization/atom.py).
"""
from __future__ import division, print_function
import os
import sys
import pickle
import collections
from copy import copy
import numpy as np
from math import sqrt, pi, log
from scipy.interpolate import splrep, splev
from ase.data import atomic_numbers, covalent_radii
from ase.units import Bohr
from hotcent.interpolation import Function, SplineFunction
from hotcent.timing import Timer
try:
    import pylab as pl
except:
    pl = None


class KSAllElectron:
    def __init__(self,
                 symbol,
                 configuration='',
                 valence=[],
                 confinement=None,
                 wf_confinement={},
                 xc='PW92',
                 convergence={'density':1e-7, 'energies':1e-7},
                 scalarrel=False,
                 rmax=100.0,
                 nodegpts=500,
                 mix=0.2,
                 itmax=200,
                 timing=False,
                 verbose=False,
                 txt=None,
                 restart=None,
                 write=None):
        """
        Run Kohn-Sham all-electron calculation for a given atom.

        Examples:
        ---------
        atom = KSAllElectron('C')
        from hotcent.confinement import PowerConfinement
        atom = KSAllElectron('C', confinement=PowerConfinement(r0=3., s=2))
        atom.run()

        Parameters:
        -----------
        symbol:         chemical symbol
        configuration:  e.g. '[He] 2s2 2p2'    
        valence:        valence orbitals, e.g. ['2s','2p']. 
        confinement:    confinement potential for the electron density 
                        (see hotcent.confinement)
        wf_confinement: dictionary with confinement potentials for the
                        valence orbitals. If empty, the same confinement
                        potential is used as for the electron density
                        (see the 'confinement' parameter).
        etol:           sp energy tolerance for eigensolver (Hartree)
        convergence:    convergence criterion dictionary
                        * density: max change for integrated |n_old-n_new|
                        * energies: max change in single-particle energy (Hartree)
        scalarrel:      Use scalar relativistic corrections
        rmax:           radial cutoff
        nodegpts:       total number of grid points is nodegpts times the max number
                        of antinodes for all orbitals
        mix:            effective potential mixing constant
        itmax:          maximum number of iterations for self-consistency.
        timing:         output of timing summary
        verbose:        increase verbosity during iterations
        txt:            output file name for log data
        write:          filename: save rgrid, effective potential and
                        density to a file for further calculations.
        restart:        filename: make an initial guess for effective
                        potential and density from another calculation.
        """
        self.symbol = symbol
        self.valence = valence
        self.confinement = confinement
        self.wf_confinement = wf_confinement
        self.xc = xc
        self.convergence = convergence
        self.scalarrel = scalarrel
        self.set_output(txt)
        self.itmax = itmax
        self.verbose = verbose
        self.nodegpts = nodegpts
        self.mix = mix
        self.timing = timing
        self.timer = Timer('KSAllElectron', txt=self.txt, enabled=self.timing)
        self.timer.start('init')
        self.restart = restart
        self.write = write

        self.Z = atomic_numbers[self.symbol]
        assert len(self.valence) > 0
     
        noble_conf = {'He':{'1s':2}}
        noble_conf['Ne'] = dict({'2s':2, '2p':6}, **noble_conf['He'])
        noble_conf['Ar'] = dict({'3s':2, '3p':6}, **noble_conf['Ne'])
        noble_conf['Kr'] = dict({'3d':10, '4s':2, '4p':6}, **noble_conf['Ar'])
        noble_conf['Xe'] = dict({'4d':10, '5s':2, '5p':6}, **noble_conf['Kr'])
        noble_conf['Rn'] = dict({'4f':14, '5d':10, '6s':2, '6p':6},
                                **noble_conf['Xe'])

        self.configuration = {}
        assert len(configuration) > 0, "Specify the electronic configuration!"
        for term in configuration.split():
            if term[0] == '[' and term[-1] == ']':
                core = term[1:-1]
                assert core in noble_conf, "[Core] config is not a noble gas!"
                conf = noble_conf[core]
            else:
                conf = {term[:2]: int(term[2:])}
            self.configuration.update(conf)

        self.nel = sum(self.configuration.values())
        self.charge = self.Z - self.nel

        self.conf = None
        self.nucl = None
        self.exc = None
        if self.xc == 'PW92':
            self.xcf = XC_PW92()
        else:
            raise NotImplementedError('Not implemented XC functional: %s' %xc)

        # technical stuff
        self.maxl = 9
        self.maxn = 9
        self.plotr = {}
        self.unlg = {}
        self.Rnlg = {}
        self.unl_fct = {}
        self.Rnl_fct = {}
        self.veff_fct = None
        self.total_energy = 0.0

        maxnodes = max( [n - l - 1 for n, l, nl in self.list_states()] )
        self.rmin = 1e-2 / self.Z
        self.rmax = rmax
        self.N = (maxnodes + 1) * self.nodegpts
        self.rmin, self.rmax, self.N = (1e-2 / self.Z, rmax, (maxnodes + 1) * self.nodegpts)

        if self.scalarrel:
            print('Using scalar relativistic corrections.', file=self.txt)

        print('max %i nodes, %i grid points' % (maxnodes, self.N), file=self.txt)
        self.xgrid = np.linspace(0, np.log(self.rmax / self.rmin), self.N)
        self.rgrid = self.rmin * np.exp(self.xgrid)
        self.grid = RadialGrid(self.rgrid)
        self.timer.stop('init')
        #print(self.get_comment(), file=self.txt)
        self.solved=False

    def __getstate__(self):
        """ Return dictionary of all pickable items. """
        d = self.__dict__.copy()
        for key in self.__dict__:
            if isinstance(d[key], collections.Callable):
                d.pop(key)
        d.pop('out')
        return d

    def set_output(self,txt):
        """ Set output channel and give greetings. """
        if txt == '-':
            self.txt = open(os.devnull,'w')
        elif txt == None:
            self.txt = sys.stdout
        else:
            self.txt = open(txt, 'a')
        print('*******************************************', file=self.txt)
        print('Kohn-Sham all-electron calculation for %2s ' % self.symbol, file=self.txt)
        print('*******************************************', file=self.txt)


    def calculate_energies(self,echo=False):
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


    def V_nuclear(self,r):
        return -self.Z / r


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
                self.veff =np.array([splev(r,v) for r in self.rgrid])
                self.dens =np.array([splev(r,d) for r in self.rgrid])
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

        confinement = self.confinement
        for nl, wf_confinement in self.wf_confinement.iteritems():
            assert nl in val, "Confinement: %s not in %s" % (nl, str(val))
            self.confinement = wf_confinement
            self._run()
            Rnlg[nl] = self.Rnlg[nl].copy()
            unlg[nl] = self.unlg[nl].copy()
            enl[nl] = self.enl[nl]

        self.confinement = confinement 
        self._run()

        self.Rnlg.update(Rnlg)
        self.unlg.update(unlg)
        self.enl.update(enl)
        for nl in val:
            self.Rnl_fct[nl] = Function('spline', self.rgrid, self.Rnlg[nl])
            self.unl_fct[nl] = Function('spline', self.rgrid, self.unlg[nl])

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

        for it in range(self.itmax):
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

            if it == self.itmax - 1:
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


    def plot_Rnl(self, filename=None):
        """ Plot radial wave functions with matplotlib.
        
        filename:  output file name + extension (extension used in matplotlib)
        """
        if pl == None:
            raise AssertionError('pylab could not be imported')

        rmax = covalent_radii[self.Z] / Bohr * 3
        ri = np.where(self.rgrid < rmax)[0][-1]
        states = len(self.list_states())
        p = np.ceil(np.sqrt(states))  # p**2 >= states subplots
        
        fig = pl.figure()        
        i = 1
        # as a function of grid points
        for n, l, nl in self.list_states():
            ax = pl.subplot(2 * p, p, i)
            pl.plot(self.Rnlg[nl])
            pl.yticks([], [])
            pl.xticks(size=5)
            
            # annotate
            c = 'k'
            if nl in self.valence: 
                c = 'r'
            pl.text(0.5, 0.4, r'$R_{%s}(r)$' % nl, transform=ax.transAxes, size=15, color=c)
            if ax.is_first_col():
                pl.ylabel(r'$R_{nl}(r)$', size=8)
            i+=1
            
        # as a function of radius
        i = p ** 2 + 1
        for n, l, nl in self.list_states():
            ax = pl.subplot(2 * p, p, i)
            pl.plot(self.rgrid[:ri], self.Rnlg[nl][:ri])
            pl.yticks([], [])
            pl.xticks(size=5)
            if ax.is_last_row():
                pl.xlabel('r (Bohr)', size=8)

            c = 'k'
            if nl in self.valence: 
                c='r'
            pl.text(0.5, 0.4, r'$R_{%s}(r)$' % nl, transform=ax.transAxes, size=15, color=c)
            if ax.is_first_col():
                pl.ylabel(r'$R_{nl}(r)$', size=8)
            i += 1
        

        file = '%s_KSAllElectron.pdf' % self.symbol
        #pl.rc('figure.subplot',wspace=0.0,hspace=0.0)
        fig.subplots_adjust(hspace=0.2, wspace=0.1)
        s = ''
        if self.confinement != None:
            s = '(confined)'
        pl.figtext(0.4, 0.95, r'$R_{nl}(r)$ for %s-%s %s' % (self.symbol, self.symbol, s))
        if filename is not None:
            file = filename
        pl.savefig(file)


    def plot_density(self, filename=None):
        """ Plot the electron density with matplotlib.
        
        filename:  output file name + extension (extension used in matplotlib)
        """
        if pl == None:
            raise AssertionError('pylab could not be imported')

        rmax = covalent_radii[self.Z] / Bohr * 3
        ri = np.where(self.rgrid < rmax)[0][-1]

        pl.clf()
        core_dens = 0
        colors = ['red', 'green', 'blue']
        for n, l, nl in self.list_states():
            dens = (self.unlg[nl] / self.rgrid) ** 2 / (4 * np.pi)
            label = r'n$_\mathrm{%s}$' % nl

            if self.configuration[nl] > 0:
                dens *= self.configuration[nl]
                ls = '-'
            else:
                ls = '--' 
                label += r'$^*$'

            if nl in self.valence:
                pl.semilogy(self.rgrid[:ri], dens[:ri], ls, color=colors[l],
                            label=label)
            else:
                core_dens += dens

        if np.max(core_dens) > 0:
            pl.semilogy(self.rgrid[:ri], core_dens[:ri], color='gray', 
                        label=r'n$_\mathrm{core}$')

        pl.semilogy(self.rgrid[:ri], self.dens[:ri], 'k-',
                    label=r'n$_\mathrm{tot}$')

        ymax = np.exp(np.ceil(np.log(np.max(self.dens))))        
        pl.ylim([1e-7, ymax])

        pl.xlabel('r (Bohr)')
        pl.grid()

        s = '' if self.confinement is None else '(confined)'
        pl.figtext(0.4, 0.95, r'Density for %s %s' % (self.symbol, s))

        pl.legend(loc='upper right', ncol=2)

        if filename is None:
            filename = '%s_density.pdf' % self.symbol
        pl.savefig(filename)


    def get_wf_range(self, nl, fractional_limit=1E-7):
        """ Return the maximum r for which |R(r)|<fractional_limit*max(|R(r)|) """
        wfmax = max(abs(self.Rnlg[nl]))
        for r, wf in zip(self.rgrid[-1::-1], self.Rnlg[nl][-1::-1]):
            if abs(wf) > fractional_limit * wfmax:
                return r


    def list_states(self):
        """ List all potential states {(n,l,'nl')}. """
        states = []
        for l in range(self.maxl + 1):
            for n in range(1, self.maxn + 1):
                nl = orbit_transform((n, l), string=True)
                #if nl in self.occu:
                if nl in self.configuration:
                    states.append((n, l, nl))
        return states


    def get_energy(self):
        return self.total_energy


    def get_epsilon(self, nl):
        """ get_eigenvalue('2p') or get_eigenvalue((2,1)) """
        nls = orbit_transform(nl, string=True)
        if not self.solved:
            raise AssertionError('run calculations first.')
        return self.enl[nls]


    def effective_potential(self, r, der=0):
        """ Return effective potential at r or its derivatives. """
        if self.veff_fct == None:
            self.veff_fct = Function('spline', self.rgrid, self.veff)
        return self.veff_fct(r, der=der)


    def get_radial_density(self):
        return self.rgrid, self.dens


    def Rnl(self, r, nl, der=0):
        """ Rnl(r,'2p') or Rnl(r,(2,1))"""
        nls = orbit_transform(nl, string=True)
        return self.Rnl_fct[nls](r, der=der)


    def unl(self, r, nl, der=0):
        """ unl(r,'2p')=Rnl(r,'2p')/r or unl(r,(2,1))..."""
        nls = orbit_transform(nl, string=True)
        return self.unl_fct[nls](r, der=der)


    def get_valence_orbitals(self):
        """ Get list of valence orbitals, e.g. ['2s','2p'] """
        return self.valence


    def get_symbol(self):
        """ Return atom's chemical symbol. """
        return self.symbol


    def get_valence_energies(self):
        """ Return list of valence energies, e.g. ['2s','2p'] --> [-39.2134,-36.9412] """
        if not self.solved:
            raise AssertionError('run calculations first.')
        return [(nl, self.enl[nl]) for nl in self.valence]


    def write_unl(self, filename, only_valence=True, step=20):
        """ Append functions unl=Rnl*r, V_effective, V_confinement into file.
            Only valence functions by default.

        Parameters:
        -----------
        filename:         output file name (e.g. XX.elm)
        only_valence:     output of only valence orbitals
        step:             step size for output grid
        """
        if not self.solved:
            raise AssertionError('run calculations first.')
        if only_valence:
            orbitals = self.valence
        else:
            orbitals = [nl for n,l,nl in self.list_states()]

        with open(filename, 'a') as f:
            for nl in orbitals:
                print('\n\nu_%s=' % nl, file=f)

                for r, u in zip(self.rgrid[::step], self.unlg[nl][::step]):
                    print(r, u, file=f)
 
            print('\n\nv_effective=', file=f)

            for r,ve in zip(self.rgrid[::step], self.veff[::step]):
                    print(r, ve, file=f)

            print('\n\nconfinement=', file=f)

            for r,vc in zip(self.rgrid[::step], self.conf[::step]):
                    print(r, vc, file=f)

            print('\n\n', file=f)


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

angular_momenta = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
def orbit_transform(nl, string):
    """ Transform orbitals into strings<->tuples, e.g. (2,1)<->'2p'. """
    if string == True and type(nl) == type(''):
        return nl  # '2p'->'2p'
    elif string == True:
        return '%i%s' % (nl[0], angular_momenta[nl[1]])  # (2,1)->'2p'
    elif string == False and type(nl) == type((2, 1)):
        return nl  # (2,1)->(2,1)
    elif string == False:
        l = angular_momenta.index(nl[1])
        n = int(nl[0])
        return (n,l)  # '2p'->(2,1)
