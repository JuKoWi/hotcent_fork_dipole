""" Defintion of the AllElectron based class (and 
supporting methods) for atomic DFT calculations.

The code below draws heavily from the Hotbit code 
written by Pekka Koskinen (https://github.com/pekkosk/
hotbit/blob/master/hotbit/parametrization/atom.py).
"""
from __future__ import division, print_function
import pickle
import collections
from copy import copy
import numpy as np
from scipy.optimize import minimize
from ase.data import atomic_numbers, covalent_radii
from ase.units import Bohr
from hotcent.interpolation import Function, SplineFunction
from hotcent.timing import Timer
try:
    import pylab as pl
except:
    pl = None


class AllElectron:
    def __init__(self,
                 symbol,
                 configuration='',
                 valence=[],
                 confinement=None,
                 wf_confinement={},
                 xcname='LDA',
                 scalarrel=False,
                 mix=0.2,
                 maxiter=200,
                 rmax=100.0,
                 nodegpts=500,
                 timing=False,
                 verbose=False,
                 txt='-'):
        """
        Base class for atomic DFT calculators

        symbol:         chemical symbol
        configuration:  e.g. '[He] 2s2 2p2'    
        valence:        valence orbitals, e.g. ['2s','2p']. 
        confinement:    confinement potential for the electron density 
                        (see hotcent.confinement)
        wf_confinement: dictionary with confinement potentials for the
                        valence orbitals. If empty, the same confinement
                        potential is used as for the electron density
                        (see the 'confinement' parameter).
        xcname:         Name of the XC functional
        scalarrel:      Use scalar relativistic corrections
        mix:            effective potential mixing constant
        maxiter:          maximum number of iterations for self-consistency.
        rmax:           radial cutoff in Bohr
        nodegpts:       total number of grid points is nodegpts times the max number
                        of antinodes for all orbitals
        timing:         output of timing summary
        verbose:        increase verbosity during iterations
        txt:            output file name for log data;
                        use '-' for stdout (default), None for /dev/null,
                        and any other string for a text file
        """
        self.symbol = symbol
        self.valence = valence
        self.confinement = confinement
        self.wf_confinement = wf_confinement
        self.xcname = xcname
        self.scalarrel = scalarrel
        self.mix = mix
        self.maxiter = maxiter
        self.rmax = rmax
        self.nodegpts = nodegpts
        self.timing = timing
        self.verbose = verbose
        self.txt = txt

        self.timer = Timer('AllElectron', txt=self.txt, enabled=self.timing)
        self.timer.start('init')

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
                conf = {term[:2]: float(term[2:])}
            self.configuration.update(conf)

        self.nel = sum(self.configuration.values())
        self.charge = self.Z - self.nel

        self.conf = None
        self.nucl = None
        self.exc = None

        self.plotr = {}
        self.unlg = {}
        self.Rnlg = {}
        self.unl_fct = {}
        self.Rnl_fct = {}
        self.veff_fct = None
        self.dens_fct = None
        self.vhar_fct = None
        self.total_energy = 0.0

        self.maxl = 9
        self.maxn = 9

        self.solved = False

    def __getstate__(self):
        """ Return dictionary of all pickable items. """
        d = self.__dict__.copy()
        for key in self.__dict__:
            if isinstance(d[key], collections.Callable):
                d.pop(key)
        d.pop('out')
        return d

    def V_nuclear(self,r):
        return -self.Z / r

    def run(self, **kwargs):
        raise NotImplementedError('Child class must implement run() method!')
        
    def plot_Rnl(self, filename=None):
        """ Plot radial wave functions with matplotlib.
        
        filename:  output file name + extension (extension used in matplotlib)
        """
        if pl is None:
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
            pl.xticks(size=5)
            
            # annotate
            c = 'k'
            if nl in self.valence: 
                c = 'r'
            pl.text(0.5, 0.4, r'$R_{%s}(r)$' % nl, transform=ax.transAxes, 
                    size=15, color=c)
            if ax.is_first_col():
                pl.ylabel(r'$R_{nl}(r)$', size=8)
            i += 1
            
        # as a function of radius
        i = p ** 2 + 1
        for n, l, nl in self.list_states():
            ax = pl.subplot(2 * p, p, i)
            pl.plot(self.rgrid[:ri], self.Rnlg[nl][:ri])
            pl.xticks(size=5)
            if ax.is_last_row():
                pl.xlabel('r (Bohr)', size=8)

            c = 'k'
            if nl in self.valence: 
                c='r'
            pl.text(0.5, 0.4, r'$R_{%s}(r)$' % nl, transform=ax.transAxes,
                    size=15, color=c)
            if ax.is_first_col():
                pl.ylabel(r'$R_{nl}(r)$', size=8)
            i += 1
        
        fig.subplots_adjust(hspace=0.2, wspace=0.1)
        s = '' if self.confinement is None else ' (confined)'
        pl.figtext(0.4, 0.95, r'$R_{nl}(r)$ for %s%s' % (self.symbol, s))

        if filename is None:
            filename = '%s_KSAllElectron.pdf' % self.symbol
        pl.savefig(filename)
        pl.clf()

    def plot_density(self, filename=None):
        """ Plot the electron density with matplotlib.
        
        filename:  output file name + extension (extension used in matplotlib)
        """
        if pl is None:
            raise AssertionError('pylab could not be imported')

        rmax = covalent_radii[self.Z] / Bohr * 3
        ri = np.where(self.rgrid < rmax)[0][-1]

        pl.clf()
        core_dens = 0
        colors = ['red', 'green', 'blue']
        for n, l, nl in self.list_states():
            if nl not in self.unlg:
                continue

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

        s = '' if self.confinement is None else ' (confined)'
        pl.figtext(0.4, 0.95, r'Density for %s%s' % (self.symbol, s))

        pl.legend(loc='upper right', ncol=2)

        if filename is None:
            filename = '%s_density.pdf' % self.symbol
        pl.savefig(filename)
        pl.clf()

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

    def get_wf_range(self, nl, fractional_limit=1e-7):
        """ Return the maximum r for which |R(r)|<fractional_limit*max(|R(r)|) """
        #wfmax = max(abs(self.Rnlg[nl]))
        wfmax = np.nanmax(np.abs(self.Rnlg[nl]))
        for r, wf in zip(self.rgrid[-1::-1], self.Rnlg[nl][-1::-1]):
            if abs(wf) > fractional_limit * wfmax:
                return r

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
        if self.veff_fct is None:
            self.veff_fct = Function('spline', self.rgrid, self.veff)
        return self.veff_fct(r, der=der)

    def electron_density(self, r, der=0):
        """ Return the all-electron density at r. """
        if self.dens_fct is None:
            self.dens_fct = Function('spline', self.rgrid, self.dens)
        return self.dens_fct(r, der=der)

    def hartree_potential(self, r):
        """ Return the Hartree potential at r. """
        if self.vhar_fct is None:
            self.vhar_fct = Function('spline', self.rgrid, self.Hartree)
        return self.vhar_fct(r)

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

    def fit_sto(self, nl, num_exp, num_pow, regularization=1e-6,
                filename=None):
        """ Fit Slater-type orbitals to the one on the grid.
            See self.write_hsd() for more information.

        Parameters:
        -----------
        nl:              the (valence) orbital of interest (e.g. '2p')
        num_exp:         number of exponents to use
        num_pow:         number of r-powers for each exponents
        regularization:  penalty to be used in the L2-regularization
        filename:        filename for a figure with the grid-based
                         and STO-fitted orbitals (to verify that the
                         fit is decent)
        """
        r = self.rgrid
        y = self.Rnlg[nl]
        l = orbit_transform(nl, False)[1]
        num_coeff = num_exp * num_pow
        num_r = len(r)

        def regression(param):
            A = np.zeros((num_r, num_coeff))

            for i in range(num_exp):
                rexp = np.exp(-param[i] * r)[:, None]
                A[:, i * num_pow : (i + 1) * num_pow] = rexp

            for j in range(num_pow):
                rpow = np.power(r, l + j)
                for i in range(num_exp):
                    A[:, num_pow * i + j] *= rpow

            A_loss = np.identity(num_coeff) * regularization
            AA = np.vstack((A, A_loss))

            y_loss = np.zeros(num_coeff)
            yy = np.hstack((y, y_loss))

            coeff, residual, rank, s = np.linalg.lstsq(AA, yy)
            values = np.dot(A, coeff)
            return coeff, values, residual[0]

        def residual(param):
            coeff, values, residual = regression(param)
            return residual

        if num_exp > 1:
            x0, x1 = self.Z, 0.5
            ratio = (x0 / x1) ** (1. / (num_exp - 1.))
            guess = [x0 / (ratio ** i) for i in range(num_exp)]
        else:
            guess = [1.]

        result = minimize(residual, guess, method='COBYLA',
                          options={'rhobeg': 0.1, 'tol': 1e-8})
        exponents = result.x
        coeff, values, residual = regression(exponents)

        integral = np.trapz((r * y) ** 2, x=r)
        if abs(integral - 1) > 1e-1:
            print('Warning -- significant deviation from unity for integral'
                  ' of grid-based %s orbital: %.5f' % (nl, integral))

        integral = np.trapz((r * values) ** 2, x=r)
        if abs(integral - 1) > 1e-1:
            print('Warning -- significant deviation from unity for integral'
                  ' of STO-based %s orbital: %.5f' % (nl, integral))

        if filename is not None:
            rmax = 3 * covalent_radii[self.Z] / Bohr
            imax = np.where(r < rmax)[0][-1]
            rmin = 1e-3 * self.Z
            imin = np.where(r < rmin)[0][-1]
            pl.plot(r[imin:imax], y[imin:imax], '-', label='On the grid')
            pl.plot(r[imin:imax], values[imin:imax], '--', label='With STOs')
            pl.xlim([0., rmax])
            pl.grid(ls='--')
            pl.legend(loc='upper right')
            pl.xlabel('r (Bohr radii)')
            pl.ylabel('Psi_%s (a.u.)' % nl)
            pl.savefig(filename)
            pl.clf()

        return exponents, coeff, values, residual

    def write_hsd(self, filename=None, num_exp=None, num_pow=4, wfthr=1e-2):
        """ Writes a HSD-format file with information on the valence
        orbitals. This includes a projection of these orbitals
        on a set of Slater-type orbitals, for post-processing
        purposes (e.g. using the Waveplot tool part of DFTB+).

        The expansion is the same as in DFTB+.
        For an atomic orbital with angular momentum l:

          R_l(r) = \sum_{i=0}^{num_exp-1} \sum_{j=0}^{num_pow-1}
                     coeff_{i,j} * r ^ (l + j) * \exp(-exponent_i * r)

        Note that also the same normalization is used as in DFTB+.
        This means that \int_{r=0}^{\infty} r^2 * |R_l(r)|^2 dr = 1.

        Starting from a reasonable initial guess, the exponents
        are optimized to reproduce the grid-based orbitals,
        with the coefficient matrix being determined by
        (L2-regularized) linear regression at each iteration.

        Parameters:
        -----------
        filename:   output file name. If None, the name
                    defaults to wcf.<Element>.hsd
        num_exp:    number of exponents to use
                    default = highest principal quantum number
        num_pow:    number of powers for each exponent
                    default = 4
        wfthr:      parameter determining the 'Cutoff' radius,
                    which will be where the orbital tail goes
                    below wfthr in absolute value
        """
        if filename is None:
            filename = 'wfc.%s.hsd' % self.symbol

        if num_exp is None:
            num_exp = max([orbit_transform(nl, False)[0]
                           for nl in self.valence])

        with open(filename, 'a') as f:
            f.write('%s = {\n' % self.symbol)
            f.write('  AtomicNumber = %d\n' % self.Z)

            for nl in self.valence:
                exp, coeff, values, resid = self.fit_sto(nl, num_exp, num_pow)
                icut = len(values) - 1
                while abs(values[icut]) < wfthr:
                    icut -= 1
                rcut = np.round(self.rgrid[icut + 1], 1)
                l = orbit_transform(nl, False)[1]

                f.write('  Orbital = {\n')
                f.write('    AngularMomentum = %d\n' % l)
                f.write('    Occupation = %.6f\n' % self.configuration[nl])
                f.write('    Cutoff = %.1f\n' % rcut)
                f.write('    Exponents = {\n    ')
                for e in exp:
                    f.write('  %.8f' % e)
                f.write('\n    }\n    Coefficients = {\n')
                for c in coeff:
                    f.write('      {: .8E}\n'.format(c))
                f.write('    }\n  }\n')
            f.write('}\n')

        return


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
