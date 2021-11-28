#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
""" Definition of the base class for atomic DFT
calculations.

The code below draws heavily from the Hotbit code
written by Pekka Koskinen (https://github.com/pekkosk/
hotbit/blob/master/hotbit/parametrization/atom.py).
"""
import os
import sys
import collections
import numpy as np
from scipy.optimize import minimize
from ase.data import atomic_numbers, covalent_radii
from ase.units import Bohr, Ha
from hotcent.confinement import Confinement, ZeroConfinement
from hotcent.interpolation import build_interpolator, CubicSplineFunction
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.phillips_kleinman import PhillipsKleinmanPP
from hotcent.timing import Timer
try:
    import matplotlib.pyplot as plt
except:
    plt = None


NOT_SOLVED_MESSAGE = 'A required attribute is missing. ' \
                     'Please call the run() method first.'


class AtomicBase:
    """
    Base class for atomic DFT calculators.

    Parameters
    ----------
    symbol : str
        Chemical symbol.
    configuration : str
        Electronic configuration, e.g. '[He] 2s2 2p2'.
    valence : list
        Valence orbitals, e.g. ['2s','2p'].
    confinement : Confinement-like object, optional
        Confinement potential for the electron density
        (see hotcent.confinement). The default None means
        that means no density confinement will be applied.
    wf_confinement : dict, optional
        Dictionary with confinement potentials for the
        valence orbitals. If None, the same confinement will
        be used as for the electron density. If a certain
        hotcent.confinement.Confinement instance is provided,
        this will be applied to all valence states. If a
        dictionary is provided, it is supposed to look like
        this: {nl: <a certain Confinement instance, or None>
        for each nl in your set of valence states}.
        For missing entries, no confinement will be applied.
    scalarrel : bool, optional
        Whether to use scalar relativistic corrections (default:
        False). Setting it to True is strongly recommended for
        all-electron calculations of heavier elements
        (atomic numbers above, say, 24).
    mix : float, optional
        Mixing coefficient for the effective potential during
        the self-consistency cycle.
    maxiter : int, optional
        Maximum number of iterations for self-consistency.
    rmax : float, optional
        Radial cutoff in Bohr.
    nodegpts : int, optional
        Total number of grid points is nodegpts times the max
        number of antinodes for all orbitals.
    timing : bool, optional
        Whether to produce a timing summary (default: False).
    verbose : bool, optional
        Whether to increase verbosity during iterations
        (default: False).
    txt : str or None or file handle, optional
        Where output should be printed. Use '-' for stdout
        (default), None for /dev/null, any other string for a
        text file, or a file handle.
    """
    def __init__(self,
                 symbol,
                 configuration='',
                 valence=[],
                 confinement=None,
                 wf_confinement=None,
                 scalarrel=False,
                 mix=0.2,
                 maxiter=200,
                 rmax=100.0,
                 nodegpts=500,
                 timing=False,
                 verbose=False,
                 txt='-'):
        self.symbol = symbol
        self.valence = valence
        self.scalarrel = scalarrel
        self.mix = mix
        self.maxiter = maxiter
        self.rmax = rmax
        self.nodegpts = nodegpts
        self.timing = timing
        self.verbose = verbose

        if txt is None:
            self.txt = open(os.devnull, 'w')
        elif isinstance(txt, str):
            if txt == '-':
                self.txt = sys.stdout
            else:
                self.txt = open(txt, 'a')
        else:
            self.txt = txt

        self.set_confinement(confinement)
        self.set_wf_confinement(wf_confinement)
        self.rcutnl = {}
        for nl, conf in self.wf_confinement.items():
            if nl in self.valence and hasattr(conf, 'rc'):
                self.rcutnl[nl] = conf.rc

        self.timer = Timer('Atomic', txt=self.txt, enabled=self.timing)

        self.Z = atomic_numbers[self.symbol]
        assert len(self.valence) > 0

        assert len(configuration) > 0, "Specify the electronic configuration!"
        self.set_configuration(configuration)

        self.maxl = 9
        self.maxn = 9
        self.unlg = {}
        self.Rnlg = {}
        self.unl_fct = {nl: None for nl in self.configuration}
        self.Rnl_fct = {nl: None for nl in self.configuration}
        self.veff_fct = None
        self.dens_fct = None
        self.densval_fct = None
        self.vhar_fct = None
        self.vharval_fct = None
        self.energies = {}
        self.solved = False
        self.basis_sets = [valence]
        self.basis_size = 'sz'

        # Set default 'pseudopotential':
        self.pp = PhillipsKleinmanPP(self.symbol)

    def set_configuration(self, configuration):
        """ Set the electron configuration

        configuration: e.g. '[He] 2s2 2p2'
        """
        self.configuration = {}
        noble_conf = {'He':{'1s':2}}
        noble_conf['Ne'] = dict({'2s':2, '2p':6}, **noble_conf['He'])
        noble_conf['Ar'] = dict({'3s':2, '3p':6}, **noble_conf['Ne'])
        noble_conf['Kr'] = dict({'3d':10, '4s':2, '4p':6}, **noble_conf['Ar'])
        noble_conf['Xe'] = dict({'4d':10, '5s':2, '5p':6}, **noble_conf['Kr'])
        noble_conf['Rn'] = dict({'4f':14, '5d':10, '6s':2, '6p':6},
                                **noble_conf['Xe'])
        noble_conf['Og'] = dict({'5f':14, '6d':10, '7s':2, '7p':6},
                                **noble_conf['Rn'])

        for term in configuration.split():
            if term[0] == '[' and term[-1] == ']':
                core = term[1:-1]
                assert core in noble_conf, "[Core] config is not a noble gas!"
                conf = noble_conf[core]
            else:
                conf = {term[:2]: float(term[2:])}
            self.configuration.update(conf)

    def set_confinement(self, confinement):
        if confinement is None:
            self.confinement = ZeroConfinement()
        else:
            self.confinement = confinement

    def set_wf_confinement(self, wf_confinement):
        if wf_confinement is None:
            self.wf_confinement = {}
        elif isinstance(wf_confinement, Confinement):
            self.wf_confinement = {nl: wf_confinement for nl in self.valence}
        elif isinstance(wf_confinement, dict):
            self.wf_confinement = {}
            for nl in self.valence:
                if nl not in wf_confinement or wf_confinement[nl] is None:
                    self.wf_confinement[nl] = ZeroConfinement()
                else:
                    self.wf_confinement[nl] = wf_confinement[nl]
        else:
            msg = "Don't know what to do with the provided wf_confinement:\n"
            msg += str(wf_confinement)
            raise ValueError(msg)

    def run(self, **kwargs):
        """ Child classes must implement a run() method which,
        in turn, is supposed to set the following attributes:

        self.solved: whether the calculations are considered to be done
        self.energies: a dictionary with the total energy and its contributions
        self.rgrid: an array with the radial grid points g
        self.dens: an array with the electron density on the radial grid
        self.vhar: an array with the Hartree potential on the radial grid
        self.veff: an array with the effective potential on the radial grid
                   (note: veff = vnuc + vhar + vxc + vconf)
        self.enl: a {'nl': eigenvalue} dictionary
        self.Rnlg: a {'nl': R_nl(g) array} dictionary
        self.unlg: a {'nl': u_nl(g) array} dictionary (u_nl = R_nl / r)
        """
        raise NotImplementedError('Child class must implement run() method!')

    def __getstate__(self):
        """ Return dictionary of all pickable items. """
        d = self.__dict__.copy()
        for key in self.__dict__:
            if isinstance(d[key], collections.Callable):
                d.pop(key)
        d.pop('out')
        return d

    def get_symbol(self):
        """ Return atom's chemical symbol. """
        return self.symbol

    def get_number_of_electrons(self, only_valence=False):
        if only_valence:
            return sum([self.configuration[nl] for nl in self.valence])
        else:
            return sum(self.configuration.values())

    def list_states(self):
        """ List all potential states {(n,l,'nl')}. """
        states = []
        for l in range(self.maxl + 1):
            for n in range(1, self.maxn + 1):
                nl = tuple2nl(n, l)
                if nl in self.configuration:
                    states.append((n, l, nl))
        return states

    def get_valence_orbitals(self):
        """ Get list of valence orbitals, e.g. ['2s','2p'] """
        return self.valence

    def get_energy(self):
        assert self.solved, NOT_SOLVED_MESSAGE
        return self.energies['total']

    def get_epsilon(self, nl):
        """ E.g. get_eigenvalue('2p') """
        assert self.solved, NOT_SOLVED_MESSAGE
        return self.enl[nl]

    def get_valence_energies(self):
        """ Return list of valence eigenenergies. """
        assert self.solved, NOT_SOLVED_MESSAGE
        return [(nl, self.enl[nl]) for nl in self.valence]

    def get_eigenvalue(self, nl):
        return self.get_epsilon(nl)

    def get_wf_range(self, nl, fractional_limit=1e-7):
        """ Return the maximum r for which |R(r)| is
        less than fractional_limit * max(|R(r)|) """
        assert self.solved, NOT_SOLVED_MESSAGE
        wfmax = np.nanmax(np.abs(self.Rnlg[nl]))
        for r, wf in zip(self.rgrid[-1::-1], self.Rnlg[nl][-1::-1]):
            if abs(wf) > fractional_limit * wfmax:
                return r

    def Rnl(self, r, nl, der=0):
        """ Rnl(r, '2p') """
        assert self.solved, NOT_SOLVED_MESSAGE
        if self.Rnl_fct[nl] is None:
            rc = self.rcutnl[nl] if nl in self.rcutnl else None
            self.Rnl_fct[nl] = build_interpolator(self.rgrid, self.Rnlg[nl], rc)
        return self.Rnl_fct[nl](r, der=der)

    def unl(self, r, nl, der=0):
        """ unl(r, '2p') = Rnl(r, '2p') / r """
        assert self.solved, NOT_SOLVED_MESSAGE
        if self.unl_fct[nl] is None:
            rc = self.rcutnl[nl] if nl in self.rcutnl else None
            self.unl_fct[nl] = build_interpolator(self.rgrid, self.unlg[nl], rc)
        return self.unl_fct[nl](r, der=der)

    def electron_density(self, r, der=0, only_valence=False):
        """ Return the all-electron density at r. """
        assert self.solved, NOT_SOLVED_MESSAGE

        if self.densval_fct is None or self.dens_fct is None:
            # Find the largest (minimal basis) wave-function cutoff radius
            rcs = [self.rcutnl[nl] for nl in self.valence if nl in self.rcutnl]
            rc = max(rcs) if len(rcs) > 0 else None
            self.dens_fct = build_interpolator(self.rgrid, self.dens, rc)
            self.densval_fct = build_interpolator(self.rgrid, self.densval, rc)

        if only_valence:
            return self.densval_fct(r, der=der)
        else:
            return self.dens_fct(r, der=der)

    def nuclear_potential(self,r):
        return -self.Z / r

    def neutral_atom_potential(self, r):
        """ Return the so-called 'neutral atom potential' at r, defined as
        the sum of the Hartree potential and the local pseudopotential
        (doi:10.1103/PhysRevB.40.3979).
        """
        assert self.solved, NOT_SOLVED_MESSAGE
        vna = self.hartree_potential(r, only_valence=False)
        vna += self.pp.local_potential(r)
        return vna

    def effective_potential(self, r, der=0):
        """ Return effective potential at r or its derivatives. """
        assert self.solved, NOT_SOLVED_MESSAGE
        if self.veff_fct is None:
            self.veff_fct = build_interpolator(self.rgrid, self.veff)
        return self.veff_fct(r, der=der)

    def hartree_potential(self, r, only_valence=False):
        """ Return the Hartree potential at r. """
        assert self.solved, NOT_SOLVED_MESSAGE
        if only_valence:
            if self.vharval_fct is None:
                self.vharval_fct = build_interpolator(self.rgrid, self.vharval)
            return self.vharval_fct(r)
        else:
            if self.vhar_fct is None:
                self.vhar_fct = build_interpolator(self.rgrid, self.vhar)
            return self.vhar_fct(r)

    def plot_Rnl(self, filename=None, only_valence=True):
        """ Plot radial wave functions with matplotlib.

        filename:  output file name + extension (extension used in matplotlib)
                   default = <Element>_Rnl.pdf
        only_valence: whether to only plot the valence states or all of them
        """
        assert plt is not None, 'Matplotlib could not be imported!'
        assert self.solved, NOT_SOLVED_MESSAGE

        rmax = 3 * covalent_radii[self.Z] / Bohr
        ri = np.where(self.rgrid < rmax)[0][-1]

        if only_valence:
            states = self.valence
        else:
            states = [x[2] for x in self.list_states()]

        p = int(np.ceil(np.sqrt(len(states))))
        q = 2 * p

        fig = plt.figure(dpi=400)
        i = 1
        # as a function of grid points
        for nl in states:
            ax = plt.subplot(q, p, i)
            plt.plot(self.Rnlg[nl])
            plt.xticks(size=5)
            plt.grid(ls='--')

            # annotate
            c = 'k'
            if nl in self.valence:
                c = 'r'
            plt.text(0.5, 0.4, r'$R_{%s}(i)$' % nl, transform=ax.transAxes,
                     size=15, color=c)
            if ax.get_subplotspec().is_first_col():
                plt.ylabel(r'$R_{nl}(i)$', size=8)
            i += 1

        # as a function of radius
        i = p ** 2 + 1
        for nl in states:
            ax = plt.subplot(q, p, i)
            plt.plot(self.rgrid[:ri], self.Rnlg[nl][:ri])
            plt.xticks(size=5)
            plt.grid(ls='--')

            # annotate
            c = 'k'
            if nl in self.valence:
                c = 'r'
            plt.text(0.5, 0.4, r'$R_{%s}(r)$' % nl, transform=ax.transAxes,
                     size=15, color=c)
            if ax.get_subplotspec().is_first_col():
                plt.ylabel(r'$R_{nl}(r)$', size=8)
            if ax.get_subplotspec().is_last_row():
                plt.xlabel('r (Bohr)', size=8)
            i += 1

        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.figtext(0.4, 0.95, r'$R_{nl}(r)$ for %s' % self.symbol)

        if filename is None:
            filename = '%s_Rnl.pdf' % self.symbol
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()

    def plot_density(self, filename=None):
        """ Plot the electron density and valence orbital densities.

        Note that the plotted electron density (rho_0) generally does
        not correspond to the sum of the valence orbital densities in
        the valence region. For this to be the case, the orbital densities
        would need to be multiplied by their occupation numbers, and
        the same confinement potential would need to be applied throughout.

        filename:  output file name + extension (extension used in matplotlib)
                   default = <Element>_rho.pdf
        """
        assert plt is not None, 'Matplotlib could not be imported!'
        assert self.solved, NOT_SOLVED_MESSAGE

        rmax = 3 * covalent_radii[self.Z] / Bohr
        ri = np.where(self.rgrid > rmax)[0][0]

        plt.figure(figsize=(6.4, 4.8), dpi=400)
        colors = ['red', 'green', 'blue', 'purple']  # for s/p/d/f

        for n, l, nl in self.list_states():
            if nl not in self.valence:
                continue

            dens = self.Rnlg[nl] ** 2 / (4 * np.pi)
            occupied = self.configuration[nl] > 0
            suffix = '' if occupied else '*'
            ls = '-' if occupied else '--'
            label = r'$|R_\mathrm{%s%s}(r) / \sqrt{4\pi}|^2$' % (nl, suffix)

            plt.semilogy(self.rgrid[:ri], dens[:ri], ls=ls, color=colors[l],
                        label=label)

        dens = self.dens[:ri]
        plt.semilogy(self.rgrid[:ri], dens, 'k-', label=r'$\rho_0(r)$')

        ymax = np.exp(np.ceil(np.log(np.max(dens))))
        plt.ylim([1e-7, ymax])
        plt.xlim([-0.05 * rmax, rmax])
        plt.xlabel('r (Bohr)')
        plt.grid(ls='--')
        plt.legend(loc='upper right', ncol=2)
        plt.title('Electron and orbital densities for %s' % self.symbol)

        if filename is None:
            filename = '%s_rho.pdf' % self.symbol
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()

    def plot_rho(self, *args, **kwargs):
        self.plot_density(*args, **kwargs)

    def write_unl(self, filename, only_valence=True, step=20):
        """ Append functions unl=Rnl*r into file.
            Only valence functions by default.

        Parameters:
        -----------
        filename:         output file name (e.g. XX.elm)
        only_valence:     output of only valence orbitals
        step:             step size for output grid
        """
        assert self.solved, NOT_SOLVED_MESSAGE
        if only_valence:
            orbitals = self.valence
        else:
            orbitals = [nl for n,l,nl in self.list_states()]

        with open(filename, 'a') as f:
            for nl in orbitals:
                f.write('\n\nu_%s=' % nl)
                for r, u in zip(self.rgrid[::step], self.unlg[nl][::step]):
                    f.write(r, u)
            f.write('\n\n')

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
        assert self.solved, NOT_SOLVED_MESSAGE
        print('Fitting Slater-type orbitals to eigenstate %s' % nl,
              file=self.txt)
        r = self.rgrid
        y = self.Rnlg[nl]
        n, l = nl2tuple(nl)
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

            coeff, residual, rank, s = np.linalg.lstsq(AA, yy, rcond=None)
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
                  ' of grid-based %s orbital: %.5f' % (nl, integral),
                  file=self.txt)

        integral = np.trapz((r * values) ** 2, x=r)
        if abs(integral - 1) > 1e-1:
            print('Warning -- significant deviation from unity for integral'
                  ' of STO-based %s orbital: %.5f' % (nl, integral),
                  file=self.txt)

        if filename is not None:
            rmax = 3 * covalent_radii[self.Z] / Bohr
            imax = np.where(r < rmax)[0][-1]
            rmin = 1e-3 * self.Z
            imin = np.where(r < rmin)[0][-1]
            plt.plot(r[imin:imax], y[imin:imax], '-', label='On the grid')
            plt.plot(r[imin:imax], values[imin:imax], '--', label='With STOs')
            plt.xlim([0., rmax])
            plt.grid(ls='--')
            plt.legend(loc='upper right')
            plt.xlabel('r (Bohr radii)')
            plt.ylabel('Psi_%s (a.u.)' % nl)
            plt.savefig(filename)
            plt.clf()

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
        assert self.solved, NOT_SOLVED_MESSAGE
        if filename is None:
            filename = 'wfc.%s.hsd' % self.symbol

        if num_exp is None:
            num_exp = max([nl2tuple(nl)[0] for nl in self.valence])

        with open(filename, 'a') as f:
            f.write('%s = {\n' % self.symbol)
            f.write('  AtomicNumber = %d\n' % self.Z)

            for nl in self.valence:
                n, l = nl2tuple(nl)
                exp, coeff, values, resid = self.fit_sto(nl, num_exp, num_pow)
                icut = len(values) - 1
                while abs(values[icut]) < wfthr:
                    icut -= 1
                rcut = np.round(self.rgrid[icut + 1], 1)

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

    def get_hubbard_value(self, nl, maxstep=1., scheme=None):
        """
        Calculates the Hubbard value of an orbital using
        (second order) finite differences.

        Note: when using perturbative confinement, the run() method
        has to be called first, because the radial functions are needed
        (and are kept fixed). This is not the case with the non-
        perturbative scheme, where a self-consistent solution is sought
        for every orbital occupancy.

        Parameters
        ----------
        nl : str
            Orbital label (e.g. '2p').
        maxstep : float, optional
            The maximal step size in the orbital occupancy.
            The default default value of 1 means not going further
            than the monovalent ions.
        scheme : None or str, optional
            The finite difference scheme, either 'central', 'forward'
            or 'backward' or None. In the last case the appropriate
            scheme will be chosen based on the orbital occupation.
        """
        assert scheme in [None, 'central', 'forward', 'backward']

        is_minimal = nl in self.valence

        if self.perturbative_confinement:
            assert self.solved, NOT_SOLVED_MESSAGE
        else:
            assert is_minimal, 'Non-minimal basis sets only implemented' + \
                   ' for the perturbative confinement scheme.'

        def get_total_energy(configuration, diff=None):
            if self.perturbative_confinement:
                configuration_original = self.configuration.copy()
                veff_original = np.copy(self.veff)
                only_valence = not isinstance(self.pp, PhillipsKleinmanPP)

                # Obtain dens & veff for the given electronic configuration
                self.configuration = configuration.copy()
                dens = self.calculate_density(self.unlg,
                                              only_valence=only_valence)
                if not is_minimal:
                    # Change density and number of electrons
                    # due to non-minimal basis function occupation
                    self.configuration[self.valence[0]] += diff
                    dens += diff * self.unlg[nl]**2 \
                            / (4 * np.pi * self.rgrid**2)
                self.veff = self.calculate_veff(dens)

                # Update the valence eigenvalues
                enl = self.enl.copy()
                for nl2 in self.valence:
                    enl[nl2] = self.get_onecenter_integrals(nl2, nl2)[0]

                # Get the total energy
                if only_valence:
                    dens_xc = self.add_core_electron_density(dens)
                else:
                    dens_xc = None
                energies = self.calculate_energies(enl, dens,
                                                   dens_xc=dens_xc,
                                                   echo='valence',
                                                   only_valence=only_valence)
                e = energies['total']
                if not is_minimal:
                    # Add non-minimal basis function eigenenergy
                    e += diff * self.get_onecenter_integrals(nl, nl)[0]

                self.configuration = configuration_original
                self.veff = veff_original
            else:
                configuration_original = self.configuration.copy()
                self.configuration = configuration.copy()
                self.run()
                self.configuration = configuration_original
                e = self.get_energy()
                self.solved = False
            return e

        if scheme is None:
            n, l = nl2tuple(nl[:2])
            max_occup = 2 * (2*l + 1)
            occup = self.configuration[nl] if is_minimal else 0
            if occup == 0:
                scheme = 'forward'
            elif occup == max_occup:
                scheme = 'backward'
            else:
                scheme = 'central'
        elif not is_minimal:
            assert scheme == 'forward'

        directions = {'forward': [0, 1, 2],
                      'central': [-1, 0, 1],
                      'backward': [-2, -1, 0]}
        delta = maxstep if scheme == 'central' else 0.5 * maxstep

        configuration = self.configuration.copy()
        energies = {}
        bar = '+' * 12

        for direction in directions[scheme]:
            diff = direction * delta
            if is_minimal:
                configuration[nl] += diff
                s = ' '.join([nl2 + '%.1f' % configuration[nl2]
                              for nl2 in self.valence])
                print('\n%s Configuration %s %s' % (bar, s, bar), file=self.txt)
                energies[direction] = get_total_energy(configuration)
                configuration[nl] -= diff
            else:
                energies[direction] = get_total_energy(configuration, diff=diff)

        # Check that the original electronic configuration has been restored
        if is_minimal:
            assert self.configuration[nl] == configuration[nl]

        if scheme in ['forward', 'central']:
            EA = (energies[0] - energies[1]) / delta
            print('\nElectron affinity = %.5f Ha (%.5f eV)' % (EA, EA * Ha),
                  file=self.txt)

        if scheme in ['backward', 'central']:
            IE = (energies[-1] - energies[0]) / delta
            print('\nIonization energy = %.5f Ha (%.5f eV)' % (IE, IE * Ha),
                  file=self.txt)
        U = 0.
        for i, d in enumerate(directions[scheme]):
            factor = 1 if i % 2 == 0 else -2
            U += energies[d] * factor / (delta ** 2)

        return U

    def generate_nonminimal_basis(self, size, tail_norm=None, l_pol=None,
                                  r_pol=None):
        """
        Adds more basis functions to the default minimal basis.

        Multiple-zeta basis functions are generated using
        a "split-valence" scheme (see Artacho et al.,
        Phys. stat. sol. (b) 215, 809 (1999)).

        Polarization functions correspond to so-called "quasi-Gaussians"
        (see Larsen et al., Phys. Rev. B 80, 105112 (2009)).

        Parameters
        ----------
        size : str
            Size of the non-minimal basis set to be generated.
            The currently allowed choices are 'dz' ('double-zeta'),
            'szp' ('single-zeta with polarization') and 'dzp'
            ('double-zeta with polarization'). The default is 'dz'.
        tail_norm : None or float, optional
            Parameter determining the radius at which a double-zeta
            function is 'split off' from the parent single-zeta in
            the split-valence scheme. This radius is chosen such that
            the norm of the corresponding tail equals the given target.
            By default, tail norms of 0.5 are chosen for hydrogen
            and 0.15 for all other elements.
        l_pol : None or float, optional
            Angular momentum of the polarizing quasi-Gaussian.
            If None (the default), the first angular momentum is
            used which does not appear in the minimal basis.
        r_pol : None or float, optional
            Characteristic radius for the polarizing quasi-Gaussian.
        """
        assert self.solved, NOT_SOLVED_MESSAGE
        assert size in ['dz', 'szp', 'dzp'], 'Unknown basis size: %s' % size

        print('Generating {0} basis for {1}'.format(size, self.symbol),
              file=self.txt)

        l_val = [ANGULAR_MOMENTUM[nl[1]] for nl in self.valence]
        assert len(set(l_val)) == len(l_val), \
               'Minimal basis should not contain multiple basis functions ' + \
               'with the same angular momentum'

        needs_dz = size in ['dz', 'dzp']
        if needs_dz:
            if tail_norm is None:
                tail_norm = 0.5 if self.symbol == 'H' else 0.15
            assert tail_norm > 0
            print('Tail norm: {0:.3f}'.format(tail_norm), file=self.txt)

        needs_pol = size in ['szp', 'dzp']
        if needs_pol:
            assert r_pol is not None, 'r_pol value is needed for polarization'
            assert r_pol > 0
            print('Characteristic radius: {0:.3f}'.format(r_pol), file=self.txt)
            if l_pol is None:
                for l_pol in sorted(ANGULAR_MOMENTUM.values()):
                    if l_pol not in l_val:
                        break
                else:
                    raise ValueError('Only polarization function functions '
                                     'up to l={0} are available'.format(l_pol))
            else:
                assert 0 <= l_pol <= 3 and l_pol not in l_val
            print('Polarization l: {0}'.format(l_pol), file=self.txt)


        def get_split_valence_unl(nl, tail_norm):
            # Find split radius based on the tail norm
            u = np.copy(self.unlg[nl])
            norm2 = 1.
            index = len(self.rgrid)
            while norm2 > (1. - tail_norm**2):
                index -= 1
                u[index] = 0.
                norm2 = self.grid.integrate(u**2)
            r_split = self.rgrid[index]

            # Fit the polynomial coefficients
            l = ANGULAR_MOMENTUM[nl[1]]
            f0 = self.Rnl(r_split, nl, der=0)
            f1 = self.Rnl(r_split, nl, der=1)
            b = (f1 - l*f0/r_split) / (-2. * r_split**(l+1))
            a = f0/r_split**l + b*r_split**2

            # Build the new radial function
            u = self.unlg[nl] - self.rgrid**(l+1) * (a - b*self.rgrid**2)
            u[index:] = 0.
            norm2 = self.grid.integrate(u**2)
            u /= np.sqrt(norm2)
            self.smoothen_tail(u, index)
            return u, r_split

        def get_quasi_gaussian_unl(nl, l_pol, r_pol, r_cut):
            alpha = 1. / r_pol**2
            alpha_rc2 = (r_cut / r_pol)**2
            a = (1 + alpha_rc2) * np.exp(-alpha_rc2)
            b = alpha * np.exp(-alpha_rc2)

            u = self.rgrid**(l_pol+1) * (np.exp(-alpha * self.rgrid**2) \
                                         - (a - b*self.rgrid**2))
            index = np.argmax(self.rgrid > r_cut)
            u[index:] = 0.
            norm2 = self.grid.integrate(u**2)
            u /= np.sqrt(norm2)
            self.smoothen_tail(u, index)
            return u

        self.basis_size = size
        self.basis_sets = [[nl for nl in self.valence]]

        if needs_dz:
            self.basis_sets.append([])

            for nl in self.valence:
                nldz = nl + '+'

                self.unlg[nldz], r_split = get_split_valence_unl(nl, tail_norm)
                print('Split radius ({0}): {1:.3f}'.format(nldz, r_split),
                      file=self.txt)

                self.unl_fct[nldz] = None
                self.Rnlg[nldz] = self.unlg[nldz] / self.rgrid
                self.Rnl_fct[nldz] = None
                self.rcutnl[nldz] = r_split
                self.basis_sets[1].append(nldz)

            print(file=self.txt)

        if needs_pol:
            nlp = '0' + 'spdf'[l_pol]
            r_cut = max([self.rcutnl[nl] for nl in self.valence])
            self.unlg[nlp] = get_quasi_gaussian_unl(nlp, l_pol, r_pol, r_cut)
            self.unl_fct[nlp] = None
            self.Rnlg[nlp] = self.unlg[nlp] / self.rgrid
            self.Rnl_fct[nlp] = None
            self.rcutnl[nlp] = r_cut
            self.basis_sets[0].append(nlp)
            print(file=self.txt)

        return

    def get_basis_set_index(self, nl):
        if nl.startswith('proj_') or nl.startswith('0'):
            assert '+' not in nl, nl
            return 0
        else:
            assert nl[:2] in self.valence, nl
            return nl.count('+')

    def smoothen_tail(self, u, N):
        """
        Smoothens any derivative kink near a cutoff radius
        by replacing the tail by a polynomial of degree 6.

        Parameters
        ----------
        u : np.ndarray
            Radial function to be smoothened.
        N : int
            Index of the grid point corresponding to the
            cutoff radius.
        """
        rgrid = self.rgrid
        M = N - 4
        tail_length = 0.1 * rgrid[N]
        while (rgrid[N] - rgrid[M]) < tail_length:
            M -= 1

        spl = CubicSplineFunction(rgrid[M-4:M+3], u[M-4:M+3],
                                  bc_type=('not-a-knot', 'not-a-knot'))
        dr = rgrid[N-1] - rgrid[M-1]
        fdr = np.array([spl(rgrid[M-1], der=der) * dr**der
                        for der in range(4)])
        c3 = np.dot([120., 60., 12, 1.], fdr) / (6*dr**3)
        c4 = -np.dot([90., 50., 11, 1.], fdr) / (2*dr**4)
        c5 = np.dot([72., 42., 10., 1.], fdr) / (2*dr**5)
        c6 = -np.dot([60., 36., 9., 1.], fdr) / (6*dr**6)

        for i in range(M, N):
            dr = rgrid[N-1] - rgrid[i]
            u[i] = c3*dr**3 + c4*dr**4 + c5*dr**5 + c6*dr**6

        norm = self.grid.integrate(u**2)
        u /= np.sqrt(norm)


SUBSHELLS = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l']


def nl2tuple(nl):
    """ Transforms e.g. '2p' into (2, 1) """
    return (int(nl[0]), SUBSHELLS.index(nl[1]))


def tuple2nl(n, l):
    """ Transforms e.g. (2, 1) into '2p' """
    return '%i%s' % (n, SUBSHELLS[l])
