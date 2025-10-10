import numpy as np
from ase.units import Bohr
from ase.data import atomic_numbers, atomic_masses, covalent_radii
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.interpolation import CubicSplineFunction
from slako_dipole import (INTEGRALS, print_integral_overview, select_integrals, NUMSK, phi3, tail_smoothening, write_skf)
import matplotlib.pyplot as plt

class Offsite2cTableDipole(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, superposition='density', xc='LDA', nr=50, stride=1, wflimit=1e-7, ntheta=150, smoothen_tails=True):

        self.print_header()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'
        assert superposition in ['density', 'potential']

        self.timer.start('run_offsite2c') # TODO check what this does
        wf_range = self.get_range(wflimit)
        Nsub = N // stride
        Rgrid = rmin + stride * dr * np.arange(Nsub)
        tables = {}

        for p, (e1, e2) in enumerate(self.pairs): # iterate over ordered element pairs
            selected = select_integrals(e1, e2)
            print_integral_overview(e1, e2, selected, file=self.txt)

            for bas1 in range(len(e1.basis_sets)):
                for bas2 in range(len(e2.basis_sets)):
                    tables[(p, bas1, bas2)] = np.zeros((Nsub, NUMSK))

        for i, R in enumerate(Rgrid):
            if R > 2 * wf_range:
                break

            grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)

            if i == Nsub - 1 or Nsub // 10 == 0 or i % (Nsub // 10) == 0:
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            for p, (e1, e2) in enumerate(self.pairs):
                selected = select_integrals(e1, e2) #tripel (label, nl, nl)
                if len(grid) > 0:
                    D = self.calculate(selected, e1, e2, R, grid, area)
                    for key in selected:
                        sk_label, nl1, nl2 = key
                        bas1 = e1.get_basis_set_index(nl1)
                        bas2 = e2.get_basis_set_index(nl2)
                        index = INTEGRALS.index(sk_label)
                        tables[(p, bas1, bas2)][i, index] = D[key] 
        
        self.Rgrid = rmin + dr * np.arange(N)

        if stride > 1:
            self.tables = {}
            for key in tables:
                self.tables[key] = np.zeros((N, NUMSK))
                for i in range(NUMSK):
                    spl = CubicSplineFunction(Rgrid, tables[key][:, i])
                    self.tables[key][:, i] = spl(self.Rgrid)
        else:
            self.tables = tables

        if smoothen_tails:
            # Smooth the curves near the cutoff
            for key in self.tables:
                for i in range(NUMSK):
                    self.tables[key][:, i] = \
                            tail_smoothening(self.Rgrid, self.tables[key][:, i])

        self.timer.stop('run_offsite2c')
        #TODO: do I have to worry about pseudopotentials, probably not

    def calculate(self, selected, e1, e2, R, grid, area):

        self.timer.start('calculate_offsite2c')

        # common for all integrals (not wf-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y - R) / r2  # cosine of theta_2
        s1 = x / r1  # sine of theta_1
        s2 = x / r2  # sine of theta_2

        self.timer.stop('prelude')
        # calculate all selected integrals
        Dl= {}
        sym1, sym2 = e1.get_symbol(), e2.get_symbol()

        for key in selected:
            integral, nl1, nl2 = key

            gphi = phi3(c1, c2, s1, s2, integral)
            aux = gphi * area * x * r1

            Rnl1 = e1.Rnl(r1, nl1)
            Rnl2 = e2.Rnl(r2, nl2)
            D = np.sum(Rnl1 * Rnl2 * aux)
            Dl[key] = D

        

        self.timer.stop('calculate_offsite2c')
        return Dl

    def write(self, eigenvalues=None, hubbardvalues=None, occupations=None,
              spe=None, offdiagonal_H=None, offdiagonal_S=None,
              filename_template='{el1}-{el2}_offsite2c-dipole.skf'):
        """
        Writes all Slater-Koster integral tables to file.
        By default the 'simple' SKF format is chosen, and the 'extended'
        SKF format is only used when necessary (i.e. when a basis set
        includes f-electrons).

        Note that the parameters related to one-center properties
        ('eigenvalues', 'hubbardvalues', ..., 'offdiagonal_S')
        are only used in the homonuclear case.

        Parameters
        ----------
        eigenvalues : None or dict, optional
            {nl: value} dictionary with valence subshell eigenvalues
            (or one-center onsite Hamiltonian integrals, if you will).
        hubbardvalues : None or dict, optional
            {nl: value} dictionary with valence subshell Hubbard values.
        occupations : None or dict, optional
            {nl: value} dictionary with valence subshell occupations.
        spe : None or (list of) float, optional
            Spin-polarization error. Needs to be a list for non-minimal
            basis sets.
        offdiagonal_H : None or dict, optional
            {(nl1, nl2): value} dictionary with the off-diagonal,
            one-center, onsite Hamiltonian integrals. Only needed for
            non-minimal basis sets.
        offdiagonal_S : None or dict, optional
            {(nl1, nl2): value} dictionary with the off-diagonal,
            one-center, onsite overlap integrals. Only needed for
            non-minimal basis sets.
        filename_template : str, optional
            Template for the names of the SKF output file(s).
            Needs to contain '{el1}' and '{el2}' fields, which will be
            filled in with the element symbols (followed, as usual,
            with '+' characters in the case of second-or-higher-zeta
            basis subsets).
        """
        def copy_dict1(dict_src, dict_dest, valence):
            if dict_src is None:
                return
            for nl in valence:
                l = nl[1]
                assert l not in dict_dest
                if dict_src is not None and nl in dict_src:
                    dict_dest[l] = dict_src[nl]

        def copy_dict2(dict_src, dict_dest, valence1, valence2):
            if dict_src is None:
                return
            for nl1 in valence1:
                l1 = nl1[1]
                for nl2 in valence2:
                    l2 = nl2[1]
                    if l1 == l2 and (nl1, nl2) in dict_src:
                        assert l1 not in dict_dest
                        dict_dest[l1] = dict_src[(nl1, nl2)]

        for p, (e1, e2) in enumerate(self.pairs): # for C-C not a true loop, two iterations at most (both directions)
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()

            for bas1, valence1 in enumerate(e1.basis_sets):
                for bas2, valence2 in enumerate(e2.basis_sets):
                    filename = filename_template.format(el1=sym1 + '+'*bas1,
                                                        el2=sym2 + '+'*bas2)
                    print('Writing to %s' % filename, file=self.txt, flush=True)

                    is_extended = any([nl[1]=='f' for nl in valence1+valence2])
                    mass = atomic_masses[atomic_numbers[sym1]]

                    eigval, hubval, occup, SPE = {}, {}, {}, 0.
                    has_diagonal_data = sym1 == sym2 and bas1 == bas2
                    if has_diagonal_data:
                        copy_dict1(eigenvalues, eigval, valence1)
                        copy_dict1(hubbardvalues, hubval, valence1)
                        copy_dict1(occupations, occup, valence1)

                    offdiag_H, offdiag_S = {}, {}
                    has_offdiagonal_data = sym1 == sym2 and bas1 != bas2
                    if has_offdiagonal_data:
                        copy_dict2(offdiagonal_H, offdiag_H, valence1, valence2)
                        copy_dict2(offdiagonal_S, offdiag_S, valence1, valence2)

                    table = self.tables[(p, bas1, bas2)]
                    with open(filename, 'w') as f:
                        write_skf(f, self.Rgrid, table, has_diagonal_data,
                                  is_extended, eigval, hubval, occup, SPE, mass,
                                  has_offdiagonal_data, offdiag_H, offdiag_S)

    def plot(self, filename=None, bas1=0, bas2=0):
        """
        Plot the Slater-Koster tables with matplotlib.

        Parameters
        ----------
        filename : str, optional
            Figure filename (default: '<el1>-<el2>_slako.pdf')
        bas1, bas2 : int, optional
            Basis set indices, when dealing with non-minimal
            basis sets.
        """
        self.timer.start('plotting')
        assert plt is not None, 'Matplotlib could not be imported!'

        fig = plt.figure()
        fig.subplots_adjust(hspace=1e-4, wspace=1e-4)

        el1 = self.ela.get_symbol()
        rmax = 6 * covalent_radii[atomic_numbers[el1]] / Bohr
        ymax = max(1, self.tables[(0, bas1, bas2)].max())
        if self.nel == 2:
            el2 = self.elb.get_symbol()
            rmax = max(rmax, 6 * covalent_radii[atomic_numbers[el2]] / Bohr)
            ymax = max(ymax, self.tables[(1, bas1, bas2)].max())

        for i in range(NUMSK):
            name = INTEGRALS[i]
            ax = plt.subplot(NUMSK//2 +1, 2, i + 1)

            for p, (e1, e2) in enumerate(self.pairs):
                s1, s2 = e1.get_symbol(), e2.get_symbol()
                key = (p, bas1, bas2)

                if p == 0:
                    s = '-'
                    lw = 1
                    alpha = 1.0
                else:
                    s = '--'
                    lw = 4
                    alpha = 0.2

                if np.all(abs(self.tables[key][:, i]) < 1e-10):
                    ax.text(0.03, 0.5 + p * 0.15,
                            'No %s integrals for <%s|%s>' % (name, s1, s2),
                            transform=ax.transAxes, size=10, va='center')

                    if not ax.get_subplotspec().is_last_row():
                        plt.xticks([], [])
                    if not ax.get_subplotspec().is_first_col():
                        plt.yticks([], [])
                else:
                    plt.plot(self.Rgrid, self.tables[key][:, i] , c='r',
                             ls=s, lw=lw, alpha=alpha)
                    plt.axhline(0, c='k', ls='--')
                    ax.text(0.8, 0.1 + p * 0.15, name, size=10,
                            transform=ax.transAxes)

                    if ax.get_subplotspec().is_last_row():
                        plt.xlabel('r (Bohr)')
                    else:
                        plt.xticks([], [])
                    if not ax.get_subplotspec().is_first_col():
                        plt.yticks([],[])

                plt.xlim([0, rmax])
                plt.ylim(-ymax, ymax)

        plt.figtext(0.3, 0.95, 'D', color='r', size=20)
        plt.figtext(0.38, 0.95, ' Slater-Koster tables', size=20)
        sym1, sym2 = self.ela.get_symbol(), self.elb.get_symbol()
        plt.figtext(0.3, 0.92, '(thin solid: <%s|%s>, wide dashed: <%s|%s>)' \
                    % (sym1, sym2, sym2, sym1), size=10)
        plt.show()

        if filename is None:
            filename = '%s-%s_slako-dipole.pdf' % (sym1 + '+'*bas1, sym2 + '+'*bas2)
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
        self.timer.stop('plotting')

    def plot_minimal(self, filename=None, bas1=0, bas2=0):
        """plot that only shows columns that are nonzero"""

        self.timer.start('plotting')
        assert plt is not None, 'Matplotlib could not be imported!'

        fig = plt.figure()
        fig.subplots_adjust(hspace=1e-4, wspace=1e-4)

        el1 = self.ela.get_symbol()
        rmax = 6 * covalent_radii[atomic_numbers[el1]] / Bohr
        ymax = max(1, self.tables[(0, bas1, bas2)].max())
        if self.nel == 2:
            el2 = self.elb.get_symbol()
            rmax = max(rmax, 6 * covalent_radii[atomic_numbers[el2]] / Bohr)
            ymax = max(ymax, self.tables[(1, bas1, bas2)].max())

        table = self.tables[(0,0,0)]
        print(np.shape(table))
        threshold = 1e-10
        nonzero_col = np.where(np.any(np.abs(table) > threshold, axis=0))[0]

        for i, col in enumerate(nonzero_col): 
            name = INTEGRALS[col]
            ax = plt.subplot(len(nonzero_col)//2 +1, 2, i + 1)

            for p, (e1, e2) in enumerate(self.pairs):
                s1, s2 = e1.get_symbol(), e2.get_symbol()
                key = (p, bas1, bas2)

                if p == 0:
                    s = '-'
                    lw = 1
                    alpha = 1.0
                else:
                    s = '--'
                    lw = 4
                    alpha = 0.2

                if np.all(abs(self.tables[key][:, col]) < 1e-10):
                    ax.text(0.03, 0.5 + p * 0.15,
                            'No %s integrals for <%s|%s>' % (name, s1, s2),
                            transform=ax.transAxes, size=10, va='center')

                    if not ax.get_subplotspec().is_last_row():
                        plt.xticks([], [])
                    if not ax.get_subplotspec().is_first_col():
                        plt.yticks([], [])
                else:
                    plt.plot(self.Rgrid, self.tables[key][:, col] , c='r',
                             ls=s, lw=lw, alpha=alpha)
                    plt.axhline(0, c='k', ls='--')
                    ax.text(0.8, 0.1 + p * 0.15, name, size=10,
                            transform=ax.transAxes)

                    if ax.get_subplotspec().is_last_row():
                        plt.xlabel('r (Bohr)')
                    else:
                        plt.xticks([], [])
                    if not ax.get_subplotspec().is_first_col():
                        plt.yticks([],[])

                plt.xlim([0, rmax])
                plt.ylim(-ymax, ymax)

        plt.figtext(0.3, 0.95, 'D', color='r', size=20)
        plt.figtext(0.38, 0.95, ' Slater-Koster tables', size=20)
        sym1, sym2 = self.ela.get_symbol(), self.elb.get_symbol()
        plt.figtext(0.3, 0.92, '(thin solid: <%s|%s>, wide dashed: <%s|%s>)' \
                    % (sym1, sym2, sym2, sym1), size=10)
        # plt.show()

        if filename is None:
            filename = '%s-%s_slako-dipole.pdf' % (sym1 + '+'*bas1, sym2 + '+'*bas2)
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
        self.timer.stop('plotting')
        
    def calculate_analytical(self, R, sk_label):
    def get_twocenter_directly(self):
        direct_table = np.zeros(np.shape(self.tables[(0,0,0)]))
        table = self.tables[(0,0,0)]
        threshold = 1e-10
        nonzero_col = np.where(np.any(np.abs(table) > threshold, axis=0))[0]
        for i, sk_label in enumerate(nonzero_col):
            for R in self.Rgrid:
                pass
                """ function needs: name of first spherical, name of second spherical, name of third spherical
                    for radial parts: subshell of first, subshell of third
                """