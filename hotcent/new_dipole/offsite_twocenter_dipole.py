import numpy as np
import sympy as sym
from ase.units import Bohr
from ase.data import atomic_numbers, atomic_masses, covalent_radii
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.new_dipole.integrals import first_center, second_center, operator, pick_quantum_number, phi, theta1, theta2
from hotcent.interpolation import CubicSplineFunction
from hotcent.new_dipole.slako_dipole import (INTEGRALS_DIPOLE, select_integrals, NUMSK, phi3, tail_smoothening, write_skf)
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

class Offsite2cTableDipole(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, nr=50, stride=1, wflimit=1e-7, ntheta=150, smoothen_tails=True, zeta=None):
        """
        Calculates off-site two-center Hamiltonian and overlap integrals.

        Parameters
        ----------
        rmin : float, optional
            Shortest interatomic separation to consider. All .skf entries for lower 
            r are set to 0. In atomic units
        dr : float, optional
            Grid spacing for the interatomic separations. Also smallest distance
            appearing in .skf. In atomic units
        N : int
            Number of grid points for the interatomic separations.
        ntheta : int, optional
            Number of angular divisions in polar grid (more dense towards
            the bonding region).
        nr : int, optional
            Number of radial divisions in polar grid (more dense towards
            the atomic centers). With p=q=2 (default powers in polar grid)
            ntheta ~ 3*nr is optimal (for a given grid size).
            With ntheta=150, nr=50 you get ~1e-4 accuracy for H integrals
            (beyond that, gain is slow with increasing grid size).
        wflimit : float, optional
            Value below which the radial wave functions are considered
            to be negligible. This determines how far the polar grids
            around the atomic centers extend in space.
        stride : int, optional
            The desired Skater-Koster table typically has quite a large
            number of points (N=500-1000), even though the integrals
            themselves are comparatively smooth. To speed up the
            construction of the SK-table, one can therefore restrict the
            expensive integrations to a subset N' = N // stride and map
            the resulting curves on the N-grid afterwards. The default
            stride (1) means that N' = N (no shortcut).
        smoothen_tails : bool, optional
            Whether to modify the 'tails' of the Slater-Koster integrals
            so that they smoothly decay to zero (default: True).
        zeta : list
            for testing. Override radial parts with functions with width zeta that make integrals
            analytically feasible. zeta gives exponents for artificial radial parts
            
        """
        # self.print_header()

        assert N is not None, 'Need to set number of grid points N!'
        assert dr >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.r_min = rmin
        self.timer.start('run_offsite2c') # TODO check what this does
        wf_range = self.get_range(wflimit)
        self.wf_range = wf_range
        Nsub = N // stride
        Rgrid = rmin + stride * dr * np.arange(Nsub)
        tables = {}

        for p, (e1, e2) in enumerate(self.pairs): # iterate over ordered element pairs
            selected = select_integrals(e1, e2)
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
                label_list = sorted(INTEGRALS_DIPOLE.keys(), key=lambda x: x[0])
                if len(grid) > 0:
                    R_operator = self.calculate(selected, e1, e2, R, grid, area, zeta=zeta)
                    for j,key in enumerate(sorted(selected, key=lambda x: x[0][0])):
                        sk_label, nl1, nl2 = key
                        idx = label_list.index(sk_label)
                        bas1 = e1.get_basis_set_index(nl1)
                        bas2 = e2.get_basis_set_index(nl2)
                        tables[(p, bas1, bas2)][i, idx] = R_operator[key]  #determines the order of sk-columns listed in sk-file?
        
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
            # Smoothen the curves near the cutoff
            for key in self.tables:
                for i in range(NUMSK):
                    self.tables[key][:, i] = \
                            tail_smoothening(self.Rgrid, self.tables[key][:, i])
        
        for p, (e1, e2) in enumerate(self.pairs):
            if e1 == e2:
                selected = select_integrals(e1, e2)
                self.atom_transition_dipole = self.calculate_atom_transitions(selected=selected, e1=e1, e2=e2)

        self.timer.stop('run_offsite2c')
    
    def calculate_atom_transitions(self, selected, e1, e2):
        position_op_dict = {}
        sym1, sym2 = e1.get_symbol(), e2.get_symbol()
        assert sym1 == sym2
        dr = 0.001
        r = np.arange(start=0, stop=self.wf_range, step=dr)
        for label in INTEGRALS_DIPOLE.keys(): 
            match = next((key for key in selected if key[0] == label), None)
            if match != None:
                integral, nl1, nl2 = match
                Rnl1 = e1.Rnl(r, nl1)
                Rnl2 = e2.Rnl(r, nl2)
                Y1 = pick_quantum_number(first_center, (integral[1], integral[2]))[0]
                Yr = pick_quantum_number(operator, (integral[3], integral[4]))[0]
                Y2 = pick_quantum_number(second_center,(integral[5], integral[6]))[0]
                Y2 = Y2.subs(theta2, theta1)
                angle_integral = sym.integrate(sym.integrate(Y1*Yr*Y2*sym.sin(theta1), (phi, 0, 2*sym.pi)), (theta1, 0, sym.pi))
                radial_integral = trapezoid(y=Rnl1 * Rnl2* r**2 * r, x=r, dx=dr)
                r_int = np.sqrt(4*np.pi/3) * radial_integral * angle_integral.evalf()
                position_op_dict[label] = r_int
            else:
                position_op_dict[label] = 0
        return position_op_dict    
        

    def calculate(self, selected, e1, e2, R, grid, area, zeta):
        """
        Calculates the selected position operator elements

        Parameters
        ----------
        selected : list of (sk_label, nl1, nl2) tuples
            Selected integrals to evaluate (for example
            [(5, 0, 0, 1, -1, 2, -1), '3d', '4d'), (...)]).
        e1 : AtomicBase-like object
            <bra| element.
        e2 : AtomicBase-like object
            |ket> element.
        R : float
            Interatomic distance (e1 is at the origin, e2 at z=R).
        grid : np.ndarray
            2D arrray of grid points in the (x, z)-plane.
        area : np.ndarray
            1D array of areas associated with the grid points.
        zeta: list
            what values to use in exponent of artificial radial part

        """

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
        Rl= {}
        sym1, sym2 = e1.get_symbol(), e2.get_symbol()

        if zeta != None:
            for key in selected:
                integral, nl1, nl2 = key

                N1 = (2 * zeta[nl1][0]/np.pi)**(3/4) * 5
                N2 = (2 * zeta[nl2][0]/np.pi)**(3/4) * 5
                gphi = phi3(c1, c2, s1, s2, integral)
                aux = gphi * area * x * r1

                Rnl1 = e1.Rnl(r1, nl1)
                Rnl1 = N1*r1**zeta[nl1][1] * np.exp(-zeta[nl1][0]*r1**2) #overwrite with gaussian for testing
                Rnl2 = e2.Rnl(r2, nl2)
                Rnl2 = N2*r2**zeta[nl2][1] * np.exp(-zeta[nl2][0]*r2**2) #overwrite with gaussian for testing

                R_operator = np.sqrt(4*np.pi/3) * np.sum(Rnl1 * Rnl2 * aux)
                Rl[key] = R_operator
        else:
            for key in selected:
                integral, nl1, nl2 = key
                gphi = phi3(c1, c2, s1, s2, integral)
                aux = gphi * area * x * r1 # other terms from formula contained in area

                Rnl1 = e1.Rnl(r1, nl1)
                Rnl2 = e2.Rnl(r2, nl2)

                R_operator = np.sqrt(4*np.pi/3) * np.sum(Rnl1 * Rnl2 * aux)
                Rl[key] = R_operator
        
        self.timer.stop('calculate_offsite2c')
        return Rl

    def write_dipole(self, filename_template='{el1}-{el2}_dipole.skf'):
        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()
            for bas1, valence1 in enumerate(e1.basis_sets):
                for bas2, valence2 in enumerate(e2.basis_sets):
                    filename = filename_template.format(el1=sym1 + '+'*bas1,
                                                        el2=sym2 + '+'*bas2)
                    mass = atomic_masses[atomic_numbers[sym1]]
                    is_extended = any([nl[1]=='f' for nl in valence1+valence2])
                    table = self.tables[(p, bas1, bas2)]
                    has_atom_transition = sym1 == sym2 and bas1 == bas2
                    atom_transitions = None
                    if has_atom_transition:
                        atom_transitions = self.atom_transition_dipole
                    with open(filename, 'w') as f:
                        write_skf(handle=f, Rgrid=self.Rgrid, table=table, mass=mass, has_atom_transition=has_atom_transition, atom_transitions=atom_transitions)

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
        threshold = 1e-10
        nonzero_col = np.where(np.any(np.abs(table) > threshold, axis=0))[0] #nonzero in skf file

        for i, col in enumerate(nonzero_col): 
            name = sorted(INTEGRALS_DIPOLE.items(), key=lambda x: x[0][0])[col]
            ax = plt.subplot(len(nonzero_col)//2 +1, 2, i + 1)

            for p, (e1, e2) in enumerate(self.pairs):
                selected = select_integrals(e1, e2)
                sk_labels = [i[0] for i in selected]
                name = sorted(sk_labels, key= lambda x: x[0])[col]
                
                
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
                    ax.text(0.5, 0.1 + p * 0.15, name[1:], size=10,
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
        # plt.figtext(0.3, 0.92, '(thin solid: <%s|%s>, wide dashed: <%s|%s>)' \
        #             % (sym1, sym2, sym2, sym1), size=10)
        # plt.show()

        if filename is None:
            filename = '%s-%s_slako-dipole.pdf' % (sym1 + '+'*bas1, sym2 + '+'*bas2)
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
        self.timer.stop('plotting')
        
