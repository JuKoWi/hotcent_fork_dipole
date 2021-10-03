#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.atomic_dft import RadialGrid
from hotcent.interpolation import CubicSplineFunction
from hotcent.orbitals import (ANGULAR_MOMENTUM, calculate_slako_coeff,
                              ORBITALS, ORBITAL_LABELS)
from hotcent.slako import INTEGRALS as INTEGRALS_2c, SlaterKosterTable
try:
    import matplotlib.pyplot as plt
except:
    plt = None


class KleinmanBylanderPP:
    """ Class representing a Kleinman-Bylander type pseudopotential
    (i.e. a norm-conserving pseudopotential in fully separable form;
    doi:10.1103/PhysRevLett.48.1425).
    """
    def __init__(self, filename, valence):
        """ Initializes a KleinmanBylanderPP object.

        Arguments:

        filename: name of a pseudopotential file in '.psf' format.
        valence: set of nl values defining the set of valence orbitals,
                 for setting the maximal angular momentum.
        """
        self.symbol = None
        self.rcore = {}
        self.Vl = {}
        self.rgrid = []
        self.rho_core = []
        self.rho_val = []
        self.rho_val_fct = None

        self.projectors = {}
        self.energies = {}
        self.overlap_fct = {}  # dict with projector-valence overlap functions

        if filename.endswith('.psf'):
            self.initialize_from_psf(filename)
        else:
            raise ValueError('Only .psf files can be used at the moment.')

        self.grid = RadialGrid(self.rgrid)
        self.normalize_valence_density()

        self.Vl_fct = {}
        for l, Vl in self.Vl.items():
            self.Vl_fct[l] = CubicSplineFunction(self.rgrid, Vl)

        self.set_lmax(valence)
        self.initialized_elements = []

    def initialize_from_psf(self, filename):
        """ Read in the given pseudopotential file in '.psf' format.

        Attributes that are initialized in this way include the
        radial grid (self.rgrid), the l-dependent potentials (self.Vl)
        and the valence charge density (self.rho_val).
        """
        with open(filename, 'r') as f:
            # First line
            items = f.readline().split()
            self.symbol, self.xcname, self.relativistic, self.pptype = items

            # Second line
            f.readline()

            # Third line
            occup = {}
            for item in f.readline().split('/'):
                if len(item.strip()) == 0:
                    continue

                try:
                    nl, occ, _, rc = item.split()
                except ValueError:
                    nlocc, _, rc = item.split()
                    nl = nlocc[:2]
                    occ = nlocc[2:]

                l = ANGULAR_MOMENTUM[nl[1]]
                occup[l] = float(occ)
                self.rcore[l] = float(rc)
                self.Vl[l] = []

            # Fourth line
            items = f.readline().split()
            Ndown, Nup, self.N, b, a, self.Zval = items
            self.N = int(self.N)
            self.Zval = float(self.Zval)
            assert np.isclose(self.Zval, sum(occup.values()))

            # Next lines with various arrays  (l-dependent potentials,
            # core density, valence density)
            label_dict = {'radial grid': 'rgrid',
                          'down pseudopotential': 'Vl',
                          'core charge': 'rho_core',
                          'valence charge': 'rho_val'}

            new_section = True
            need_l = False
            counter = 0
            for line in f:
                if new_section:
                    assert 'follows' in line
                    label = ' '.join(line.lower().split()[:2])
                    assert label in label_dict
                    key = label_dict[label]
                    l_dependent = key == 'Vl'
                    need_l = l_dependent
                    new_section = False
                    array = []
                    counter = 0

                elif need_l:
                    l = int(line)
                    need_l = False

                else:
                    items = list(map(float, line.split()))
                    if l_dependent:
                        array.extend(items)
                    else:
                        array.extend(items)

                    counter += len(items)
                    assert counter <= self.N

                    if counter == self.N:
                        # Reached the end of the section
                        if key == 'Vl':
                            # Check values beyond rcore
                            limit = -2. * self.Zval  # in Ry
                            index = np.argmax(self.rgrid > self.rcore[l])
                            valid = np.allclose(array[index:], limit, atol=5e-2)
                            assert valid, array[index:]

                            array = np.array(array) / self.rgrid / 2.  # in Ha
                            getattr(self, key)[l] = array
                        else:
                            setattr(self, key, np.array(array))
                        new_section = True

        # Final checks
        for key, val in label_dict.items():
            assert hasattr(self, val)

        assert np.allclose(self.rho_core, 0.), \
               'Non-zero core densities are not yet implemented'

    def set_lmax(self, valence):
        """ Sets self.lmax to the highest angular momentum  among
        the given valence states.

        Note: V_loc is taken equal to V_l[lmax+1]. Hence, only projectors
        up to self.lmax will be considered.
        """
        lmax = max([ANGULAR_MOMENTUM[nl[1]] for nl in valence])

        assert lmax <= 4, 'Can only handle projectors with angular momenta ' \
                          'up to f'

        ps_lmax = max(self.Vl.keys())
        assert lmax+1 <= ps_lmax, 'Pseudopotential only contains V_l up to ' \
                                  'l=%d and we need l=%d' % (ps_lmax, lmax+1)

        self.lmax = lmax

    def get_valence_density(self, r, der=0):
        """ Evaluates the valence electron density at the given radius. """
        if self.rho_val_fct is None:
            self.rho_val_fct = CubicSplineFunction(self.rgrid, self.rho_val)
        return self.rho_val_fct(r, der=der)

    def normalize_valence_density(self):
        """ Normalizes the valence electron density. """
        norm = self.grid.integrate(self.rho_val)
        assert abs(norm - self.Zval) < 1e-3, (norm, self.Zval)
        self.rho_val /= norm / self.Zval
        self.rho_val /= 4. * np.pi * self.rgrid ** 2

    def get_max_rcore(self):
        """ Returns the largest of the different core radii. """
        return max(self.rcore.values())

    def local_potential(self, r):
        """ Returns the local part of the pseudopotential at r. """
        return self.Vl_fct[self.lmax+1](r)

    def nonlocal_potential(self, r, l, der=0):
        """ Returns the chosen l-dependent nonlocal potential at r. """
        return self.Vl_fct[l](r, der=der) \
               - self.Vl_fct[self.lmax+1](r, der=der)

    def plot_Vl(self, filename=None):
        """ Plots the l-dependent potentials. """
        assert plt is not None, 'Matplotlib could not be imported!'

        for l, Vl in self.Vl.items():
            plt.plot(self.rgrid, Vl, label=str(l))

        plt.xlim([0., 3.*self.get_max_rcore()])
        plt.grid()
        plt.xlabel(r'r [a$_0$]')
        plt.ylabel(r'$V_\ell$ [Ha]')
        plt.legend()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.clf()

    def plot_valence_density(self, logscale=True, filename=None):
        """ Plots the valence electron density. """
        assert plt is not None, 'Matplotlib could not be imported!'

        plot = plt.semilogy if logscale else plt.plot
        plot(self.rgrid, self.rho_val)

        plt.xlim([0., 3.*self.get_max_rcore()])
        plt.ylim([1e-5, None])
        plt.grid()
        plt.xlabel(r'r [a$_0$]')
        plt.ylabel(r'$\rho_{val}$ [Ha]')
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.clf()

    def build_projectors(self, e3):
        """ Build the pseudopotential projectors and the
        associated energies.

        Note: currently only one projector per angular momentum
              is considered.
        """
        assert e3.get_symbol() == self.symbol

        self.energies = {}
        self.projectors = {}

        for l in range(self.lmax+1):
            for nl in e3.valence:
                if nl[1] == 'spdf'[l]:
                    break
            else:
                raise ValueError('Could not find valence state for l=%d' % l)

            Rnl = e3.Rnl(e3.rgrid, nl)
            dVl = self.nonlocal_potential(e3.rgrid, l)

            self.projectors[l] = CubicSplineFunction(e3.rgrid, Rnl * dVl)
            self.energies[l] = e3.grid.integrate(Rnl * Rnl * dVl * e3.rgrid**2,
                                                 use_dV=False)

    def build_overlaps(self, e1, e3, rmin=1e-2, rmax=None, dr=0.1,
                       wflimit=1e-7):
        """ Builds the projector-valence overlap integral interpolators.

        Assumes that the projectors have already been constructed
        through a previous call to build_projectors().

        Arguments:

        e1: AtomicDFT object (for the valence orbitals)
        e3: AtomicDFT object (for the projectors)
        rmin, rmax, dr: defines the linear grid on which the overlaps
                        will be evaluated (and used for interpolation)
        wflimit: wave function cutoff for setting the range of the 2D
                 integration grids
        """
        sym1, sym3 = e1.get_symbol(), e3.get_symbol()

        assert sym3 == self.symbol
        assert len(self.energies) > 0
        assert len(self.projectors) > 0

        sk = SlaterKosterTable(e1, e3, txt=None, timing=False)
        sk.wf_range = sk.get_range(wflimit)

        if rmax is None:
            rmax = 2. * sk.wf_range

        numr = min(100, int(np.ceil((rmax - rmin) / dr)))
        rval = np.linspace(rmin, rmax, num=numr, endpoint=True)

        for nl1 in e1.valence:
            l1 = ANGULAR_MOMENTUM[nl1[1]]

            for l3 in range(self.lmax+1):
                assert l3 in self.projectors, 'No projector for l=%d' % l3

                nl3 = 'proj_' + 'spdf'[l3]
                e3.Rnl_fct[nl3] = self.projectors[l3]

                for tau in range(min(l1, l3) + 1):
                    key = (sym1, sym3, nl1, nl3, tau)
                    if key in self.overlap_fct:
                        continue

                    print('Calculating overlaps for ', key)

                    if l1 < l3:
                        sk_integral = nl1[1] + nl3[-1] + 'spdf'[tau]
                        sk_selected = [(sk_integral, nl1, nl3)]
                    else:
                        sk_integral = nl3[-1] + nl1[1] + 'spdf'[tau]
                        sk_selected = [(sk_integral, nl3, nl1)]

                    iint = INTEGRALS_2c.index(sk_integral)

                    sval = []
                    for r13 in rval:
                        grid, area = sk.make_grid(r13, nt=150, nr=50)
                        if l1 < l3:
                            s = sk.calculate_mels(sk_selected, e1, e3, r13,
                                              grid, area, only_overlap=True)
                        else:
                            s = sk.calculate_mels(sk_selected, e3, e1, r13,
                                              grid, area, only_overlap=True)
                        if len(grid) == 0:
                            assert abs(s[iint]) < 1e-24
                        sval.append(s[iint])

                    self.overlap_fct[key] = CubicSplineFunction(rval, sval)

                del e3.Rnl_fct[nl3]

        self.initialized_elements.append(sym1)

    def get_overlap(self, sym1, sym2, nl1, nl2, tau, r):
        """ Returns the overlap integral, evaluated by interpolation. """
        key = (sym1, sym2, nl1, nl2, tau)
        s = self.overlap_fct[key](r, der=0)
        return s

    def get_nonlocal_integral(self, sym1, sym2, sym3, x0, z0, R, nl1, nl2,
                              lm1, lm2):
        """ Returns the nonlocal pseudopotential integral involving
        orbitals on the first 2 atoms and a pseudopotential on the 3rd atom
        (sum_proj3 sum_m <phi_1|chi_proj3> e_proj3 <chi_proj3|phi_2>).

        Assumes that the projector-valence overlap interpolators have
        already been constructed through a previous call to build_overlaps().

        Arguments:

        sym1, sym2, sym3: chemical symbols of the three atoms
        x0, z0, R: defines the atomic positions (all in the xz-plane)
        nl1, nl2: nl orbital labels ('1s', '2p', ...)
        lm1, lm2: lm orbital labels ('s', 'px', ...)
        """
        assert sym1 in self.initialized_elements
        assert sym2 in self.initialized_elements
        assert sym3 == self.symbol

        tol = 1e-6
        x3, y3, z3 = x0, 0., z0

        v13 = np.array([x3, y3, z3])
        r13 = np.linalg.norm(v13)
        coincide13 = r13 < tol
        if coincide13:
            v13[:] = 0.
            r13 = 0.
        else:
            v13 /= r13

        v23 = np.array([x3, y3, z3-R])
        r23 = np.linalg.norm(v23)
        coincide23 = r23 < tol
        if coincide23:
            v23[:] = 0.
            r23 = 0.
        else:
            v23 /= r23

        l1 = ANGULAR_MOMENTUM[nl1[1]]
        l2 = ANGULAR_MOMENTUM[nl2[1]]

        result = 0.
        for l3 in range(self.lmax+1):
            nl3 = 'proj_' + 'spdf'[l3]

            for lm3 in ORBITALS[l3]:
                term = 1. / self.energies[l3]
                ilm3 = ORBITAL_LABELS.index(lm3)

                # Atom1
                if coincide13:
                    S3 = self.energies[l3] if lm1 == lm3 else 0.
                else:
                    S3 = 0.
                    x, y, z = v13
                    ilm = ORBITAL_LABELS.index(lm1)
                    minl = min(l1, l3)
                    for tau in range(minl+1):
                        skint = self.get_overlap(sym1, sym3, nl1, nl3, tau, r13)
                        if ilm3 >= ilm:
                           coef = calculate_slako_coeff(x, y, z, ilm+1,
                                                        ilm3+1, tau+1)
                        else:
                           coef = calculate_slako_coeff(x, y, z, ilm3+1,
                                                        ilm+1, tau+1)
                           coef *= (-1)**(l1 + l3)
                        S3 += coef * skint
                term *= S3

                # Atom2
                if coincide23:
                    S3 = self.energies[l3] if lm2 == lm3 else 0.
                else:
                    S3 = 0.
                    x, y, z = v23
                    ilm = ORBITAL_LABELS.index(lm2)
                    minl = min(l2, l3)
                    for tau in range(minl+1):
                        skint = self.get_overlap(sym2, sym3, nl2, nl3, tau, r23)
                        if ilm3 >= ilm:
                            coef = calculate_slako_coeff(x, y, z, ilm+1,
                                                         ilm3+1, tau+1)
                        else:
                            coef = calculate_slako_coeff(x, y, z, ilm3+1,
                                                         ilm+1, tau+1)
                            coef *= (-1)**(l2 + l3)
                        S3 += coef * skint
                term *= S3

                result += term

        return result


if __name__ == '__main__':
    """ Example:

    python kleinman_bylander.py C.psf 2s 2p
    """
    import sys
    pp = KleinmanBylanderPP(sys.argv[1], sys.argv[2:])
    pp.plot_Vl()
    pp.plot_valence_density()
