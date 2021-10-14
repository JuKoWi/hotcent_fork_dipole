#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.interpolation import CubicSplineFunction
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.radial_grid import RadialGrid
from hotcent.separable_pseudopotential import SeparablePP
try:
    import matplotlib.pyplot as plt
except:
    plt = None


class KleinmanBylanderPP(SeparablePP):
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
        SeparablePP.__init__(self)
        self.subshells = []
        self.rcore = {}
        self.Vl = {}
        self.rgrid = []
        self.rho_core = []
        self.rho_val = []
        self.rho_val_fct = None

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

    def initialize_from_psf(self, filename):
        """ Read in the given pseudopotential file in '.psf' format.

        Attributes that are initialized in this way include the chemical
        symbol (self.symbol), the radial grid (self.rgrid), the l-dependent
        potentials (self.Vl) and the valence charge density (self.rho_val).
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

                self.subshells.append(nl)
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

    def get_subshells(self):
        return [nl for nl in self.subshells
                if ANGULAR_MOMENTUM[nl[1]] <= self.lmax]

    def set_lmax(self, valence):
        """ Sets self.lmax to the highest angular momentum  among
        the given valence states.

        Note: V_loc is taken equal to V_l[lmax+1]. Hence, only projectors
        up to self.lmax will be considered.
        """
        assert all([nl in self.subshells for nl in valence]), \
               'Not all valence states are supported by this pseudopotential'

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
            for nl in self.subshells:
                if nl[1] == 'spdf'[l]:
                    break
            else:
                raise ValueError('Could not find valence state for l=%d' % l)

            Rnl_free = e3.Rnl_free(e3.rgrid, nl)
            dVl = self.nonlocal_potential(e3.rgrid, l)

            self.projectors[nl] = CubicSplineFunction(e3.rgrid, Rnl_free * dVl)
            integrand = Rnl_free**2 * dVl * e3.rgrid**2
            self.energies[nl] = 1. / e3.grid.integrate(integrand, use_dV=False)

            if nl in e3.valence:
                integrand = Rnl_free * dVl * e3.rgrid**2 * e3.Rnl(e3.rgrid, nl)
                self.overlap_onsite[nl] = e3.grid.integrate(integrand,
                                                            use_dV=False)
            else:
                self.overlap_onsite[nl] = 1. / self.energies[nl]


if __name__ == '__main__':
    """ Example:

    python kleinman_bylander.py C.psf 2s 2p
    """
    import sys
    pp = KleinmanBylanderPP(sys.argv[1], sys.argv[2:])
    pp.plot_Vl()
    pp.plot_valence_density()
