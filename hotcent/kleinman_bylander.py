#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
import _hotcent
from hotcent.interpolation import build_interpolator, CubicSplineFunction
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.radial_grid import RadialGrid
from hotcent.separable_pseudopotential import SeparablePP
try:
    import matplotlib.pyplot as plt
except:
    plt = None


class KleinmanBylanderPP(SeparablePP):
    """
    Class representing a Kleinman-Bylander type pseudopotential
    (i.e. a norm-conserving pseudopotential in fully separable form;
    Kleinman and Bylander, Phys. Rev. Lett. (1982),
    doi:10.1103/PhysRevLett.48.1425).

    Parameters
    ----------
    filename : str
        Name of a pseudopotential file in '.psf' format.
    valence : list of str, optional
        Set of nl values defining the set of minimal valence subshells,
        for setting the maximal angular momentum if the lmax argument
        is None.
    with_polarization : bool, optional
        Whether polarization functions will be included in the complete
        basis set (for setting the maximal angular momentum if the lmax
        argument is None).
    local_component : str, optional
        Describes how to construct the local potential. For the
        default 'siesta' value, the same exponential distribution is
        used as in the Siesta code (doi:10.1088/0953-8984/14/11/302).
        Alternatively, specific subshells can be chosen (e.g. 'd').
    lmax : int, optional
        Maximum angular momentum for the KB projectors (needed if the
        valence or local_component keywords are None).
    rcore : float, optional
        Radius used for the building the local potential when the
        local_component argument is 'siesta'. If None, the largest
        core semilocal potential radius of the included angular momenta
        is used.
    verbose : bool, optional
        Verbosity flag (default: False).
    """
    def __init__(self, filename, valence=None, with_polarization=None,
                 local_component='siesta', lmax=None, rcore=None,
                 verbose=False):
        SeparablePP.__init__(self, verbose=verbose)
        self.local_component = local_component

        self.subshells = []
        self.cosines = {}
        self.rcore = {}
        self.Vl = {}
        self.Vl_fct = {}
        self.Vloc = []
        self.Vloc_fct = None
        self.rgrid = []
        self.rho_core = []
        self.rho_core_fct = None
        self.rho_val = []
        self.rho_val_fct = None
        self.rho_loc = []
        self.rho_loc_fct = None

        if filename.endswith('.psf'):
            self.initialize_from_psf(filename)
        else:
            raise ValueError('Only .psf files can be used at the moment.')

        self.grid = RadialGrid(self.rgrid)

        self.normalize_valence_density()
        if self.has_nonzero_rho_core:
            print('PP: non-linear core corrections for %s' % self.symbol)
            self.normalize_core_density()

        for l, Vl in self.Vl.items():
            self.Vl_fct[l] = CubicSplineFunction(self.rgrid, Vl)

        if lmax is None:
            assert valence is not None
            assert with_polarization is not None
            self.set_lmax(valence, with_polarization)
        else:
            assert valence is None
            assert with_polarization is None
            self.lmax = lmax

        self.check_lmax()

        if self.verbose:
            print('PP: lmax for {0} set to {1}'.format(self.symbol, self.lmax))

        self.initialize_local_potential(rcore=rcore)

    def initialize_from_psf(self, filename):
        """ Read in the given pseudopotential file in '.psf' format.

        Attributes that are initialized in this way include the chemical
        symbol (self.symbol), the radial grid (self.rgrid), the l-dependent
        potentials (self.Vl), the core and valence charge densities
        (self.rho_core, self.rho_val) and the self.has_nonzero_rho_core flag.
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
                            if not valid:
                                msg = 'The (reduced) semilocal potential ' \
                                      'for l={0} is not sufficiently close ' \
                                      'to the -2*Zval limit at rcore: {1}'
                                raise ValueError(msg.format(l, array[index:]))

                            array = np.array(array) / self.rgrid / 2.  # in Ha
                            getattr(self, key)[l] = array
                        else:
                            setattr(self, key, np.array(array))
                        new_section = True

        self.has_nonzero_rho_core = not np.allclose(self.rho_core, 0.)

        # Final checks
        for key, val in label_dict.items():
            assert hasattr(self, val)

    def all_zero_onsite_overlaps(self):
        """ Returns whether all on-site overlaps of the (valence)
        states with the PP projectors are zero.
        """
        return False

    def initialize_local_potential(self, rcore=None):
        """ Builds self.Vloc and self.Vloc_fct, respectively the
        local potential on the grid and its interpolator.

        Parameters
        ----------
        rcore : float, optional
            Radius determining the 'a' parameter in the 'siesta'
            scheme.
        """
        if self.local_component == 'siesta':
            if rcore is None:
                rcore = self.get_max_rcore()
            a, b = 1.82 / rcore, 1.
            self.rho_loc = np.exp(-(np.sinh(a*b*self.rgrid) / np.sinh(b))**2)
            norm = self.grid.integrate(self.rho_loc, use_dV=True)
            self.rho_loc /= -norm / self.Zval

            dV = self.grid.get_dvolumes()
            r, r0 = self.rgrid, self.grid.get_r0grid()
            N = len(self.rho_loc)
            n0 = 0.5 * (self.rho_loc[1:] + self.rho_loc[:-1])
            n0 *= -self.Zval / np.sum(n0 * dV)
            self.Vloc = _hotcent.hartree(n0, dV, r, r0, N)
        else:
            self.Vloc = np.copy(self.Vl[self.lmax+1])

        self.Vloc_fct = CubicSplineFunction(self.rgrid, self.Vloc)
        return

    def get_subshells(self):
        return [nl for nl in self.subshells
                if ANGULAR_MOMENTUM[nl[1]] <= self.lmax]

    def set_lmax(self, valence, with_polarization):
        """ Sets self.lmax (the highest angular momentum for which
        projectors will be considered.

        Self.lmax will be set to the highest angular momentum
        in the given set of valence states, plus 1. If, however,
        self.local_component corresponds to a subshell of this
        angular momentum, the self.lmax value is again decremented
        by 1.

        Note: V_loc is taken equal to V_l[lmax+1]. Hence, only projectors
        up to self.lmax will be considered.
        """
        allowed_components = ['siesta'] + [nl[1] for nl in self.subshells]
        assert self.local_component in allowed_components, \
               'Unknown type of local component: ' + self.local_component

        assert all([nl in self.subshells for nl in valence]), \
               'Not all valence states are supported by this pseudopotential'

        lval = [ANGULAR_MOMENTUM[nl[1]] for nl in valence]
        if with_polarization:
            # Check whether there will be polarization functions
            # for which we need to raise self.lmax
            for l in range(max(lval)+2):
                if l not in lval:
                    lval.append(l)
                    break

        self.lmax = max(lval) + 1

        if self.local_component != 'siesta':
            l = ANGULAR_MOMENTUM[self.local_component]
            if l == self.lmax:
                self.lmax -= 1
        return

    def check_lmax(self):
        ps_lmax = max(self.Vl.keys())
        assert self.lmax <= ps_lmax, 'Pseudopotential only contains V_l up ' \
                              'to l=%d and we need l=%d' % (ps_lmax, self.lmax)

        assert self.lmax <= 4, 'Can only handle projectors with angular ' \
                               'momenta up to f'
        return

    def get_core_density(self, r, der=0):
        """ Evaluates the core electron density at the given radius. """
        if self.rho_core_fct is None:
            self.rho_core_fct = CubicSplineFunction(self.rgrid, self.rho_core)

        if self.has_nonzero_rho_core:
            return self.rho_core_fct(r, der=der)
        else:
            return 0.

    def get_local_density(self, r, der=0):
        """ Evaluates the local electron density at the given radius. """
        if self.rho_loc_fct is None:
            self.rho_loc_fct = CubicSplineFunction(self.rgrid, self.rho_loc)
        return self.rho_loc_fct(r, der=der)

    def get_valence_density(self, r, der=0):
        """ Evaluates the valence electron density at the given radius. """
        if self.rho_val_fct is None:
            self.rho_val_fct = CubicSplineFunction(self.rgrid, self.rho_val)
        return self.rho_val_fct(r, der=der)

    def normalize_core_density(self):
        """ Normalizes the core electron density. """
        self.rho_core /= 4. * np.pi * self.rgrid ** 2

    def normalize_valence_density(self):
        """ Normalizes the valence electron density. """
        norm = self.grid.integrate(self.rho_val)
        assert abs(norm - self.Zval) < 1e-3, (norm, self.Zval)
        self.rho_val /= norm / self.Zval
        self.rho_val /= 4. * np.pi * self.rgrid ** 2

    def get_max_rcore(self):
        """ Returns the largest of the different core radii. """
        return max(self.rcore.values())

    def get_cutoff_radius(self, threshold=1e-12):
        """ Returns the radius beyond which all KB projectors are
        considered to be negligible.

        Parameters
        ----------
        threshold : float
            Threshold value for difference between the local
            potential and -Z/r.

        Returns
        -------
        rc : float
            The cutoff radius.
        """
        rcore_max = self.get_max_rcore()
        i = np.argmax(self.rgrid > rcore_max)

        diff = np.inf
        while abs(diff) > threshold:
            rc = self.rgrid[i]
            diff = self.local_potential(rc) + self.Zval/rc
            i = i + 1
        return rc

    def get_self_energy(self):
        """ Returns the 'self energy' as defined in
        Soler et al., J. Phys.: Condens. Matter (2002),
        doi:10.1088/0953-8984/14/11/302.
        """
        e_self = self.grid.integrate(self.Vloc * self.rho_loc, use_dV=True) / 2
        return e_self

    def local_potential(self, r):
        """ Returns the local part of the pseudopotential at r. """
        return self.Vloc_fct(r)

    def semilocal_potential(self, r, l, der=0):
        """ Returns the chosen l-dependent semilocal potential at r. """
        return self.Vl_fct[l](r, der=der) - self.Vloc_fct(r, der=der)

    def plot_potentials(self, filename=None):
        """ Plots the l-dependent and local potentials. """
        assert plt is not None, 'Matplotlib could not be imported!'

        vmin = np.inf
        imin = np.argmax(self.rgrid > 0.2)

        for l, Vl in self.Vl.items():
            if l <= self.lmax:
                plt.plot(self.rgrid, Vl, label=r'$\ell$='+str(l))
                vmin = min(vmin, Vl[imin])

        plt.plot(self.rgrid, self.Vloc, color='k', label='local')
        vmin = min(vmin, self.Vloc[imin])

        Vz = -self.Zval/self.rgrid
        imin = np.argmax(Vz > vmin)
        plt.plot(self.rgrid[imin:], Vz[imin:], ls='--', color='gray',
                 label=r'-Z$_{val}$/r', zorder=0)

        plt.xlim([0., self.get_cutoff_radius()])
        plt.grid()
        plt.xlabel(r'r [a$_0$]')
        plt.ylabel(r'$V$ [Ha]')
        plt.legend()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.clf()

    def plot_density(self, valence=True, logscale=True, filename=None):
        assert plt is not None, 'Matplotlib could not be imported!'

        plot = plt.semilogy if logscale else plt.plot
        rho = self.rho_val if valence else self.rho_core
        plot(self.rgrid, rho)

        plt.xlim([0., self.get_cutoff_radius()])
        plt.ylim([1e-5, None])
        plt.grid()
        plt.xlabel(r'r [a$_0$]')
        label = r'\rho_{val}' if valence else r'\rho_{core}'
        plt.ylabel(r'$%s$ [a.u.]' % label)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.clf()

    def plot_valence_density(self, logscale=True, filename=None):
        """ Plots the valence electron density. """
        self.plot_density(valence=True, logscale=logscale, filename=filename)

    def plot_core_density(self, logscale=True, filename=None):
        """ Plots the core electron density. """
        self.plot_density(valence=False, logscale=logscale, filename=filename)

    def build_projectors(self, e3):
        """ Build the pseudopotential projectors and the
        associated energies.

        Note: currently only one projector per angular momentum
              is considered.
        """
        assert e3.get_symbol() == self.symbol

        self.energies = {}
        self.projectors = {}
        rc = self.get_cutoff_radius()

        for l in range(self.lmax+1):
            for nl in self.subshells:
                if ANGULAR_MOMENTUM[nl[1]] == l:
                    break
            else:
                raise ValueError('Could not find valence state for l=%d' % l)

            Rnl_free = e3.Rnl_free(e3.rgrid, nl)
            dVl = self.semilocal_potential(e3.rgrid, l)

            projector = Rnl_free * dVl
            integrand = projector**2 * e3.rgrid**2
            projector /= np.sqrt(e3.grid.integrate(integrand, use_dV=False))
            self.projectors[nl] = build_interpolator(e3.rgrid, projector, rc)
            integrand = Rnl_free**2 * dVl**2 * e3.rgrid**2
            self.energies[nl] = e3.grid.integrate(integrand, use_dV=False)
            integrand = Rnl_free**2 * dVl * e3.rgrid**2
            self.energies[nl] /= e3.grid.integrate(integrand, use_dV=False)

            integrand = Rnl_free * projector * e3.rgrid**2
            self.cosines[nl] = e3.grid.integrate(integrand, use_dV=False)

            if self.verbose:
                energy = self.energies[nl]
                print('PP: E_KB for %s_%s: %.6f' % (self.symbol, nl, energy))
                cosine = self.cosines[nl]
                print('PP: KB_cos for %s_%s: %.6f' % (self.symbol, nl, cosine))

            # Calculate onsite overlaps
            for valence in e3.basis_sets:
                for nl2 in valence:
                    if ANGULAR_MOMENTUM[nl2[1]] != l:
                        continue

                    integrand = projector * e3.rgrid**2 * e3.Rnl(e3.rgrid, nl2)
                    self.overlap_onsite[(nl, nl2)] = \
                                    e3.grid.integrate(integrand, use_dV=False)


if __name__ == '__main__':
    import argparse
    import os

    description = 'Tool for plotting (semi)local potentials and densities.'
    usage = """
    python kleinman_bylander.py --help

    python kleinman_bylander.py C.psf --valence=2s,2p --local-component=siesta

    python kleinman_bylander.py Au.psf --valence=5d,6s --with-polarization
                                       --local-component=f
    """
    parser = argparse.ArgumentParser(description=description, usage=usage)
    parser.add_argument('filename', type=str)
    parser.add_argument('--lmax', type=int, default=None)
    parser.add_argument('--local-component', type=str, default='siesta')
    parser.add_argument('--rcore', type=float, default=None)
    parser.add_argument('--valence', type=str, default=None)
    parser.add_argument('--with-polarization', action='store_true')
    args = parser.parse_args()

    valence = None if args.valence is None else args.valence.split(',')
    pp = KleinmanBylanderPP(args.filename, valence=valence,
                            with_polarization=args.with_polarization,
                            local_component=args.local_component,
                            lmax=args.lmax, rcore=args.rcore, verbose=True)

    prefix, _ = os.path.splitext(os.path.basename(args.filename))
    pp.plot_potentials(filename='{0}_potentials.png'.format(prefix))
    pp.plot_valence_density(filename='{0}_rhovalence.png'.format(prefix))
    if pp.has_nonzero_rho_core:
        pp.plot_core_density(filename='{0}_rhocore.png'.format(prefix))
