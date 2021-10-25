#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.interpolation import CubicSplineFunction
from hotcent.orbitals import (ANGULAR_MOMENTUM, calculate_slako_coeff,
                              ORBITALS, ORBITAL_LABELS)
from hotcent.slako import INTEGRALS as INTEGRALS_2c, SlaterKosterTable


class SeparablePP:
    """ Abstract class representing a separable pseudopotential. """
    def __init__(self):
        self.symbol = None
        self.projectors = {}
        self.energies = {}
        self.overlap_fct = {}  # dict with core-valence overlap functions
        self.overlap_onsite = {}
        self.initialized_elements = []

    def all_zero_onsite_overlaps(self):
        """ Returns whether all on-site overlaps of the (valence)
        states with the PP projectors are zero.
        """
        raise NotImplementedError

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

            for nl3, projector in self.projectors.items():
                l3 = ANGULAR_MOMENTUM[nl3[1]]
                nl3 = 'proj_' + nl3
                e3.Rnl_fct[nl3] = projector  # temporary

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

    def assert_initialized(self, sym):
        """ Checks whether the PP has been initialized
        for the given element.
        """
        msg = 'Pseudopotential for {0} has not been initialized for use ' \
              'together with element {1}. Please apply the build_overlaps() ' \
              'and build_projectors() functions.'
        assert sym in self.initialized_elements, msg.format(self.symbol, sym)

    def get_nonlocal_integral(self, sym1, sym2, sym3, x0, z0, R, nl1, nl2,
                              lm1, lm2):
        """ Returns the nonlocal pseudopotential integral involving
        orbitals on the first 2 atoms and a pseudopotential on the 3rd atom
        (sum_proj3 sum_m <phi_1|chi_proj3> e_proj3 <chi_proj3|phi_2>).

        Assumes that the projector-valence overlap interpolators have
        already been constructed through a previous call to build_overlaps().
        This is not needed if the pseudopotential is located on one of the
        first 2 atoms and all on-site overlaps with the projectors are zero
        (as is the case in the Phillips-Kleinman scheme).

        Arguments:

        sym1, sym2, sym3: chemical symbols of the three atoms
        x0, z0, R: defines the atomic positions (all in the xz-plane)
        nl1, nl2: nl orbital labels ('1s', '2p', ...)
        lm1, lm2: lm orbital labels ('s', 'px', ...)
        """
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

        if self.all_zero_onsite_overlaps() and (coincide13 or coincide23):
            # Quick return
            return 0.

        self.assert_initialized(sym1)
        self.assert_initialized(sym2)

        l1 = ANGULAR_MOMENTUM[nl1[1]]
        l2 = ANGULAR_MOMENTUM[nl2[1]]

        result = 0.
        for nl3, e3 in self.energies.items():
            l3 = ANGULAR_MOMENTUM[nl3[1]]
            prj = 'proj_' + nl3

            for lm3 in ORBITALS[l3]:
                term = e3
                ilm3 = ORBITAL_LABELS.index(lm3)

                # Atom1
                if coincide13:
                    S3 = self.overlap_onsite[nl3] if lm1 == lm3 else 0.
                else:
                    S3 = 0.
                    x, y, z = v13
                    ilm = ORBITAL_LABELS.index(lm1)
                    minl = min(l1, l3)
                    for tau in range(minl+1):
                        skint = self.get_overlap(sym1, sym3, nl1, prj, tau, r13)
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
                    S3 = self.overlap_onsite[nl3] if lm2 == lm3 else 0.
                else:
                    S3 = 0.
                    x, y, z = v23
                    ilm = ORBITAL_LABELS.index(lm2)
                    minl = min(l2, l3)
                    for tau in range(minl+1):
                        skint = self.get_overlap(sym2, sym3, nl2, prj, tau, r23)
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
