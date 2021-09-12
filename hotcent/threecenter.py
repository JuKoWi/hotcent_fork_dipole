#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.orbitals import ORBITAL_LABELS


INTEGRAL_PAIRS = {'%s_%s' % (lm1, lm2): (lm1, lm2)
                  for lm1 in ORBITAL_LABELS for lm2 in ORBITAL_LABELS}

INTEGRALS = INTEGRAL_PAIRS.keys()

XZ_SYMMETRIC_ORBITALS = ['s', 'px', 'pz', 'dxz', 'dx2-y2', 'dz2']

XZ_ANTISYMMETRIC_ORBITALS = ['py', 'dxy', 'dyz']


def select_integrals(e1, e2):
    """ Return list of non-zero integrals (integral, nl1, nl2)
    to be evaluated for element pair e1, e2. """
    selected = []
    val1, val2 = e1.get_valence_orbitals(), e2.get_valence_orbitals()

    for integral in INTEGRALS:
        nl1, nl2 = select_orbitals(val1, val2, integral)
        if nl1 is None or nl2 is None:
            continue
        else:
            lm1, lm2 = integral.split('_')
            if lm1 in XZ_ANTISYMMETRIC_ORBITALS:
                is_nonzero = lm2 in XZ_ANTISYMMETRIC_ORBITALS
            else:
                is_nonzero = lm2 not in XZ_ANTISYMMETRIC_ORBITALS
            if is_nonzero:
                selected.append((integral, nl1, nl2))

    return selected


def select_orbitals(val1, val2, integral):
    """ Select orbitals from given valences to evaluate the given integral.
    e.g. ['2s', '2p'], ['4s', '3d'], 's_dxy' --> ('2s', '3d')
    """
    nl1 = None
    for nl in val1:
        if nl[1] == integral.split('_')[0][0]:
            nl1 = nl

    nl2 = None
    for nl in val2:
        if nl[1] == integral.split('_')[1][0]:
            nl2 = nl

    return nl1, nl2


def sph_nophi(lm, c, s):
    """ Spherical harmonics without phi-dependent factors. """
    if lm == 's':
        return 0.5 / np.sqrt(np.pi)
    elif lm == 'px':
        return 0.5 * np.sqrt(3. / np.pi) * s
    elif lm == 'py':
        return 0.5 * np.sqrt(3. / np.pi) * s
    elif lm == 'pz':
        return 0.5 * np.sqrt(3. / np.pi) * c
    elif lm == 'dxy':
        return 0.25 * np.sqrt(15. / np.pi) * s**2
    elif lm == 'dyz':
        return 0.5 * np.sqrt(15. / np.pi) * s * c
    elif lm == 'dxz':
        return 0.5 * np.sqrt(15. / np.pi) * s * c
    elif lm == 'dx2-y2':
        return 0.25 * np.sqrt(15. / np.pi) * s**2
    elif lm == 'dz2':
        return 0.25 * np.sqrt(5. / np.pi) * (3*c**2 - 1.)


def sph_phi(lm, phi):
    """ Phi-dependent spherical harmonics factors. """
    if lm == 's':
        return 1.
    elif lm == 'px':
        return np.cos(phi)
    elif lm == 'py':
        return np.sin(phi)
    elif lm == 'pz':
        return 1.
    elif lm == 'dxy':
        return np.sin(2*phi)
    elif lm == 'dyz':
        return np.sin(phi)
    elif lm == 'dxz':
        return np.cos(phi)
    elif lm == 'dx2-y2':
        return np.cos(2*phi)
    elif lm == 'dz2':
        return 1.
