#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
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


def write_3cf(filename, Rgrid, Sgrid, Tgrid, data, fmt='%.8e'):
    """
    Writes a parameter file in '.3cf' format.

    Parameters
    ----------
    filename : str
        File name.
    Rgrid, Sgrid, Tgrid : list or array
        Lists with distances defining the three-atom geometries.
    data : dict
        Dictionary with the tabulated values for each integral type.
    fmt : str, optional
        Formatting string for the integrals.
    """
    numR = len(Rgrid)
    numS = len(Sgrid)
    numT = len(Tgrid)

    with open(filename, 'w') as f:
        # Header
        f.write('%.6f %.6f %d\n' % (Rgrid[0], Rgrid[-1], numR))
        f.write('%.6f %.6f %d\n' % (Sgrid[0], Sgrid[-1], numS))
        f.write('%d\n' % numT)

        keys = list(data.keys())
        f.write(' '.join(keys) + '\n')

        # Body
        for i in range(numR):
            for j in range(1 + numS*numT):
                f.write(' '.join([fmt % data[key][i][j] for key in keys]))
                f.write('\n')
