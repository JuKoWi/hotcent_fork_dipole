#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.orbitals import ANGULAR_MOMENTUM, ORBITALS, ORBITAL_LABELS


NUMINT_2CL = 4*4  # number of subshell-resolved integrals in .2cl files
NUMINT_2CM = 16*16  # number of orbital-resolved integrals in .2cm files


def select_subshells(e1, e2):
    """ Returns a list of subshell pairs (nl1, nl2)
    to be evaluated for the given element pair. """
    selected = []
    for ival1, valence1 in enumerate(e1.basis_sets):
        for nl1 in valence1:
            for ival2, valence2 in enumerate(e2.basis_sets):
                for nl2 in valence2:
                    selected.append((nl1, nl2))
    return selected


def select_orbitals(e1, e2):
    """ Returns a list of orbital pairs ((nl1, lm1), (nl2, lm2))
    to be evaluated for the given element pair. """
    orbitals1 = []
    for valence in e1.basis_sets:
        for nl in valence:
            l = ANGULAR_MOMENTUM[nl[1]]
            for lm in ORBITALS[l]:
                orbitals1.append((nl, lm))

    orbitals2 = []
    for valence in e2.basis_sets:
        for nl in valence:
            l = ANGULAR_MOMENTUM[nl[1]]
            for lm in ORBITALS[l]:
                orbitals2.append((nl, lm))

    selected = []
    for i, orbital1 in enumerate(orbitals1):
        for j, orbital2 in enumerate(orbitals2):
            selected.append((orbital1, orbital2))
    return selected


def write_2cl(handle, Rgrid, table, angmom1, angmom2):
    """
    Writes a parameter file in '.2cl' format.

    Parameters
    ----------
    angmom1: list of int
        Included angular momenta for the first element.
    angmom2: list of int
        Included angular momenta for the second element.

    Other parameters
    ----------------
    See slako.write_skf()
    """
    grid_dist = Rgrid[1] - Rgrid[0]
    grid_npts, numint = np.shape(table)
    assert numint == NUMINT_2CL

    nzeros = int(np.round(Rgrid[0] / grid_dist)) - 1
    assert nzeros >= 0

    print("%.12f, %d" % (grid_dist, grid_npts + nzeros), file=handle)

    for i in range(nzeros):
        print(' '.join(['0.0'] * NUMINT_2CL), file=handle)

    formats = []
    for i in range(NUMINT_2CL):
        l1 = i // 4
        l2 = i % 4
        fmt = '%1.12e' if l1 in angmom1 and l2 in angmom2 else '%.1f'
        formats.append(fmt)

    np.savetxt(handle, table, fmt=formats)
    return


def write_2cm(handle, Rgrid, table, angmom1, angmom2):
    """
    Writes a parameter file in '.2cm' format.

    Parameters
    ----------
    angmom1: list of int
        Included angular momenta for the first element.
    angmom2: list of int
        Included angular momenta for the second element.

    Other parameters
    ----------------
    See slako.write_skf()
    """
    grid_dist = Rgrid[1] - Rgrid[0]
    grid_npts, numint = np.shape(table)
    assert numint == NUMINT_2CM

    nzeros = int(np.round(Rgrid[0] / grid_dist)) - 1
    assert nzeros >= 0

    print("%.12f, %d" % (grid_dist, grid_npts + nzeros), file=handle)

    for i in range(nzeros):
        print(' '.join(['0.0'] * NUMINT_2CM), file=handle)

    formats = []
    for i in range(NUMINT_2CM):
        lm1 = ORBITAL_LABELS[i // 16]
        lm2 = ORBITAL_LABELS[i % 16]
        l1 = ANGULAR_MOMENTUM[lm1[0]]
        l2 = ANGULAR_MOMENTUM[lm2[0]]
        fmt = '%1.12e' if l1 in angmom1 and l2 in angmom2 else '%.1f'
        formats.append(fmt)

    np.savetxt(handle, table, fmt=formats)
    return
