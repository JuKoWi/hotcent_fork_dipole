#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np


NUMINT_2CL = 4*4  # number of subshell-resolved integrals in .2cl files


def select_subshells(e1, e2):
    """ Returns a list of subshell pairs (nl1, nl2)
    to be evaluated for the given element pair e1. """
    selected = []
    for ival1, valence1 in enumerate(e1.basis_sets):
        for nl1 in valence1:
            for ival2, valence2 in enumerate(e2.basis_sets):
                for nl2 in valence2:
                    selected.append((nl1, nl2))
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
