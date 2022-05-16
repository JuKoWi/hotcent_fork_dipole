#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.orbitals import ANGULAR_MOMENTUM, ORBITALS


NUMLM_1CM = 16  # total number of orbitals considered for .1cm files


def select_orbitals(el):
    """ Returns a list of orbitals (nl, lm) to be evaluated
    for the given element. """
    orbitals = []
    for valence in el.basis_sets:
        for nl in valence:
            l = ANGULAR_MOMENTUM[nl[1]]
            for lm in ORBITALS[l]:
                orbitals.append((nl, lm))

    selected = []
    for i, orbital1 in enumerate(orbitals):
        for j, orbital2 in enumerate(orbitals):
            if j <= i:
                selected.append((orbital1, orbital2))
    return selected


def write_1cm(handle, table):
    """
    Writes a parameter file in '.1cm' format.

    Parameters
    ----------
    handle : file handle
        Handle of an open file.
    table : nd.ndarray
        Two-dimensional table of (NUMLM_1CM, NUMLM_1CM) shape.
    """
    assert np.size(table) == NUMLM_1CM**2
    for row in table:
        for item in row:
            handle.write('0 ' if abs(item) < 1e-16 else '%1.12e ' % item)
        handle.write('\n')
