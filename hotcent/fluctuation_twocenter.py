#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.orbitals import ANGULAR_MOMENTUM, ORBITAL_LABELS, ORBITALS


NUMINT_2CL = 4*4  # number of subshell-resolved integrals in .2cl files

NUML_2CK = 3  # number of subshells included in .2ck files (3 = up to d)

NUMLM_2CM = 9  # number of orbitals considered for .2cm files (9 = up to d)

INTEGRALS_2CK = [
    'sss', 'sps', 'sds', 'pss', 'pps', 'ppp', 'pds', 'pdp',
    'dss', 'dps', 'dpp', 'dds', 'ddp', 'ddd',
]

NUMSK_2CK = len(INTEGRALS_2CK)


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
    Writes a parameter file in '2cl' format.

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


def write_2ck(handle, Rgrid, table, point_kernels=None):
    """
    Writes a parameter file in '2ck' format.

    Parameters
    ----------
    point_kernels : None or np.ndarray, optional
        Point multipole kernel values for every integral,
        evaluated at R = 1 Bohr. If None, a zero-valued array
        will be used.

    Other parameters
    ----------------
    See slako.write_skf()
    """
    grid_dist = Rgrid[1] - Rgrid[0]
    grid_npts, numint = np.shape(table)
    assert numint == NUMSK_2CK

    nzeros = int(np.round(Rgrid[0] / grid_dist)) - 1
    assert nzeros >= 0

    print("%.12f, %d" % (grid_dist, grid_npts + nzeros), file=handle)
    if point_kernels is None:
        np.savetxt(handle, np.zeros((1, NUMSK_2CK)), fmt='%.12f')
    else:
        np.savetxt(handle, [point_kernels], fmt='%.12f')

    for i in range(nzeros):
        print(' '.join(['0.0'] * numint), file=handle)

    np.savetxt(handle, table, fmt='%1.12e')
    return


def write_2cm(handle, Rgrid, table):
    """
    Writes a parameter file in '2cm' format.

    Parameters
    ----------
    handle : file handle
        Handle of an open file.
    Rgrid : np.ndarray
        Array of interatomic distances.
    table : nd.ndarray
        Three-dimensional table.
    """
    grid_dist = Rgrid[1] - Rgrid[0]
    grid_npts = len(Rgrid)
    nzeros = int(np.round(Rgrid[0] / grid_dist)) - 1
    assert nzeros >= 0
    assert table.ndim == 4

    print("%.12f, %d" % (grid_dist, grid_npts + nzeros), file=handle)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            header = '# {0}_{1}'.format(ORBITAL_LABELS[i], ORBITAL_LABELS[j])
            print(header, file=handle)

            for k in range(nzeros):
                print(' '.join(['0.0'] * table.shape[3]), file=handle)

            np.savetxt(handle, table[i, j, :, :], fmt='%1.12e')
    return
