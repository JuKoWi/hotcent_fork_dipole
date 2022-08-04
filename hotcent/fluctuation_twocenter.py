#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.orbitals import ANGULAR_MOMENTUM, ORBITALS


NUMINT_2CL = 4*4  # number of subshell-resolved integrals in .2cl files

NUML_2CK = 3  # number of subshells included in .2ck files (3 = up to d)

NUML_2CM = 3  # number of subshells included in .2cm files (3 = up to d)

INTEGRALS_2CK = [
    'sss', 'sps', 'sds', 'pss', 'pps', 'ppp', 'pds', 'pdp',
    'dss', 'dps', 'dpp', 'dds', 'ddp', 'ddd',
]

NUMSK_2CK = len(INTEGRALS_2CK)

INTEGRALS_2CM = [
    'sss', 'sps', 'sds', 'sfs', 'sgs', 'pss', 'pps', 'ppp', 'pds', 'pdp',
    'pfs', 'pfp', 'pgs', 'pgp', 'dss', 'dps', 'dpp', 'dds', 'ddp', 'ddd',
    'dfs', 'dfp', 'dfd', 'dgs', 'dgp', 'dgd', 'fss', 'fps', 'fpp', 'fds',
    'fdp', 'fdd', 'ffs', 'ffp', 'ffd', 'fff', 'fgs', 'fgp', 'fgd', 'fgf',
    'gss', 'gps', 'gpp', 'gds', 'gdp', 'gdd', 'gfs', 'gfp', 'gfd', 'gff',
    'ggs', 'ggp', 'ggd', 'ggf', 'ggg',
]

NUMSK_2CM = len(INTEGRALS_2CM)


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


def write_2ck(handle, Rgrid, table):
    """
    Writes a parameter file in '2ck' format.

    Parameters
    ----------
    See slako.write_skf()
    """
    grid_dist = Rgrid[1] - Rgrid[0]
    grid_npts, numint = np.shape(table)
    assert numint == NUMSK_2CK

    nzeros = int(np.round(Rgrid[0] / grid_dist)) - 1
    assert nzeros >= 0

    print("%.12f, %d" % (grid_dist, grid_npts + nzeros), file=handle)

    for i in range(nzeros):
        print(' '.join(['0.0'] * numint), file=handle)

    np.savetxt(handle, table, fmt='%1.12e')
    return


def write_2cm(handle, Rgrid, table, l1, l2=None):
    """
    Writes (part of) a parameter file in '2cm' format.

    Parameters
    ----------
    l1 : int or None
        Agular momentum associated with the first element.
    l2 : int or None, optional
        Agular momentum associated with the second element.

    Other parameters
    ----------------
    See slako.write_skf()
    """
    grid_dist = Rgrid[1] - Rgrid[0]
    grid_npts, numint = np.shape(table)
    assert numint == NUMSK_2CM

    if l1 is not None:
        header = '# {0}'.format('spdf'[l1])
        if l2 is not None:
            header += '_{0}'.format('spdf'[l2])
        print(header, file=handle)

    nzeros = int(np.round(Rgrid[0] / grid_dist)) - 1
    assert nzeros >= 0

    print("%.12f, %d" % (grid_dist, grid_npts + nzeros), file=handle)

    for i in range(nzeros):
        print(' '.join(['0.0'] * numint), file=handle)

    np.savetxt(handle, table, fmt='%1.12e')
    return
