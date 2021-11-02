#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
"""Utilities related to Slater-Koster integrals."""
import numpy as np


INTEGRALS = ['ffs', 'ffp', 'ffd', 'fff',
             'dfs', 'dfp', 'dfd', 'dds', 'ddp', 'ddd',
             'pfs', 'pfp', 'pds', 'pdp', 'pps', 'ppp',
             'sfs', 'sds', 'sps', 'sss']

NUMSK = len(INTEGRALS)

INTEGRAL_PAIRS = {
    'sss': ('s', 's'),
    'sps': ('s', 'pz'),
    'sds': ('s', 'dz2'),
    'sfs': ('s', 'fz3'),
    'pps': ('pz', 'pz'),
    'ppp': ('px', 'px'),
    'pds': ('pz', 'dz2'),
    'pdp': ('px', 'dxz'),
    'pfs': ('pz', 'fz3'),
    'pfp': ('px', 'fxz2'),
    'dds': ('dz2', 'dz2'),
    'ddp': ('dxz', 'dxz'),
    'ddd': ('dxy', 'dxy'),
    'dfs': ('dz2', 'fz3'),
    'dfp': ('dxz', 'fxz2'),
    'dfd': ('dxy', 'fxyz'),
    'ffs': ('fz3', 'fz3'),
    'ffp': ('fxz2', 'fxz2'),
    'ffd': ('fxyz', 'fxyz'),
    'fff': ('fx(x2-3y2)', 'fx(x2-3y2)'),
}


def search_integrals(lm1, lm2):
    """ Returns the sigma/pi/... integrals to be considered for the
    given pair of orbitals. """
    integrals, ordered = [], []

    for integral, pair in INTEGRAL_PAIRS.items():
        if pair == (lm1, lm2):
            integrals.append(integral)
            ordered.append(True)
        elif pair == (lm2, lm1):
            integrals.append(integral)
            ordered.append(False)

    return integrals, ordered


def select_orbitals(val1, val2, integral):
    """ Select orbitals from given valences to calculate given integral.
    e.g. ['2s', '2p'], ['4s', '3d'], 'sds' --> ('2s', '3d')
    """
    nl1 = None
    for nl in val1:
        if nl[1] == integral[0]:
            nl1 = nl

    nl2 = None
    for nl in val2:
        if nl[1] == integral[1]:
            nl2 = nl

    return nl1, nl2


def select_integrals(e1, e2):
    """ Return list of integrals (integral, nl1, nl2)
    to be done for element pair e1, e2. """
    selected = []
    val1, val2 = e1.get_valence_orbitals(), e2.get_valence_orbitals()

    for integral in INTEGRALS:
        nl1, nl2 = select_orbitals(val1, val2, integral)
        if nl1 is None or nl2 is None:
            continue
        else:
            selected.append((integral, nl1, nl2))

    return selected


def g(c1, c2, s1, s2, integral):
    """ Returns the angle-dependent part of the given two-center integral,
    with c1 and s1 (c2 and s2) the sine and cosine of theta_1 (theta_2)
    for the atom at origin (atom at z=Rz). These expressions are obtained
    by integrating analytically over phi.
    """
    if integral == 'ffs':
        return 7. / 8 * c1 * (5 * c1**2 - 3) * c2 * (5 * c2**2 - 3)
    elif integral == 'ffp':
        return 21. / 32 * (5 * c1**2 - 1) * s1 * (5 * c2**2 - 1) * s2
    elif integral == 'ffd':
        return 105. / 16 * s1**2 * c1 * s2**2 * c2
    elif integral == 'fff':
        return 35. / 32 * s1**3 * s2**3
    elif integral == 'dfs':
        return np.sqrt(35.) / 8 * (3 * c1**2 - 1) * c2 * (5 * c2**2 - 3)
    elif integral == 'dfp':
        return 3. / 8 * np.sqrt(17.5) * s1 * c1 * s2 * (5 * c2**2 - 1)
    elif integral == 'dfd':
        return 15. / 16 * np.sqrt(7.) * s1**2 * s2**2 * c2
    elif integral == 'dds':
        return 5. / 8 * (3 * c1**2 - 1) * (3 * c2**2 - 1)
    elif integral == 'ddp':
        return 15. / 4 * s1 * c1 * s2 * c2
    elif integral == 'ddd':
        return 15. / 16 * s1**2 * s2**2
    elif integral == 'pfs':
        return np.sqrt(21.) / 4 * c1 * c2 * (5 * c2**2 - 3)
    elif integral == 'pfp':
        return 3. / 8 * np.sqrt(3.5) * s1 * s2 * (5 * c2**2 - 1)
    elif integral == 'pds':
        return np.sqrt(15.) / 4 * c1 * (3 * c2**2 - 1)
    elif integral == 'pdp':
        return np.sqrt(45.) / 4 * s1 * s2 * c2
    elif integral == 'pps':
        return 3. / 2 * c1 * c2
    elif integral == 'ppp':
        return 3. / 4 * s1 * s2
    elif integral == 'sfs':
        return np.sqrt(7.) / 4 * c2 * (5 * c2**2 - 3)
    elif integral == 'sds':
        return np.sqrt(5.) / 4 * (3 * c2**2 - 1)
    elif integral == 'sps':
        return np.sqrt(3.) / 2 * c2
    elif integral == 'sss':
        return 0.5 * np.ones_like(c1)


def dg(c1, c2, s1, s2, integral):
    """ Returns an array with the c1, c2, s1 and s2 derivatives
    of g(c1, c2, s1, s2) for the given integral.
    """
    if integral == 'sss':
        return [0, 0, 0, 0]
    elif integral == 'sps':
        return [0, np.sqrt(3)/2, 0, 0]
    elif integral == 'sds':
        return [0, 3*np.sqrt(5)*c2/2, 0, 0]
    elif integral == 'sfs':
        return [0, 3*np.sqrt(7)*(5*c2**2 - 1)/4, 0, 0]
    elif integral == 'pps':
        return [3*c2/2, 3*c1/2, 0, 0]
    elif integral == 'ppp':
        return [0, 0, 3*s2/4, 3*s1/4]
    elif integral == 'pds':
        return [np.sqrt(15)*(3*c2**2 - 1)/4, 3*np.sqrt(15)*c1*c2/2, 0, 0]
    elif integral == 'pdp':
        return [0, 3*np.sqrt(5)*s1*s2/4, 3*np.sqrt(5)*c2*s2/4,
                3*np.sqrt(5)*c2*s1/4]
    elif integral == 'pfs':
        return [np.sqrt(21)*c2*(5*c2**2 - 3)/4,
                3*np.sqrt(21)*c1*(5*c2**2 - 1)/4, 0, 0]
    elif integral == 'pfp':
        return [0, 15*np.sqrt(14)*c2*s1*s2/8,
                3*np.sqrt(14)*s2*(5*c2**2 - 1)/16,
                3*np.sqrt(14)*s1*(5*c2**2 - 1)/16]
    elif integral == 'dds':
        return [15*c1*(3*c2**2 - 1)/4, 15*c2*(3*c1**2 - 1)/4, 0, 0]
    elif integral == 'ddp':
        return [15*c2*s1*s2/4, 15*c1*s1*s2/4, 15*c1*c2*s2/4, 15*c1*c2*s1/4]
    elif integral == 'ddd':
        return [0, 0, 15*s1*s2**2/8, 15*s1**2*s2/8]
    elif integral == 'dfs':
        return [3*np.sqrt(35)*c1*c2*(5*c2**2 - 3)/4,
                3*np.sqrt(35)*(3*c1**2 - 1)*(5*c2**2 - 1)/8, 0, 0]
    elif integral == 'dfp':
        return [3*np.sqrt(70)*s1*s2*(5*c2**2 - 1)/16,
                15*np.sqrt(70)*c1*c2*s1*s2/8,
                3*np.sqrt(70)*c1*s2*(5*c2**2 - 1)/16,
                3*np.sqrt(70)*c1*s1*(5*c2**2 - 1)/16]
    elif integral == 'dfd':
        return [0, 15*np.sqrt(7)*s1**2*s2**2/16, 15*np.sqrt(7)*c2*s1*s2**2/8,
                15*np.sqrt(7)*c2*s1**2*s2/8]
    elif integral == 'ffs':
        return [21*c2*(5*c1**2 - 1)*(5*c2**2 - 3)/8,
                21*c1*(5*c1**2 - 3)*(5*c2**2 - 1)/8, 0, 0]
    elif integral == 'ffp':
        return [105*c1*s1*s2*(5*c2**2 - 1)/16, 105*c2*s1*s2*(5*c1**2 - 1)/16,
                21*s2*(5*c1**2 - 1)*(5*c2**2 - 1)/32,
                21*s1*(5*c1**2 - 1)*(5*c2**2 - 1)/32]
    elif integral == 'ffd':
        return [105*c2*s1**2*s2**2/16, 105*c1*s1**2*s2**2/16,
                105*c1*c2*s1*s2**2/8, 105*c1*c2*s1**2*s2/8]
    elif integral == 'fff':
        return [0, 0, 105*s1**2*s2**3/32, 105*s1**3*s2**2/32]


def tail_smoothening(x, y):
    """ For given grid-function y(x), make smooth tail.

    Aim is to get (e.g. for Slater-Koster tables and repulsions) smoothly
    behaving energies and forces near cutoff region.

    Make is such that y and y' go smoothly exactly to zero at last point.
    Method: take largest neighboring points y_k and y_(k+1) (k<N-3) such
    that line through them passes zero below x_(N-1). Then fit
    third-order polynomial through points y_k, y_k+1 and y_N-1.

    Return:
    smoothed y-function on same grid.
    """
    if np.all(abs(y) < 1e-10):
        return y

    Nzero = 0
    for i in range(len(y) - 1, 1, -1):
        if abs(y[i]) < 1e-60:
            Nzero += 1
        else:
            break

    N = len(y) - Nzero
    y = y[:N]
    xmax = x[:N][-1]

    for i in range(N - 3, 1, -1):
        x0i = x[i] - y[i] / ((y[i + 1] - y[i]) /(x[i + 1] - x[i]))
        if x0i < xmax:
            k = i
            break
    else:
        print('N:', N, 'len(y):', len(y))
        for i in range(len(y)):
            print(x[i], y[i])
        raise RuntimeError('Problem with tail smoothening')

    if k < N / 4:
        for i in range(N):
            print(x[i], y[i])
        msg = 'Problem with tail smoothening: requires too large tail.'
        raise RuntimeError(msg)

    if k == N - 3:
        y[-1] = 0.
        y = np.append(y, np.zeros(Nzero))
        return y
    else:
        # g(x)=c2*(xmax-x)**m + c3*(xmax-x)**(m+1) goes through
        # (xk,yk),(xk+1,yk+1) and (xmax,0)
        # Try different m if g(x) should change sign (this we do not want)
        sgn = np.sign(y[k])
        for m in range(2, 10):
            a1, a2 = (xmax - x[k]) ** m, (xmax - x[k]) ** (m + 1)
            b1, b2 = (xmax-  x[k + 1]) ** m, (xmax - x[k + 1]) ** (m + 1)
            c3 = (y[k] - a1 * y[k + 1] / b1) / (a2 - a1 * b2 / b1)
            c2 = (y[k] - a2 * c3) / a1

            for i in range(k + 2,N):
                y[i] = c2 * (xmax - x[i]) ** 2 + c3 * (xmax - x[i]) ** 3

            y[-1] = 0.  # once more explicitly

            if np.all(y[k:] * sgn >= 0):
                y = np.append(y, np.zeros(Nzero))
                break

            if m == 9:
                msg = 'Problems with smoothening; need for new algorithm?'
                raise RuntimeError(msg)

    return y
