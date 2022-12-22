#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
"""Module for Gaunt coefficients."""
from math import sqrt, pi
from hotcent.orbitals import ANGULAR_MOMENTUM, ORBITAL_LABELS


def get_gaunt_coefficient(lm1, lm2, lm3):
    """
    Wrapper around calculate_gaunt_coefficient(), returning the Gaunt
    coefficient for the chosen spherical harmonic combination.

    Parameters
    ----------
    lm1, lm2, lm3 : str
        Orbital labels (e.g. 'px').

    Returns
    -------
    c : int or float
        The rotation coefficient.
    """
    ilm = sorted([ORBITAL_LABELS.index(lm) for lm in [lm1, lm2, lm3]])
    l = [ANGULAR_MOMENTUM[ORBITAL_LABELS[lm][0]] for lm in ilm]

    assert l[0] <= 2 and l[1] <= 2 and l[2] <= 4, \
           'Gaunt coefficients are only implemented up to d-d-g combinations.'

    c = calculate_gaunt_coefficient(ilm[0]+1, ilm[1]+1, ilm[2]+1)
    return c


def calculate_gaunt_coefficient(ilm, jlm, klm):
    """
    Returns the Gaunt coefficient for the chosen spherical harmonic
    combination.

    These have been generated with the help of [SymPy](
    https://www.sympy.org).

    Parameters
    ----------
    ilm, jlm, klm : int
        Orbital indices, starting from 1. Assumed to be in ascending
        order, i.e. ilm <= jlm <= klm.

    Returns
    -------
    c : float
        The Gaunt coefficient.
    """
    c = 0

    if (ilm == 1):
        if (jlm == 1):
            if (klm == 1):
                # s s s
                c = 1./(2*sqrt(pi))
            elif (klm == 2):
                # s s px
                c = 0
            elif (klm == 3):
                # s s py
                c = 0
            elif (klm == 4):
                # s s pz
                c = 0
            elif (klm == 5):
                # s s dxy
                c = 0
            elif (klm == 6):
                # s s dyz
                c = 0
            elif (klm == 7):
                # s s dxz
                c = 0
            elif (klm == 8):
                # s s dx2-y2
                c = 0
            elif (klm == 9):
                # s s dz2
                c = 0
            elif (klm == 10):
                # s s fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # s s fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # s s fz(x2-y2)
                c = 0
            elif (klm == 13):
                # s s fxyz
                c = 0
            elif (klm == 14):
                # s s fyz2
                c = 0
            elif (klm == 15):
                # s s fxz2
                c = 0
            elif (klm == 16):
                # s s fz3
                c = 0
            elif (klm == 17):
                # s s g1
                c = 0
            elif (klm == 18):
                # s s g2
                c = 0
            elif (klm == 19):
                # s s g3
                c = 0
            elif (klm == 20):
                # s s g4
                c = 0
            elif (klm == 21):
                # s s g5
                c = 0
            elif (klm == 22):
                # s s g6
                c = 0
            elif (klm == 23):
                # s s g7
                c = 0
            elif (klm == 24):
                # s s g8
                c = 0
            elif (klm == 25):
                # s s g9
                c = 0

        elif (jlm == 2):
            if (klm == 2):
                # s px px
                c = 1./(2*sqrt(pi))
            elif (klm == 3):
                # s px py
                c = 0
            elif (klm == 4):
                # s px pz
                c = 0
            elif (klm == 5):
                # s px dxy
                c = 0
            elif (klm == 6):
                # s px dyz
                c = 0
            elif (klm == 7):
                # s px dxz
                c = 0
            elif (klm == 8):
                # s px dx2-y2
                c = 0
            elif (klm == 9):
                # s px dz2
                c = 0
            elif (klm == 10):
                # s px fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # s px fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # s px fz(x2-y2)
                c = 0
            elif (klm == 13):
                # s px fxyz
                c = 0
            elif (klm == 14):
                # s px fyz2
                c = 0
            elif (klm == 15):
                # s px fxz2
                c = 0
            elif (klm == 16):
                # s px fz3
                c = 0
            elif (klm == 17):
                # s px g1
                c = 0
            elif (klm == 18):
                # s px g2
                c = 0
            elif (klm == 19):
                # s px g3
                c = 0
            elif (klm == 20):
                # s px g4
                c = 0
            elif (klm == 21):
                # s px g5
                c = 0
            elif (klm == 22):
                # s px g6
                c = 0
            elif (klm == 23):
                # s px g7
                c = 0
            elif (klm == 24):
                # s px g8
                c = 0
            elif (klm == 25):
                # s px g9
                c = 0

        elif (jlm == 3):
            if (klm == 3):
                # s py py
                c = 1./(2*sqrt(pi))
            elif (klm == 4):
                # s py pz
                c = 0
            elif (klm == 5):
                # s py dxy
                c = 0
            elif (klm == 6):
                # s py dyz
                c = 0
            elif (klm == 7):
                # s py dxz
                c = 0
            elif (klm == 8):
                # s py dx2-y2
                c = 0
            elif (klm == 9):
                # s py dz2
                c = 0
            elif (klm == 10):
                # s py fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # s py fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # s py fz(x2-y2)
                c = 0
            elif (klm == 13):
                # s py fxyz
                c = 0
            elif (klm == 14):
                # s py fyz2
                c = 0
            elif (klm == 15):
                # s py fxz2
                c = 0
            elif (klm == 16):
                # s py fz3
                c = 0
            elif (klm == 17):
                # s py g1
                c = 0
            elif (klm == 18):
                # s py g2
                c = 0
            elif (klm == 19):
                # s py g3
                c = 0
            elif (klm == 20):
                # s py g4
                c = 0
            elif (klm == 21):
                # s py g5
                c = 0
            elif (klm == 22):
                # s py g6
                c = 0
            elif (klm == 23):
                # s py g7
                c = 0
            elif (klm == 24):
                # s py g8
                c = 0
            elif (klm == 25):
                # s py g9
                c = 0

        elif (jlm == 4):
            if (klm == 4):
                # s pz pz
                c = 1./(2*sqrt(pi))
            elif (klm == 5):
                # s pz dxy
                c = 0
            elif (klm == 6):
                # s pz dyz
                c = 0
            elif (klm == 7):
                # s pz dxz
                c = 0
            elif (klm == 8):
                # s pz dx2-y2
                c = 0
            elif (klm == 9):
                # s pz dz2
                c = 0
            elif (klm == 10):
                # s pz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # s pz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # s pz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # s pz fxyz
                c = 0
            elif (klm == 14):
                # s pz fyz2
                c = 0
            elif (klm == 15):
                # s pz fxz2
                c = 0
            elif (klm == 16):
                # s pz fz3
                c = 0
            elif (klm == 17):
                # s pz g1
                c = 0
            elif (klm == 18):
                # s pz g2
                c = 0
            elif (klm == 19):
                # s pz g3
                c = 0
            elif (klm == 20):
                # s pz g4
                c = 0
            elif (klm == 21):
                # s pz g5
                c = 0
            elif (klm == 22):
                # s pz g6
                c = 0
            elif (klm == 23):
                # s pz g7
                c = 0
            elif (klm == 24):
                # s pz g8
                c = 0
            elif (klm == 25):
                # s pz g9
                c = 0

        elif (jlm == 5):
            if (klm == 5):
                # s dxy dxy
                c = 1./(2*sqrt(pi))
            elif (klm == 6):
                # s dxy dyz
                c = 0
            elif (klm == 7):
                # s dxy dxz
                c = 0
            elif (klm == 8):
                # s dxy dx2-y2
                c = 0
            elif (klm == 9):
                # s dxy dz2
                c = 0
            elif (klm == 10):
                # s dxy fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # s dxy fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # s dxy fz(x2-y2)
                c = 0
            elif (klm == 13):
                # s dxy fxyz
                c = 0
            elif (klm == 14):
                # s dxy fyz2
                c = 0
            elif (klm == 15):
                # s dxy fxz2
                c = 0
            elif (klm == 16):
                # s dxy fz3
                c = 0
            elif (klm == 17):
                # s dxy g1
                c = 0
            elif (klm == 18):
                # s dxy g2
                c = 0
            elif (klm == 19):
                # s dxy g3
                c = 0
            elif (klm == 20):
                # s dxy g4
                c = 0
            elif (klm == 21):
                # s dxy g5
                c = 0
            elif (klm == 22):
                # s dxy g6
                c = 0
            elif (klm == 23):
                # s dxy g7
                c = 0
            elif (klm == 24):
                # s dxy g8
                c = 0
            elif (klm == 25):
                # s dxy g9
                c = 0

        elif (jlm == 6):
            if (klm == 6):
                # s dyz dyz
                c = 1./(2*sqrt(pi))
            elif (klm == 7):
                # s dyz dxz
                c = 0
            elif (klm == 8):
                # s dyz dx2-y2
                c = 0
            elif (klm == 9):
                # s dyz dz2
                c = 0
            elif (klm == 10):
                # s dyz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # s dyz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # s dyz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # s dyz fxyz
                c = 0
            elif (klm == 14):
                # s dyz fyz2
                c = 0
            elif (klm == 15):
                # s dyz fxz2
                c = 0
            elif (klm == 16):
                # s dyz fz3
                c = 0
            elif (klm == 17):
                # s dyz g1
                c = 0
            elif (klm == 18):
                # s dyz g2
                c = 0
            elif (klm == 19):
                # s dyz g3
                c = 0
            elif (klm == 20):
                # s dyz g4
                c = 0
            elif (klm == 21):
                # s dyz g5
                c = 0
            elif (klm == 22):
                # s dyz g6
                c = 0
            elif (klm == 23):
                # s dyz g7
                c = 0
            elif (klm == 24):
                # s dyz g8
                c = 0
            elif (klm == 25):
                # s dyz g9
                c = 0

        elif (jlm == 7):
            if (klm == 7):
                # s dxz dxz
                c = 1./(2*sqrt(pi))
            elif (klm == 8):
                # s dxz dx2-y2
                c = 0
            elif (klm == 9):
                # s dxz dz2
                c = 0
            elif (klm == 10):
                # s dxz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # s dxz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # s dxz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # s dxz fxyz
                c = 0
            elif (klm == 14):
                # s dxz fyz2
                c = 0
            elif (klm == 15):
                # s dxz fxz2
                c = 0
            elif (klm == 16):
                # s dxz fz3
                c = 0
            elif (klm == 17):
                # s dxz g1
                c = 0
            elif (klm == 18):
                # s dxz g2
                c = 0
            elif (klm == 19):
                # s dxz g3
                c = 0
            elif (klm == 20):
                # s dxz g4
                c = 0
            elif (klm == 21):
                # s dxz g5
                c = 0
            elif (klm == 22):
                # s dxz g6
                c = 0
            elif (klm == 23):
                # s dxz g7
                c = 0
            elif (klm == 24):
                # s dxz g8
                c = 0
            elif (klm == 25):
                # s dxz g9
                c = 0

        elif (jlm == 8):
            if (klm == 8):
                # s dx2-y2 dx2-y2
                c = 1./(2*sqrt(pi))
            elif (klm == 9):
                # s dx2-y2 dz2
                c = 0
            elif (klm == 10):
                # s dx2-y2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # s dx2-y2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # s dx2-y2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # s dx2-y2 fxyz
                c = 0
            elif (klm == 14):
                # s dx2-y2 fyz2
                c = 0
            elif (klm == 15):
                # s dx2-y2 fxz2
                c = 0
            elif (klm == 16):
                # s dx2-y2 fz3
                c = 0
            elif (klm == 17):
                # s dx2-y2 g1
                c = 0
            elif (klm == 18):
                # s dx2-y2 g2
                c = 0
            elif (klm == 19):
                # s dx2-y2 g3
                c = 0
            elif (klm == 20):
                # s dx2-y2 g4
                c = 0
            elif (klm == 21):
                # s dx2-y2 g5
                c = 0
            elif (klm == 22):
                # s dx2-y2 g6
                c = 0
            elif (klm == 23):
                # s dx2-y2 g7
                c = 0
            elif (klm == 24):
                # s dx2-y2 g8
                c = 0
            elif (klm == 25):
                # s dx2-y2 g9
                c = 0

        elif (jlm == 9):
            if (klm == 9):
                # s dz2 dz2
                c = 1./(2*sqrt(pi))
            elif (klm == 10):
                # s dz2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # s dz2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # s dz2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # s dz2 fxyz
                c = 0
            elif (klm == 14):
                # s dz2 fyz2
                c = 0
            elif (klm == 15):
                # s dz2 fxz2
                c = 0
            elif (klm == 16):
                # s dz2 fz3
                c = 0
            elif (klm == 17):
                # s dz2 g1
                c = 0
            elif (klm == 18):
                # s dz2 g2
                c = 0
            elif (klm == 19):
                # s dz2 g3
                c = 0
            elif (klm == 20):
                # s dz2 g4
                c = 0
            elif (klm == 21):
                # s dz2 g5
                c = 0
            elif (klm == 22):
                # s dz2 g6
                c = 0
            elif (klm == 23):
                # s dz2 g7
                c = 0
            elif (klm == 24):
                # s dz2 g8
                c = 0
            elif (klm == 25):
                # s dz2 g9
                c = 0

    elif (ilm == 2):
        if (jlm == 2):
            if (klm == 2):
                # px px px
                c = 0
            elif (klm == 3):
                # px px py
                c = 0
            elif (klm == 4):
                # px px pz
                c = 0
            elif (klm == 5):
                # px px dxy
                c = 0
            elif (klm == 6):
                # px px dyz
                c = 0
            elif (klm == 7):
                # px px dxz
                c = 0
            elif (klm == 8):
                # px px dx2-y2
                c = sqrt(15)/(10*sqrt(pi))
            elif (klm == 9):
                # px px dz2
                c = -sqrt(5)/(10*sqrt(pi))
            elif (klm == 10):
                # px px fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # px px fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # px px fz(x2-y2)
                c = 0
            elif (klm == 13):
                # px px fxyz
                c = 0
            elif (klm == 14):
                # px px fyz2
                c = 0
            elif (klm == 15):
                # px px fxz2
                c = 0
            elif (klm == 16):
                # px px fz3
                c = 0
            elif (klm == 17):
                # px px g1
                c = 0
            elif (klm == 18):
                # px px g2
                c = 0
            elif (klm == 19):
                # px px g3
                c = 0
            elif (klm == 20):
                # px px g4
                c = 0
            elif (klm == 21):
                # px px g5
                c = 0
            elif (klm == 22):
                # px px g6
                c = 0
            elif (klm == 23):
                # px px g7
                c = 0
            elif (klm == 24):
                # px px g8
                c = 0
            elif (klm == 25):
                # px px g9
                c = 0

        elif (jlm == 3):
            if (klm == 3):
                # px py py
                c = 0
            elif (klm == 4):
                # px py pz
                c = 0
            elif (klm == 5):
                # px py dxy
                c = sqrt(15)/(10*sqrt(pi))
            elif (klm == 6):
                # px py dyz
                c = 0
            elif (klm == 7):
                # px py dxz
                c = 0
            elif (klm == 8):
                # px py dx2-y2
                c = 0
            elif (klm == 9):
                # px py dz2
                c = 0
            elif (klm == 10):
                # px py fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # px py fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # px py fz(x2-y2)
                c = 0
            elif (klm == 13):
                # px py fxyz
                c = 0
            elif (klm == 14):
                # px py fyz2
                c = 0
            elif (klm == 15):
                # px py fxz2
                c = 0
            elif (klm == 16):
                # px py fz3
                c = 0
            elif (klm == 17):
                # px py g1
                c = 0
            elif (klm == 18):
                # px py g2
                c = 0
            elif (klm == 19):
                # px py g3
                c = 0
            elif (klm == 20):
                # px py g4
                c = 0
            elif (klm == 21):
                # px py g5
                c = 0
            elif (klm == 22):
                # px py g6
                c = 0
            elif (klm == 23):
                # px py g7
                c = 0
            elif (klm == 24):
                # px py g8
                c = 0
            elif (klm == 25):
                # px py g9
                c = 0

        elif (jlm == 4):
            if (klm == 4):
                # px pz pz
                c = 0
            elif (klm == 5):
                # px pz dxy
                c = 0
            elif (klm == 6):
                # px pz dyz
                c = 0
            elif (klm == 7):
                # px pz dxz
                c = sqrt(15)/(10*sqrt(pi))
            elif (klm == 8):
                # px pz dx2-y2
                c = 0
            elif (klm == 9):
                # px pz dz2
                c = 0
            elif (klm == 10):
                # px pz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # px pz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # px pz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # px pz fxyz
                c = 0
            elif (klm == 14):
                # px pz fyz2
                c = 0
            elif (klm == 15):
                # px pz fxz2
                c = 0
            elif (klm == 16):
                # px pz fz3
                c = 0
            elif (klm == 17):
                # px pz g1
                c = 0
            elif (klm == 18):
                # px pz g2
                c = 0
            elif (klm == 19):
                # px pz g3
                c = 0
            elif (klm == 20):
                # px pz g4
                c = 0
            elif (klm == 21):
                # px pz g5
                c = 0
            elif (klm == 22):
                # px pz g6
                c = 0
            elif (klm == 23):
                # px pz g7
                c = 0
            elif (klm == 24):
                # px pz g8
                c = 0
            elif (klm == 25):
                # px pz g9
                c = 0

        elif (jlm == 5):
            if (klm == 5):
                # px dxy dxy
                c = 0
            elif (klm == 6):
                # px dxy dyz
                c = 0
            elif (klm == 7):
                # px dxy dxz
                c = 0
            elif (klm == 8):
                # px dxy dx2-y2
                c = 0
            elif (klm == 9):
                # px dxy dz2
                c = 0
            elif (klm == 10):
                # px dxy fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # px dxy fy(3x2-y2)
                c = 3*sqrt(14)/(28*sqrt(pi))
            elif (klm == 12):
                # px dxy fz(x2-y2)
                c = 0
            elif (klm == 13):
                # px dxy fxyz
                c = 0
            elif (klm == 14):
                # px dxy fyz2
                c = -sqrt(210)/(140*sqrt(pi))
            elif (klm == 15):
                # px dxy fxz2
                c = 0
            elif (klm == 16):
                # px dxy fz3
                c = 0
            elif (klm == 17):
                # px dxy g1
                c = 0
            elif (klm == 18):
                # px dxy g2
                c = 0
            elif (klm == 19):
                # px dxy g3
                c = 0
            elif (klm == 20):
                # px dxy g4
                c = 0
            elif (klm == 21):
                # px dxy g5
                c = 0
            elif (klm == 22):
                # px dxy g6
                c = 0
            elif (klm == 23):
                # px dxy g7
                c = 0
            elif (klm == 24):
                # px dxy g8
                c = 0
            elif (klm == 25):
                # px dxy g9
                c = 0

        elif (jlm == 6):
            if (klm == 6):
                # px dyz dyz
                c = 0
            elif (klm == 7):
                # px dyz dxz
                c = 0
            elif (klm == 8):
                # px dyz dx2-y2
                c = 0
            elif (klm == 9):
                # px dyz dz2
                c = 0
            elif (klm == 10):
                # px dyz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # px dyz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # px dyz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # px dyz fxyz
                c = sqrt(21)/(14*sqrt(pi))
            elif (klm == 14):
                # px dyz fyz2
                c = 0
            elif (klm == 15):
                # px dyz fxz2
                c = 0
            elif (klm == 16):
                # px dyz fz3
                c = 0
            elif (klm == 17):
                # px dyz g1
                c = 0
            elif (klm == 18):
                # px dyz g2
                c = 0
            elif (klm == 19):
                # px dyz g3
                c = 0
            elif (klm == 20):
                # px dyz g4
                c = 0
            elif (klm == 21):
                # px dyz g5
                c = 0
            elif (klm == 22):
                # px dyz g6
                c = 0
            elif (klm == 23):
                # px dyz g7
                c = 0
            elif (klm == 24):
                # px dyz g8
                c = 0
            elif (klm == 25):
                # px dyz g9
                c = 0

        elif (jlm == 7):
            if (klm == 7):
                # px dxz dxz
                c = 0
            elif (klm == 8):
                # px dxz dx2-y2
                c = 0
            elif (klm == 9):
                # px dxz dz2
                c = 0
            elif (klm == 10):
                # px dxz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # px dxz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # px dxz fz(x2-y2)
                c = sqrt(21)/(14*sqrt(pi))
            elif (klm == 13):
                # px dxz fxyz
                c = 0
            elif (klm == 14):
                # px dxz fyz2
                c = 0
            elif (klm == 15):
                # px dxz fxz2
                c = 0
            elif (klm == 16):
                # px dxz fz3
                c = -3*sqrt(35)/(70*sqrt(pi))
            elif (klm == 17):
                # px dxz g1
                c = 0
            elif (klm == 18):
                # px dxz g2
                c = 0
            elif (klm == 19):
                # px dxz g3
                c = 0
            elif (klm == 20):
                # px dxz g4
                c = 0
            elif (klm == 21):
                # px dxz g5
                c = 0
            elif (klm == 22):
                # px dxz g6
                c = 0
            elif (klm == 23):
                # px dxz g7
                c = 0
            elif (klm == 24):
                # px dxz g8
                c = 0
            elif (klm == 25):
                # px dxz g9
                c = 0

        elif (jlm == 8):
            if (klm == 8):
                # px dx2-y2 dx2-y2
                c = 0
            elif (klm == 9):
                # px dx2-y2 dz2
                c = 0
            elif (klm == 10):
                # px dx2-y2 fx(x2-3y2)
                c = 3*sqrt(14)/(28*sqrt(pi))
            elif (klm == 11):
                # px dx2-y2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # px dx2-y2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # px dx2-y2 fxyz
                c = 0
            elif (klm == 14):
                # px dx2-y2 fyz2
                c = 0
            elif (klm == 15):
                # px dx2-y2 fxz2
                c = -sqrt(210)/(140*sqrt(pi))
            elif (klm == 16):
                # px dx2-y2 fz3
                c = 0
            elif (klm == 17):
                # px dx2-y2 g1
                c = 0
            elif (klm == 18):
                # px dx2-y2 g2
                c = 0
            elif (klm == 19):
                # px dx2-y2 g3
                c = 0
            elif (klm == 20):
                # px dx2-y2 g4
                c = 0
            elif (klm == 21):
                # px dx2-y2 g5
                c = 0
            elif (klm == 22):
                # px dx2-y2 g6
                c = 0
            elif (klm == 23):
                # px dx2-y2 g7
                c = 0
            elif (klm == 24):
                # px dx2-y2 g8
                c = 0
            elif (klm == 25):
                # px dx2-y2 g9
                c = 0

        elif (jlm == 9):
            if (klm == 9):
                # px dz2 dz2
                c = 0
            elif (klm == 10):
                # px dz2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # px dz2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # px dz2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # px dz2 fxyz
                c = 0
            elif (klm == 14):
                # px dz2 fyz2
                c = 0
            elif (klm == 15):
                # px dz2 fxz2
                c = 3*sqrt(70)/(70*sqrt(pi))
            elif (klm == 16):
                # px dz2 fz3
                c = 0
            elif (klm == 17):
                # px dz2 g1
                c = 0
            elif (klm == 18):
                # px dz2 g2
                c = 0
            elif (klm == 19):
                # px dz2 g3
                c = 0
            elif (klm == 20):
                # px dz2 g4
                c = 0
            elif (klm == 21):
                # px dz2 g5
                c = 0
            elif (klm == 22):
                # px dz2 g6
                c = 0
            elif (klm == 23):
                # px dz2 g7
                c = 0
            elif (klm == 24):
                # px dz2 g8
                c = 0
            elif (klm == 25):
                # px dz2 g9
                c = 0

    elif (ilm == 3):
        if (jlm == 3):
            if (klm == 3):
                # py py py
                c = 0
            elif (klm == 4):
                # py py pz
                c = 0
            elif (klm == 5):
                # py py dxy
                c = 0
            elif (klm == 6):
                # py py dyz
                c = 0
            elif (klm == 7):
                # py py dxz
                c = 0
            elif (klm == 8):
                # py py dx2-y2
                c = -sqrt(15)/(10*sqrt(pi))
            elif (klm == 9):
                # py py dz2
                c = -sqrt(5)/(10*sqrt(pi))
            elif (klm == 10):
                # py py fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # py py fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # py py fz(x2-y2)
                c = 0
            elif (klm == 13):
                # py py fxyz
                c = 0
            elif (klm == 14):
                # py py fyz2
                c = 0
            elif (klm == 15):
                # py py fxz2
                c = 0
            elif (klm == 16):
                # py py fz3
                c = 0
            elif (klm == 17):
                # py py g1
                c = 0
            elif (klm == 18):
                # py py g2
                c = 0
            elif (klm == 19):
                # py py g3
                c = 0
            elif (klm == 20):
                # py py g4
                c = 0
            elif (klm == 21):
                # py py g5
                c = 0
            elif (klm == 22):
                # py py g6
                c = 0
            elif (klm == 23):
                # py py g7
                c = 0
            elif (klm == 24):
                # py py g8
                c = 0
            elif (klm == 25):
                # py py g9
                c = 0

        elif (jlm == 4):
            if (klm == 4):
                # py pz pz
                c = 0
            elif (klm == 5):
                # py pz dxy
                c = 0
            elif (klm == 6):
                # py pz dyz
                c = sqrt(15)/(10*sqrt(pi))
            elif (klm == 7):
                # py pz dxz
                c = 0
            elif (klm == 8):
                # py pz dx2-y2
                c = 0
            elif (klm == 9):
                # py pz dz2
                c = 0
            elif (klm == 10):
                # py pz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # py pz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # py pz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # py pz fxyz
                c = 0
            elif (klm == 14):
                # py pz fyz2
                c = 0
            elif (klm == 15):
                # py pz fxz2
                c = 0
            elif (klm == 16):
                # py pz fz3
                c = 0
            elif (klm == 17):
                # py pz g1
                c = 0
            elif (klm == 18):
                # py pz g2
                c = 0
            elif (klm == 19):
                # py pz g3
                c = 0
            elif (klm == 20):
                # py pz g4
                c = 0
            elif (klm == 21):
                # py pz g5
                c = 0
            elif (klm == 22):
                # py pz g6
                c = 0
            elif (klm == 23):
                # py pz g7
                c = 0
            elif (klm == 24):
                # py pz g8
                c = 0
            elif (klm == 25):
                # py pz g9
                c = 0

        elif (jlm == 5):
            if (klm == 5):
                # py dxy dxy
                c = 0
            elif (klm == 6):
                # py dxy dyz
                c = 0
            elif (klm == 7):
                # py dxy dxz
                c = 0
            elif (klm == 8):
                # py dxy dx2-y2
                c = 0
            elif (klm == 9):
                # py dxy dz2
                c = 0
            elif (klm == 10):
                # py dxy fx(x2-3y2)
                c = -3*sqrt(14)/(28*sqrt(pi))
            elif (klm == 11):
                # py dxy fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # py dxy fz(x2-y2)
                c = 0
            elif (klm == 13):
                # py dxy fxyz
                c = 0
            elif (klm == 14):
                # py dxy fyz2
                c = 0
            elif (klm == 15):
                # py dxy fxz2
                c = -sqrt(210)/(140*sqrt(pi))
            elif (klm == 16):
                # py dxy fz3
                c = 0
            elif (klm == 17):
                # py dxy g1
                c = 0
            elif (klm == 18):
                # py dxy g2
                c = 0
            elif (klm == 19):
                # py dxy g3
                c = 0
            elif (klm == 20):
                # py dxy g4
                c = 0
            elif (klm == 21):
                # py dxy g5
                c = 0
            elif (klm == 22):
                # py dxy g6
                c = 0
            elif (klm == 23):
                # py dxy g7
                c = 0
            elif (klm == 24):
                # py dxy g8
                c = 0
            elif (klm == 25):
                # py dxy g9
                c = 0

        elif (jlm == 6):
            if (klm == 6):
                # py dyz dyz
                c = 0
            elif (klm == 7):
                # py dyz dxz
                c = 0
            elif (klm == 8):
                # py dyz dx2-y2
                c = 0
            elif (klm == 9):
                # py dyz dz2
                c = 0
            elif (klm == 10):
                # py dyz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # py dyz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # py dyz fz(x2-y2)
                c = -sqrt(21)/(14*sqrt(pi))
            elif (klm == 13):
                # py dyz fxyz
                c = 0
            elif (klm == 14):
                # py dyz fyz2
                c = 0
            elif (klm == 15):
                # py dyz fxz2
                c = 0
            elif (klm == 16):
                # py dyz fz3
                c = -3*sqrt(35)/(70*sqrt(pi))
            elif (klm == 17):
                # py dyz g1
                c = 0
            elif (klm == 18):
                # py dyz g2
                c = 0
            elif (klm == 19):
                # py dyz g3
                c = 0
            elif (klm == 20):
                # py dyz g4
                c = 0
            elif (klm == 21):
                # py dyz g5
                c = 0
            elif (klm == 22):
                # py dyz g6
                c = 0
            elif (klm == 23):
                # py dyz g7
                c = 0
            elif (klm == 24):
                # py dyz g8
                c = 0
            elif (klm == 25):
                # py dyz g9
                c = 0

        elif (jlm == 7):
            if (klm == 7):
                # py dxz dxz
                c = 0
            elif (klm == 8):
                # py dxz dx2-y2
                c = 0
            elif (klm == 9):
                # py dxz dz2
                c = 0
            elif (klm == 10):
                # py dxz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # py dxz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # py dxz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # py dxz fxyz
                c = sqrt(21)/(14*sqrt(pi))
            elif (klm == 14):
                # py dxz fyz2
                c = 0
            elif (klm == 15):
                # py dxz fxz2
                c = 0
            elif (klm == 16):
                # py dxz fz3
                c = 0
            elif (klm == 17):
                # py dxz g1
                c = 0
            elif (klm == 18):
                # py dxz g2
                c = 0
            elif (klm == 19):
                # py dxz g3
                c = 0
            elif (klm == 20):
                # py dxz g4
                c = 0
            elif (klm == 21):
                # py dxz g5
                c = 0
            elif (klm == 22):
                # py dxz g6
                c = 0
            elif (klm == 23):
                # py dxz g7
                c = 0
            elif (klm == 24):
                # py dxz g8
                c = 0
            elif (klm == 25):
                # py dxz g9
                c = 0

        elif (jlm == 8):
            if (klm == 8):
                # py dx2-y2 dx2-y2
                c = 0
            elif (klm == 9):
                # py dx2-y2 dz2
                c = 0
            elif (klm == 10):
                # py dx2-y2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # py dx2-y2 fy(3x2-y2)
                c = 3*sqrt(14)/(28*sqrt(pi))
            elif (klm == 12):
                # py dx2-y2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # py dx2-y2 fxyz
                c = 0
            elif (klm == 14):
                # py dx2-y2 fyz2
                c = sqrt(210)/(140*sqrt(pi))
            elif (klm == 15):
                # py dx2-y2 fxz2
                c = 0
            elif (klm == 16):
                # py dx2-y2 fz3
                c = 0
            elif (klm == 17):
                # py dx2-y2 g1
                c = 0
            elif (klm == 18):
                # py dx2-y2 g2
                c = 0
            elif (klm == 19):
                # py dx2-y2 g3
                c = 0
            elif (klm == 20):
                # py dx2-y2 g4
                c = 0
            elif (klm == 21):
                # py dx2-y2 g5
                c = 0
            elif (klm == 22):
                # py dx2-y2 g6
                c = 0
            elif (klm == 23):
                # py dx2-y2 g7
                c = 0
            elif (klm == 24):
                # py dx2-y2 g8
                c = 0
            elif (klm == 25):
                # py dx2-y2 g9
                c = 0

        elif (jlm == 9):
            if (klm == 9):
                # py dz2 dz2
                c = 0
            elif (klm == 10):
                # py dz2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # py dz2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # py dz2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # py dz2 fxyz
                c = 0
            elif (klm == 14):
                # py dz2 fyz2
                c = 3*sqrt(70)/(70*sqrt(pi))
            elif (klm == 15):
                # py dz2 fxz2
                c = 0
            elif (klm == 16):
                # py dz2 fz3
                c = 0
            elif (klm == 17):
                # py dz2 g1
                c = 0
            elif (klm == 18):
                # py dz2 g2
                c = 0
            elif (klm == 19):
                # py dz2 g3
                c = 0
            elif (klm == 20):
                # py dz2 g4
                c = 0
            elif (klm == 21):
                # py dz2 g5
                c = 0
            elif (klm == 22):
                # py dz2 g6
                c = 0
            elif (klm == 23):
                # py dz2 g7
                c = 0
            elif (klm == 24):
                # py dz2 g8
                c = 0
            elif (klm == 25):
                # py dz2 g9
                c = 0

    elif (ilm == 4):
        if (jlm == 4):
            if (klm == 4):
                # pz pz pz
                c = 0
            elif (klm == 5):
                # pz pz dxy
                c = 0
            elif (klm == 6):
                # pz pz dyz
                c = 0
            elif (klm == 7):
                # pz pz dxz
                c = 0
            elif (klm == 8):
                # pz pz dx2-y2
                c = 0
            elif (klm == 9):
                # pz pz dz2
                c = sqrt(5)/(5*sqrt(pi))
            elif (klm == 10):
                # pz pz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # pz pz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # pz pz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # pz pz fxyz
                c = 0
            elif (klm == 14):
                # pz pz fyz2
                c = 0
            elif (klm == 15):
                # pz pz fxz2
                c = 0
            elif (klm == 16):
                # pz pz fz3
                c = 0
            elif (klm == 17):
                # pz pz g1
                c = 0
            elif (klm == 18):
                # pz pz g2
                c = 0
            elif (klm == 19):
                # pz pz g3
                c = 0
            elif (klm == 20):
                # pz pz g4
                c = 0
            elif (klm == 21):
                # pz pz g5
                c = 0
            elif (klm == 22):
                # pz pz g6
                c = 0
            elif (klm == 23):
                # pz pz g7
                c = 0
            elif (klm == 24):
                # pz pz g8
                c = 0
            elif (klm == 25):
                # pz pz g9
                c = 0

        elif (jlm == 5):
            if (klm == 5):
                # pz dxy dxy
                c = 0
            elif (klm == 6):
                # pz dxy dyz
                c = 0
            elif (klm == 7):
                # pz dxy dxz
                c = 0
            elif (klm == 8):
                # pz dxy dx2-y2
                c = 0
            elif (klm == 9):
                # pz dxy dz2
                c = 0
            elif (klm == 10):
                # pz dxy fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # pz dxy fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # pz dxy fz(x2-y2)
                c = 0
            elif (klm == 13):
                # pz dxy fxyz
                c = sqrt(21)/(14*sqrt(pi))
            elif (klm == 14):
                # pz dxy fyz2
                c = 0
            elif (klm == 15):
                # pz dxy fxz2
                c = 0
            elif (klm == 16):
                # pz dxy fz3
                c = 0
            elif (klm == 17):
                # pz dxy g1
                c = 0
            elif (klm == 18):
                # pz dxy g2
                c = 0
            elif (klm == 19):
                # pz dxy g3
                c = 0
            elif (klm == 20):
                # pz dxy g4
                c = 0
            elif (klm == 21):
                # pz dxy g5
                c = 0
            elif (klm == 22):
                # pz dxy g6
                c = 0
            elif (klm == 23):
                # pz dxy g7
                c = 0
            elif (klm == 24):
                # pz dxy g8
                c = 0
            elif (klm == 25):
                # pz dxy g9
                c = 0

        elif (jlm == 6):
            if (klm == 6):
                # pz dyz dyz
                c = 0
            elif (klm == 7):
                # pz dyz dxz
                c = 0
            elif (klm == 8):
                # pz dyz dx2-y2
                c = 0
            elif (klm == 9):
                # pz dyz dz2
                c = 0
            elif (klm == 10):
                # pz dyz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # pz dyz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # pz dyz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # pz dyz fxyz
                c = 0
            elif (klm == 14):
                # pz dyz fyz2
                c = sqrt(210)/(35*sqrt(pi))
            elif (klm == 15):
                # pz dyz fxz2
                c = 0
            elif (klm == 16):
                # pz dyz fz3
                c = 0
            elif (klm == 17):
                # pz dyz g1
                c = 0
            elif (klm == 18):
                # pz dyz g2
                c = 0
            elif (klm == 19):
                # pz dyz g3
                c = 0
            elif (klm == 20):
                # pz dyz g4
                c = 0
            elif (klm == 21):
                # pz dyz g5
                c = 0
            elif (klm == 22):
                # pz dyz g6
                c = 0
            elif (klm == 23):
                # pz dyz g7
                c = 0
            elif (klm == 24):
                # pz dyz g8
                c = 0
            elif (klm == 25):
                # pz dyz g9
                c = 0

        elif (jlm == 7):
            if (klm == 7):
                # pz dxz dxz
                c = 0
            elif (klm == 8):
                # pz dxz dx2-y2
                c = 0
            elif (klm == 9):
                # pz dxz dz2
                c = 0
            elif (klm == 10):
                # pz dxz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # pz dxz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # pz dxz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # pz dxz fxyz
                c = 0
            elif (klm == 14):
                # pz dxz fyz2
                c = 0
            elif (klm == 15):
                # pz dxz fxz2
                c = sqrt(210)/(35*sqrt(pi))
            elif (klm == 16):
                # pz dxz fz3
                c = 0
            elif (klm == 17):
                # pz dxz g1
                c = 0
            elif (klm == 18):
                # pz dxz g2
                c = 0
            elif (klm == 19):
                # pz dxz g3
                c = 0
            elif (klm == 20):
                # pz dxz g4
                c = 0
            elif (klm == 21):
                # pz dxz g5
                c = 0
            elif (klm == 22):
                # pz dxz g6
                c = 0
            elif (klm == 23):
                # pz dxz g7
                c = 0
            elif (klm == 24):
                # pz dxz g8
                c = 0
            elif (klm == 25):
                # pz dxz g9
                c = 0

        elif (jlm == 8):
            if (klm == 8):
                # pz dx2-y2 dx2-y2
                c = 0
            elif (klm == 9):
                # pz dx2-y2 dz2
                c = 0
            elif (klm == 10):
                # pz dx2-y2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # pz dx2-y2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # pz dx2-y2 fz(x2-y2)
                c = sqrt(21)/(14*sqrt(pi))
            elif (klm == 13):
                # pz dx2-y2 fxyz
                c = 0
            elif (klm == 14):
                # pz dx2-y2 fyz2
                c = 0
            elif (klm == 15):
                # pz dx2-y2 fxz2
                c = 0
            elif (klm == 16):
                # pz dx2-y2 fz3
                c = 0
            elif (klm == 17):
                # pz dx2-y2 g1
                c = 0
            elif (klm == 18):
                # pz dx2-y2 g2
                c = 0
            elif (klm == 19):
                # pz dx2-y2 g3
                c = 0
            elif (klm == 20):
                # pz dx2-y2 g4
                c = 0
            elif (klm == 21):
                # pz dx2-y2 g5
                c = 0
            elif (klm == 22):
                # pz dx2-y2 g6
                c = 0
            elif (klm == 23):
                # pz dx2-y2 g7
                c = 0
            elif (klm == 24):
                # pz dx2-y2 g8
                c = 0
            elif (klm == 25):
                # pz dx2-y2 g9
                c = 0

        elif (jlm == 9):
            if (klm == 9):
                # pz dz2 dz2
                c = 0
            elif (klm == 10):
                # pz dz2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # pz dz2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # pz dz2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # pz dz2 fxyz
                c = 0
            elif (klm == 14):
                # pz dz2 fyz2
                c = 0
            elif (klm == 15):
                # pz dz2 fxz2
                c = 0
            elif (klm == 16):
                # pz dz2 fz3
                c = 3*sqrt(105)/(70*sqrt(pi))
            elif (klm == 17):
                # pz dz2 g1
                c = 0
            elif (klm == 18):
                # pz dz2 g2
                c = 0
            elif (klm == 19):
                # pz dz2 g3
                c = 0
            elif (klm == 20):
                # pz dz2 g4
                c = 0
            elif (klm == 21):
                # pz dz2 g5
                c = 0
            elif (klm == 22):
                # pz dz2 g6
                c = 0
            elif (klm == 23):
                # pz dz2 g7
                c = 0
            elif (klm == 24):
                # pz dz2 g8
                c = 0
            elif (klm == 25):
                # pz dz2 g9
                c = 0

    elif (ilm == 5):
        if (jlm == 5):
            if (klm == 5):
                # dxy dxy dxy
                c = 0
            elif (klm == 6):
                # dxy dxy dyz
                c = 0
            elif (klm == 7):
                # dxy dxy dxz
                c = 0
            elif (klm == 8):
                # dxy dxy dx2-y2
                c = 0
            elif (klm == 9):
                # dxy dxy dz2
                c = -sqrt(5)/(7*sqrt(pi))
            elif (klm == 10):
                # dxy dxy fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dxy dxy fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dxy dxy fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dxy dxy fxyz
                c = 0
            elif (klm == 14):
                # dxy dxy fyz2
                c = 0
            elif (klm == 15):
                # dxy dxy fxz2
                c = 0
            elif (klm == 16):
                # dxy dxy fz3
                c = 0
            elif (klm == 17):
                # dxy dxy g1
                c = 0
            elif (klm == 18):
                # dxy dxy g2
                c = 0
            elif (klm == 19):
                # dxy dxy g3
                c = 0
            elif (klm == 20):
                # dxy dxy g4
                c = 0
            elif (klm == 21):
                # dxy dxy g5
                c = 1./(14*sqrt(pi))
            elif (klm == 22):
                # dxy dxy g6
                c = 0
            elif (klm == 23):
                # dxy dxy g7
                c = 0
            elif (klm == 24):
                # dxy dxy g8
                c = 0
            elif (klm == 25):
                # dxy dxy g9
                c = -sqrt(35)/(14*sqrt(pi))

        elif (jlm == 6):
            if (klm == 6):
                # dxy dyz dyz
                c = 0
            elif (klm == 7):
                # dxy dyz dxz
                c = sqrt(15)/(14*sqrt(pi))
            elif (klm == 8):
                # dxy dyz dx2-y2
                c = 0
            elif (klm == 9):
                # dxy dyz dz2
                c = 0
            elif (klm == 10):
                # dxy dyz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dxy dyz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dxy dyz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dxy dyz fxyz
                c = 0
            elif (klm == 14):
                # dxy dyz fyz2
                c = 0
            elif (klm == 15):
                # dxy dyz fxz2
                c = 0
            elif (klm == 16):
                # dxy dyz fz3
                c = 0
            elif (klm == 17):
                # dxy dyz g1
                c = 0
            elif (klm == 18):
                # dxy dyz g2
                c = 0
            elif (klm == 19):
                # dxy dyz g3
                c = 0
            elif (klm == 20):
                # dxy dyz g4
                c = 0
            elif (klm == 21):
                # dxy dyz g5
                c = 0
            elif (klm == 22):
                # dxy dyz g6
                c = -sqrt(10)/(28*sqrt(pi))
            elif (klm == 23):
                # dxy dyz g7
                c = 0
            elif (klm == 24):
                # dxy dyz g8
                c = -sqrt(70)/(28*sqrt(pi))
            elif (klm == 25):
                # dxy dyz g9
                c = 0

        elif (jlm == 7):
            if (klm == 7):
                # dxy dxz dxz
                c = 0
            elif (klm == 8):
                # dxy dxz dx2-y2
                c = 0
            elif (klm == 9):
                # dxy dxz dz2
                c = 0
            elif (klm == 10):
                # dxy dxz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dxy dxz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dxy dxz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dxy dxz fxyz
                c = 0
            elif (klm == 14):
                # dxy dxz fyz2
                c = 0
            elif (klm == 15):
                # dxy dxz fxz2
                c = 0
            elif (klm == 16):
                # dxy dxz fz3
                c = 0
            elif (klm == 17):
                # dxy dxz g1
                c = 0
            elif (klm == 18):
                # dxy dxz g2
                c = sqrt(70)/(28*sqrt(pi))
            elif (klm == 19):
                # dxy dxz g3
                c = 0
            elif (klm == 20):
                # dxy dxz g4
                c = -sqrt(10)/(28*sqrt(pi))
            elif (klm == 21):
                # dxy dxz g5
                c = 0
            elif (klm == 22):
                # dxy dxz g6
                c = 0
            elif (klm == 23):
                # dxy dxz g7
                c = 0
            elif (klm == 24):
                # dxy dxz g8
                c = 0
            elif (klm == 25):
                # dxy dxz g9
                c = 0

        elif (jlm == 8):
            if (klm == 8):
                # dxy dx2-y2 dx2-y2
                c = 0
            elif (klm == 9):
                # dxy dx2-y2 dz2
                c = 0
            elif (klm == 10):
                # dxy dx2-y2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dxy dx2-y2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dxy dx2-y2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dxy dx2-y2 fxyz
                c = 0
            elif (klm == 14):
                # dxy dx2-y2 fyz2
                c = 0
            elif (klm == 15):
                # dxy dx2-y2 fxz2
                c = 0
            elif (klm == 16):
                # dxy dx2-y2 fz3
                c = 0
            elif (klm == 17):
                # dxy dx2-y2 g1
                c = sqrt(35)/(14*sqrt(pi))
            elif (klm == 18):
                # dxy dx2-y2 g2
                c = 0
            elif (klm == 19):
                # dxy dx2-y2 g3
                c = 0
            elif (klm == 20):
                # dxy dx2-y2 g4
                c = 0
            elif (klm == 21):
                # dxy dx2-y2 g5
                c = 0
            elif (klm == 22):
                # dxy dx2-y2 g6
                c = 0
            elif (klm == 23):
                # dxy dx2-y2 g7
                c = 0
            elif (klm == 24):
                # dxy dx2-y2 g8
                c = 0
            elif (klm == 25):
                # dxy dx2-y2 g9
                c = 0

        elif (jlm == 9):
            if (klm == 9):
                # dxy dz2 dz2
                c = 0
            elif (klm == 10):
                # dxy dz2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dxy dz2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dxy dz2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dxy dz2 fxyz
                c = 0
            elif (klm == 14):
                # dxy dz2 fyz2
                c = 0
            elif (klm == 15):
                # dxy dz2 fxz2
                c = 0
            elif (klm == 16):
                # dxy dz2 fz3
                c = 0
            elif (klm == 17):
                # dxy dz2 g1
                c = 0
            elif (klm == 18):
                # dxy dz2 g2
                c = 0
            elif (klm == 19):
                # dxy dz2 g3
                c = sqrt(15)/(14*sqrt(pi))
            elif (klm == 20):
                # dxy dz2 g4
                c = 0
            elif (klm == 21):
                # dxy dz2 g5
                c = 0
            elif (klm == 22):
                # dxy dz2 g6
                c = 0
            elif (klm == 23):
                # dxy dz2 g7
                c = 0
            elif (klm == 24):
                # dxy dz2 g8
                c = 0
            elif (klm == 25):
                # dxy dz2 g9
                c = 0

    elif (ilm == 6):
        if (jlm == 6):
            if (klm == 6):
                # dyz dyz dyz
                c = 0
            elif (klm == 7):
                # dyz dyz dxz
                c = 0
            elif (klm == 8):
                # dyz dyz dx2-y2
                c = -sqrt(15)/(14*sqrt(pi))
            elif (klm == 9):
                # dyz dyz dz2
                c = sqrt(5)/(14*sqrt(pi))
            elif (klm == 10):
                # dyz dyz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dyz dyz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dyz dyz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dyz dyz fxyz
                c = 0
            elif (klm == 14):
                # dyz dyz fyz2
                c = 0
            elif (klm == 15):
                # dyz dyz fxz2
                c = 0
            elif (klm == 16):
                # dyz dyz fz3
                c = 0
            elif (klm == 17):
                # dyz dyz g1
                c = 0
            elif (klm == 18):
                # dyz dyz g2
                c = 0
            elif (klm == 19):
                # dyz dyz g3
                c = 0
            elif (klm == 20):
                # dyz dyz g4
                c = 0
            elif (klm == 21):
                # dyz dyz g5
                c = -2./(7*sqrt(pi))
            elif (klm == 22):
                # dyz dyz g6
                c = 0
            elif (klm == 23):
                # dyz dyz g7
                c = -sqrt(5)/(7*sqrt(pi))
            elif (klm == 24):
                # dyz dyz g8
                c = 0
            elif (klm == 25):
                # dyz dyz g9
                c = 0

        elif (jlm == 7):
            if (klm == 7):
                # dyz dxz dxz
                c = 0
            elif (klm == 8):
                # dyz dxz dx2-y2
                c = 0
            elif (klm == 9):
                # dyz dxz dz2
                c = 0
            elif (klm == 10):
                # dyz dxz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dyz dxz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dyz dxz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dyz dxz fxyz
                c = 0
            elif (klm == 14):
                # dyz dxz fyz2
                c = 0
            elif (klm == 15):
                # dyz dxz fxz2
                c = 0
            elif (klm == 16):
                # dyz dxz fz3
                c = 0
            elif (klm == 17):
                # dyz dxz g1
                c = 0
            elif (klm == 18):
                # dyz dxz g2
                c = 0
            elif (klm == 19):
                # dyz dxz g3
                c = sqrt(5)/(7*sqrt(pi))
            elif (klm == 20):
                # dyz dxz g4
                c = 0
            elif (klm == 21):
                # dyz dxz g5
                c = 0
            elif (klm == 22):
                # dyz dxz g6
                c = 0
            elif (klm == 23):
                # dyz dxz g7
                c = 0
            elif (klm == 24):
                # dyz dxz g8
                c = 0
            elif (klm == 25):
                # dyz dxz g9
                c = 0

        elif (jlm == 8):
            if (klm == 8):
                # dyz dx2-y2 dx2-y2
                c = 0
            elif (klm == 9):
                # dyz dx2-y2 dz2
                c = 0
            elif (klm == 10):
                # dyz dx2-y2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dyz dx2-y2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dyz dx2-y2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dyz dx2-y2 fxyz
                c = 0
            elif (klm == 14):
                # dyz dx2-y2 fyz2
                c = 0
            elif (klm == 15):
                # dyz dx2-y2 fxz2
                c = 0
            elif (klm == 16):
                # dyz dx2-y2 fz3
                c = 0
            elif (klm == 17):
                # dyz dx2-y2 g1
                c = 0
            elif (klm == 18):
                # dyz dx2-y2 g2
                c = sqrt(70)/(28*sqrt(pi))
            elif (klm == 19):
                # dyz dx2-y2 g3
                c = 0
            elif (klm == 20):
                # dyz dx2-y2 g4
                c = sqrt(10)/(28*sqrt(pi))
            elif (klm == 21):
                # dyz dx2-y2 g5
                c = 0
            elif (klm == 22):
                # dyz dx2-y2 g6
                c = 0
            elif (klm == 23):
                # dyz dx2-y2 g7
                c = 0
            elif (klm == 24):
                # dyz dx2-y2 g8
                c = 0
            elif (klm == 25):
                # dyz dx2-y2 g9
                c = 0

        elif (jlm == 9):
            if (klm == 9):
                # dyz dz2 dz2
                c = 0
            elif (klm == 10):
                # dyz dz2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dyz dz2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dyz dz2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dyz dz2 fxyz
                c = 0
            elif (klm == 14):
                # dyz dz2 fyz2
                c = 0
            elif (klm == 15):
                # dyz dz2 fxz2
                c = 0
            elif (klm == 16):
                # dyz dz2 fz3
                c = 0
            elif (klm == 17):
                # dyz dz2 g1
                c = 0
            elif (klm == 18):
                # dyz dz2 g2
                c = 0
            elif (klm == 19):
                # dyz dz2 g3
                c = 0
            elif (klm == 20):
                # dyz dz2 g4
                c = sqrt(30)/(14*sqrt(pi))
            elif (klm == 21):
                # dyz dz2 g5
                c = 0
            elif (klm == 22):
                # dyz dz2 g6
                c = 0
            elif (klm == 23):
                # dyz dz2 g7
                c = 0
            elif (klm == 24):
                # dyz dz2 g8
                c = 0
            elif (klm == 25):
                # dyz dz2 g9
                c = 0

    elif (ilm == 7):
        if (jlm == 7):
            if (klm == 7):
                # dxz dxz dxz
                c = 0
            elif (klm == 8):
                # dxz dxz dx2-y2
                c = sqrt(15)/(14*sqrt(pi))
            elif (klm == 9):
                # dxz dxz dz2
                c = sqrt(5)/(14*sqrt(pi))
            elif (klm == 10):
                # dxz dxz fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dxz dxz fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dxz dxz fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dxz dxz fxyz
                c = 0
            elif (klm == 14):
                # dxz dxz fyz2
                c = 0
            elif (klm == 15):
                # dxz dxz fxz2
                c = 0
            elif (klm == 16):
                # dxz dxz fz3
                c = 0
            elif (klm == 17):
                # dxz dxz g1
                c = 0
            elif (klm == 18):
                # dxz dxz g2
                c = 0
            elif (klm == 19):
                # dxz dxz g3
                c = 0
            elif (klm == 20):
                # dxz dxz g4
                c = 0
            elif (klm == 21):
                # dxz dxz g5
                c = -2./(7*sqrt(pi))
            elif (klm == 22):
                # dxz dxz g6
                c = 0
            elif (klm == 23):
                # dxz dxz g7
                c = sqrt(5)/(7*sqrt(pi))
            elif (klm == 24):
                # dxz dxz g8
                c = 0
            elif (klm == 25):
                # dxz dxz g9
                c = 0

        elif (jlm == 8):
            if (klm == 8):
                # dxz dx2-y2 dx2-y2
                c = 0
            elif (klm == 9):
                # dxz dx2-y2 dz2
                c = 0
            elif (klm == 10):
                # dxz dx2-y2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dxz dx2-y2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dxz dx2-y2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dxz dx2-y2 fxyz
                c = 0
            elif (klm == 14):
                # dxz dx2-y2 fyz2
                c = 0
            elif (klm == 15):
                # dxz dx2-y2 fxz2
                c = 0
            elif (klm == 16):
                # dxz dx2-y2 fz3
                c = 0
            elif (klm == 17):
                # dxz dx2-y2 g1
                c = 0
            elif (klm == 18):
                # dxz dx2-y2 g2
                c = 0
            elif (klm == 19):
                # dxz dx2-y2 g3
                c = 0
            elif (klm == 20):
                # dxz dx2-y2 g4
                c = 0
            elif (klm == 21):
                # dxz dx2-y2 g5
                c = 0
            elif (klm == 22):
                # dxz dx2-y2 g6
                c = -sqrt(10)/(28*sqrt(pi))
            elif (klm == 23):
                # dxz dx2-y2 g7
                c = 0
            elif (klm == 24):
                # dxz dx2-y2 g8
                c = sqrt(70)/(28*sqrt(pi))
            elif (klm == 25):
                # dxz dx2-y2 g9
                c = 0

        elif (jlm == 9):
            if (klm == 9):
                # dxz dz2 dz2
                c = 0
            elif (klm == 10):
                # dxz dz2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dxz dz2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dxz dz2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dxz dz2 fxyz
                c = 0
            elif (klm == 14):
                # dxz dz2 fyz2
                c = 0
            elif (klm == 15):
                # dxz dz2 fxz2
                c = 0
            elif (klm == 16):
                # dxz dz2 fz3
                c = 0
            elif (klm == 17):
                # dxz dz2 g1
                c = 0
            elif (klm == 18):
                # dxz dz2 g2
                c = 0
            elif (klm == 19):
                # dxz dz2 g3
                c = 0
            elif (klm == 20):
                # dxz dz2 g4
                c = 0
            elif (klm == 21):
                # dxz dz2 g5
                c = 0
            elif (klm == 22):
                # dxz dz2 g6
                c = sqrt(30)/(14*sqrt(pi))
            elif (klm == 23):
                # dxz dz2 g7
                c = 0
            elif (klm == 24):
                # dxz dz2 g8
                c = 0
            elif (klm == 25):
                # dxz dz2 g9
                c = 0

    elif (ilm == 8):
        if (jlm == 8):
            if (klm == 8):
                # dx2-y2 dx2-y2 dx2-y2
                c = 0
            elif (klm == 9):
                # dx2-y2 dx2-y2 dz2
                c = -sqrt(5)/(7*sqrt(pi))
            elif (klm == 10):
                # dx2-y2 dx2-y2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dx2-y2 dx2-y2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dx2-y2 dx2-y2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dx2-y2 dx2-y2 fxyz
                c = 0
            elif (klm == 14):
                # dx2-y2 dx2-y2 fyz2
                c = 0
            elif (klm == 15):
                # dx2-y2 dx2-y2 fxz2
                c = 0
            elif (klm == 16):
                # dx2-y2 dx2-y2 fz3
                c = 0
            elif (klm == 17):
                # dx2-y2 dx2-y2 g1
                c = 0
            elif (klm == 18):
                # dx2-y2 dx2-y2 g2
                c = 0
            elif (klm == 19):
                # dx2-y2 dx2-y2 g3
                c = 0
            elif (klm == 20):
                # dx2-y2 dx2-y2 g4
                c = 0
            elif (klm == 21):
                # dx2-y2 dx2-y2 g5
                c = 1./(14*sqrt(pi))
            elif (klm == 22):
                # dx2-y2 dx2-y2 g6
                c = 0
            elif (klm == 23):
                # dx2-y2 dx2-y2 g7
                c = 0
            elif (klm == 24):
                # dx2-y2 dx2-y2 g8
                c = 0
            elif (klm == 25):
                # dx2-y2 dx2-y2 g9
                c = sqrt(35)/(14*sqrt(pi))

        elif (jlm == 9):
            if (klm == 9):
                # dx2-y2 dz2 dz2
                c = 0
            elif (klm == 10):
                # dx2-y2 dz2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dx2-y2 dz2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dx2-y2 dz2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dx2-y2 dz2 fxyz
                c = 0
            elif (klm == 14):
                # dx2-y2 dz2 fyz2
                c = 0
            elif (klm == 15):
                # dx2-y2 dz2 fxz2
                c = 0
            elif (klm == 16):
                # dx2-y2 dz2 fz3
                c = 0
            elif (klm == 17):
                # dx2-y2 dz2 g1
                c = 0
            elif (klm == 18):
                # dx2-y2 dz2 g2
                c = 0
            elif (klm == 19):
                # dx2-y2 dz2 g3
                c = 0
            elif (klm == 20):
                # dx2-y2 dz2 g4
                c = 0
            elif (klm == 21):
                # dx2-y2 dz2 g5
                c = 0
            elif (klm == 22):
                # dx2-y2 dz2 g6
                c = 0
            elif (klm == 23):
                # dx2-y2 dz2 g7
                c = sqrt(15)/(14*sqrt(pi))
            elif (klm == 24):
                # dx2-y2 dz2 g8
                c = 0
            elif (klm == 25):
                # dx2-y2 dz2 g9
                c = 0

    elif (ilm == 9):
        if (jlm == 9):
            if (klm == 9):
                # dz2 dz2 dz2
                c = sqrt(5)/(7*sqrt(pi))
            elif (klm == 10):
                # dz2 dz2 fx(x2-3y2)
                c = 0
            elif (klm == 11):
                # dz2 dz2 fy(3x2-y2)
                c = 0
            elif (klm == 12):
                # dz2 dz2 fz(x2-y2)
                c = 0
            elif (klm == 13):
                # dz2 dz2 fxyz
                c = 0
            elif (klm == 14):
                # dz2 dz2 fyz2
                c = 0
            elif (klm == 15):
                # dz2 dz2 fxz2
                c = 0
            elif (klm == 16):
                # dz2 dz2 fz3
                c = 0
            elif (klm == 17):
                # dz2 dz2 g1
                c = 0
            elif (klm == 18):
                # dz2 dz2 g2
                c = 0
            elif (klm == 19):
                # dz2 dz2 g3
                c = 0
            elif (klm == 20):
                # dz2 dz2 g4
                c = 0
            elif (klm == 21):
                # dz2 dz2 g5
                c = 3./(7*sqrt(pi))
            elif (klm == 22):
                # dz2 dz2 g6
                c = 0
            elif (klm == 23):
                # dz2 dz2 g7
                c = 0
            elif (klm == 24):
                # dz2 dz2 g8
                c = 0
            elif (klm == 25):
                # dz2 dz2 g9
                c = 0
    return c
