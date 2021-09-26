#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np


def sph_nophi(lm, c, s):
    """ Evaluates the phi-independent part of the chosen spherical harmonic,
    as function of the cosine and sine of the theta angle.
    """
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
    """ Evaluates the phi-dependent part of the chosen spherical harmonic. """
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


def sph(lm, c, s, phi):
    """ Evaluates the chosen spherical harmonic based on cos(theta),
    sin(theta) and phi.
    """
    return sph_nophi(lm, c, s) * sph_phi(lm, phi)


def sph_cartesian(x, y, z, r, lm):
    """ Evaluates the chosen spherical harmonic in cartesian coordinates. """
    if lm == 's':
        return 0.5 / np.sqrt(np.pi)
    elif lm == 'px':
        return np.sqrt(3. / (4.*np.pi)) * x/r
    elif lm == 'py':
        return np.sqrt(3. / (4.*np.pi)) * y/r
    elif lm == 'pz':
        return np.sqrt(3. / (4.*np.pi)) * z/r
    elif lm == 'dxy':
        return np.sqrt(15. / (4.*np.pi)) * x*y/r**2
    elif lm == 'dyz':
        return np.sqrt(15. / (4.*np.pi)) * y*z/r**2
    elif lm == 'dxz':
        return np.sqrt(15. / (4.*np.pi)) * x*z/r**2
    elif lm == 'dx2-y2':
        return np.sqrt(15. / (16.*np.pi)) * (x**2-y**2)/r**2
    elif lm == 'dz2':
        return np.sqrt(5. / (16.*np.pi)) * (2*z**2-x**2-y**2)/r**2
