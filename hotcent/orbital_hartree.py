#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from scipy.linalg import solve_banded
from hotcent.interpolation import build_interpolator
from hotcent.orbitals import ORBITALS
from hotcent.radial_grid import RadialGrid
from hotcent.spherical_harmonics import sph


class OrbitalHartreePotential:
    """
    Calculator for Hartree potentials associated with
    atomic orbital densities.

    The overall approach is similar to that of
    Becke and Dickson, J. Chem. Phys. (1988), doi:10.1063/1.455005.

    The selected orbital density \rho_{\ell,m} is first expanded in
    a basis of spherical harmonics:

    \rho_{\ell,m}(\mathbf{r}) = |R_{nl}(r) * Y_{\ell,m}|**2
                              = \sum_\ell'=0,\ell'_{max} \sum_{m' \in \ell'}
                                    \rho_{\ell',m'}(r) Y_{\ell',m'}

    with \ell'_{max} = 2 * \ell and

    \rho_{\ell',m'}(r) = \int \rho_{\ell,m}(\mathbf{r}) Y_{\ell',m'} d\Omega

    \rho_{\ell,m}(r) = |R_{nl}(r)|**2
                       * \int |Y_{\ell,m}|**2 Y_{\ell',m'} d\Omega.

    The Hartree potential is obtained as:

    V_{har} = \sum_\ell \sum_{m \in \ell} V_{har,\ell,m}(r) Y_{\ell,m}

    where the V_{har,\ell,m}(r) functions are the solutions of the
    radial Poisson equation for the density \rho_{\ell,m}(r) Y_{\ell,m}.

    Parameters
    ----------
    rmin : float
        Smallest radius in the radial grid.
    xgrid : np.ndarray
        Equidistant logarithmic grid.
    Rl : np.ndarray
        Radial component of the orbital density on the grid
        (i.e. the square of that of the associated atomic orbital).
    lmax : int
        Maximum angular momentum for which to solve the Poisson equation.
    """
    def __init__(self, rmin, xgrid, Rl, lmax):
        self.rmin = rmin
        self.xgrid = xgrid
        self.lmax = lmax
        self.rgrid = self.rmin * np.exp(self.xgrid)
        self.grid = RadialGrid(self.rgrid)
        self.build_potentials(Rl)

    def build_potentials(self, Rl):
        """ Solves all the needed radial Poisson equations and
        builds the corresponding V_{har,\ell,m}(r) interpolators.

        Parameters
        ----------
        Rlm : np.ndarray
            Radial component of the orbital density on the grid.
        """
        self.vhar_fct = {}
        for l in range(self.lmax+1):
            vhar_l = self.solve_poisson(Rl, l)
            if vhar_l is None:
                self.vhar_fct[l] = lambda r: 0.
            else:
                self.vhar_fct[l] = build_interpolator(self.rgrid, vhar_l)
        return

    def solve_poisson(self, Rl, l):
        """ Returns the radial component of the Hartree potential
        associated with the given subshell density by solving the
        corresponding Poisson equation.

        Parameters
        ----------
        Rl : np.ndarray
            Radial component of the orbital density on the grid.
        l : int
            Subshell index of the spherical harmonic.

        Returns
        -------
        vhar_l : np.ndarray or None
            Hartree potential on the grid.
        """

        c0 = -l * (l + 1) / self.rgrid**2
        c1 = -1. / self.rgrid**2
        c2 = 1. / self.rgrid**2
        source = -4 * np.pi * self.rgrid * Rl

        h = self.xgrid[1] - self.xgrid[0]

        if l == 0:
            v0 = self.grid.integrate(Rl / self.rgrid, use_dV=True)
            u0 = self.rmin * np.exp(-h) * v0
        else:
            u0 = 0.

        nel = self.grid.integrate(Rl * self.rgrid**2, use_dV=False)
        u1 = 4 * np.pi * nel if l == 0 else 0.

        u_l = solve_radial_dgl(c0, c1, c2, source, u0, u1, h)
        vhar_l = u_l / self.rgrid
        return vhar_l

    def __call__(self, r, c, s, phi, lm):
        """ Evaluates the Hartree potential in spherical coordinates,
        using interpolation.

        Parameters
        ----------
        r : np.ndarray
            Radial distances.
        c : np.ndarray
            Cosines of the theta angle.
        s : np.ndarray
            Sines of the theta angle.
        phi : np.ndarray, optional
            Phi angles.
        lm : str
            Orbital label for the density (e.g. 'px').

        Returns
        -------
        value : float
            Hartree potential at the given points.
        """
        value = np.zeros_like(r)

        for l in range(self.lmax+1):
            vhar_l = self.vhar_fct[l](r)

            for m in range(2*l+1):
                lm2 = ORBITALS[l][m]
                coeff = get_density_expansion_coefficient(lm, lm2)
                value += vhar_l * coeff * sph(lm2, c, s, phi)

        return value


def get_density_expansion_coefficient(lm1, lm2):
    """ Calculates the expansion coefficient of a density associated
    with an lm1 orbital, for an lm2 spherical harmonic:

    coeff = \int |Y_{\ell1,m1}|^2 * Y_{\ell2,m2} d\Omega

    Parameters
    ----------
    lm1 : str
        Orbital label for the density (e.g. 'px').
    lm2 : str
        Orbital label for the spherical harmonic (e.g. 'px').

    Returns
    -------
    coeff : float
        The expansion coefficient.
    """
    if lm1[0] not in 'spd' or lm2[0] not in 'spdfg':
        raise NotImplementedError('{0}-{1} not implemented'.format(lm1, lm2))

    if lm1 == 's' and lm2 == 's':
        coeff = 1./(2*np.sqrt(np.pi))
    elif lm1 == 'px' and lm2 == 's':
        coeff = 1./(2*np.sqrt(np.pi))
    elif lm1 == 'px' and lm2 == 'dx2-y2':
        coeff = np.sqrt(15)/(10*np.sqrt(np.pi))
    elif lm1 == 'px' and lm2 == 'dz2':
        coeff = -np.sqrt(5)/(10*np.sqrt(np.pi))
    elif lm1 == 'py' and lm2 == 's':
        coeff = 1./(2*np.sqrt(np.pi))
    elif lm1 == 'py' and lm2 == 'dx2-y2':
        coeff = -np.sqrt(15)/(10*np.sqrt(np.pi))
    elif lm1 == 'py' and lm2 == 'dz2':
        coeff = -np.sqrt(5)/(10*np.sqrt(np.pi))
    elif lm1 == 'pz' and lm2 == 's':
        coeff = 1./(2*np.sqrt(np.pi))
    elif lm1 == 'pz' and lm2 == 'dz2':
        coeff = np.sqrt(5)/(5*np.sqrt(np.pi))
    elif lm1 == 'dxy' and lm2 == 's':
        coeff = 1./(2*np.sqrt(np.pi))
    elif lm1 == 'dxy' and lm2 == 'dz2':
        coeff = -np.sqrt(5)/(7*np.sqrt(np.pi))
    elif lm1 == 'dxy' and lm2 == 'g5':
        coeff = 1./(14*np.sqrt(np.pi))
    elif lm1 == 'dxy' and lm2 == 'g9':
        coeff = -np.sqrt(35)/(14*np.sqrt(np.pi))
    elif lm1 == 'dyz' and lm2 == 's':
        coeff = 1./(2*np.sqrt(np.pi))
    elif lm1 == 'dyz' and lm2 == 'dx2-y2':
        coeff = -np.sqrt(15)/(14*np.sqrt(np.pi))
    elif lm1 == 'dyz' and lm2 == 'dz2':
        coeff = np.sqrt(5)/(14*np.sqrt(np.pi))
    elif lm1 == 'dyz' and lm2 == 'g5':
        coeff = -2./(7*np.sqrt(np.pi))
    elif lm1 == 'dyz' and lm2 == 'g7':
        coeff = -np.sqrt(5)/(7*np.sqrt(np.pi))
    elif lm1 == 'dxz' and lm2 == 's':
        coeff = 1./(2*np.sqrt(np.pi))
    elif lm1 == 'dxz' and lm2 == 'dx2-y2':
        coeff = np.sqrt(15)/(14*np.sqrt(np.pi))
    elif lm1 == 'dxz' and lm2 == 'dz2':
        coeff = np.sqrt(5)/(14*np.sqrt(np.pi))
    elif lm1 == 'dxz' and lm2 == 'g5':
        coeff = -2./(7*np.sqrt(np.pi))
    elif lm1 == 'dxz' and lm2 == 'g7':
        coeff = np.sqrt(5)/(7*np.sqrt(np.pi))
    elif lm1 == 'dx2-y2' and lm2 == 's':
        coeff = 1./(2*np.sqrt(np.pi))
    elif lm1 == 'dx2-y2' and lm2 == 'dz2':
        coeff = -np.sqrt(5)/(7*np.sqrt(np.pi))
    elif lm1 == 'dx2-y2' and lm2 == 'g5':
        coeff = 1./(14*np.sqrt(np.pi))
    elif lm1 == 'dx2-y2' and lm2 == 'g9':
        coeff = np.sqrt(35)/(14*np.sqrt(np.pi))
    elif lm1 == 'dz2' and lm2 == 's':
        coeff = 1./(2*np.sqrt(np.pi))
    elif lm1 == 'dz2' and lm2 == 'dz2':
        coeff = np.sqrt(5)/(7*np.sqrt(np.pi))
    elif lm1 == 'dz2' and lm2 == 'g5':
        coeff = 3./(7*np.sqrt(np.pi))
    else:
        coeff = 0
    return coeff


def diagonal2banded(l_and_u, a):
    """
    Converts the given matrix to the diagonal ordered form.
    From https://github.com/scipy/scipy/pull/11344.

    Parameters
    ----------
    l_and_u : (int, int)
        Number of non-zero lower and upper diagonals.
    a : np.ndarray
        Matrix to be converted.

    Returns
    -------
    diagonal_ordered : np.ndarray
        Matrix in diagonal ordered form.
    """
    n = a.shape[1]
    if a.shape != (n, n):
        raise ValueError("Matrix must be square (has shape %s)" % (a.shape,))
    (nlower, nupper) = l_and_u

    if nlower >= n or nupper >= n:
        msg = "Number of nonzero diagonals must be less than square dimension"
        raise ValueError(msg)

    diagonal_ordered = np.empty((nlower + nupper + 1, n), dtype=a.dtype)

    for i in range(1, nupper + 1):
        for j in range(n - i):
            diagonal_ordered[nupper - i, i + j] = a[j, i + j]

    for i in range(n):
        diagonal_ordered[nupper, i] = a[i, i]

    for i in range(nlower):
        for j in range(n - i - 1):
            diagonal_ordered[nupper + 1 + i, j] = a[i + j + 1, j]

    return diagonal_ordered


def solve_radial_dgl(c0, c1, c2, source, u0, u1, h):
    """
    Solves the following differential equation on a equidistant grid:

    c_0(x) u(x) + c_1(x) \frac{d u}{d x} + c_2(x) \frac{d^2 u}{d x^2}
        = source(x)

    with left- and right-boundary conditions u0 and u1 (corresponding
    to the u(x) values for "ghost" points beyond the grid).

    Adapted from https://github.com/humeniuka/becke_multicenter_integration.

    See also Bickley, The Mathematical Gazette (1941), doi:10.2307/3606475.
    """
    N = len(c0)
    # operators d/dx and d^2/dx^2
    D1 = np.zeros((N, N))
    D2 = np.zeros((N, N))
    # terms from boundary conditions
    b1 = np.zeros(N)
    b2 = np.zeros(N)
    # non-centered five-point formulae for i=0
    D1[0, 0:4] = np.array([-20., 36., -12., 2.]) / (24.*h)
    b1[0] = -6./(24.*h) * u0
    D2[0, 0:4] = np.array([-20.,  6.,  +4., -1.]) / (12.*h**2)
    b2[0] = 11.0/(12.*h**2) * u0
    # non-centered six-point formulae for i=1
    D1[1, 0:5] = np.array([-60., -40., 120., -30., 4.]) / (120.*h)
    b1[1] = 6./(120.*h) * u0
    D2[1, 0:5] = np.array([80., -150., 80., -5., 0.]) / (60.*h**2)
    b2[1] = -5./(60.*h**2) * u0
    # centered seven-point formulae for i=2
    D1[2, 0:6] = np.array([108., -540., 0., 540., -108., 12.]) / (720.*h)
    b1[2] = -12./(720.*h) * u0
    D2[2, 0:6] = np.array([-54., 540., -980., 540., -54., 4.]) / (360.*h**2)
    b2[2] = 4./(360.*h**2) * u0
    # centered seven-point formulae for i=3,...,N-4
    for i in range(3, N-3):
        D1[i, i-3:i+4] = np.array([-12., 108., -540., 0., 540., -108., 12.]) \
                         / (720.*h)
        D2[i, i-3:i+4] = np.array([4., -54., 540., -980., 540., -54., 4.]) \
                         / (360.*h**2)
    # centered seven-point formulae for i=N-3
    D1[N-3, N-6:] = np.array([-12., 108., -540., 0., 540., -108.]) / (720.*h)
    b1[N-3] = 12./(720.*h) * u1
    D2[N-3, N-6:] = np.array([4., -54., 540., -980., 540., -54.]) / (360.*h**2)
    b2[N-3] = 4./(360.*h**2) * u1
    # non-centered six-point formulae for i=N-2
    D1[N-2, N-5:] = np.array([-4., 30., -120., 40., 60.]) / (120.*h)
    b1[N-2] = -6./(120.*h) * u1
    D2[N-2, N-5:] = np.array([0., -5., 80., -150., 80.]) / (60.*h**2)
    b2[N-2] = -5./(60.*h**2) * u1
    # non-centered five-point formulae for i=N-1
    D1[N-1, N-4:] = np.array([-2., 12., -36., 20.]) / (24.*h)
    b1[N-1] = 6./(24.*h) * u1
    D2[N-1, N-4:] = np.array([-1., 4., 6., -20.]) / (12.*h**2)
    b2[N-1] = 11./(12.*h**2) * u1
    # build matrix A on the left hand side of the equation
    A = np.zeros((N, N))
    for i in range(0, N):
        A[i, i] = c0[i]
        A[i, :] += c1[i]*D1[i, :] + c2[i]*D2[i, :]
    # right hand side
    rhs = source - c1*b1 - c2*b2

    # solve matrix equation
    ab = diagonal2banded((3, 3), A)
    u = solve_banded((3, 3), ab, rhs)
    return u
