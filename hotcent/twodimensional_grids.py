#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import _hotcent
except ModuleNotFoundError:
    print('Warning: C-extensions not available')
    _hotcent = None


def make_bipolar_grid(Rz, wf_range, nt, nr, p=2, q=2, view=False):
    """
    Construct a bipolar grid.

    Parameters
    ----------
    Rz : float
        Distance between the two grid centers (element #1 is at origin,
        element #2 is at z=Rz).
    wf_range : float
        Maximal radial extent of the grid.
    nt : int
        Number of theta grid points.
    nr : int
        Number of radial grid points.
    p : int
        Power describing the angular distribution of grid points
        (larger puts more weight towards theta=0).
    q : int
        Power describing the radial disribution of grid points
        (larger puts more weight towards centers).
    view : bool
        View the distribution of grid points with matplotlib.

    Returns
    -------
    grid : np.ndarray
        2D array with the x- and z-coordinates of the grid points.
    area : np.ndarray
        1D array with the area element associated with each grid point.


    Plane at R/2 divides two polar grids.


     ^ (z-axis)
     |--------_____               phi_j
     |       /     ----__         *
     |      /            \       /  *
     |     /               \    /  X *                X=coordinates of the center of area element(z,d),
     |    /                  \  \-----* phi_(j+1)     area=(r_(i+1)**2-r_i**2)*(phi_(j+1)-phi_j)/2
     |   /                    \  r_i   r_(i+1)
     |  /                      \
     | /                       |
     *2------------------------|           polar centered on atom 2
     | \                       |
     |  \                     /                                                     1
     |   \                   /                                                     /  \
     |-------------------------- z=h -line         ordering of sector slice       /     \
     |   /                   \                                      points:      /        \
     |  /                     \                                                 /          \
     | /                       |                                               /     0       4
     *1------------------------|--->      polar centered on atom 1            2            /
     | \                       |    (r_perpendicular (xy-plane) = 'd-axis')    \        /
     |  \                      /                                                 \   /
     |   \                    /                                                    3
     |    \                  /
     |     \               /
     |      \           /
     |       \ ___ ---
     |---------
    """
    rmin, rmax = 1e-7, wf_range
    h = Rz / 2

    if _hotcent is not None:
        grid, area = _hotcent.make_grid(Rz, rmin, rmax, nt, nr, p, q)
    else:
        max_range = wf_range
        T = np.linspace(0, 1, nt) ** p * np.pi
        R = rmin + np.linspace(0, 1, nr) ** q * (rmax - rmin)

        area = np.array([])
        d = np.array([])
        z = np.array([])

        # first calculate grid for polar centered on atom 1:
        # the z=h-like starts cutting full elements starting from point (1)
        Tj0 = T[:nt - 1]
        Tj1 = T[1: nt]

        for i in range(nr - 1):
            # corners of area element
            z1 = R[i + 1] * np.cos(Tj0)
            z2 = R[i] * np.cos(Tj0)
            z3 = R[i] * np.cos(Tj1)
            z4 = R[i + 1] * np.cos(Tj1)

            cond_list = [z1 <= h,  # area fully inside region
                 (z1 > h) * (z2 <= h) * (z4 <= h),  # corner 1 outside region
                 (z1 > h) * (z2 > h) * (z3 <= h) * (z4 <= h),  # 1 & 2 outside
                 (z1 > h) * (z2 > h) * (z3 <= h) * (z4 > h),  # only 3 inside
                 (z1 > h) * (z2 <= h) * (z3 <= h) * (z4 > h),  # 1 & 4 outside
                 (z1 > h) * (z3 > h) * ~((z2 <= h) * (z4 > h))]

            r0_list = [0.5 * (R[i] + R[i + 1]),
                       0.5 * (R[i] + R[i + 1]),
                       0.5 * (R[i] + R[i + 1]),
                       lambda x: 0.5 * (R[i] + h / np.cos(x)),
                       lambda x: 0.5 * (R[i] + h / np.cos(x)),
                       0,
                       np.nan]
            r0 = np.piecewise(Tj1, cond_list, r0_list)

            Th0 = np.piecewise(h / R[i], [np.abs(h / R[i]) > 1],
                               [np.nan, lambda x: np.arccos(x)])
            Th1 = np.piecewise(h / R[i + 1], [np.abs(h / R[i + 1]) > 1],
                               [np.nan, lambda x: np.arccos(x)])

            t0_list = [lambda x: 0.5 * x,
                       0.5 * Th1,
                       0.5 * Th1,
                       0.5 * Th0,
                       lambda x: 0.5 * x,
                       0,
                       np.nan]
            t0 = 0.5 * Tj1
            t0 += np.piecewise(Tj0, cond_list, t0_list)

            rr = 0.5 * (R[i + 1] ** 2 - R[i] ** 2)
            A_list0 = [lambda x: rr * -x,
                       lambda x: rr * -x - 0.5 * R[i + 1] ** 2 * (Th1 - x) \
                                 + 0.5 * h ** 2 * (np.tan(Th1) - np.tan(x)),
                       lambda x: rr * -x - (rr * -x + 0.5 * R[i + 1] ** 2 \
                                 * (Th1 - Th0)),
                       0.,
                       lambda x: 0.5 * h ** 2 * -np.tan(x) \
                                 - 0.5 * R[i] ** 2 * -x,
                       -1,
                       np.nan]
            A = np.piecewise(Tj0, cond_list, A_list0)

            A_list1 = [lambda x: rr * x,
                       lambda x: rr * x,
                       lambda x: rr * x - (rr * Th0 - 0.5 * h ** 2 \
                                 * (np.tan(Th1) - np.tan(Th0))),
                       lambda x: 0.5 * h ** 2 * (np.tan(x) - np.tan(Th0)) \
                                 - 0.5 * R[i] ** 2 * (x - Th0),
                       lambda x: 0.5 * h ** 2 * np.tan(x) \
                                 - 0.5 * R[i] ** 2 * x,
                       0,
                       np.nan]
            A += np.piecewise(Tj1, cond_list, A_list1)

            dd = r0 * np.sin(t0)
            zz = r0 * np.cos(t0)
            select = np.sqrt(dd ** 2 + zz ** 2) < max_range
            select *= np.sqrt(dd ** 2 + (Rz - zz) ** 2) < max_range
            select *= A > 0
            area = np.hstack((area, A[select]))
            d = np.hstack((d, dd[select]))
            z = np.hstack((z, zz[select]))
        grid = np.array([d, z]).T

    # calculate the polar centered on atom 2 by mirroring the other grid
    grid2 = grid.copy()
    grid2[:, 1] = -grid[:, 1]
    shift = np.zeros_like(grid)
    shift[:, 1] = 2 * h
    grid = np.concatenate((grid, grid2 + shift))
    area = np.concatenate((area, area))

    if view:
        assert plt is not None, 'Matplotlib could not be imported!'
        plt.plot([h, h ,h])
        plt.scatter(grid[:, 0], grid[:, 1], s=10 * area / np.max(area))
        plt.show()
        plt.clf()

    return grid, area
