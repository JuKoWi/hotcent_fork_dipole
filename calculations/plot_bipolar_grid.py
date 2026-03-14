import numpy as np
import matplotlib.pyplot as plt
from hotcent.new_dipole.utils import angstrom_to_bohr, bohr_to_angstrom
plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.bbox'] = 'tight'

def plot_bipolar(nr, nt, Rz):
    h = Rz/2
    rmax = 5.2
    rmin = 1e-7
    T = np.linspace(0, 1, nt) ** 2 * np.pi
    R = rmin + np.linspace(0, 1, nr) ** 2 * (rmax - rmin)

    fig, ax = plt.subplots(figsize=(6,9))

    #plot circles
    for r in R:
        theta = np.linspace(0, np.pi, 400)
        x1 = r*np.sin(theta)
        z1 = r*np.cos(theta)
        z2 = Rz - r*np.cos(theta)
        mask = z1 <= h
        ax.plot(bohr_to_angstrom(x1[mask]), bohr_to_angstrom(z1[mask]), 'black')
        ax.plot(bohr_to_angstrom(x1[mask]), bohr_to_angstrom(z2[mask]), 'black')

    # radial lines
    for t in T:
        x1 = np.linspace(rmin*np.sin(t), rmax*np.sin(t), 100)
        z1 = np.linspace(rmin*np.cos(t), rmax*np.cos(t), 100)
        z2 = Rz - z1

        mask = z1 <= h
        ax.plot(bohr_to_angstrom(x1[mask]), bohr_to_angstrom(z1[mask]), 'black')
        ax.plot(bohr_to_angstrom(x1[mask]), bohr_to_angstrom(z2[mask]), 'black')


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
        select = np.sqrt(dd ** 2 + zz ** 2) < rmax
        select *= np.sqrt(dd ** 2 + (Rz - zz) ** 2) < rmax
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
    ax.plot(bohr_to_angstrom(grid[:,0]), bohr_to_angstrom(grid[:,1]), "o", ms=4, color='red')

    ax.axhline(bohr_to_angstrom(h), color="black")

    ax.set_aspect("equal")
    ax.set_xlabel(r'$\rho$ [$\mathrm{\AA}$]')
    ax.set_ylabel(r'$z$ [$\mathrm{\AA}$]')
    ax.set_xlim(left=0)
    plt.savefig('bipolar_plot.pdf')
    plt.show()

plot_bipolar(nr=20,nt=20,Rz=angstrom_to_bohr(1.5))
