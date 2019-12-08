#cython: language_level=3
import numpy as np

DTYPE = np.float64


def shoot(double[:] u, double dx, double[:] c2, double[:] c1, double[:] c0,
          int N):
    """ Cython version of shoot.py for faster atomic DFT calculations """
    cdef Py_ssize_t u_len = u.shape[0]
    assert u_len == N
    cdef int nodes = 0
    cdef int ctp = 0
    cdef double A = 0.
    cdef Py_ssize_t i, j

    u_new = np.zeros(N, dtype=DTYPE)
    cdef double[:] u_view = u_new
    for i in range(N):
        u_view[i] = u[i]

    fp = np.zeros(N, dtype=DTYPE)
    fm = np.zeros(N, dtype=DTYPE)
    f0 = np.zeros(N, dtype=DTYPE)
    cdef double[:] fp_view = fp
    cdef double[:] fm_view = fm
    cdef double[:] f0_view = f0

    cdef int all_negative = 1
    for i in range(N):
        fp_view[i] = c2[i] / dx ** 2 + 0.5 * c1[i] / dx
        fm_view[i] = c2[i] / dx ** 2 - 0.5 * c1[i] / dx
        f0_view[i] = c0[i] - 2 * c2[i] / (dx ** 2)
        if c0[i] > 0:
            all_negative = 0

    # backward integration down to classical turning point ctp
    # (or one point beyond to get derivative)
    # If no ctp, integrate half-way
    u_view[-1] = 1.0
    u_view[-2] = u_view[-1] * f0_view[-1] / fm_view[-1]
    for i in range(N - 2 , 0, -1):
        u_view[i - 1] = -fp_view[i] * u_view[i + 1] - f0_view[i] * u_view[i]
        u_view[i - 1] /= fm_view[i]
        if abs(u_view[i - 1]) > 1e10:
            for j in range(i - 1, N):
                u_view[j] *= 1e-10  # numerical stability
        if c0[i] > 0:
            ctp = i
            break
        if all_negative > 0 and i == N // 2:
            ctp = N // 2
            break

    cdef double utp, utp1, dright
    utp = u_view[ctp]
    utp1 = u_view[ctp + 1]
    dright = (u_view[ctp + 1] - u_view[ctp - 1]) / (2 * dx)

    for i in range(1, ctp + 1):
        u_view[i + 1] = -f0_view[i] * u_view[i] - fm_view[i] * u_view[i - 1]
        u_view[i + 1] /= fp_view[i]

    cdef double dleft, scale
    dleft = (u_view[ctp + 1] - u_view[ctp - 1]) / (2 * dx)
    scale = utp / u_view[ctp]
    for i in range(ctp + 1):
        u_view[i] *= scale
    u_view[ctp + 1] = utp1  # above overrode
    dleft *= scale

    if u_view[1] < 0:
        for i in range(N):
            u_view[i] *= -1

    for i in range(ctp - 1):
        if u_view[i] * u_view[i+1] < 0:
            nodes += 1

    A = (dright - dleft) * utp
    return u_new, nodes, A, ctp


def hartree(double[:] rho, double[:] dV, double[:] r, double[:] r0, int N):
    """ Calculate Hartree potential from radial density """
    vhar = np.zeros(N, dtype=DTYPE)
    lo = np.zeros(N, dtype=DTYPE)
    hi = np.zeros(N, dtype=DTYPE)
    cdef double[:] lo_view = lo
    cdef double[:] hi_view = hi
    cdef double[:] vhar_view = vhar
    cdef Py_ssize_t i

    for i in range(1, N):
        lo_view[i] = lo_view[i-1] + dV[i-1] * rho[i-1]

    for i in range(N - 2, -1, -1):
        hi_view[i] = hi_view[i + 1] + rho[i] * dV[i] / r0[i]

    for i in range(N):
        vhar_view[i] = lo_view[i] / r[i] + hi_view[i]

    return vhar
