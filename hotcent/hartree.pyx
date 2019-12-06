#cython: language_level=3
import numpy as np

DTYPE = np.float64


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
