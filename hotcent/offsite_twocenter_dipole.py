import numpy as np
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.orbitals import ANGULAR_MOMENTUM
from slako_dipole import (INTEGRALS, print_integral_overview, select_integrals, NUMSK, phi3)
import matplotlib.pyplot as plt

class Offsite2cTableDipole(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, superposition='density', xc='LDA', nr=50, stride=1, wflimit=1e-7, ntheta=150):

        self.print_header()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'
        assert superposition in ['density', 'potential']

        self.timer.start('run_offsite2c') # TODO check what this does
        wf_range = self.get_range(wflimit)
        Nsub = N // stride
        Rgrid = rmin + stride * dr * np.arange(Nsub)
        tables = {}

        for p, (e1, e2) in enumerate(self.pairs): # iterate over ordered element pairs
            selected = select_integrals(e1, e2)
            print_integral_overview(e1, e2, selected, file=self.txt)

            for bas1 in range(len(e1.basis_sets)):
                for bas2 in range(len(e2.basis_sets)):
                    tables[(p, bas1, bas2)] = np.zeros((Nsub, 2*NUMSK))

        for i, R in enumerate(Rgrid):
            if R > 2 * wf_range:
                break

            grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)

            if i == Nsub - 1 or Nsub // 10 == 0 or i % (Nsub // 10) == 0:
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            for p, (e1, e2) in enumerate(self.pairs):
                selected = select_integrals(e1, e2)
                if len(grid) > 0:
                    D = self.calculate(selected, e1, e2, R, grid, area,
                                             xc=xc, superposition=superposition)
                    for key in selected:
                        integral, nl1, nl2 = key
                        bas1 = e1.get_basis_set_index(nl1)
                        bas2 = e2.get_basis_set_index(nl2)
                        index = INTEGRALS.index(integral)
                        tables[(p, bas1, bas2)][i, index] = H[key]
                        tables[(p, bas1, bas2)][i, NUMSK+index] = S[key]
        #TODO: do I have to worry about pseudopotentials, probably not

        self.timer.stop('run_offsite2c')

    def calculate(self, selected, e1, e2, R, grid, area):

        self.timer.start('calculate_offsite2c')

        # common for all integrals (not wf-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y - R) / r2  # cosine of theta_2
        s1 = x / r1  # sine of theta_1
        s2 = x / r2  # sine of theta_2

        # calculate all selected integrals
        Dl= {}
        sym1, sym2 = e1.get_symbol(), e2.get_symbol()

        for key in selected:
            integral, nl1, nl2 = key

            gphi = phi3(c1, c2, s1, s2, integral)
            aux = gphi * area * x * r1

            Rnl1 = e1.Rnl(r1, nl1)
            Rnl2 = e2.Rnl(r2, nl2)
            D = np.sum(Rnl1 * Rnl2 * aux)
            Dl[key] = D

        self.timer.stop('calculate_offsite2c')
        return Dl
