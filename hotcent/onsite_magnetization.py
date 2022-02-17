#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.fluctuation_twocenter import (NUMINT_2CL, select_subshells,
                                           write_2cl)
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.slako import tail_smoothening
from hotcent.xc import LibXC


class Onsite2cWTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='monopolar',
                                     **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA_X+LDA_C_PW', smoothen_tails=True):
        """
        Calculates on-site, distance dependent "W" values as matrix
        elements of the two-center-expanded spin-polarized XC kernel.

        Parameters
        ----------
        See Onsite2cTable.run().
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Onsite-W table construction for %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_onsiteW')
        wf_range = self.get_range(wflimit)
        grid, area = self.make_grid(wf_range, nt=ntheta, nr=nr)

        self.Rgrid = rmin + dr * np.arange(N)
        self.tables = {}

        e1, e2 = self.ela, self.elb
        selected = select_subshells(e1, e2)

        for bas1a in range(len(e1.basis_sets)):
            for bas1b in range(len(e1.basis_sets)):
                self.tables[(bas1a, bas1b)] = np.zeros((N, NUMINT_2CL))

        for i, R in enumerate(self.Rgrid):
            if R > 2 * wf_range:
                break

            if i == N - 1 or N // 10 == 0 or i % (N // 10) == 0:
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            W = self.calculate(selected, e1, e2, R, grid, area, xc=xc)

            for key in selected:
                nl1a, nl1b = key
                bas1a = e1.get_basis_set_index(nl1a)
                bas1b = e1.get_basis_set_index(nl1b)
                index = ANGULAR_MOMENTUM[nl1a[1]] * 4
                index += ANGULAR_MOMENTUM[nl1b[1]]
                self.tables[(bas1a, bas1b)][i, index] = W[key]

        if smoothen_tails:
            # Smooth the curves near the cutoff
            for key in self.tables:
                for i in range(NUMINT_2CL):
                    self.tables[key][:, i] = \
                            tail_smoothening(self.Rgrid, self.tables[key][:, i])

        self.timer.stop('run_onsiteW')

    def calculate(self, selected, e1, e2, R, grid, area, xc='LDA_X+LDA_C_PW'):
        """
        Calculates the selected integrals involving the magnetization kernel.

        Parameters
        ----------
        See Onsite2cTable.calculate().

        Returns
        -------
        W: dict
            Dictionary containing the integral for each selected
            subshell pair.
        """
        self.timer.start('calculate_onsiteW')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        aux = 2 * np.pi * area * x

        xc = LibXC(xc, spin_polarized=True)

        rho1_up = e1.electron_density(r1) / 2.
        rho1_down = np.copy(rho1_up)
        rho2_up = e2.electron_density(r2) / 2.
        rho2_down = np.copy(rho2_up)
        rho12_up = rho1_up + rho2_up
        rho12_down = rho1_down + rho2_down

        if xc.add_gradient_corrections:
            raise NotImplementedError('GGA functionals are not yet implemented')
        else:
            sigma1_up, sigma1_updown, sigma1_down = None, None, None
            sigma12_up, sigma12_updown, sigma12_down = None, None, None

        self.timer.stop('prelude')

        self.timer.start('fxc')
        out = xc.compute_vxc_polarized(rho12_up, rho12_down,
                                       sigma_up=sigma12_up,
                                       sigma_updown=sigma12_updown,
                                       sigma_down=sigma12_down, fxc=True)
        v2rho2 = (out['v2rho2_up'] - out['v2rho2_updown']) / 2.

        out = xc.compute_vxc_polarized(rho1_up, rho1_down, sigma_up=sigma1_up,
                                       sigma_updown=sigma1_updown,
                                       sigma_down=sigma1_down, fxc=True)
        v2rho2 -= (out['v2rho2_up'] - out['v2rho2_updown']) / 2.
        self.timer.stop('fxc')

        W = {}
        for key in selected:
            nl1a, nl1b = key
            dens_nl1a = e1.Rnl(r1, nl1a)**2 / (4 * np.pi)
            dens_nl1b = e1.Rnl(r1, nl1b)**2 / (4 * np.pi)
            W[key] = np.sum(dens_nl1a * v2rho2 * dens_nl1b * aux)

            if xc.add_gradient_corrections:
                raise NotImplementedError('GGA functionals not yet implemented')

        self.timer.stop('calculate_onsiteW')
        return W

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el1>_onsiteW_<el2>.2cl'.
        """
        sym1, sym2 = self.ela.get_symbol(), self.elb.get_symbol()

        for bas1a, valence1a in enumerate(self.ela.basis_sets):
            angmom1a = [ANGULAR_MOMENTUM[nl[1]] for nl in valence1a]

            for bas1b, valence1b in enumerate(self.ela.basis_sets):
                angmom1b = [ANGULAR_MOMENTUM[nl[1]] for nl in valence1b]

                template = '%s-%s_onsiteW_%s.2cl'
                filename = template % (sym1 + '+'*bas1a, sym1 + '+'*bas1b, sym2)
                print('Writing to %s' % filename, file=self.txt, flush=True)

                table = self.tables[(bas1a, bas1b)]
                with open(filename, 'w') as f:
                    write_2cl(f, self.Rgrid, table, angmom1a, angmom1b)
        return
