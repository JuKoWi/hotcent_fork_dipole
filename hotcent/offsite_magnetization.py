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


class Offsite2cWTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA', smoothen_tails=True):
        """
        Calculates off-site, distance dependent "W" values as matrix
        elements of the two-center-expanded spin-polarized XC kernel.

        Parameters
        ----------
        See Offsite2cTable.run().
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Offsite-W table construction for %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_offsiteW')
        wf_range = self.get_range(wflimit)
        self.Rgrid = rmin + dr * np.arange(N)
        self.tables = {}

        for p, (e1, e2) in enumerate(self.pairs):
            for bas1 in range(len(e1.basis_sets)):
                for bas2 in range(len(e2.basis_sets)):
                    self.tables[(p, bas1, bas2)] = np.zeros((N, NUMINT_2CL))

        for i, R in enumerate(self.Rgrid):
            if R > 2 * wf_range:
                break

            grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)

            if i == N - 1 or N // 10 == 0 or i % (N // 10) == 0:
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            if len(grid) > 0:
                for p, (e1, e2) in enumerate(self.pairs):
                    selected = select_subshells(e1, e2)

                    W = self.calculate(selected, e1, e2, R, grid, area, xc=xc)

                    for key in selected:
                        nl1, nl2 = key
                        bas1 = e1.get_basis_set_index(nl1)
                        bas2 = e2.get_basis_set_index(nl2)
                        index = ANGULAR_MOMENTUM[nl1[1]] * 4
                        index += ANGULAR_MOMENTUM[nl2[1]]
                        self.tables[(p, bas1, bas2)][i, index] = W[key]

        if smoothen_tails:
            # Smooth the curves near the cutoff
            for key in self.tables:
                for i in range(NUMINT_2CL):
                    self.tables[key][:, i] = \
                            tail_smoothening(self.Rgrid, self.tables[key][:, i])

        self.timer.stop('run_offsiteW')

    def calculate(self, selected, e1, e2, R, grid, area, xc='LDA'):
        """
        Calculates the selected integrals involving the magnetization kernel.

        Parameters
        ----------
        See Offsite2cTable.calculate().

        Returns
        -------
        W: dict
            Dictionary containing the integral for each selected
            subshell pair.
        """
        self.timer.start('calculate_offsiteW')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        aux = 2 * np.pi * area * x

        xc = LibXC('LDA_X+LDA_C_PW' if xc == 'LDA' else xc,
                   spin_polarized=True)

        rho_up = (e1.electron_density(r1) + e2.electron_density(r2)) / 2.
        rho_down = np.copy(rho_up)

        if xc.add_gradient_corrections:
            drho1_up = e1.electron_density(r1, der=1) / 2.
            drho2_up = e2.electron_density(r2, der=1) / 2.
            c1 = y / r1  # cosine of theta_1
            c2 = (y - R) / r2  # cosine of theta_2
            s1 = x / r1  # sine of theta_1
            s2 = x / r2  # sine of theta_2

            grad_rho_x_up = drho1_up * s1 + drho2_up * s2
            grad_rho_y_up = drho1_up * c1 + drho2_up * c2
            grad_rho_x_down = np.copy(grad_rho_x_up)
            grad_rho_y_down = np.copy(grad_rho_y_up)
            sigma_up = grad_rho_x_up**2 + grad_rho_y_up**2
            sigma_updown = grad_rho_x_up * grad_rho_x_down \
                           + grad_rho_y_up * grad_rho_y_down
            sigma_down = grad_rho_x_down**2 + grad_rho_y_down**2
        else:
            sigma_up, sigma_updown, sigma_down = None, None, None
        self.timer.stop('prelude')

        self.timer.start('fxc')
        out = xc.compute_vxc_polarized(rho_up, rho_down, sigma_up=sigma_up,
                                       sigma_updown=sigma_updown,
                                       sigma_down=sigma_down, fxc=True)
        self.timer.stop('fxc')

        W = {}
        for key in selected:
            nl1, nl2 = key
            dens_nl1 = e1.Rnl(r1, nl1)**2 / (4 * np.pi)
            dens_nl2 = e2.Rnl(r2, nl2)**2 / (4 * np.pi)

            integrand = (out['v2rho2_up'] - out['v2rho2_updown']) \
                        * dens_nl1 * dens_nl2

            if xc.add_gradient_corrections:
                dnl1 = e1.Rnl(r1, nl1) / (2 * np.pi) * e1.Rnl(r1, nl1, der=1)
                dnl2 = e2.Rnl(r2, nl2) / (2 * np.pi) * e2.Rnl(r2, nl2, der=1)
                grad_nl1_grad_nl2 = (dnl1*s1 * dnl2*s2) + (dnl1*c1 * dnl2*c2)

                grad_nl1_grad_rho_up = (dnl1*s1 * grad_rho_x_up) \
                                       + (dnl1*c1 * grad_rho_y_up)
                grad_nl2_grad_rho_up = (dnl2*s2 * grad_rho_x_up) \
                                       + (dnl2*c2 * grad_rho_y_up)
                grad_nl1_grad_rho_down = (dnl1*s1 * grad_rho_x_down) \
                                         + (dnl1*c1 * grad_rho_y_down)
                grad_nl2_grad_rho_down = (dnl2*s2 * grad_rho_x_down) \
                                         + (dnl2*c2 * grad_rho_y_down)

                # up-up contribution
                integrand += 2. * out['v2rhosigma_up_up'] \
                             * (grad_nl1_grad_rho_up * dens_nl2 \
                                + dens_nl1 * grad_nl2_grad_rho_up)

                integrand += 4. * out['v2sigma2_up_up'] \
                             * grad_nl1_grad_rho_up * grad_nl2_grad_rho_up
                integrand += 2. * out['v2sigma2_up_updown'] \
                             * grad_nl1_grad_rho_down * grad_nl2_grad_rho_up
                integrand += 2. * out['v2sigma2_up_updown'] \
                             * grad_nl1_grad_rho_up * grad_nl2_grad_rho_down
                integrand += 1. * out['v2sigma2_updown_updown'] \
                             * grad_nl1_grad_rho_down * grad_nl2_grad_rho_down
                integrand += 2. * out['vsigma_up'] * grad_nl1_grad_nl2

                # up-down contrbution
                integrand -= 2. * out['v2rhosigma_up_down'] \
                             * (grad_nl1_grad_rho_down * dens_nl2 \
                                + dens_nl1 * grad_nl2_grad_rho_down)

                integrand -= 4. * out['v2sigma2_up_down'] \
                             * grad_nl1_grad_rho_down * grad_nl2_grad_rho_up
                integrand -= 2. * out['v2sigma2_up_updown'] \
                             * grad_nl1_grad_rho_down * grad_nl2_grad_rho_up
                integrand -= 2. * out['v2sigma2_updown_down'] \
                             * grad_nl1_grad_rho_down * grad_nl2_grad_rho_down
                integrand -= 1. * out['v2sigma2_updown_updown'] \
                             * grad_nl1_grad_rho_up * grad_nl2_grad_rho_down
                integrand -= 1. * out['vsigma_updown'] * grad_nl1_grad_nl2

            W[key] = np.sum(integrand * aux) / 2.

        self.timer.stop('calculate_offsiteW')
        return W

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el2>_offsiteW.2cl'.
        """
        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()

            for bas1, valence1 in enumerate(e1.basis_sets):
                angmom1 = [ANGULAR_MOMENTUM[nl[1]] for nl in valence1]

                for bas2, valence2 in enumerate(e2.basis_sets):
                    angmom2 = [ANGULAR_MOMENTUM[nl[1]] for nl in valence2]

                    template = '%s-%s_offsiteW.2cl'
                    filename = template % (sym1 + '+'*bas1, sym2  + '+'*bas2)
                    print('Writing to %s' % filename, file=self.txt, flush=True)

                    table = self.tables[(p, bas1, bas2)]
                    with open(filename, 'w') as f:
                        write_2cl(f, self.Rgrid, table, angmom1, angmom2)
        return
