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


class Onsite2cGammaTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='monopolar',
                                     **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA', smoothen_tails=True):
        """
        Calculates on-site, distance dependent "Gamma" values as matrix
        elements of the two-center-expanded XC kernel.

        Parameters
        ----------
        See Onsite2cTable.run().
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Onsite-Gamma table construction for %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_onsiteG')
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

            G = self.calculate(selected, e1, e2, R, grid, area, xc=xc)

            for key in selected:
                nl1a, nl1b = key
                bas1a = e1.get_basis_set_index(nl1a)
                bas1b = e1.get_basis_set_index(nl1b)
                index = ANGULAR_MOMENTUM[nl1a[1]] * 4
                index += ANGULAR_MOMENTUM[nl1b[1]]
                self.tables[(bas1a, bas1b)][i, index] = G[key]

        if smoothen_tails:
            # Smooth the curves near the cutoff
            for key in self.tables:
                for i in range(NUMINT_2CL):
                    self.tables[key][:, i] = \
                            tail_smoothening(self.Rgrid, self.tables[key][:, i])

        self.timer.stop('run_onsiteG')

    def calculate(self, selected, e1, e2, R, grid, area, xc='LDA'):
        """
        Calculates the selected integrals involving the magnetization kernel.

        Parameters
        ----------
        See Onsite2cTable.calculate().

        Returns
        -------
        G: dict
            Dictionary containing the integral for each selected
            subshell pair.
        """
        self.timer.start('calculate_onsiteG')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        aux = 2 * np.pi * area * x

        xc = LibXC('LDA_X+LDA_C_PW' if xc == 'LDA' else xc)

        rho1 = e1.electron_density(r1)
        rho2 = e2.electron_density(r2)
        rho12 = rho1 + rho2

        if xc.add_gradient_corrections:
            drho1 = e1.electron_density(r1, der=1)
            drho2 = e2.electron_density(r2, der=1)
            c1 = y / r1  # cosine of theta_1
            c2 = (y - R) / r2  # cosine of theta_2
            s1 = x / r1  # sine of theta_1
            s2 = x / r2  # sine of theta_2

            grad_rho1_x = drho1 * s1
            grad_rho1_y = drho1 * c1
            sigma1 = grad_rho1_x**2 + grad_rho1_y**2

            grad_rho12_x = drho1 * s1 + drho2 * s2
            grad_rho12_y = drho1 * c1 + drho2 * c2
            sigma12 = grad_rho12_x**2 + grad_rho12_y**2
        else:
            sigma1 = None
            sigma12 = None
        self.timer.stop('prelude')

        self.timer.start('fxc')
        out1 = xc.compute_vxc(rho1, sigma=sigma1, fxc=True)
        out12 = xc.compute_vxc(rho12, sigma=sigma12, fxc=True)
        self.timer.stop('fxc')

        G = {}
        for key in selected:
            nl1a, nl1b = key
            dens_nl1a = e1.Rnl(r1, nl1a)**2 / (4 * np.pi)
            dens_nl1b = e1.Rnl(r1, nl1b)**2 / (4 * np.pi)

            integrand = (out12['v2rho2'] - out1['v2rho2']) \
                        * dens_nl1a * dens_nl1b

            if xc.add_gradient_corrections:
                dnl1a = e1.Rnl(r1, nl1a) / (2 * np.pi) * e1.Rnl(r1, nl1a, der=1)
                dnl1b = e1.Rnl(r1, nl1b) / (2 * np.pi) * e1.Rnl(r1, nl1b, der=1)
                grad_nl1a_grad_nl1b = (dnl1a*s1 * dnl1b*s1) \
                                      + (dnl1a*c1 * dnl1b*c1)

                grad_nl1a_grad_rho12 = (dnl1a*s1 * grad_rho12_x) \
                                       + (dnl1a*c1 * grad_rho12_y)
                grad_nl1b_grad_rho12 = (dnl1b*s1 * grad_rho12_x) \
                                       + (dnl1b*c1 * grad_rho12_y)

                integrand += 2. * out12['v2rhosigma'] \
                             * (grad_nl1a_grad_rho12 * dens_nl1b \
                                + grad_nl1b_grad_rho12 * dens_nl1a)
                integrand += 4. * out12['v2sigma2'] \
                             * grad_nl1a_grad_rho12 * grad_nl1b_grad_rho12
                integrand += 2. * out12['vsigma'] * grad_nl1a_grad_nl1b

                grad_nl1a_grad_rho1 = (dnl1a*s1 * grad_rho1_x) \
                                      + (dnl1a*c1 * grad_rho1_y)
                grad_nl1b_grad_rho1 = (dnl1b*s1 * grad_rho1_x) \
                                      + (dnl1b*c1 * grad_rho1_y)

                integrand -= 2. * out1['v2rhosigma'] \
                             * (grad_nl1a_grad_rho1 * dens_nl1b \
                                + grad_nl1b_grad_rho1 * dens_nl1a)
                integrand -= 4. * out1['v2sigma2'] \
                             * grad_nl1a_grad_rho1 * grad_nl1b_grad_rho1
                integrand -= 2. * out1['vsigma'] * grad_nl1a_grad_nl1b

            G[key] = np.sum(integrand * aux)

        self.timer.stop('calculate_onsiteG')
        return G

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el1>_onsiteG_<el2>.2cl'.
        """
        sym1, sym2 = self.ela.get_symbol(), self.elb.get_symbol()

        for bas1a, valence1a in enumerate(self.ela.basis_sets):
            angmom1a = [ANGULAR_MOMENTUM[nl[1]] for nl in valence1a]

            for bas1b, valence1b in enumerate(self.ela.basis_sets):
                angmom1b = [ANGULAR_MOMENTUM[nl[1]] for nl in valence1b]

                template = '%s-%s_onsiteG_%s.2cl'
                filename = template % (sym1 + '+'*bas1a, sym1 + '+'*bas1b, sym2)
                print('Writing to %s' % filename, file=self.txt, flush=True)

                table = self.tables[(bas1a, bas1b)]
                with open(filename, 'w') as f:
                    write_2cl(f, self.Rgrid, table, angmom1a, angmom1b)
        return
