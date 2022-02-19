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
from hotcent.interpolation import CubicSplineFunction
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.slako import tail_smoothening
from hotcent.xc import LibXC


class Offsite2cGammaTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA', smoothen_tails=True, shift=False):
        """
        Calculates off-site, distance dependent "Gamma" values as matrix
        elements of the two-center-expanded Hartree-XC kernel.

        Parameters
        ----------
        See Offsite2cTable.run() and Onsite2cTable.run().
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Offsite-Gamma table construction for %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_offsiteG')
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

            for term in ['hartree', 'xc']:
                if term == 'hartree':
                    self.grid_type = 'monopolar'
                    grid, area = self.make_grid(wf_range, nt=ntheta, nr=nr)
                elif term == 'xc':
                    self.grid_type = 'bipolar'
                    grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)

                if i == N - 1 or N // 10 == 0 or i % (N // 10) == 0:
                    print('R=%8.2f, %s, %i grid points ...' % \
                          (R, term, len(grid)), file=self.txt, flush=True)

                if len(grid) > 0:
                    for p, (e1, e2) in enumerate(self.pairs):
                        selected = select_subshells(e1, e2)

                        if term == 'hartree':
                            G = self.calculate_hartree(selected, e1, e2, R,
                                                       grid, area)
                        else:
                            G = self.calculate_xc(selected, e1, e2, R, grid,
                                                  area, xc=xc)

                        for key in selected:
                            nl1, nl2 = key
                            bas1 = e1.get_basis_set_index(nl1)
                            bas2 = e2.get_basis_set_index(nl2)
                            index = ANGULAR_MOMENTUM[nl1[1]] * 4
                            index += ANGULAR_MOMENTUM[nl2[1]]
                            self.tables[(p, bas1, bas2)][i, index] += G[key]

        for key in self.tables:
            for i in range(NUMINT_2CL):
                if shift and not np.allclose(self.tables[key][:, i], 0):
                    for j in range(N-1, 1, -1):
                        if abs(self.tables[key][j, i]) > 0:
                            self.tables[key][:j+1, i] -= self.tables[key][j, i]
                            break

                if smoothen_tails:
                    for key in self.tables:
                        self.tables[key][:, i] = \
                            tail_smoothening(self.Rgrid, self.tables[key][:, i])

        self.timer.stop('run_offsiteG')

    def calculate_xc(self, selected, e1, e2, R, grid, area, xc='LDA'):
        """
        Calculates the selected integrals involving the XC kernel.

        Parameters
        ----------
        See Offsite2cTable.calculate().

        Returns
        -------
        G: dict
            Dictionary containing the integral for each selected
            subshell pair.
        """
        self.timer.start('calculate_offsiteG_xc')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        aux = 2 * np.pi * area * x

        xc = LibXC('LDA_X+LDA_C_PW' if xc == 'LDA' else xc)

        rho = e1.electron_density(r1) + e2.electron_density(r2)

        if xc.add_gradient_corrections:
            drho1 = e1.electron_density(r1, der=1)
            drho2 = e2.electron_density(r2, der=1)
            c1 = y / r1  # cosine of theta_1
            c2 = (y - R) / r2  # cosine of theta_2
            s1 = x / r1  # sine of theta_1
            s2 = x / r2  # sine of theta_2

            grad_rho_x = drho1 * s1 + drho2 * s2
            grad_rho_y = drho1 * c1 + drho2 * c2
            sigma = grad_rho_x**2 + grad_rho_y**2
        else:
            sigma = None
        self.timer.stop('prelude')

        self.timer.start('fxc')
        out = xc.compute_vxc(rho, sigma=sigma, fxc=True)
        self.timer.stop('fxc')

        G = {}
        for key in selected:
            nl1, nl2 = key
            dens_nl1 = e1.Rnl(r1, nl1)**2 / (4 * np.pi)
            dens_nl2 = e2.Rnl(r2, nl2)**2 / (4 * np.pi)
            integrand = out['v2rho2'] * dens_nl1 * dens_nl2

            if xc.add_gradient_corrections:
                dnl1 = e1.Rnl(r1, nl1) / (2 * np.pi) * e1.Rnl(r1, nl1, der=1)
                dnl2 = e2.Rnl(r2, nl2) / (2 * np.pi) * e2.Rnl(r2, nl2, der=1)
                grad_nl1_grad_nl2 = (dnl1*s1 * dnl2*s2) + (dnl1*c1 * dnl2*c2)

                grad_nl1_grad_rho = (dnl1*s1 * grad_rho_x) \
                                    + (dnl1*c1 * grad_rho_y)
                grad_nl2_grad_rho = (dnl2*s2 * grad_rho_x) \
                                    + (dnl2*c2 * grad_rho_y)

                integrand += 2. * out['v2rhosigma'] \
                             * (grad_nl1_grad_rho * dens_nl2 \
                                + grad_nl2_grad_rho * dens_nl1)
                integrand += 4. * out['v2sigma2'] \
                             * grad_nl1_grad_rho * grad_nl2_grad_rho
                integrand += 2. * out['vsigma'] * grad_nl1_grad_nl2

            G[key] = np.sum(integrand * aux)

        self.timer.stop('calculate_offsiteG_xc')
        return G

    def calculate_hartree(self, selected, e1, e2, R, grid, area):
        """
        Calculates the selected integrals involving the Hartree kernel.

        Parameters
        ----------
        See Offsite2cTable.calculate().

        Returns
        -------
        G: dict
            Dictionary containing the integral for each selected
            subshell pair.
        """
        self.timer.start('calculate_offsiteG_hartree')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        aux = 2 * np.pi * area * x
        self.timer.stop('prelude')

        G = {}
        for key in selected:
            nl1, nl2 = key
            dens_nl1 = e1.Rnl(r1, nl1)**2 / (4 * np.pi)

            dens_nl2 = e2.Rnlg[nl2]**2 / (4 * np.pi)
            vhar2 = e2.calculate_hartree_potential(dens_nl2, nel=1.)
            spl = CubicSplineFunction(e2.rgrid, vhar2)

            G[key] = np.sum(dens_nl1 * spl(r2) * aux) - 1./R

        self.timer.stop('calculate_offsiteG_hartree')
        return G

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el2>_offsiteG.2cl'.
        """
        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()

            for bas1, valence1 in enumerate(e1.basis_sets):
                angmom1 = [ANGULAR_MOMENTUM[nl[1]] for nl in valence1]

                for bas2, valence2 in enumerate(e2.basis_sets):
                    angmom2 = [ANGULAR_MOMENTUM[nl[1]] for nl in valence2]

                    template = '%s-%s_offsiteG.2cl'
                    filename = template % (sym1 + '+'*bas1, sym2  + '+'*bas2)
                    print('Writing to %s' % filename, file=self.txt, flush=True)

                    table = self.tables[(p, bas1, bas2)]
                    with open(filename, 'w') as f:
                        write_2cl(f, self.Rgrid, table, angmom1, angmom2)
        return
