#-----------------------------------------------------------------------------#
#   Hotcent: a tool for generating tight-binding parameter files              #
#   Copyright 2018-2024 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from collections import OrderedDict
from ase.units import Ha
from hotcent.fluctuation_onecenter import NUML_1CK, write_1ck
from hotcent.fluctuation_twocenter import (
                INTEGRALS_2CK, NUMINT_2CL, NUML_2CK, NUMSK_2CK,
                select_subshells, write_2cl, write_2ck)
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.singleatom_integrator import SingleAtomIntegrator
from hotcent.slako import (get_integral_pair, get_twocenter_phi_integral,
                           get_twocenter_phi_integrals_derivatives,
                           tail_smoothening)
from hotcent.xc import LibXC


class Onsite1cWTable:
    """
    Convenience wrapper around the Onsite1cWMainTable
    and Onsite1cWAuxiliaryTable classes.

    Parameters
    ----------
    basis : str
        Whether to derive parameters from the main basis set in
        the monopole approximation (basis='main') or from the
        (possibly multipolar) auxiliary basis set (basis='auxiliary').
    """
    def __init__(self, *args, basis=None, **kwargs):
        if basis == 'main':
            self.calc = Onsite1cWMainTable(*args, **kwargs)
        elif basis == 'auxiliary':
            self.calc = Onsite1cWAuxiliaryTable(*args, **kwargs)
        else:
            raise ValueError('Unknown basis: {0}'.format(basis))

    def __getattr__(self, attr):
        return getattr(self.calc, attr)

    def run(self, *args, **kwargs):
        self.calc.run(*args, **kwargs)
        return

    def write(self, *args, **kwargs):
        self.calc.write(*args, **kwargs)
        return


class Onsite1cWMainTable(SingleAtomIntegrator):
    """
    Calculator for the "W" integrals as matrix elements of the
    one-center-expanded spin-polarized XC kernel and the main basis set,
    in a monopole approximation.

    Parameters
    ----------
    el : AtomicBase-like object
        Object with atomic properties.

    Other parameters
    ----------------
    See SingleAtomIntegrator
    """
    def __init__(self, el, **kwargs):
        SingleAtomIntegrator.__init__(self, el, **kwargs)

    def run(self, method='analytical', maxstep=0.25):
        """
        Calculates the onsite, one-center "W" integrals.

        Parameters
        ----------
        method : str, optional
            Whether to calculate the integrals analytically
            ('analytical') or via numerical differentiation
            ('numerical').
        maxstep : float, optional
            Step size to use for the integrals, if evaluated
            via numerical differentiation.
        """
        self.print_header()

        self.method = method
        self.tables = OrderedDict()

        for valence1 in self.el.basis_sets:
            for nl1 in valence1:
                for valence2 in self.el.basis_sets:
                    for nl2 in valence2:
                        if method == 'analytical':
                            W = self.el.get_analytical_spin_constant(nl1, nl2)
                        elif method == 'numerical':
                            W = self.el.get_spin_constant(nl1, nl2, scheme=None,
                                                          maxstep=maxstep)
                        else:
                            raise ValueError('Unknown method:', method)
                        self.tables[(nl1, nl2)] = W
        return

    def write(self):
        """
        Writes the spin constants to file.

        The filename template corresponds to '<el>_spin_constants.txt'.
        """
        sym = self.el.get_symbol()

        filename = '%s_spin_constants.txt' % sym
        print('Writing to %s' % filename, file=self.txt)

        with open(filename, 'w') as f:
            f.write('Method: {0}\n'.format(self.method))

            f.write('Spin constant table:\n')
            template = '%16s: %10.6f [%s] %10.6f [%s]\n'

            Ws = []
            for (nl1, nl2), W in self.tables.items():
                W_eV = W * Ha
                key = 'W_%s-%s' % (nl1, nl2)
                f.write(template % (key, W, 'Ha', W_eV, 'eV'))
                Ws.append(W_eV)

            # Repeat in list form for convenience (only eV)
            f.write('Spin constant list [eV]:\n')
            Ws_str = '[' + ', '.join(list(map(lambda x: '%.3f' % x, Ws))) + ']'
            f.write('    %s: %s\n' % (sym, Ws_str))
        return


class Onsite1cWAuxiliaryTable(SingleAtomIntegrator):
    """
    Calculator for (parts of) the "W" integrals as matrix elements of the
    one-center-expanded spin-polarized XC kernel and the auxiliary basis set,
    in a multipole expansion.

    Parameters
    ----------
    el : AtomicBase-like object
        Object with atomic properties.

    Other parameters
    ----------------
    See SingleAtomIntegrator
    """
    def __init__(self, el, **kwargs):
        SingleAtomIntegrator.__init__(self, el, **kwargs)
        assert self.el.aux_basis.get_lmax() < NUML_1CK

    def run(self, xc='LDA'):
        """
        Calculates the onsite, one-center "W" integrals.

        Parameters
        ----------
        xc : str, optional
            Name of the exchange-correlation functional (default: LDA).
        """
        self.print_header()

        self.tables = {}
        for bas1 in range(self.el.aux_basis.get_nzeta()):
            for bas2 in range(self.el.aux_basis.get_nzeta()):
                self.tables[(bas1, bas2)] = np.zeros((2, NUML_1CK))

        selected = self.el.aux_basis.select_radial_functions()
        print('Selected subshells:', selected, file=self.txt)

        for nl1 in selected:
            for nl2 in selected:
                W, radmom = self.calculate(nl1, nl2, xc=xc)
                bas1 = self.el.aux_basis.get_zeta_index(nl1)
                bas2 = self.el.aux_basis.get_zeta_index(nl2)
                self.tables[(bas1, bas2)][0, :] = W
                self.tables[(bas1, bas2)][1, :] = radmom
        return

    def calculate(self, nl1, nl2, xc='LDA'):
        """
        Calculates the selected integrals involving the spin-polarized
        XC kernel.

        Parameters
        ----------
        nl1, nl2 : str
            Subshells defining the radial functions.
        xc : str, optional
            Name of the exchange-correlation functional (default: LDA).

        Returns
        -------
        W: np.ndarray
            Array with the integral for each multipole.
        radmom : np.ndarray
            Array with the radial moments of the auxiliary basis function
            if nl1 equals nl2 (\int \chi_{nl} r^{l+2} dr).
        """
        xc = LibXC('LDA_X+LDA_C_PW' if xc == 'LDA' else xc,
                   spin_polarized=True)

        rho_up = self.el.electron_density(self.el.rgrid) / 2.
        rho_down = np.copy(rho_up)

        if xc.add_gradient_corrections:
            drho = self.el.electron_density(self.el.rgrid, der=1) / 2.
            sigma = drho**2
        else:
            sigma = None

        out = xc.compute_vxc_polarized(rho_up, rho_down, sigma_up=sigma,
                                       sigma_updown=sigma, sigma_down=sigma,
                                       fxc=True)

        W = np.zeros(NUML_1CK)
        radmom = np.zeros(NUML_1CK)

        for l in range(NUML_1CK):
            if l > self.el.aux_basis.get_lmax():
                continue

            Anl1 = self.el.aux_basis(self.el.rgrid, nl1, l)
            Anl2 = self.el.aux_basis(self.el.rgrid, nl2, l)

            integrand = (out['v2rho2_up'] - out['v2rho2_updown']) * Anl1 * Anl2
            W[l] = self.el.grid.integrate(integrand * self.el.rgrid**2,
                                          use_dV=False)

            if xc.add_gradient_corrections:
                dnl1 = self.el.aux_basis(self.el.rgrid, nl1, l, der=1)
                dnl2 = self.el.aux_basis(self.el.rgrid, nl2, l, der=1)
                grad_nl1_grad_rho = dnl1 * drho
                grad_nl2_grad_rho = dnl2 * drho

                products = grad_nl1_grad_rho * Anl2 \
                           + grad_nl2_grad_rho * Anl1
                integrand = 2. * products \
                        * (out['v2rhosigma_up_up'] - out['v2rhosigma_up_down'])

                products = grad_nl1_grad_rho * grad_nl2_grad_rho
                integrand += 4. * products \
                        * (out['v2sigma2_up_up'] - out['v2sigma2_up_down'])
                integrand += 2. * products \
                    * (out['v2sigma2_up_updown'] - out['v2sigma2_updown_down'])

                products = dnl1 * dnl2
                integrand += products \
                             * (2. * out['vsigma_up'] - out['vsigma_updown'])

                W[l] += self.el.grid.integrate(integrand * self.el.rgrid**2,
                                               use_dV=False)

                integrand = (2. * out['vsigma_up'] - out['vsigma_updown']) \
                            * Anl1 * Anl2
                W[l] += self.el.grid.integrate(integrand * l * (l+1),
                                               use_dV=False)

            if nl1 == nl2:
                integrand = Anl1 * self.el.rgrid**(l+2)
                radmom[l] = self.el.grid.integrate(integrand, use_dV=False)
            else:
                radmom[l] = 0

        W /= 2
        return (W, radmom)

    def write(self):
        """
        Writes the integrals to file.

        The filename template corresponds to '<el>-<el>_onsiteW.1ck'.
        """
        sym = self.el.get_symbol()

        for bas1 in range(self.el.aux_basis.get_nzeta()):
            for bas2 in range(self.el.aux_basis.get_nzeta()):
                template = '%s-%s_onsiteW.1ck'
                filename = template % (sym + '+'*bas1, sym + '+'*bas2)
                print('Writing to %s' % filename, file=self.txt, flush=True)

                table = self.tables[(bas1, bas2)]
                with open(filename, 'w') as f:
                    write_1ck(f, table[1, :], table[0, :])
        return


class Onsite2cWTable:
    """
    Convenience wrapper around the Onsite2cWMainTable
    and Onsite2cWAuxiliaryTable classes.

    Parameters
    ----------
    basis : str
        Whether to derive parameters from the main basis set in
        the monopole approximation (basis='main') or from the
        (possibly multipolar) auxiliary basis set (basis='auxiliary').
    """
    def __init__(self, *args, basis=None, **kwargs):
        if basis == 'main':
            self.calc = Onsite2cWMainTable(*args, **kwargs)
        elif basis == 'auxiliary':
            self.calc = Onsite2cWAuxiliaryTable(*args, **kwargs)
        else:
            raise ValueError('Unknown basis: {0}'.format(basis))

    def __getattr__(self, attr):
        return getattr(self.calc, attr)

    def run(self, *args, **kwargs):
        self.calc.run(*args, **kwargs)
        return

    def write(self, *args, **kwargs):
        self.calc.write(*args, **kwargs)
        return


class Onsite2cWMainTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='monopolar',
                                     **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA', smoothen_tails=True):
        """
        Calculates onsite, distance dependent "W" values as matrix
        elements of the two-center-expanded spin-polarized XC kernel
        and the main basis set, in the monopole approximation.

        Parameters
        ----------
        See Onsite2cTable.run().
        """
        self.print_header()

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
        selected = select_subshells(e1, e1)

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

    def calculate(self, selected, e1, e2, R, grid, area, xc='LDA'):
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

        xc = LibXC('LDA_X+LDA_C_PW' if xc == 'LDA' else xc,
                   spin_polarized=True)

        rho1_up = e1.electron_density(r1) / 2.
        rho1_down = np.copy(rho1_up)
        rho2_up = e2.electron_density(r2) / 2.
        rho2_down = np.copy(rho2_up)
        rho12_up = rho1_up + rho2_up
        rho12_down = rho1_down + rho2_down

        if xc.add_gradient_corrections:
            drho1_up = e1.electron_density(r1, der=1) / 2.
            drho2_up = e2.electron_density(r2, der=1) / 2.
            c1 = y / r1  # cosine of theta_1
            c2 = (y - R) / r2  # cosine of theta_2
            s1 = x / r1  # sine of theta_1
            s2 = x / r2  # sine of theta_2

            grad_rho1_x_up = drho1_up * s1
            grad_rho1_y_up = drho1_up * c1
            grad_rho1_x_down = np.copy(grad_rho1_x_up)
            grad_rho1_y_down = np.copy(grad_rho1_y_up)
            sigma1_up = grad_rho1_x_up**2 + grad_rho1_y_up**2
            sigma1_updown = grad_rho1_x_up * grad_rho1_x_down \
                            + grad_rho1_y_up * grad_rho1_y_down
            sigma1_down = grad_rho1_x_down**2 + grad_rho1_y_down**2

            grad_rho12_x_up = drho1_up * s1 + drho2_up * s2
            grad_rho12_y_up = drho1_up * c1 + drho2_up * c2
            grad_rho12_x_down = np.copy(grad_rho12_x_up)
            grad_rho12_y_down = np.copy(grad_rho12_y_up)
            sigma12_up = grad_rho12_x_up**2 + grad_rho12_y_up**2
            sigma12_updown = grad_rho12_x_up * grad_rho12_x_down \
                             + grad_rho12_y_up * grad_rho12_y_down
            sigma12_down = grad_rho12_x_down**2 + grad_rho12_y_down**2
        else:
            sigma1_up, sigma1_updown, sigma1_down = None, None, None
            sigma12_up, sigma12_updown, sigma12_down = None, None, None

        self.timer.stop('prelude')

        self.timer.start('fxc')
        out1 = xc.compute_vxc_polarized(rho1_up, rho1_down, sigma_up=sigma1_up,
                                        sigma_updown=sigma1_updown,
                                        sigma_down=sigma1_down, fxc=True)
        out12 = xc.compute_vxc_polarized(rho12_up, rho12_down,
                                         sigma_up=sigma12_up,
                                         sigma_updown=sigma12_updown,
                                         sigma_down=sigma12_down, fxc=True)
        self.timer.stop('fxc')

        W = {}
        for key in selected:
            nl1a, nl1b = key
            dens_nl1a = e1.Rnl(r1, nl1a)**2 / (4 * np.pi)
            dens_nl1b = e1.Rnl(r1, nl1b)**2 / (4 * np.pi)

            integrand = (out12['v2rho2_up'] - out12['v2rho2_updown']) \
                        * dens_nl1a * dens_nl1b

            integrand -= (out1['v2rho2_up'] - out1['v2rho2_updown']) \
                         * dens_nl1a * dens_nl1b

            if xc.add_gradient_corrections:
                dnl1a = e1.Rnl(r1, nl1a) / (2 * np.pi) * e1.Rnl(r1, nl1a, der=1)
                dnl1b = e1.Rnl(r1, nl1b) / (2 * np.pi) * e1.Rnl(r1, nl1b, der=1)
                grad_nl1a_grad_nl1b = (dnl1a*s1 * dnl1b*s1) \
                                      + (dnl1a*c1 * dnl1b*c1)

                # rho12
                grad_nl1a_grad_rho12_up = (dnl1a*s1 * grad_rho12_x_up) \
                                          + (dnl1a*c1 * grad_rho12_y_up)
                grad_nl1b_grad_rho12_up = (dnl1b*s1 * grad_rho12_x_up) \
                                          + (dnl1b*c1 * grad_rho12_y_up)
                grad_nl1a_grad_rho12_down = (dnl1a*s1 * grad_rho12_x_down) \
                                            + (dnl1a*c1 * grad_rho12_y_down)
                grad_nl1b_grad_rho12_down = (dnl1b*s1 * grad_rho12_x_down) \
                                            + (dnl1b*c1 * grad_rho12_y_down)

                ## up-up contribution
                integrand += 2. * out12['v2rhosigma_up_up'] \
                             * (grad_nl1a_grad_rho12_up * dens_nl1b \
                                + dens_nl1a * grad_nl1b_grad_rho12_up)

                integrand += 4. * out12['v2sigma2_up_up'] \
                             * grad_nl1a_grad_rho12_up \
                             * grad_nl1b_grad_rho12_up
                integrand += 2. * out12['v2sigma2_up_updown'] \
                             * grad_nl1a_grad_rho12_down \
                             * grad_nl1b_grad_rho12_up
                integrand += 2. * out12['v2sigma2_up_updown'] \
                             * grad_nl1a_grad_rho12_up \
                             * grad_nl1b_grad_rho12_down
                integrand += 1. * out12['v2sigma2_updown_updown'] \
                             * grad_nl1a_grad_rho12_down \
                             * grad_nl1b_grad_rho12_down
                integrand += 2. * out12['vsigma_up'] * grad_nl1a_grad_nl1b

                ## up-down contrbution
                integrand -= 2. * out12['v2rhosigma_up_down'] \
                             * (grad_nl1a_grad_rho12_down * dens_nl1b \
                                + dens_nl1a * grad_nl1b_grad_rho12_down)

                integrand -= 4. * out12['v2sigma2_up_down'] \
                             * grad_nl1a_grad_rho12_down \
                             * grad_nl1b_grad_rho12_up
                integrand -= 2. * out12['v2sigma2_up_updown'] \
                             * grad_nl1a_grad_rho12_down \
                             * grad_nl1b_grad_rho12_up
                integrand -= 2. * out12['v2sigma2_updown_down'] \
                             * grad_nl1a_grad_rho12_down \
                             * grad_nl1b_grad_rho12_down
                integrand -= 1. * out12['v2sigma2_updown_updown'] \
                             * grad_nl1a_grad_rho12_up \
                             * grad_nl1b_grad_rho12_down
                integrand -= 1. * out12['vsigma_updown'] * grad_nl1a_grad_nl1b

                # rho1
                grad_nl1a_grad_rho1_up = (dnl1a*s1 * grad_rho1_x_up) \
                                          + (dnl1a*c1 * grad_rho1_y_up)
                grad_nl1b_grad_rho1_up = (dnl1b*s1 * grad_rho1_x_up) \
                                          + (dnl1b*c1 * grad_rho1_y_up)
                grad_nl1a_grad_rho1_down = (dnl1a*s1 * grad_rho1_x_down) \
                                            + (dnl1a*c1 * grad_rho1_y_down)
                grad_nl1b_grad_rho1_down = (dnl1b*s1 * grad_rho1_x_down) \
                                            + (dnl1b*c1 * grad_rho1_y_down)

                ## up-up contribution
                integrand -= 2. * out1['v2rhosigma_up_up'] \
                             * (grad_nl1a_grad_rho1_up * dens_nl1b \
                                + dens_nl1a * grad_nl1b_grad_rho1_up)

                integrand -= 4. * out1['v2sigma2_up_up'] \
                             * grad_nl1a_grad_rho1_up \
                             * grad_nl1b_grad_rho1_up
                integrand -= 2. * out1['v2sigma2_up_updown'] \
                             * grad_nl1a_grad_rho1_down \
                             * grad_nl1b_grad_rho1_up
                integrand -= 2. * out1['v2sigma2_up_updown'] \
                             * grad_nl1a_grad_rho1_up \
                             * grad_nl1b_grad_rho1_down
                integrand -= 1. * out1['v2sigma2_updown_updown'] \
                             * grad_nl1a_grad_rho1_down \
                             * grad_nl1b_grad_rho1_down
                integrand -= 2. * out1['vsigma_up'] * grad_nl1a_grad_nl1b

                ## up-down contrbution
                integrand += 2. * out1['v2rhosigma_up_down'] \
                             * (grad_nl1a_grad_rho1_down * dens_nl1b \
                                + dens_nl1a * grad_nl1b_grad_rho1_down)

                integrand += 4. * out1['v2sigma2_up_down'] \
                             * grad_nl1a_grad_rho1_down \
                             * grad_nl1b_grad_rho1_up
                integrand += 2. * out1['v2sigma2_up_updown'] \
                             * grad_nl1a_grad_rho1_down \
                             * grad_nl1b_grad_rho1_up
                integrand += 2. * out1['v2sigma2_updown_down'] \
                             * grad_nl1a_grad_rho1_down \
                             * grad_nl1b_grad_rho1_down
                integrand += 1. * out1['v2sigma2_updown_updown'] \
                             * grad_nl1a_grad_rho1_up \
                             * grad_nl1b_grad_rho1_down
                integrand += 1. * out1['vsigma_updown'] * grad_nl1a_grad_nl1b

            W[key] = np.sum(integrand * aux) / 2.

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


class Onsite2cWAuxiliaryTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='monopolar',
                                     **kwargs)
        assert self.ela.aux_basis.get_lmax() < NUML_2CK
        assert self.elb.aux_basis.get_lmax() < NUML_2CK

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA', smoothen_tails=True):
        """
        Calculates onsite, orbital- and distance-dependent "W" values
        as matrix elements of the two-center-expanded spin-polarized XC
        kernel and the auxiliary basis set, in a multipole expansion.

        Parameters
        ----------
        See Onsite2cTable.run().
        """
        self.print_header()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_onsiteW')
        wf_range = self.get_range(wflimit)
        grid, area = self.make_grid(wf_range, nt=ntheta, nr=nr)
        self.Rgrid = rmin + dr * np.arange(N)

        e1, e2 = self.ela, self.elb

        selected = e1.aux_basis.select_radial_functions()
        print('Selected subshells:', selected, file=self.txt)

        self.tables = {}
        for bas1a in range(e1.aux_basis.get_nzeta()):
            for bas1b in range(e1.aux_basis.get_nzeta()):
                self.tables[(bas1a, bas1b)] = np.zeros((N, NUMSK_2CK))

        for i, R in enumerate(self.Rgrid):
            if R > 2 * wf_range:
                break

            if i == N - 1 or N // 10 == 0 or i % (N // 10) == 0:
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            W = self.calculate(e1, e2, R, grid, area, selected, xc=xc)

            for key in W:
                nl1a, nl1b = key
                bas1a = e1.aux_basis.get_zeta_index(nl1a)
                bas1b = e1.aux_basis.get_zeta_index(nl1b)
                self.tables[(bas1a, bas1b)][i, :] = W[key][:]

        if smoothen_tails:
            # Smooth the curves near the cutoff
            for key in self.tables:
                for i in range(NUMSK_2CK):
                    self.tables[key][:, i] = tail_smoothening(self.Rgrid,
                                                         self.tables[key][:, i])

        self.timer.stop('run_onsiteW')
        return

    def calculate(self, e1, e2, R, grid, area, selected, xc='LDA'):
        """
        Calculates the selected integrals involving the spin-polarized
        XC kernel.

        Parameters
        ----------
        selected : list
            List of subshells to use as radial functions.

        Other parameters
        ----------------
        See Onsite2cTable.calculate().

        Returns
        -------
        W : dict of np.ndarray
            Dictionary containing the needed Slater-Koster integrals.
        """
        self.timer.start('calculate_onsiteW')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        c1 = y / r1  # cosine of theta_1
        s1 = x / r1  # sine of theta_1
        aux = area * x

        xc = LibXC('LDA_X+LDA_C_PW' if xc == 'LDA' else xc,
                   spin_polarized=True)

        rho1 = e1.electron_density(r1) / 2
        rho2 = e2.electron_density(r2) / 2
        rho12 = rho1 + rho2
        Anl1 = {(nl, l): e1.aux_basis(r1, nl, l) for nl in selected
                for l in e1.aux_basis.get_angular_momenta()}

        if xc.add_gradient_corrections:
            drho1 = e1.electron_density(r1, der=1) / 2
            drho2 = e2.electron_density(r2, der=1) / 2
            c2 = (y - R) / r2  # cosine of theta_2
            s2 = x / r2  # sine of theta_2

            drho1dx = drho1 * s1
            drho1dy = drho1 * c1
            sigma1 = drho1dx**2 + drho1dy**2

            drho12dx = drho1 * s1 + drho2 * s2
            drho12dy = drho1 * c1 + drho2 * c2
            sigma12 = drho12dx**2 + drho12dy**2

            dr1dx = x/r1
            ds1dx = (r1 - x*dr1dx) / r1**2
            dr1dy = y/r1
            ds1dy = -x*dr1dy / r1**2
            dtheta1dx = ds1dx / c1
            dtheta1dy = ds1dy / c1

            grad_r1_grad_rho12 = dr1dx * drho12dx + dr1dy * drho12dy
            grad_theta1_grad_rho12 = dtheta1dx * drho12dx + dtheta1dy * drho12dy
            grad_r1_grad_theta1 = dr1dx * dtheta1dx + dr1dy * dtheta1dy

            grad_r1_grad_rho1 = dr1dx * drho1dx + dr1dy * drho1dy
            grad_theta1_grad_rho1 = dtheta1dx * drho1dx + dtheta1dy * drho1dy

            dAnl1dr1 = {(nl, l): e1.aux_basis(r1, nl, l, der=1)
                        for nl in selected
                        for l in e1.aux_basis.get_angular_momenta()}
        else:
            sigma1 = None
            sigma12 = None
        self.timer.stop('prelude')

        self.timer.start('fxc')
        out1 = xc.compute_vxc_polarized(rho1, rho1, sigma_up=sigma1,
                                        sigma_updown=sigma1,
                                        sigma_down=sigma1, fxc=True)
        out12 = xc.compute_vxc_polarized(rho12, rho12, sigma_up=sigma12,
                                         sigma_updown=sigma12,
                                         sigma_down=sigma12, fxc=True)
        self.timer.stop('fxc')

        keys = [(nl1a, nl1b) for nl1a in selected for nl1b in selected]
        W = {key: np.zeros(NUMSK_2CK) for key in keys}

        for index, integral in enumerate(INTEGRALS_2CK):
            lm1a, lm1b = get_integral_pair(integral)
            gphi = get_twocenter_phi_integral(lm1a, lm1b, c1, c1, s1, s1)

            if xc.add_gradient_corrections:
                dgphi = get_twocenter_phi_integrals_derivatives(lm1a, lm1b, c1,
                                                                c1, s1, s1)

            l1a = ANGULAR_MOMENTUM[lm1a[0]]
            l1b = ANGULAR_MOMENTUM[lm1b[0]]
            if l1a > e1.aux_basis.get_lmax() or l1b > e1.aux_basis.get_lmax():
                continue

            for key in keys:
                nl1a, nl1b = key

                integrand = (out12['v2rho2_up'] - out12['v2rho2_updown']) \
                            * Anl1[(nl1a, l1a)] * Anl1[(nl1b, l1b)] * gphi
                integrand -= (out1['v2rho2_up'] - out1['v2rho2_updown']) \
                             * Anl1[(nl1a, l1a)] * Anl1[(nl1b, l1b)] * gphi

                if xc.add_gradient_corrections:
                    products = dAnl1dr1[(nl1a, l1a)] * grad_r1_grad_rho12 \
                               * Anl1[(nl1b, l1b)] * gphi
                    products += dAnl1dr1[(nl1b, l1b)] * grad_r1_grad_rho12 \
                                * Anl1[(nl1a, l1a)] * gphi
                    products += Anl1[(nl1a, l1a)] * Anl1[(nl1b, l1b)] \
                                * grad_theta1_grad_rho12 * dgphi[2]
                    products += Anl1[(nl1b, l1b)] * Anl1[(nl1a, l1a)] \
                                * grad_theta1_grad_rho12 * dgphi[3]
                    integrand += 2 * products * (out12['v2rhosigma_up_up'] \
                                                 - out12['v2rhosigma_up_down'])

                    products = dAnl1dr1[(nl1a, l1a)] * grad_r1_grad_rho1 \
                               * Anl1[(nl1b, l1b)] * gphi
                    products += dAnl1dr1[(nl1b, l1b)] * grad_r1_grad_rho1 \
                                * Anl1[(nl1a, l1a)] * gphi
                    products += Anl1[(nl1a, l1a)] * Anl1[(nl1b, l1b)] \
                                * grad_theta1_grad_rho1 * dgphi[2]
                    products += Anl1[(nl1b, l1b)] * Anl1[(nl1a, l1a)] \
                                * grad_theta1_grad_rho1 * dgphi[3]
                    integrand -= 2 * products * (out1['v2rhosigma_up_up'] \
                                                 - out1['v2rhosigma_up_down'])

                    products = dAnl1dr1[(nl1a, l1a)] * dAnl1dr1[(nl1b, l1b)] \
                               * grad_r1_grad_rho12**2 * gphi
                    products += dAnl1dr1[(nl1b, l1b)] * grad_r1_grad_rho12 \
                                * Anl1[(nl1a, l1a)] * grad_theta1_grad_rho12 \
                                * dgphi[2]
                    products += dAnl1dr1[(nl1a, l1a)] * grad_r1_grad_rho12 \
                                * Anl1[(nl1b, l1b)] * grad_theta1_grad_rho12 \
                                * dgphi[3]
                    products += Anl1[(nl1a, l1a)] * Anl1[(nl1b, l1b)] \
                                * grad_theta1_grad_rho12**2 * dgphi[0]
                    integrand += 4 * products * (out12['v2sigma2_up_up'] \
                                                 - out12['v2sigma2_up_down'])
                    integrand += 2 * products * (out12['v2sigma2_up_updown'] \
                                             - out12['v2sigma2_updown_down'])

                    products = dAnl1dr1[(nl1a, l1a)] * dAnl1dr1[(nl1b, l1b)] \
                               * grad_r1_grad_rho1**2 * gphi
                    products += dAnl1dr1[(nl1b, l1b)] * grad_r1_grad_rho1 \
                                * Anl1[(nl1a, l1a)] * grad_theta1_grad_rho1 \
                                * dgphi[2]
                    products += dAnl1dr1[(nl1a, l1a)] * grad_r1_grad_rho1 \
                                * Anl1[(nl1b, l1b)] * grad_theta1_grad_rho1 \
                                * dgphi[3]
                    products += Anl1[(nl1a, l1a)] * Anl1[(nl1b, l1b)] \
                                * grad_theta1_grad_rho1**2 * dgphi[0]
                    integrand -= 4 * products * (out1['v2sigma2_up_up'] \
                                                 - out1['v2sigma2_up_down'])
                    integrand -= 2 * products * (out1['v2sigma2_up_updown'] \
                                                 - out1['v2sigma2_updown_down'])

                    products = dAnl1dr1[(nl1a, l1a)] * dAnl1dr1[(nl1b, l1b)] \
                               * (dr1dx**2 + dr1dy**2) * gphi
                    products += Anl1[(nl1a, l1a)] * grad_r1_grad_theta1 \
                                * dAnl1dr1[(nl1b, l1b)] * dgphi[2]
                    products += Anl1[(nl1b, l1b)] * grad_r1_grad_theta1 \
                                * dAnl1dr1[(nl1a, l1a)] * dgphi[3]
                    products += Anl1[(nl1a, l1a)] * Anl1[(nl1b, l1b)] \
                                * ((dtheta1dx**2 + dtheta1dy**2) * dgphi[0] \
                                   + dgphi[1] / x**2)
                    integrand += products * (2 * out12['vsigma_up'] \
                                             - out12['vsigma_updown'])
                    integrand -= products * (2 * out1['vsigma_up'] \
                                             - out1['vsigma_updown'])

                W[key][index] = np.sum(integrand * aux) / 2

        self.timer.stop('calculate_onsiteW')
        return W

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el1>_onsiteW_<el2>.2ck'.
        """
        sym1, sym2 = self.ela.get_symbol(), self.elb.get_symbol()

        for bas1a in range(self.ela.aux_basis.get_nzeta()):
            for bas1b in range(self.ela.aux_basis.get_nzeta()):
                template = '%s-%s_onsiteW_%s.2ck'
                filename = template % (sym1+'+'*bas1a, sym1+'+'*bas1b, sym2)
                print('Writing to %s' % filename, file=self.txt, flush=True)

                with open(filename, 'w') as f:
                    write_2ck(f, self.Rgrid, self.tables[(bas1a, bas1b)])
        return
