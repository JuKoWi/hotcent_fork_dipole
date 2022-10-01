#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.fluctuation_onecenter import (
                NUML_1CK, select_radial_function, write_1ck)
from hotcent.fluctuation_twocenter import (
                INTEGRALS_2CK, NUMINT_2CL, NUMSK_2CK, select_subshells,
                write_2cl, write_2ck)
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.slako import (get_integral_pair, get_twocenter_phi_integral,
                           get_twocenter_phi_integrals_derivatives,
                           tail_smoothening)
from hotcent.xc import LibXC


class Onsite2cWTable:
    """
    Convenience wrapper around the Onsite2cWMonopoleTable
    and Onsite2cWMultipoleTable classes.

    Parameters
    ----------
    use_multipoles : bool
        Whether to consider a multipole expansion of the magnetization
        density (rather than a monopole approximation).
    """
    def __init__(self, *args, use_multipoles=None, **kwargs):
        msg = '"use_multipoles" is required and must be either True or False'
        assert use_multipoles is not None, msg

        if use_multipoles:
            self.calc = Onsite2cWMultipoleTable(*args, **kwargs)
        else:
            self.calc = Onsite2cWMonopoleTable(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.calc, attr)

    def run(self, *args, **kwargs):
        self.calc.run(*args, **kwargs)
        return

    def write(self, *args, **kwargs):
        self.calc.write(*args, **kwargs)
        return


class Onsite2cWMonopoleTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='monopolar',
                                     **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA', smoothen_tails=True):
        """
        Calculates onsite, distance dependent "W" values as matrix
        elements of the two-center-expanded spin-polarized XC kernel
        in the monopole approximation.

        Parameters
        ----------
        See Onsite2cTable.run().
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Monopole onsite-W table construction for %s and %s' % \
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


class Onsite1cWTable:
    """
    Calculator for (parts of) the onsite, one-center "W" integrals
    as matrix elements of the spin-polarized XC kernel.

    Parameters
    ----------
    el : AtomicBase-like object
        Object with atomic properties.
    txt : None or filehandle
        Where to print output to (None for stdout).
    """
    def __init__(self, el, txt=None):
        self.el = el
        self.txt = txt

    def run(self, nl=None, xc='LDA'):
        """
        Calculates onsite, one-center "W" values.

        Parameters
        ----------
        nl : str, optional
            Subshell defining the radial function. If None, the subshell
            with the lowest angular momentum will be chosen from the
            minimal valence set.
        xc : str, optional
            Name of the exchange-correlation functional (default: LDA).
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Multipole onsite-W table construction for %s' % \
              self.el.get_symbol(), file=self.txt)
        print('***********************************************', file=self.txt)

        if nl is None:
            nl = select_radial_function(self.el)

        self.table, self.radmom = self.calculate(nl, xc=xc)
        return

    def calculate(self, nl, xc='LDA'):
        """
        Calculates the selected integrals involving the spin-polarized
        XC kernel.

        Parameters
        ----------
        nl : str
            Subshell defining the radial function.
        xc : str, optional
            Name of the exchange-correlation functional (default: LDA).

        Returns
        -------
        W: np.ndarray
            Array with the integral for each multipole.
        radmom : np.ndarray
            Array with the radial moments of the associated density
            (\int R_{nl}^2 r^{l+2} dr, l = 0, 1, ...).
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

        Rnl = np.copy(self.el.Rnlg[nl])
        dens_nl = Rnl**2

        integrand = (out['v2rho2_up'] - out['v2rho2_updown']) * dens_nl**2
        W[:] = self.el.grid.integrate(integrand * self.el.rgrid**2,
                                      use_dV=False)


        if xc.add_gradient_corrections:
            dnl = 2 * Rnl * self.el.Rnl(self.el.rgrid, nl, der=1)
            grad_nl_grad_rho = dnl * drho

            products =  2 * grad_nl_grad_rho * dens_nl
            integrand = 2. * products \
                        * (out['v2rhosigma_up_up'] - out['v2rhosigma_up_down'])

            products = grad_nl_grad_rho**2
            integrand += 4. * products \
                         * (out['v2sigma2_up_up'] - out['v2sigma2_up_down'])
            integrand += 2. * products \
                     * (out['v2sigma2_up_updown'] - out['v2sigma2_updown_down'])

            products = dnl**2
            integrand += products \
                         * (2. * out['vsigma_up'] - out['vsigma_updown'])

            W[:] += self.el.grid.integrate(integrand * self.el.rgrid**2,
                                           use_dV=False)

            for l in range(NUML_1CK):
                integrand = (2. * out['vsigma_up'] - out['vsigma_updown']) \
                            * dens_nl**2
                W[l] += self.el.grid.integrate(integrand * l * (l+1),
                                               use_dV=False)

        for l in range(NUML_1CK):
            radmom[l] = self.el.grid.integrate(dens_nl * self.el.rgrid**(l+2),
                                               use_dV=False)

        W /= 2
        return (W, radmom)

    def write(self):
        """
        Writes the integrals to file.

        The filename template corresponds to '<el1>-<el1>_onsiteW.1ck'.
        """
        sym = self.el.get_symbol()
        template = '%s-%s_onsiteW.1ck'
        filename = template % (sym, sym)
        print('Writing to %s' % filename, file=self.txt, flush=True)

        with open(filename, 'w') as f:
            write_1ck(f, self.radmom, self.table)
        return


class Onsite2cWMultipoleTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='monopolar',
                                     **kwargs)

    def run(self, nl=None, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50,
            wflimit=1e-7, xc='LDA', smoothen_tails=True):
        """
        Calculates onsite, orbital- and distance-dependent "W" values
        as matrix elements of the two-center-expanded spin-polarized XC
        kernel in a multipole expansion.

        Parameters
        ----------
        nl : str, optional
            Subshell defining the radial function. If None, the subshell
            with the lowest angular momentum will be chosen from the
            minimal valence set.

        Other parameters
        ----------------
        See Onsite2cTable.run().
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Multipole onsite-W table construction for %s and %s' % \
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

        e1, e2 = self.ela, self.elb
        self.tables = np.zeros((N, NUMSK_2CK))

        if nl is None:
            nl = select_radial_function(self.ela)

        for i, R in enumerate(self.Rgrid):
            if R > 2 * wf_range:
                break

            if i == N - 1 or N // 10 == 0 or i % (N // 10) == 0:
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            W = self.calculate(e1, e2, R, grid, area, nl, xc=xc)
            self.tables[i, :] = W[:]

        if smoothen_tails:
            # Smooth the curves near the cutoff
            for i in range(NUMSK_2CK):
                self.tables[:, i] = tail_smoothening(self.Rgrid,
                                                     self.tables[:, i])

        self.timer.stop('run_onsiteW')
        return

    def calculate(self, e1, e2, R, grid, area, nl, xc='LDA'):
        """
        Calculates the selected integrals involving the spin-polarized
        XC kernel.

        Parameters
        ----------
        nl : str
            Subshell defining the radial function.

        Other parameters
        ----------------
        See Onsite2cTable.calculate().

        Returns
        -------
        W : np.ndarray
            Array containing the needed Slater-Koster integrals.
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
        Rnl1 = e1.Rnl(r1, nl)**2

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

            dRnl1dr1 = 2 * e1.Rnl(r1, nl) * e1.Rnl(r1, nl, der=1)
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

        W = np.zeros(NUMSK_2CK)

        for index, integral in enumerate(INTEGRALS_2CK):
            lm1a, lm1b = get_integral_pair(integral)
            gphi = get_twocenter_phi_integral(lm1a, lm1b, c1, c1, s1, s1)

            integrand = (out12['v2rho2_up'] - out12['v2rho2_updown']) \
                        * Rnl1**2 * gphi
            integrand -= (out1['v2rho2_up'] - out1['v2rho2_updown']) \
                         * Rnl1**2 * gphi

            if xc.add_gradient_corrections:
                dgphi = get_twocenter_phi_integrals_derivatives(lm1a, lm1b, c1,
                                                                c1, s1, s1)

                products = 2 * dRnl1dr1 * grad_r1_grad_rho12 * Rnl1 * gphi
                products += Rnl1**2 * grad_theta1_grad_rho12 \
                            * (dgphi[2] + dgphi[3])
                integrand += 2 * products * (out12['v2rhosigma_up_up'] \
                                             - out12['v2rhosigma_up_down'])

                products = 2 * dRnl1dr1 * grad_r1_grad_rho1 * Rnl1 * gphi
                products += Rnl1**2 * grad_theta1_grad_rho1 \
                            * (dgphi[2] + dgphi[3])
                integrand -= 2 * products * (out1['v2rhosigma_up_up'] \
                                             - out1['v2rhosigma_up_down'])

                products = dRnl1dr1**2 * grad_r1_grad_rho12**2 * gphi
                products += dRnl1dr1 * grad_r1_grad_rho12 * Rnl1 \
                            * grad_theta1_grad_rho12 * (dgphi[2] + dgphi[3])
                products += Rnl1**2 * grad_theta1_grad_rho12**2 * dgphi[0]
                integrand += 4 * products * (out12['v2sigma2_up_up'] \
                                             - out12['v2sigma2_up_down'])
                integrand += 2 * products * (out12['v2sigma2_up_updown'] \
                                             - out12['v2sigma2_updown_down'])

                products = dRnl1dr1**2 * grad_r1_grad_rho1**2 * gphi
                products += dRnl1dr1 * grad_r1_grad_rho1 * Rnl1 \
                            * grad_theta1_grad_rho1 * (dgphi[2] + dgphi[3])
                products += Rnl1**2 * grad_theta1_grad_rho1**2 * dgphi[0]
                integrand -= 4 * products * (out1['v2sigma2_up_up'] \
                                             - out1['v2sigma2_up_down'])
                integrand -= 2 * products * (out1['v2sigma2_up_updown'] \
                                             - out1['v2sigma2_updown_down'])

                products = dRnl1dr1**2 * (dr1dx**2 + dr1dy**2) * gphi
                products += Rnl1 * grad_r1_grad_theta1 * dRnl1dr1 \
                            * (dgphi[2] + dgphi[3])
                products += Rnl1**2 * ((dtheta1dx**2 + dtheta1dy**2) * dgphi[0]\
                                       + dgphi[1] / x**2)
                integrand += products * (2 * out12['vsigma_up'] \
                                         - out12['vsigma_updown'])
                integrand -= products * (2 * out1['vsigma_up'] \
                                         - out1['vsigma_updown'])

            W[index] = np.sum(integrand * aux)

        W /= 2
        self.timer.stop('calculate_onsiteW')
        return W

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el1>_onsiteW_<el2>.2ck'.
        """
        sym1, sym2 = self.ela.get_symbol(), self.elb.get_symbol()
        template = '%s-%s_onsiteW_%s.2ck'
        filename = template % (sym1, sym1, sym2)
        print('Writing to %s' % filename, file=self.txt, flush=True)

        with open(filename, 'w') as f:
            write_2ck(f, self.Rgrid, self.tables)
        return
