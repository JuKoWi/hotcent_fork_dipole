#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.fluctuation_onecenter import (
                NUML_1CK, NUML_1CM, select_radial_functions,
                write_1ck, write_1cm)
from hotcent.fluctuation_twocenter import (
                INTEGRALS_2CK, NUMINT_2CL, NUMSK_2CK, select_subshells,
                write_2cl, write_2ck)
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.orbital_hartree import OrbitalHartreePotential
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.slako import (get_integral_pair, get_twocenter_phi_integral,
                           get_twocenter_phi_integrals_derivatives,
                           tail_smoothening)
from hotcent.xc import LibXC


class Onsite2cUTable:
    """
    Convenience wrapper around the Onsite2cUMonopoleTable
    and Onsite2cUMultipoleTable classes.

    Parameters
    ----------
    use_multipoles : bool
        Whether to consider a multipole expansion of the difference
        density (rather than a monopole approximation).
    """
    def __init__(self, *args, use_multipoles=None, **kwargs):
        msg = '"use_multipoles" is required and must be either True or False'
        assert use_multipoles is not None, msg

        if use_multipoles:
            self.calc = Onsite2cUMultipoleTable(*args, **kwargs)
        else:
            self.calc = Onsite2cUMonopoleTable(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.calc, attr)

    def run(self, *args, **kwargs):
        self.calc.run(*args, **kwargs)
        return

    def write(self, *args, **kwargs):
        self.calc.write(*args, **kwargs)
        return


class Onsite2cUMonopoleTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='monopolar',
                                     **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA', smoothen_tails=True):
        """
        Calculates on-site, distance dependent U (or "Gamma") values
        as matrix elements of the two-center-expanded XC kernel
        in the monopole approximation.

        Parameters
        ----------
        See Onsite2cTable.run().
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Monopole onsite-U table construction for %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_onsiteU')
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

            U = self.calculate(selected, e1, e2, R, grid, area, xc=xc)

            for key in selected:
                nl1a, nl1b = key
                bas1a = e1.get_basis_set_index(nl1a)
                bas1b = e1.get_basis_set_index(nl1b)
                index = ANGULAR_MOMENTUM[nl1a[1]] * 4
                index += ANGULAR_MOMENTUM[nl1b[1]]
                self.tables[(bas1a, bas1b)][i, index] = U[key]

        if smoothen_tails:
            # Smooth the curves near the cutoff
            for key in self.tables:
                for i in range(NUMINT_2CL):
                    self.tables[key][:, i] = \
                            tail_smoothening(self.Rgrid, self.tables[key][:, i])

        self.timer.stop('run_onsiteU')

    def calculate(self, selected, e1, e2, R, grid, area, xc='LDA'):
        """
        Calculates the selected integrals involving the XC kernel.

        Parameters
        ----------
        See Onsite2cTable.calculate().

        Returns
        -------
        U : dict
            Dictionary containing the integral for each selected
            subshell pair.
        """
        self.timer.start('calculate_onsiteU')

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

        U = {}
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

            U[key] = np.sum(integrand * aux)

        self.timer.stop('calculate_onsiteU')
        return U

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el1>_onsiteU_<el2>.2cl'.
        """
        sym1, sym2 = self.ela.get_symbol(), self.elb.get_symbol()

        for bas1a, valence1a in enumerate(self.ela.basis_sets):
            angmom1a = [ANGULAR_MOMENTUM[nl[1]] for nl in valence1a]

            for bas1b, valence1b in enumerate(self.ela.basis_sets):
                angmom1b = [ANGULAR_MOMENTUM[nl[1]] for nl in valence1b]

                template = '%s-%s_onsiteU_%s.2cl'
                filename = template % (sym1 + '+'*bas1a, sym1 + '+'*bas1b, sym2)
                print('Writing to %s' % filename, file=self.txt, flush=True)

                table = self.tables[(bas1a, bas1b)]
                with open(filename, 'w') as f:
                    write_2cl(f, self.Rgrid, table, angmom1a, angmom1b)
        return


class Onsite1cUTable:
    """
    Calculator for (parts of) the on-site, one-center "U" integrals
    as matrix elements of the Hartree-XC kernel.

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

    def run(self, subshells=None, xc='LDA'):
        """
        Calculates on-site, one-center "U" values.

        Parameters
        ----------
        subshells : list, optional
            Specific subshells to use as radial functions (one for
            every 'basis subset'). By default, the subshell with lowest
            angular momentum is chosen from each subset.
        xc : str, optional
            Name of the exchange-correlation functional (default: LDA).
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Multipole onsite-U table construction for %s' % \
              self.el.get_symbol(), file=self.txt)
        print('***********************************************', file=self.txt)

        self.tables = {}
        for bas1 in range(len(self.el.basis_sets)):
            for bas2 in range(len(self.el.basis_sets)):
                self.tables[(bas1, bas2)] = np.zeros((2, NUML_1CK))

        if subshells is None:
            selected = select_radial_functions(self.el)
        else:
            assert len(subshells) == len(self.el.basis_sets), \
                   'Expecting one subshell per basis subset'
            selected = subshells
        print('Selected subshells:', selected, file=self.txt)

        for nl1 in selected:
            for nl2 in selected:
                U, radmom = self.calculate(nl1, nl2, xc=xc)
                bas1 = self.el.get_basis_set_index(nl1)
                bas2 = self.el.get_basis_set_index(nl2)
                self.tables[(bas1, bas2)][0, :] = U
                self.tables[(bas1, bas2)][1, :] = radmom
        return

    def calculate(self, nl1, nl2, xc='LDA'):
        """
        Calculates the selected integrals involving the Hartree-XC kernel.

        Parameters
        ----------
        nl1, nl2 : str
            Subshells defining the radial functions.
        xc : str, optional
            Name of the exchange-correlation functional (default: LDA).

        Returns
        -------
        U: np.ndarray
            Array with the integral for each multipole.
        radmom : np.ndarray
            Array with the radial moments of the associated density
            (\int R_{nl}^2 r^{l+2} dr, l = 0, 1, ...).
        """
        xc = LibXC('LDA_X+LDA_C_PW' if xc == 'LDA' else xc)
        rho = self.el.electron_density(self.el.rgrid)

        if xc.add_gradient_corrections:
            drho = self.el.electron_density(self.el.rgrid, der=1)
            sigma = drho**2
        else:
            sigma = None

        out = xc.compute_vxc(rho, sigma=sigma, fxc=True)

        U = np.zeros(NUML_1CK)
        radmom = np.zeros(NUML_1CK)

        Rnl1 = np.copy(self.el.Rnlg[nl1])
        dens_nl1 = Rnl1**2
        Rnl2 = np.copy(self.el.Rnlg[nl2])
        dens_nl2 = Rnl2**2

        integrand = out['v2rho2'] * dens_nl1 * dens_nl2
        U[:] = self.el.grid.integrate(integrand * self.el.rgrid**2,
                                      use_dV=False)

        if xc.add_gradient_corrections:
            dnl1 = 2 * Rnl1 * self.el.Rnl(self.el.rgrid, nl1, der=1)
            dnl2 = 2 * Rnl2 * self.el.Rnl(self.el.rgrid, nl2, der=1)
            grad_nl1_grad_rho = dnl1 * drho
            grad_nl2_grad_rho = dnl2 * drho
            integrand = 2. * out['v2rhosigma'] * grad_nl1_grad_rho * dens_nl2
            integrand += 2. * out['v2rhosigma'] * grad_nl2_grad_rho * dens_nl1
            integrand += 4. * out['v2sigma2'] * grad_nl1_grad_rho \
                         * grad_nl2_grad_rho
            integrand += 2. * out['vsigma'] * dnl1 * dnl2
            U[:] += self.el.grid.integrate(integrand * self.el.rgrid**2,
                                           use_dV=False)

            for l in range(NUML_1CK):
                integrand = 2. * out['vsigma'] * dens_nl1 * dens_nl2
                U[l] += self.el.grid.integrate(integrand * l * (l+1),
                                               use_dV=False)

        for l in range(NUML_1CK):
            ohp = OrbitalHartreePotential(self.el.rmin, self.el.xgrid,
                                          dens_nl1, lmax=NUML_1CK-1)
            vhar = ohp.vhar_fct[l](self.el.rgrid)
            integrand = vhar * dens_nl2 * self.el.rgrid**2
            U[l] += self.el.grid.integrate(integrand, use_dV=False)

            integrand = Rnl1 * Rnl2 * self.el.rgrid**(l+2)
            radmom[l] = self.el.grid.integrate(integrand, use_dV=False)

        return (U, radmom)

    def write(self):
        """
        Writes the integrals to file.

        The filename template corresponds to '<el>-<el>_onsiteU.1ck'.
        """
        sym = self.el.get_symbol()

        for bas1, valence1 in enumerate(self.el.basis_sets):
            for bas2, valence2 in enumerate(self.el.basis_sets):
                template = '%s-%s_onsiteU.1ck'
                filename = template % (sym + '+'*bas1, sym + '+'*bas2)
                print('Writing to %s' % filename, file=self.txt, flush=True)

                table = self.tables[(bas1, bas2)]
                with open(filename, 'w') as f:
                    write_1ck(f, table[1, :], table[0, :])
        return


class Onsite1cMTable:
    """
    Class for calculations involving on-site moment integrals "M"
    (Boleininger, Guilbert and Horsfield (2016), doi:10.1063/1.4964391):

    $M_{\mu,\nu,lm} = \int \phi_\mu(\mathbf{r}) Y_{lm}(\mathbf{r})
                           \phi_\nu(\mathbf{r}) d\mathbf{r}$

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

    def run(self):
        """
        Calculates the on-site, subshell-dependent integrals

        $\int R_{nl1}(r) R_{nl2}(r) r^2 dr$

        from which the moment integrals M can be obtained
        by multiplication with the appropriate Gaunt coefficients.
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Multipole onsite-M table construction for %s' % \
              self.el.get_symbol(), file=self.txt)
        print('***********************************************', file=self.txt)

        selected = select_subshells(self.el, self.el)

        self.tables = {}
        for bas1 in range(len(self.el.basis_sets)):
            for bas2 in range(len(self.el.basis_sets)):
                shape = (NUML_1CM, NUML_1CM)
                self.tables[(bas1, bas2)] = np.zeros(shape)

        M = self.calculate(selected)

        for key in selected:
            nl1, nl2 = key
            bas1 = self.el.get_basis_set_index(nl1)
            bas2 = self.el.get_basis_set_index(nl2)
            i = ANGULAR_MOMENTUM[nl1[1]]
            j = ANGULAR_MOMENTUM[nl2[1]]
            self.tables[(bas1, bas2)][i, j] = M[key]
            if i != j or bas1 != bas2:
                self.tables[(bas2, bas1)][j, i] = M[key]
        return

    def calculate(self, selected):
        """
        Calculates the selected integrals.

        Parameters
        ----------
        selected : list of 2-tuples of 2-tuples
            Sets of subshell pairs to evaluate.

        Returns
        -------
        M: dict
            Dictionary containing the integral for each selected
            subshell pair.
        """
        M = {}
        for key in selected:
            nl1, nl2 = key
            Rnl1 = np.copy(self.el.Rnlg[nl1])
            Rnl2 = np.copy(self.el.Rnlg[nl2])
            integrand = Rnl1 * Rnl2 * self.el.rgrid**2
            M[key] = self.el.grid.integrate(integrand, use_dV=False)
        return M

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el1>_onsiteM.1cm'.
        """
        sym = self.el.get_symbol()

        for bas1, valence1 in enumerate(self.el.basis_sets):
            for bas2, valence2 in enumerate(self.el.basis_sets):
                template = '%s-%s_onsiteM.1cm'
                filename = template % (sym + '+'*bas1, sym + '+'*bas2)
                print('Writing to %s' % filename, file=self.txt, flush=True)

                with open(filename, 'w') as f:
                    table = self.tables[(bas1, bas2)][:, :]
                    write_1cm(f, table)
        return


class Onsite2cUMultipoleTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='monopolar',
                                     **kwargs)

    def run(self, subshells=None, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50,
            wflimit=1e-7, xc='LDA', smoothen_tails=True):
        """
        Calculates on-site, orbital- and distance-dependent "U" values
        as matrix elements of the two-center-expanded XC kernel
        in a multipole expansion.

        Parameters
        ----------
        subshells : list, optional
            Specific subshells to use as radial functions (one for
            every 'basis subset'). By default, the subshell with lowest
            angular momentum is chosen from each subset.

        Other parameters
        ----------------
        See Onsite2cTable.run().
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Multipole onsite-U table construction for %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_onsiteU')
        wf_range = self.get_range(wflimit)
        grid, area = self.make_grid(wf_range, nt=ntheta, nr=nr)
        self.Rgrid = rmin + dr * np.arange(N)

        e1, e2 = self.ela, self.elb

        if subshells is None:
            selected = select_radial_functions(e1)
        else:
            assert len(subshells) == len(e1.basis_sets), \
                   'Need one subshell per basis subset'
            selected = subshells
        print('Selected subshells:', selected, file=self.txt)

        self.tables = {}
        for bas1a in range(len(e1.basis_sets)):
            for bas1b in range(len(e1.basis_sets)):
                self.tables[(bas1a, bas1b)] = np.zeros((N, NUMSK_2CK))

        for i, R in enumerate(self.Rgrid):
            if R > 2 * wf_range:
                break

            if i == N - 1 or N // 10 == 0 or i % (N // 10) == 0:
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            U = self.calculate(e1, e2, R, grid, area, selected, xc=xc)

            for key in U:
                nl1a, nl1b = key
                bas1a = e1.get_basis_set_index(nl1a)
                bas1b = e1.get_basis_set_index(nl1b)
                self.tables[(bas1a, bas1b)][i, :] = U[key][:]

        if smoothen_tails:
            # Smooth the curves near the cutoff
            for key in self.tables:
                for i in range(NUMSK_2CK):
                    self.tables[key][:, i] = tail_smoothening(self.Rgrid,
                                                         self.tables[key][:, i])

        self.timer.stop('run_onsiteU')
        return

    def calculate(self, e1, e2, R, grid, area, selected, xc='LDA'):
        """
        Calculates the selected integrals involving the XC kernel.

        Parameters
        ----------
        selected : dict
            List of subshells to use as radial functions
            (one for every 'basis subset') for every element.

        Other parameters
        ----------------
        See Onsite2cTable.calculate().

        Returns
        -------
        U: dict of np.ndarray
            Dictionary containing the needed Slater-Koster integrals.
        """
        self.timer.start('calculate_onsiteU')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        c1 = y / r1  # cosine of theta_1
        s1 = x / r1  # sine of theta_1
        aux = area * x

        xc = LibXC('LDA_X+LDA_C_PW' if xc == 'LDA' else xc)

        rho1 = e1.electron_density(r1)
        rho2 = e2.electron_density(r2)
        rho12 = rho1 + rho2
        Rnl1 = {nl: e1.Rnl(r1, nl)**2 for nl in selected}

        if xc.add_gradient_corrections:
            drho1 = e1.electron_density(r1, der=1)
            drho2 = e2.electron_density(r2, der=1)
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

            dRnl1dr1 = {nl: 2 * e1.Rnl(r1, nl) * e1.Rnl(r1, nl, der=1)
                        for nl in selected}
        else:
            sigma1 = None
            sigma12 = None
        self.timer.stop('prelude')

        self.timer.start('fxc')
        out1 = xc.compute_vxc(rho1, sigma=sigma1, fxc=True)
        out12 = xc.compute_vxc(rho12, sigma=sigma12, fxc=True)
        self.timer.stop('fxc')

        keys = [(nl1a, nl1b) for nl1a in selected for nl1b in selected]
        U = {key: np.zeros(NUMSK_2CK) for key in keys}

        for index, integral in enumerate(INTEGRALS_2CK):
            lm1a, lm1b = get_integral_pair(integral)
            gphi = get_twocenter_phi_integral(lm1a, lm1b, c1, c1, s1, s1)

            if xc.add_gradient_corrections:
                dgphi = get_twocenter_phi_integrals_derivatives(lm1a, lm1b, c1,
                                                                c1, s1, s1)

            for key in keys:
                nl1a, nl1b = key

                integrand = (out12['v2rho2'] - out1['v2rho2']) \
                            * Rnl1[nl1a] * Rnl1[nl1b] * gphi

                if xc.add_gradient_corrections:
                    products = dRnl1dr1[nl1a] * grad_r1_grad_rho12 \
                               * Rnl1[nl1b] * gphi
                    products += dRnl1dr1[nl1b] * grad_r1_grad_rho12 \
                                * Rnl1[nl1a] * gphi
                    products += Rnl1[nl1a] * Rnl1[nl1b] \
                                * grad_theta1_grad_rho12 * dgphi[2]
                    products += Rnl1[nl1b] * Rnl1[nl1a] \
                                * grad_theta1_grad_rho12 * dgphi[3]
                    integrand += 2 * products * out12['v2rhosigma']

                    products = dRnl1dr1[nl1a] * grad_r1_grad_rho1 \
                               * Rnl1[nl1b] * gphi
                    products += dRnl1dr1[nl1b] * grad_r1_grad_rho1 \
                                * Rnl1[nl1a] * gphi
                    products += Rnl1[nl1a] * Rnl1[nl1b] \
                                * grad_theta1_grad_rho1 * dgphi[2]
                    products += Rnl1[nl1b] * Rnl1[nl1a] \
                                * grad_theta1_grad_rho1 * dgphi[3]
                    integrand -= 2 * products * out1['v2rhosigma']

                    products = dRnl1dr1[nl1a] * dRnl1dr1[nl1b] \
                               * grad_r1_grad_rho12**2 * gphi
                    products += dRnl1dr1[nl1b] * grad_r1_grad_rho12 \
                                * Rnl1[nl1a] * grad_theta1_grad_rho12 * dgphi[2]
                    products += dRnl1dr1[nl1a] * grad_r1_grad_rho12 \
                                * Rnl1[nl1b] * grad_theta1_grad_rho12 * dgphi[3]
                    products += Rnl1[nl1a] * Rnl1[nl1b] \
                                * grad_theta1_grad_rho12**2 * dgphi[0]
                    integrand += 4 * products * out12['v2sigma2']

                    products = dRnl1dr1[nl1a] * dRnl1dr1[nl1b] \
                               * grad_r1_grad_rho1**2 * gphi
                    products += dRnl1dr1[nl1b] * grad_r1_grad_rho1 \
                                * Rnl1[nl1a] * grad_theta1_grad_rho1 * dgphi[2]
                    products += dRnl1dr1[nl1a] * grad_r1_grad_rho1 \
                                 * Rnl1[nl1b] * grad_theta1_grad_rho1 * dgphi[3]
                    products += Rnl1[nl1a] * Rnl1[nl1b] \
                                * grad_theta1_grad_rho1**2 * dgphi[0]
                    integrand -= 4 * products * out1['v2sigma2']

                    products = dRnl1dr1[nl1a] * dRnl1dr1[nl1b] \
                               * (dr1dx**2 + dr1dy**2) * gphi
                    products += Rnl1[nl1a] * grad_r1_grad_theta1 \
                                * dRnl1dr1[nl1b] * dgphi[2]
                    products += Rnl1[nl1b] * grad_r1_grad_theta1 \
                                * dRnl1dr1[nl1a] * dgphi[3]
                    products += Rnl1[nl1a] * Rnl1[nl1b] \
                                * ((dtheta1dx**2 + dtheta1dy**2) * dgphi[0] \
                                   + dgphi[1] / x**2)
                    integrand += 2 * products \
                                 * (out12['vsigma'] - out1['vsigma'])

                U[key][index] = np.sum(integrand * aux)

        self.timer.stop('calculate_onsiteU')
        return U

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el1>_onsiteU_<el2>.2ck'.
        """
        sym1, sym2 = self.ela.get_symbol(), self.elb.get_symbol()

        for bas1a in range(len(self.ela.basis_sets)):
            for bas1b in range(len(self.ela.basis_sets)):
                template = '%s-%s_onsiteU_%s.2ck'
                filename = template % (sym1+'+'*bas1a, sym1+'+'*bas1b, sym2)
                print('Writing to %s' % filename, file=self.txt, flush=True)

                with open(filename, 'w') as f:
                    write_2ck(f, self.Rgrid, self.tables[(bas1a, bas1b)])
        return
