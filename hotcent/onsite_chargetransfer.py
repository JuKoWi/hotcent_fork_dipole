#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.fluctuation_onecenter import (
                NUML_1CK, NUML_1CM, select_radial_function, write_1ck,
                write_1cm)
from hotcent.fluctuation_twocenter import (
                INTEGRALS_2CK, NUMINT_2CL, NUMSK_2CK, select_subshells,
                write_2cl, write_2ck)
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.orbital_hartree import OrbitalHartreePotential
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.slako import (get_integral_pair, get_twocenter_phi_integral,
                           tail_smoothening)
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
        Calculates the selected integrals involving the XC kernel.

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

    def run(self, nl=None, xc='LDA'):
        """
        Calculates on-site, one-center "U" values.

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
        print('Onsite-U table construction for %s' % self.el.get_symbol(),
              file=self.txt)
        print('***********************************************', file=self.txt)

        if nl is None:
            nl = select_radial_function(self.el)

        self.table = self.calculate(nl, xc=xc)
        return

    def calculate(self, nl, xc='LDA'):
        """
        Calculates the selected integrals involving the Hartree-XC kernel.

        Parameters
        ----------
        nl : str
            Subshell defining the radial function.
        xc : str, optional
            Name of the exchange-correlation functional (default: LDA).

        Returns
        -------
        U: np.ndarray
            Array with the integral for each multipole.
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
        Rnl = np.copy(self.el.Rnlg[nl])
        dens_nl = Rnl**2

        integrand = out['v2rho2'] * dens_nl * dens_nl
        U[:] = self.el.grid.integrate(integrand * self.el.rgrid**2,
                                      use_dV=False)

        if xc.add_gradient_corrections:
            dnl = 2 * Rnl * self.el.Rnl(self.el.rgrid, nl, der=1)
            grad_nl_grad_rho = dnl * drho
            integrand = 2. * out['v2rhosigma'] * 2 * grad_nl_grad_rho * dens_nl
            integrand += 4. * out['v2sigma2'] * grad_nl_grad_rho**2
            integrand += 2. * out['vsigma'] * dnl**2
            U[:] += self.el.grid.integrate(integrand * self.el.rgrid**2,
                                           use_dV=False)

            for l in range(NUML_1CK):
                integrand = 2. * out['vsigma'] * dens_nl**2
                U[l] += self.el.grid.integrate(integrand * l * (l+1),
                                               use_dV=False)

        for l in range(NUML_1CK):
            ohp = OrbitalHartreePotential(self.el.rmin, self.el.xgrid,
                                          dens_nl, lmax=NUML_1CK-1)
            vhar = ohp.vhar_fct[l](self.el.rgrid)
            integrand = vhar * dens_nl * self.el.rgrid**2
            U[l] += self.el.grid.integrate(integrand, use_dV=False)

        return U

    def write(self):
        """
        Writes the integrals to file.

        The filename template corresponds to '<el1>-<el1>_onsiteU.1ck'.
        """
        sym = self.el.get_symbol()
        template = '%s-%s_onsiteU.1ck'
        filename = template % (sym, sym)
        print('Writing to %s' % filename, file=self.txt, flush=True)

        with open(filename, 'w') as f:
            write_1ck(f, self.table)
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
        print('Onsite-M table construction for %s' % \
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


class Onsite2cUTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='monopolar',
                                     **kwargs)

    def run(self, nl=None, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50,
            wflimit=1e-7, xc='LDA', smoothen_tails=True):
        """
        Calculates on-site, orbital- and distance-dependent "U" values
        as matrix elements of the two-center-expanded XC kernel.

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
        print('Onsite-U table construction for %s and %s' % \
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
        self.tables = np.zeros((N, NUMSK_2CK))

        if nl is None:
            nl = select_radial_function(self.ela)

        for i, R in enumerate(self.Rgrid):
            if R > 2 * wf_range:
                break

            if i == N - 1 or N // 10 == 0 or i % (N // 10) == 0:
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            U = self.calculate(e1, e2, R, grid, area, nl, xc=xc)
            self.tables[i, :] = U[:]

        if smoothen_tails:
            # Smooth the curves near the cutoff
            for i in range(NUMSK_2CK):
                self.tables[:, i] = tail_smoothening(self.Rgrid,
                                                     self.tables[:, i])

        self.timer.stop('run_onsiteU')
        return

    def calculate(self, e1, e2, R, grid, area, nl, xc='LDA'):
        """
        Calculates the selected integrals involving the XC kernel.

        Parameters
        ----------
        nl : str
            Subshell defining the radial function.

        Other parameters
        ----------------
        See Onsite2cTable.calculate().

        Returns
        -------
        U: np.ndarray
            Array containing the needed Slater-Koster integrals.
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

        if xc.add_gradient_corrections:
            raise NotImplementedError
        else:
            sigma1 = None
            sigma12 = None
        self.timer.stop('prelude')

        self.timer.start('fxc')
        out1 = xc.compute_vxc(rho1, sigma=sigma1, fxc=True)
        out12 = xc.compute_vxc(rho12, sigma=sigma12, fxc=True)
        self.timer.stop('fxc')

        #U = np.zeros(NUMSK_2CM)
        U = np.zeros(NUMSK_2CK)
        dens_nl = e1.Rnl(r1, nl)**2

        for index, integral in enumerate(INTEGRALS_2CK):
            lm1a, lm1b = get_integral_pair(integral)
            gphi = get_twocenter_phi_integral(lm1a, lm1b, c1, c1, s1, s1)

            integrand = (out12['v2rho2'] - out1['v2rho2']) * dens_nl**2 * gphi

            if xc.add_gradient_corrections:
                raise NotImplementedError

            U[index] = np.sum(integrand * aux)

        self.timer.stop('calculate_onsiteU')
        return U

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el1>_onsiteU_<el2>.2ck'.
        """
        sym1, sym2 = self.ela.get_symbol(), self.elb.get_symbol()
        template = '%s-%s_onsiteU_%s.2ck'
        filename = template % (sym1, sym1, sym2)
        print('Writing to %s' % filename, file=self.txt, flush=True)

        with open(filename, 'w') as f:
            write_2ck(f, self.Rgrid, self.tables)
        return
