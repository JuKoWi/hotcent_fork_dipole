#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.fluctuation_onecenter import select_radial_function
from hotcent.fluctuation_twocenter import (
                INTEGRALS_2CK, INTEGRALS_2CM, NUMINT_2CL, NUML_2CK, NUML_2CM,
                NUMSK_2CK, NUMSK_2CM, select_subshells, write_2cl, write_2ck,
                write_2cm)
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.interpolation import CubicSplineFunction
from hotcent.orbital_hartree import OrbitalHartreePotential
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.slako import (get_integral_pair, get_twocenter_phi_integral,
                           get_twocenter_phi_integrals_derivatives,
                           tail_smoothening)
from hotcent.xc import LibXC


class Offsite2cUTable:
    """
    Convenience wrapper around the Offsite2cUMonopoleTable
    and Offsite2cUMultipoleTable classes.

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
            self.calc = Offsite2cUMultipoleTable(*args, **kwargs)
        else:
            self.calc = Offsite2cUMonopoleTable(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.calc, attr)

    def run(self, *args, **kwargs):
        self.calc.run(*args, **kwargs)
        return

    def write(self, *args, **kwargs):
        self.calc.write(*args, **kwargs)
        return


class Offsite2cUMonopoleTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA', smoothen_tails=True, shift=False):
        """
        Calculates off-site, distance dependent U (or "Gamma") values
        as matrix elements of the two-center-expanded Hartree-XC kernel
        in the monopole approximation.

        Parameters
        ----------
        See Offsite2cTable.run() and Onsite2cTable.run().
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Monopole offsite-U table construction for %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_offsiteU')
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
                            U = self.calculate_hartree(selected, e1, e2, R,
                                                       grid, area)
                        else:
                            U = self.calculate_xc(selected, e1, e2, R, grid,
                                                  area, xc=xc)

                        for key in selected:
                            nl1, nl2 = key
                            bas1 = e1.get_basis_set_index(nl1)
                            bas2 = e2.get_basis_set_index(nl2)
                            index = ANGULAR_MOMENTUM[nl1[1]] * 4
                            index += ANGULAR_MOMENTUM[nl2[1]]
                            self.tables[(p, bas1, bas2)][i, index] += U[key]

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

        self.timer.stop('run_offsiteU')

    def calculate_xc(self, selected, e1, e2, R, grid, area, xc='LDA'):
        """
        Calculates the selected integrals involving the XC kernel.

        Parameters
        ----------
        See Offsite2cTable.calculate().

        Returns
        -------
        U : dict
            Dictionary containing the integral for each selected
            subshell pair.
        """
        self.timer.start('calculate_offsiteU_xc')

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

        U = {}
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

            U[key] = np.sum(integrand * aux)

        self.timer.stop('calculate_offsiteU_xc')
        return U

    def calculate_hartree(self, selected, e1, e2, R, grid, area):
        """
        Calculates the selected integrals involving the Hartree kernel.

        Parameters
        ----------
        See Offsite2cTable.calculate().

        Returns
        -------
        U : dict
            Dictionary containing the integral for each selected
            subshell pair.
        """
        self.timer.start('calculate_offsiteU_hartree')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        aux = 2 * np.pi * area * x
        self.timer.stop('prelude')

        U = {}
        for key in selected:
            nl1, nl2 = key
            dens_nl1 = e1.Rnl(r1, nl1)**2 / (4 * np.pi)

            dens_nl2 = e2.Rnlg[nl2]**2 / (4 * np.pi)
            vhar2 = e2.calculate_hartree_potential(dens_nl2, nel=1.)
            spl = CubicSplineFunction(e2.rgrid, vhar2)

            U[key] = np.sum(dens_nl1 * spl(r2) * aux) - 1./R

        self.timer.stop('calculate_offsiteU_hartree')
        return U

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el2>_offsiteU.2cl'.
        """
        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()

            for bas1, valence1 in enumerate(e1.basis_sets):
                angmom1 = [ANGULAR_MOMENTUM[nl[1]] for nl in valence1]

                for bas2, valence2 in enumerate(e2.basis_sets):
                    angmom2 = [ANGULAR_MOMENTUM[nl[1]] for nl in valence2]

                    template = '%s-%s_offsiteU.2cl'
                    filename = template % (sym1 + '+'*bas1, sym2  + '+'*bas2)
                    print('Writing to %s' % filename, file=self.txt, flush=True)

                    table = self.tables[(p, bas1, bas2)]
                    with open(filename, 'w') as f:
                        write_2cl(f, self.Rgrid, table, angmom1, angmom2)
        return


class Offsite2cUMultipoleTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, nl=None, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50,
            wflimit=1e-7, xc='LDA', smoothen_tails=True, shift=False,
            subtract_delta=True):
        """
        Calculates off-site, orbital- and distance-dependent "U" values
        as matrix elements of the two-center-expanded Hartree-XC kernel
        in a multipole expansion.

        Parameters
        ----------
        nl : tuple of str, optional
            Two-tuple with the subshells defining the radial functions
            for each element. If None, the subshells with the lowest angular
            momentum will be chosen from the minimal valence sets.

        subtract_delta : bool, optional
            Whether to subtract the point multipole contributions from
            the kernel integrals (default: True). Setting it to False
            is only useful for debugging purposes.

        Other parameters
        ----------------
        See Offsite2cTable.run() and Onsite2cTable.run().
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Multipole offsite-U table construction for %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_offsiteU')
        wf_range = self.get_range(wflimit)
        self.Rgrid = rmin + dr * np.arange(N)
        self.tables = {}

        if nl is None:
            nl1 = select_radial_function(self.ela)
            nl2 = select_radial_function(self.elb)
            nl = (nl1, nl2)
        else:
            assert isinstance(nl, tuple)

        self.build_ohp(nl)
        self.build_int1c(nl)

        for p, (e1, e2) in enumerate(self.pairs):
            self.tables[p] = np.zeros((N, NUMSK_2CK))

        for i, R in enumerate(self.Rgrid):
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
                        nl1, nl2 = nl if p == 0 else nl[::-1]

                        if term == 'hartree':
                            U = self.calculate_hartree(e1, e2, R, grid, area,
                                                       nl1, nl2, subtract_delta)
                        else:
                            U = self.calculate_xc(e1, e2, R, grid, area,
                                                  nl1, nl2, xc=xc)

                        self.tables[p][i, :] += U[:]

        for key in self.tables:
            for i in range(NUMSK_2CK):
                all0 = np.allclose(self.tables[key][:, i], 0)
                if shift and not all0:
                    for j in range(N-1, 1, -1):
                        if abs(self.tables[key][j, i]) > 0:
                            self.tables[key][:j+1, i] -= self.tables[key][j, i]
                            break

                if smoothen_tails:
                    for key in self.tables:
                        self.tables[key][:, i] = \
                            tail_smoothening(self.Rgrid, self.tables[key][:, i])

        self.timer.stop('run_offsiteU')
        return

    def build_ohp(self, nl):
        """
        Populates the self.ohp_dict dictionary with the needed
        Hartree potential interpolators.
        """
        self.timer.start('build_ohp')
        print('Building the Hartree potential interpolators', flush=True,
              file=self.txt)

        self.ohp_dict = {}
        for p, (e1, e2) in enumerate(self.pairs):
            sym1 = e1.get_symbol()
            sym2 = e2.get_symbol()
            nl1, nl2 = nl if p == 0 else nl[::-1]

            if (sym1, nl1) not in self.ohp_dict:
                lmax = NUML_2CK
                rho = e1.Rnlg[nl1]**2
                self.ohp_dict[(sym1, nl1)] = \
                    OrbitalHartreePotential(e1.rmin, e1.xgrid, rho, lmax)

            if (sym2, nl2) not in self.ohp_dict:
                lmax = NUML_2CK
                rho = e2.Rnlg[nl2]**2
                self.ohp_dict[(sym2, nl2)] = \
                    OrbitalHartreePotential(e2.rmin, e2.xgrid, rho, lmax)

        self.timer.stop('build_ohp')
        return

    def evaluate_ohp(self, sym, nl, l, r):
        """
        Evaluates one component of the Hartree potential
        for the given orbital and atom.

        Parameters
        ----------
        sym : str
            Chemical symbol for the atomic orbital.
        nl : str
            Subshell label for the atomic orbital.
        l : str
            Angular momentum label for the Hartree potential component.
        r : np.ndarray
            Distances from the atomic center.

        Returns
        -------
        v : np.ndarray
            Hartree potential at the given distances.
        """
        v = self.ohp_dict[(sym, nl)].vhar_fct[l](r)
        return v

    def build_int1c(self, nl):
        """
        Populates the self.int1c_dict dictionary with the needed
        one-center integrals for the point multipole contributions.
        """
        self.int1c_dict = {}
        sym1 = self.ela.get_symbol()
        sym2 = self.elb.get_symbol()
        nl1, nl2 = nl

        Rnl = self.ela.Rnlg[nl1]**2
        self.int1c_dict[sym1] = []
        for l in range(NUML_2CK):
            int1c = self.ela.grid.integrate(Rnl * self.ela.rgrid**(l+2),
                                            use_dV=False)
            self.int1c_dict[sym1].append(int1c)

        if sym2 != sym1:
            Rnl = self.elb.Rnlg[nl2]**2
            self.int1c_dict[sym2] = []
            for l in range(NUML_2CK):
                int1c = self.elb.grid.integrate(Rnl * self.elb.rgrid**(l+2),
                                                use_dV=False)
                self.int1c_dict[sym2].append(int1c)
        return

    def calculate_xc(self, e1, e2, R, grid, area, nl1, nl2, xc='LDA'):
        """
        Calculates the selected integrals involving the XC kernel.

        Parameters
        ----------
        nl1, nl2 : str
            Subshells defining the radial functions.

        Other parameters
        ----------------
        See Offsite2cTable.calculate().

        Returns
        -------
        U: np.ndarray
            Array with the needed Slater-Koster integrals.
        """
        self.timer.start('calculate_offsiteU_xc')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y - R) / r2  # cosine of theta_2
        s1 = x / r1  # sine of theta_1
        s2 = x / r2  # sine of theta_2
        aux = area * x

        xc = LibXC('LDA_X+LDA_C_PW' if xc == 'LDA' else xc)

        rho1 = e1.electron_density(r1)
        rho2 = e2.electron_density(r2)
        rho12 = rho1 + rho2
        Rnl1 = e1.Rnl(r1, nl1)**2
        Rnl2 = e2.Rnl(r2, nl2)**2

        if xc.add_gradient_corrections:
            drho1 = e1.electron_density(r1, der=1)
            drho2 = e2.electron_density(r2, der=1)
            c2 = (y - R) / r2  # cosine of theta_2
            s2 = x / r2  # sine of theta_2

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
            dRnl1dr1 = 2 * e1.Rnl(r1, nl1) * e1.Rnl(r1, nl1, der=1)

            dr2dx = x/r2
            ds2dx = (r2 - x*dr2dx) / r2**2
            dr2dy = (y - R)/r2
            ds2dy = -x*dr2dy / r2**2
            dtheta2dx = ds2dx / c2
            dtheta2dy = ds2dy / c2

            grad_r2_grad_rho12 = dr2dx * drho12dx + dr2dy * drho12dy
            grad_theta2_grad_rho12 = dtheta2dx * drho12dx + dtheta2dy * drho12dy
            dRnl2dr2 = 2 * e2.Rnl(r2, nl2) * e2.Rnl(r2, nl2, der=1)

            grad_r1_grad_r2 = dr1dx*dr2dx + dr1dy*dr2dy
            grad_r1_grad_theta2 = dr1dx * dtheta2dx + dr1dy * dtheta2dy
            grad_r2_grad_theta1 = dr2dx * dtheta1dx + dr2dy * dtheta1dy
            grad_theta1_grad_theta2 = dtheta1dx*dtheta2dx + dtheta1dy*dtheta2dy
        else:
            sigma12 = None
        self.timer.stop('prelude')

        self.timer.start('fxc')
        out12 = xc.compute_vxc(rho12, sigma=sigma12, fxc=True)
        self.timer.stop('fxc')

        U = np.zeros(NUMSK_2CK)

        for index, integral in enumerate(INTEGRALS_2CK):
            lm1, lm2 = get_integral_pair(integral)
            gphi = get_twocenter_phi_integral(lm1, lm2, c1, c2, s1, s2)
            integrand = out12['v2rho2'] * Rnl1 * Rnl2 * gphi

            if xc.add_gradient_corrections:
                dgphi = get_twocenter_phi_integrals_derivatives(lm1, lm2, c1,
                                                                c2, s1, s2)

                products = dRnl1dr1 * grad_r1_grad_rho12 * Rnl2 * gphi
                products += dRnl2dr2 * grad_r2_grad_rho12 * Rnl1 * gphi
                products += Rnl1 * Rnl2 * grad_theta1_grad_rho12 * dgphi[2]
                products += Rnl1 * Rnl2 * grad_theta2_grad_rho12 * dgphi[3]
                integrand += 2 * products * out12['v2rhosigma']

                products = dRnl1dr1 * dRnl2dr2 * grad_r1_grad_rho12 \
                           * grad_r2_grad_rho12 * gphi
                products += dRnl2dr2 * grad_r2_grad_rho12 * Rnl1 \
                            * grad_theta1_grad_rho12 * dgphi[2]
                products += dRnl1dr1 * grad_r1_grad_rho12 * Rnl2 \
                            * grad_theta2_grad_rho12 * dgphi[3]
                products += Rnl1 * Rnl2 * grad_theta1_grad_rho12 \
                            * grad_theta2_grad_rho12 * dgphi[0]
                integrand += 4 * products * out12['v2sigma2']

                products = dRnl1dr1 * dRnl2dr2 * grad_r1_grad_r2 * gphi
                products += Rnl1 * grad_r2_grad_theta1 * dRnl2dr2 * dgphi[2]
                products += Rnl2 * grad_r1_grad_theta2 * dRnl1dr1 * dgphi[3]
                products += Rnl1 * Rnl2 * (grad_theta1_grad_theta2 * dgphi[0] \
                                           + dgphi[1] / x**2)
                integrand += 2 * products * out12['vsigma']

            U[index] = np.sum(integrand * aux)

        self.timer.stop('calculate_offsiteU_xc')
        return U

    def evaluate_point_multipole_hartree(self, sym1, sym2, integral, R):
        """
        Evaluates the point multipole contribution for the given Hartree
        kernel integral (which is also the value that it should approach
        at large distance R).

        Parameters
        ----------
        sym1, sym2 : str
            Chemical symbols of the first and second element.
        integral : str
            Integral label.
        R : float
            Interatomic distance

        Returns
        -------
        U_delta : float
            The point multipole contribution.
        """
        if integral == 'sss':
            U_delta = 4 * np.pi
        elif integral == 'sps':
            U_delta = -4 * np.pi / np.sqrt(3)
        elif integral == 'sds':
            U_delta = 4 * np.pi / np.sqrt(5)
        elif integral == 'pss':
            U_delta = 4 * np.pi / np.sqrt(3)
        elif integral == 'pps':
            U_delta = -8 * np.pi / 3.
        elif integral == 'ppp':
            U_delta = 4 * np.pi / 3.
        elif integral == 'pds':
            U_delta = 12 * np.pi / np.sqrt(15)
        elif integral == 'pdp':
            U_delta = -4 * np.pi / np.sqrt(5)
        elif integral == 'dss':
            U_delta = 4 * np.pi / np.sqrt(5)
        elif integral == 'dps':
            U_delta = -12 * np.pi / np.sqrt(15)
        elif integral == 'dpp':
            U_delta = 4 * np.pi / np.sqrt(5)
        elif integral == 'dds':
            U_delta = 24 * np.pi / 5.
        elif integral == 'ddp':
            U_delta = -16 * np.pi / 5.
        elif integral == 'ddd':
            U_delta = 4 * np.pi / 5
        else:
            raise NotImplementedError(integral)

        lm1, lm2 = get_integral_pair(integral)
        l1 = ANGULAR_MOMENTUM[lm1[0]]
        l2 = ANGULAR_MOMENTUM[lm2[0]]
        U_delta *= self.int1c_dict[sym1][l1] * self.int1c_dict[sym2][l2]
        U_delta /= R**(1 + l1 + l2)
        return U_delta

    def calculate_hartree(self, e1, e2, R, grid, area, nl1, nl2,
                          subtract_delta):
        """
        Calculates the selected integrals involving the Hartree kernel.

        Parameters
        ----------
        nl1, nl2 : str
            Subshells defining the radial functions.
        subtract_delta : bool
            Whether to subtract the point multipole contributions from
            the kernel integrals.

        Other parameters
        ----------------
        See Offsite2cTable.calculate().

        Returns
        -------
        U: np.ndarray
            Array with the needed Slater-Koster integrals.
        """
        self.timer.start('calculate_offsiteU_hartree')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y - R) / r2  # cosine of theta_2
        s1 = x / r1  # sine of theta_1
        s2 = x / r2  # sine of theta_2
        aux = area * x
        self.timer.stop('prelude')

        sym1 = e1.get_symbol()
        sym2 = e2.get_symbol()
        U = np.zeros(NUMSK_2CK)
        dens_nl1 = e1.Rnl(r1, nl1)**2

        for index, integral in enumerate(INTEGRALS_2CK):
            lm1, lm2 = get_integral_pair(integral)
            gphi = get_twocenter_phi_integral(lm1, lm2, c1, c2, s1, s2)

            l2 = ANGULAR_MOMENTUM[lm2[0]]
            vhar = self.evaluate_ohp(sym2, nl2, l2, r2)
            U[index] = np.sum(vhar * dens_nl1 * aux * gphi)

            if subtract_delta:
                U_delta = self.evaluate_point_multipole_hartree(sym1, sym2,
                                                                integral, R)
                U[index] -= U_delta

        self.timer.stop('calculate_offsiteU_hartree')
        return U

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el2>_offsiteU.2ck'.
        """
        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()

            template = '%s-%s_offsiteU.2ck'
            filename = template % (sym1, sym2)
            print('Writing to %s' % filename, file=self.txt, flush=True)

            with open(filename, 'w') as f:
                point_kernels = np.zeros(NUMSK_2CK)
                for i, integral in enumerate(INTEGRALS_2CK):
                    point_kernels[i] = self.evaluate_point_multipole_hartree(
                                                    sym1, sym2, integral, 1)

                write_2ck(f, self.Rgrid, self.tables[p],
                          point_kernels=point_kernels)
        return


class Offsite2cMTable(MultiAtomIntegrator):
    """
    Calculator for (parts of) the off-site moment integrals "M"
    (Boleininger, Guilbert and Horsfield (2016), doi:10.1063/1.4964391):

    $M_{\mu,\nu,lm} = \int \phi_\mu(\mathbf{r_1}) Y_{lm}(\mathbf{r_1})
                           \phi_\nu(\mathbf{r_2}) d\mathbf{r}$
    """
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            smoothen_tails=True):
        """
        Calculates the required off-site Slater-Koster integrals for

        $\int R_{nl1}(\mathbf{r}) Y_{lm}(\mathbf{r_1}
              \phi_\nu(\mathbf{r_2}) d\mathbf{r}$

        from which the moment integrals M can be obtained by rotation
        after multiplication with the appropriate Gaunt coefficients.

        Parameters
        ----------
        See Offsite2cTable.run() and Onsite2cTable.run().
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Multipole offsite-M table construction for %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        self.timer.start('run_offsiteM')
        wf_range = self.get_range(wflimit)
        self.Rgrid = rmin + dr * np.arange(N)
        self.tables = {}

        for p, (e1, e2) in enumerate(self.pairs):
            for bas1 in range(len(e1.basis_sets)):
                for bas2 in range(len(e2.basis_sets)):
                    shape = (NUML_2CM, N, NUMSK_2CM)
                    self.tables[(p, bas1, bas2)] = np.zeros(shape)

        for i, R in enumerate(self.Rgrid):
            if R > 2 * wf_range:
                break

            grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)

            if i == N - 1 or N // 10 == 0 or i % (N // 10) == 0:
                print('R=%8.2f, %i grid points ...' % \
                      (R, len(grid)), file=self.txt, flush=True)

            if len(grid) > 0:
                for p, (e1, e2) in enumerate(self.pairs):
                    selected = self.select_integrals(e1, e2)

                    M = self.calculate(selected, e1, e2, R, grid, area)

                    for key in selected:
                        nl1, nl2, integrals = key
                        bas1 = e1.get_basis_set_index(nl1)
                        bas2 = e2.get_basis_set_index(nl2)
                        l1 = ANGULAR_MOMENTUM[nl1[1]]
                        self.tables[(p, bas1, bas2)][l1, i, :] += M[key][:]

        if smoothen_tails:
            for key in self.tables:
                for l1 in range(NUML_2CM):
                    for i in range(NUMSK_2CM):
                            for key in self.tables:
                                self.tables[key][l1, :, i] = \
                                    tail_smoothening(self.Rgrid,
                                                     self.tables[key][l1, :, i])

        self.timer.stop('run_offsiteM')
        return

    def select_integrals(self, e1, e2, expand2=True):
        """ Returns the list of (nl1, nl2, integrals) tuples with
        the integrals to evaluate for every subshell pair of the given
        elements. """
        selected = []
        for ival1, valence1 in enumerate(e1.basis_sets):
            for nl1 in valence1:
                lmax1 = 4

                for ival2, valence2 in enumerate(e2.basis_sets):
                    for nl2 in valence2:
                        angmom2 = [ANGULAR_MOMENTUM[nl2[1]]]

                        integrals = []
                        for integral in INTEGRALS_2CM:
                            l1 = ANGULAR_MOMENTUM[integral[0]]
                            l2 = ANGULAR_MOMENTUM[integral[1]]
                            if l1 <= lmax1 and l2 in angmom2:
                                integrals.append(integral)

                        selected.append((nl1, nl2, tuple(integrals)))
        return selected

    def calculate(self, selected, e1, e2, R, grid, area):
        """
        Calculates the selected integrals.

        Parameters
        ----------
        See Offsite2cTable.calculate().

        Returns
        -------
        M: dict of np.ndarray
            Dictionary containing the integrals for each selected
            orbital pair.
        """
        self.timer.start('calculate_offsiteM')

        # common for all integrals (not subshell-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y - R) / r2  # cosine of theta_2
        s1 = x / r1  # sine of theta_1
        s2 = x / r2  # sine of theta_2
        aux = area * x
        self.timer.stop('prelude')

        M = {}
        for key in selected:
            nl1, nl2, integrals = key
            M[key] = np.zeros(NUMSK_2CM)

            Rnl1 = e1.Rnl(r1, nl1)
            Rnl2 = e2.Rnl(r2, nl2)

            for integral in integrals:
                lm1, lm2 = get_integral_pair(integral)
                gphi = get_twocenter_phi_integral(lm1, lm2, c1, c2, s1, s2)

                index = INTEGRALS_2CM.index(integral)
                M[key][index] = np.sum(Rnl1 * Rnl2 * aux * gphi)

        self.timer.stop('calculate_offsiteM')
        return M

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el2>_offsiteM.2cm'.
        """
        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()

            for bas1, valence1 in enumerate(e1.basis_sets):
                for bas2, valence2 in enumerate(e2.basis_sets):
                    template = '%s-%s_offsiteM.2cm'
                    filename = template % (sym1 + '+'*bas1, sym2  + '+'*bas2)
                    print('Writing to %s' % filename, file=self.txt, flush=True)

                    with open(filename, 'a') as f:
                        # ensure that we start from empty file
                        f.truncate(0)

                        for l1 in range(NUML_2CM):
                            table = self.tables[(p, bas1, bas2)][l1, :, :]
                            write_2cm(f, self.Rgrid, table, l1)
        return
