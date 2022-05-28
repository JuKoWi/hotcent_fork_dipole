#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from hotcent.fluctuation_twocenter import (NUMINT_2CL, NUMINT_2CM,
                                           select_orbitals, select_subshells,
                                           write_2cl, write_2cm)
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.interpolation import CubicSplineFunction
from hotcent.orbital_hartree import (get_density_expansion_coefficient,
                                     OrbitalHartreePotential)
from hotcent.orbitals import ANGULAR_MOMENTUM, ORBITAL_LABELS, ORBITALS
from hotcent.slako import get_twocenter_phi_integral, tail_smoothening
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


class Offsite2cUTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA', smoothen_tails=True, shift=False, use_becke=False):
        """
        Calculates off-site, orbital- and distance-dependent "U" values
        as matrix elements of the two-center-expanded Hartree-XC kernel.

        Parameters
        ----------
        use_becke : bool, optional
            Whether to use the 'becke' package to evaluate the
            kernel contributions (for debugging purposes).

        Other Parameters
        ----------------
        See Offsite2cTable.run() and Onsite2cTable.run().
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Orbital-resolved offsite-U table construction for %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'

        if not use_becke:
            self.build_ohp()

        self.timer.start('run_offsiteU')
        wf_range = self.get_range(wflimit)
        self.Rgrid = rmin + dr * np.arange(N)
        self.tables = {}

        for p, (e1, e2) in enumerate(self.pairs):
            for bas1 in range(len(e1.basis_sets)):
                for bas2 in range(len(e2.basis_sets)):
                    self.tables[(p, bas1, bas2)] = np.zeros((N, NUMINT_2CM))

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
                        selected = select_orbitals(e1, e2)

                        if term == 'hartree':
                            U = self.calculate_hartree(selected, e1, e2, R,
                                            grid, area, use_becke=use_becke)
                        else:
                            U = self.calculate_xc(selected, e1, e2, R, grid,
                                            area, xc=xc, use_becke=use_becke)

                        for key in selected:
                            (nl1, lm1), (nl2, lm2) = key
                            bas1 = e1.get_basis_set_index(nl1)
                            bas2 = e2.get_basis_set_index(nl2)
                            index = ORBITAL_LABELS.index(lm1) * 16
                            index += ORBITAL_LABELS.index(lm2)
                            self.tables[(p, bas1, bas2)][i, index] += U[key]

        for key in self.tables:
            for i in range(NUMINT_2CM):
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
        return

    def build_ohp(self):
        """
        Populates the self.ohp_dict dictionary with the Hartree
        potential interpolators for every needed orbital.
        """
        self.timer.start('build_ohp')
        print('Building orbital-hartree-potentials', flush=True, file=self.txt)

        self.ohp_dict = {}
        for p, (e1, e2) in enumerate(self.pairs):
            selected = select_orbitals(e1, e2)

            for key in selected:
                (nl1, lm1), (nl2, lm2) = key
                sym2 = e2.get_symbol()

                if (sym2, nl2, lm2) not in self.ohp_dict:
                    dens_nl2 = e2.Rnlg[nl2]**2
                    self.ohp_dict[(sym2, nl2, lm2)] = \
                        OrbitalHartreePotential(e2.rmin, e2.xgrid,
                                                dens_nl2, lm2)
        self.timer.stop('build_ohp')
        return

    def evaluate_ohp(self, sym, nl, lm, l, m, r):
        """
        Evaluates one component of the Hartree potential
        for the given orbital and atom.

        Parameters
        ----------
        sym : str
            Chemical symbol for the atomic orbital.
        nl : str
            Subshell label for the atomic orbital.
        lm : str
            Orbital label for the atomic orbital.
        l : str
            Angular momentum label for the Hartree potential component.
        m : str
            Orbital label for the Hartree potential component.
        r : np.ndarray
            Distances from the atomic center.

        Returns
        -------
        v : np.ndarray
            Hartree potential at the given distances.
        """
        v = self.ohp_dict[(sym, nl, lm)].vhar_fct[(l, m)](r)
        return v

    def calculate_xc(self, selected, e1, e2, R, grid, area, xc='LDA',
                     use_becke=False):
        """
        Calculates the selected integrals involving the XC kernel.

        Parameters
        ----------
        use_becke : bool, optional
            See run().

        Other Parameters
        ----------------
        See Offsite2cTable.calculate().

        Returns
        -------
        U: dict
            Dictionary containing the integral for each selected
            ornital pair.
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

        rho = e1.electron_density(r1) + e2.electron_density(r2)

        if xc.add_gradient_corrections:
            drho1 = e1.electron_density(r1, der=1)
            drho2 = e2.electron_density(r2, der=1)
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
            (nl1, lm1), (nl2, lm2) = key
            dens_nl1 = e1.Rnl(r1, nl1)**2
            dens_nl2 = e2.Rnl(r2, nl2)**2
            integrand = out['v2rho2'] * dens_nl1 * dens_nl2
            integrand *= self.get_angular_integral(lm1, lm2, c1, c2, s1, s2)

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

    def calculate_hartree(self, selected, e1, e2, R, grid, area,
                          use_becke=False):
        """
        Calculates the selected integrals involving the Hartree kernel.

        Parameters
        ----------
        use_becke : bool, optional
            See run().

        Other Parameters
        ----------------
        See Offsite2cTable.calculate().

        Returns
        -------
        U: dict
            Dictionary containing the integral for each selected
            orbital pair.
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

        sym2 = e2.get_symbol()
        U = {}
        for key in selected:
            (nl1, lm1), (nl2, lm2) = key
            U[key] = -1. / R

            if use_becke:
                import becke
                from hotcent.spherical_harmonics import sph_cartesian
                becke.settings.radial_grid_factor = 5
                becke.settings.lebedev_order = 27
                atoms = [(e1.Z, (0., 0., 0.)), (e2.Z, (0., 0., R))]

                def rho1(x, y, z):
                    r = np.sqrt(x**2 + y**2 + z**2)
                    return (e1.Rnl(r, nl1) * sph_cartesian(x, y, z, r, lm1))**2

                def rho2(x, y, z):
                    Z = z - R
                    r = np.sqrt(x**2 + y**2 + Z**2)
                    return (e2.Rnl(r, nl2) * sph_cartesian(x, y, Z, r, lm2))**2

                U[key] += becke.integral(atoms, integrand)
            else:
                dens_nl1 = e1.Rnl(r1, nl1)**2
                lmax2 = self.ohp_dict[(sym2, nl2, lm2)].lmax

                for l in range(lmax2+1):
                    for m in range(2*l+1):
                        lm = ORBITALS[l][m]

                        lmax1 = 2*ANGULAR_MOMENTUM[lm1[0]]
                        for ll in range(lmax1+1):
                            for mm in range(2*ll+1):
                                llmm = ORBITALS[ll][mm]
                                coeff = get_density_expansion_coefficient(lm1,
                                                                          llmm)
                                if coeff == 0:
                                    continue

                                gphi = get_twocenter_phi_integral(lm, llmm, c1,
                                                                  c2, s1, s2)
                                if np.allclose(gphi, 0.):
                                    continue

                                integrand = self.evaluate_ohp(sym2, nl2, lm2,
                                                              l, m, r2)
                                integrand *= dens_nl1 * coeff * aux * gphi
                                U[key] += np.sum(integrand)

        self.timer.stop('calculate_offsiteU_hartree')
        return U

    def write(self):
        """
        Writes all integral tables to file.

        The filename template corresponds to '<el1>-<el2>_offsiteU.2cm'.
        """
        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()

            for bas1, valence1 in enumerate(e1.basis_sets):
                angmom1 = [ANGULAR_MOMENTUM[nl[1]] for nl in valence1]

                for bas2, valence2 in enumerate(e2.basis_sets):
                    angmom2 = [ANGULAR_MOMENTUM[nl[1]] for nl in valence2]

                    template = '%s-%s_offsiteU.2cm'
                    filename = template % (sym1 + '+'*bas1, sym2  + '+'*bas2)
                    print('Writing to %s' % filename, file=self.txt, flush=True)

                    table = self.tables[(p, bas1, bas2)]
                    with open(filename, 'w') as f:
                        write_2cm(f, self.Rgrid, table, angmom1, angmom2)
        return

    def get_angular_integral(self, lm1, lm2, c1, c2, s1, s2):
        """ Returns \int |Y_{lm1}|^2 |Ylm_{lm1}|^2 d\phi.

        Parameters
        ----------
        lm1, lm2 : str
            Orbital labels (e.g. 'px').
        c1, c2, s1, s2 : np.ndarray
            Cosine (c1 and c2) and sine (s1 and s2) of the
            theta_1 and theta2 angles, respectively.

        Returns
        -------
        integral : float
            Value of the integral.
        """
        if lm1[0] not in 'spd' or lm2[0] not in 'spd':
            raise NotImplementedError('Only for angular momenta up to d.')

        if lm1 == 's' and lm2 == 's':
            integral = 1/(8*np.pi)
        elif lm1 == 's' and lm2 == 'px':
            integral = 3*s2**2/(16*np.pi)
        elif lm1 == 's' and lm2 == 'py':
            integral = 3*s2**2/(16*np.pi)
        elif lm1 == 's' and lm2 == 'pz':
            integral = 3*c2**2/(8*np.pi)
        elif lm1 == 's' and lm2 == 'dxy':
            integral = 15*s2**4/(64*np.pi)
        elif lm1 == 's' and lm2 == 'dyz':
            integral = 15*(-8*c2**4 + 8*c2**2)/(128*np.pi)
        elif lm1 == 's' and lm2 == 'dxz':
            integral = 15*(-8*c2**4 + 8*c2**2)/(128*np.pi)
        elif lm1 == 's' and lm2 == 'dx2-y2':
            integral = 15*s2**4/(64*np.pi)
        elif lm1 == 's' and lm2 == 'dz2':
            integral = 5*(3*c2**2 - 1)**2/(32*np.pi)
        elif lm1 == 'px' and lm2 == 's':
            integral = 3*s1**2/(16*np.pi)
        elif lm1 == 'px' and lm2 == 'px':
            integral = 27*s2**2*s1**2/(64*np.pi)
        elif lm1 == 'px' and lm2 == 'py':
            integral = 9*s2**2*s1**2/(64*np.pi)
        elif lm1 == 'px' and lm2 == 'pz':
            integral = 9*s1**2*c2**2/(16*np.pi)
        elif lm1 == 'px' and lm2 == 'dxy':
            integral = 45*s2**4*s1**2/(128*np.pi)
        elif lm1 == 'px' and lm2 == 'dyz':
            integral = 45*s2**2*s1**2*c2**2/(64*np.pi)
        elif lm1 == 'px' and lm2 == 'dxz':
            integral = 135*s2**2*s1**2*c2**2/(64*np.pi)
        elif lm1 == 'px' and lm2 == 'dx2-y2':
            integral = 45*s2**4*s1**2/(128*np.pi)
        elif lm1 == 'px' and lm2 == 'dz2':
            integral = 15*(3*c2**2 - 1)**2*s1**2/(64*np.pi)
        elif lm1 == 'py' and lm2 == 's':
            integral = 3*s1**2/(16*np.pi)
        elif lm1 == 'py' and lm2 == 'px':
            integral = 9*s2**2*s1**2/(64*np.pi)
        elif lm1 == 'py' and lm2 == 'py':
            integral = 27*s2**2*s1**2/(64*np.pi)
        elif lm1 == 'py' and lm2 == 'pz':
            integral = 9*s1**2*c2**2/(16*np.pi)
        elif lm1 == 'py' and lm2 == 'dxy':
            integral = 45*s2**4*s1**2/(128*np.pi)
        elif lm1 == 'py' and lm2 == 'dyz':
            integral = 135*s2**2*s1**2*c2**2/(64*np.pi)
        elif lm1 == 'py' and lm2 == 'dxz':
            integral = 45*s2**2*s1**2*c2**2/(64*np.pi)
        elif lm1 == 'py' and lm2 == 'dx2-y2':
            integral = 45*s2**4*s1**2/(128*np.pi)
        elif lm1 == 'py' and lm2 == 'dz2':
            integral = 15*(3*c2**2 - 1)**2*s1**2/(64*np.pi)
        elif lm1 == 'pz' and lm2 == 's':
            integral = 3*c1**2/(8*np.pi)
        elif lm1 == 'pz' and lm2 == 'px':
            integral = 9*s2**2*c1**2/(16*np.pi)
        elif lm1 == 'pz' and lm2 == 'py':
            integral = 9*s2**2*c1**2/(16*np.pi)
        elif lm1 == 'pz' and lm2 == 'pz':
            integral = 9*c2**2*c1**2/(8*np.pi)
        elif lm1 == 'pz' and lm2 == 'dxy':
            integral = 45*s2**4*c1**2/(64*np.pi)
        elif lm1 == 'pz' and lm2 == 'dyz':
            integral = 45*s2**2*c2**2*c1**2/(16*np.pi)
        elif lm1 == 'pz' and lm2 == 'dxz':
            integral = 45*s2**2*c2**2*c1**2/(16*np.pi)
        elif lm1 == 'pz' and lm2 == 'dx2-y2':
            integral = 45*s2**4*c1**2/(64*np.pi)
        elif lm1 == 'pz' and lm2 == 'dz2':
            integral = 15*(3*c2**2 - 1)**2*c1**2/(32*np.pi)
        elif lm1 == 'dxy' and lm2 == 's':
            integral = 15*s1**4/(64*np.pi)
        elif lm1 == 'dxy' and lm2 == 'px':
            integral = 45*s2**2*s1**4/(128*np.pi)
        elif lm1 == 'dxy' and lm2 == 'py':
            integral = 45*s2**2*s1**4/(128*np.pi)
        elif lm1 == 'dxy' and lm2 == 'pz':
            integral = 45*s1**4*c2**2/(64*np.pi)
        elif lm1 == 'dxy' and lm2 == 'dxy':
            integral = 675*s2**4*s1**4/(1024*np.pi)
        elif lm1 == 'dxy' and lm2 == 'dyz':
            integral = 225*(2 - 2*c1**2)**2*(-8*c2**4 + 8*c2**2)/(4096*np.pi)
        elif lm1 == 'dxy' and lm2 == 'dxz':
            integral = 225*(2 - 2*c1**2)**2*(-8*c2**4 + 8*c2**2)/(4096*np.pi)
        elif lm1 == 'dxy' and lm2 == 'dx2-y2':
            integral = 225*s2**4*s1**4/(1024*np.pi)
        elif lm1 == 'dxy' and lm2 == 'dz2':
            integral = 75*(3*c2**2 - 1)**2*s1**4/(256*np.pi)
        elif lm1 == 'dyz' and lm2 == 's':
            integral = 15*(-8*c1**4 + 8*c1**2)/(128*np.pi)
        elif lm1 == 'dyz' and lm2 == 'px':
            integral = 45*s2**2*s1**2*c1**2/(64*np.pi)
        elif lm1 == 'dyz' and lm2 == 'py':
            integral = 135*s2**2*s1**2*c1**2/(64*np.pi)
        elif lm1 == 'dyz' and lm2 == 'pz':
            integral = 45*s1**2*c2**2*c1**2/(16*np.pi)
        elif lm1 == 'dyz' and lm2 == 'dxy':
            integral = 225*(2 - 2*c2**2)**2*(-8*c1**4 + 8*c1**2)/(4096*np.pi)
        elif lm1 == 'dyz' and lm2 == 'dyz':
            integral = 675*s2**2*s1**2*c2**2*c1**2/(64*np.pi)
        elif lm1 == 'dyz' and lm2 == 'dxz':
            integral = 225*s2**2*s1**2*c2**2*c1**2/(64*np.pi)
        elif lm1 == 'dyz' and lm2 == 'dx2-y2':
            integral = 225*(2 - 2*c2**2)**2*(-8*c1**4 + 8*c1**2)/(4096*np.pi)
        elif lm1 == 'dyz' and lm2 == 'dz2':
            integral = 75*(6*c2**2 - 2)**2*(-8*c1**4 + 8*c1**2)/(2048*np.pi)
        elif lm1 == 'dxz' and lm2 == 's':
            integral = 15*(-8*c1**4 + 8*c1**2)/(128*np.pi)
        elif lm1 == 'dxz' and lm2 == 'px':
            integral = 135*s2**2*s1**2*c1**2/(64*np.pi)
        elif lm1 == 'dxz' and lm2 == 'py':
            integral = 45*s2**2*s1**2*c1**2/(64*np.pi)
        elif lm1 == 'dxz' and lm2 == 'pz':
            integral = 45*s1**2*c2**2*c1**2/(16*np.pi)
        elif lm1 == 'dxz' and lm2 == 'dxy':
            integral = 225*(2 - 2*c2**2)**2*(-8*c1**4 + 8*c1**2)/(4096*np.pi)
        elif lm1 == 'dxz' and lm2 == 'dyz':
            integral = 225*s2**2*s1**2*c2**2*c1**2/(64*np.pi)
        elif lm1 == 'dxz' and lm2 == 'dxz':
            integral = 675*s2**2*s1**2*c2**2*c1**2/(64*np.pi)
        elif lm1 == 'dxz' and lm2 == 'dx2-y2':
            integral = 225*(2 - 2*c2**2)**2*(-8*c1**4 + 8*c1**2)/(4096*np.pi)
        elif lm1 == 'dxz' and lm2 == 'dz2':
            integral = 75*(6*c2**2 - 2)**2*(-8*c1**4 + 8*c1**2)/(2048*np.pi)
        elif lm1 == 'dx2-y2' and lm2 == 's':
            integral = 15*s1**4/(64*np.pi)
        elif lm1 == 'dx2-y2' and lm2 == 'px':
            integral = 45*s2**2*s1**4/(128*np.pi)
        elif lm1 == 'dx2-y2' and lm2 == 'py':
            integral = 45*s2**2*s1**4/(128*np.pi)
        elif lm1 == 'dx2-y2' and lm2 == 'pz':
            integral = 45*s1**4*c2**2/(64*np.pi)
        elif lm1 == 'dx2-y2' and lm2 == 'dxy':
            integral = 225*s2**4*s1**4/(1024*np.pi)
        elif lm1 == 'dx2-y2' and lm2 == 'dyz':
            integral = 225*(2 - 2*c1**2)**2*(-8*c2**4 + 8*c2**2)/(4096*np.pi)
        elif lm1 == 'dx2-y2' and lm2 == 'dxz':
            integral = 225*(2 - 2*c1**2)**2*(-8*c2**4 + 8*c2**2)/(4096*np.pi)
        elif lm1 == 'dx2-y2' and lm2 == 'dx2-y2':
            integral = 675*s2**4*s1**4/(1024*np.pi)
        elif lm1 == 'dx2-y2' and lm2 == 'dz2':
            integral = 75*(3*c2**2 - 1)**2*s1**4/(256*np.pi)
        elif lm1 == 'dz2' and lm2 == 's':
            integral = 5*(3*c1**2 - 1)**2/(32*np.pi)
        elif lm1 == 'dz2' and lm2 == 'px':
            integral = 15*(3*c1**2 - 1)**2*s2**2/(64*np.pi)
        elif lm1 == 'dz2' and lm2 == 'py':
            integral = 15*(3*c1**2 - 1)**2*s2**2/(64*np.pi)
        elif lm1 == 'dz2' and lm2 == 'pz':
            integral = 15*(3*c1**2 - 1)**2*c2**2/(32*np.pi)
        elif lm1 == 'dz2' and lm2 == 'dxy':
            integral = 75*(3*c1**2 - 1)**2*s2**4/(256*np.pi)
        elif lm1 == 'dz2' and lm2 == 'dyz':
            integral = 75*(-8*c2**4 + 8*c2**2)*(6*c1**2 - 2)**2/(2048*np.pi)
        elif lm1 == 'dz2' and lm2 == 'dxz':
            integral = 75*(-8*c2**4 + 8*c2**2)*(6*c1**2 - 2)**2/(2048*np.pi)
        elif lm1 == 'dz2' and lm2 == 'dx2-y2':
            integral = 75*(3*c1**2 - 1)**2*s2**4/(256*np.pi)
        elif lm1 == 'dz2' and lm2 == 'dz2':
            integral = 25*(3*c2**2 - 1)**2*(3*c1**2 - 1)**2/(128*np.pi)
        else:
            integral = 0
        return integral
