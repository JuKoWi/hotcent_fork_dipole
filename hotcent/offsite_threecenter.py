#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from scipy.integrate import quad_vec
from hotcent.slako import SlaterKosterTable
from hotcent.spherical_harmonics import (sph_nophi, sph_nophi_der,
                                         sph_phi, sph_phi_der)
from hotcent.threecenter import select_integrals, write_3cf
from hotcent.xc import EXC_PW92_Spline, LibXC, VXC_PW92_Spline


class Offsite3cTable(SlaterKosterTable):
    def run(self, e3, Rgrid, Sgrid, Tgrid, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA', write=True, filename=None):
        """
        Calculates off-site three-center Hamiltonian integrals.

        Parameters
        ----------
        e3 : AtomicBase-like object
            Object with atomic properties for the third atom.
        Rgrid, Sgrid, Tgrid : list or array
            Lists with distances defining the three-atom geometries.
        write : bool, optional
            Whether to write the integrals to file (the default)
            or return them as a dictionary instead.
        filename : str, optional
            File name to use in case write=True. The default (None)
            implies that a '<el1>-<el2>_offsite3c_<el3>.3cf'
            template is used.
        ntheta, nr, wflimit, xc :
            See SlaterKosterTable.run().

        Returns
        -------
        output : dict of dict of np.ndarray, optional
            Dictionary with the values for each el1-el2 pair
            and integral type. Only returned if write=False.
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Off-site three-center calculations with %s' % e3.get_symbol(),
              file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        self.timer.start('run_offsite3c')
        wf_range = self.get_range(wflimit)
        numST = len(Sgrid) * len(Tgrid)
        output = {}

        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2, sym3 = e1.get_symbol(), e2.get_symbol(), e3.get_symbol()

            selected = select_integrals(e1, e2)
            print('Integrals:', end=' ', file=self.txt)
            for s in selected:
                print(s[0], end=' ', file=self.txt)
            print(file=self.txt, flush=True)

            output[(sym1, sym2)] = {integral: []
                                    for (integral, nl1, nl2) in selected}

            for i, R in enumerate(Rgrid):
                print('Starting for R=%.3f' % R, file=self.txt, flush=True)

                d = None
                if R < 2 * wf_range:
                    grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)
                    if len(grid) > 0:
                        d = self.calculate(selected, e1, e2, e3, R, grid, area,
                                           Sgrid=Sgrid, Tgrid=Tgrid, xc=xc)

                if d is None:
                    d = {key: np.zeros(1 + numST) for key in selected}

                for key in selected:
                    integral, nl1, nl2 = key
                    output[(sym1, sym2)][integral].append(d[key])

            if write:
                if filename is None:
                    fname = '%s-%s_offsite3c_%s.3cf' % (sym1, sym2, sym3)
                else:
                    fname = filename
                print('Writing to %s' % fname, file=self.txt, flush=True)
                write_3cf(fname, Rgrid, Sgrid, Tgrid, output[(sym1, sym2)])

        self.timer.stop('run_offsite3c')
        if not write:
            return output


    def calculate(self, selected, e1, e2, e3, R, grid, area, Sgrid, Tgrid,
                  xc='LDA'):
        self.timer.start('calculate_offsite3c')

        # TODO: boilerplate
        # common for all integrals (not wf-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y -R)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y - R) / r2  # cosine of theta_2
        s1 = x / r1  # sine of theta_1
        s2 = x / r2  # sine of theta_2
        aux = area * x

        rho12 = e1.electron_density(r1) + e2.electron_density(r2)

        self.timer.start('vxc')
        if xc in ['LDA', 'PW92']:
            xc = VXC_PW92_Spline()
            vxc12 = xc(rho12)
        else:
            xc = LibXC(xc)
            drho1 = e1.electron_density(r1, der=1)
            drho2 = e2.electron_density(r2, der=1)
            grad_rho12_x = drho1 * s1 + drho2 * s2
            grad_rho12_y = drho1 * c1 + drho2 * c2
            sigma12 = grad_rho12_x**2 + grad_rho12_y**2
            out = xc.compute_vxc(rho12, sigma12)
            vxc12 = out['vrho']
            if xc.add_gradient_corrections:
                vsigma12 = out['vsigma']
        self.timer.stop('vxc')
        self.timer.stop('prelude')


        def integrands(phi):
            rA = np.sqrt((x - x0*np.cos(phi))**2 + (y - y0 )**2 \
                         + (x0*np.sin(phi))**2)

            V = e3.neutral_atom_potential(rA) - vxc12

            rho3 = e3.electron_density(rA)
            rho123 = rho12 + rho3

            if isinstance(xc, VXC_PW92_Spline):
                V += xc(rho123)
            else:
                drdx = (x - x0*np.cos(phi)) / rA
                drdy = (y - y0) / rA
                drdphi = ((x - x0*np.cos(phi))*x0*np.sin(phi) \
                          + x0*np.sin(phi)*x0*np.cos(phi)) / rA

                drho3 = e3.electron_density(rA, der=1)
                grad_rho3_phi = drho3 * drdphi / x

                grad_rho123_x = grad_rho12_x + drho3 * drdx
                grad_rho123_y = grad_rho12_y + drho3 * drdy
                sigma123 = grad_rho123_x**2 + grad_rho123_y**2 \
                           + grad_rho3_phi**2
                out = xc.compute_vxc(rho123, sigma123)
                V += out['vrho']
                if xc.add_gradient_corrections:
                    vsigma123 = out['vsigma']

            V *= aux

            vals = np.zeros(len(selected))
            for i, key in enumerate(selected):
                integral, nl1, nl2 = key
                lm1, lm2 = integral.split('_')
                Phi1 = sph_phi(lm1, phi)
                Phi2 = sph_phi(lm2, phi)
                vals[i] = np.dot(Rnl12[key], V) * Phi1 * Phi2

                if xc.add_gradient_corrections:
                    dPhi1 = sph_phi_der(lm1, phi)
                    dPhi2 = sph_phi_der(lm2, phi)
                    grad_phi_x = grad_phi_x_nophi[key] * Phi1 * Phi2
                    grad_phi_y = grad_phi_y_nophi[key] * Phi1 * Phi2
                    grad_phi_phi = Rnl12[key] \
                                   * (dPhi1 * Phi2 + Phi1 * dPhi2) / x
                    grad_rho123_grad_phi = grad_rho123_x * grad_phi_x \
                                           + grad_rho123_y * grad_phi_y \
                                           + grad_rho3_phi * grad_phi_phi
                    grad_rho12_grad_phi = grad_rho12_x * grad_phi_x \
                                           + grad_rho12_y * grad_phi_y
                    vals[i] += 2. * np.dot(vsigma123 * grad_rho123_grad_phi \
                                           - vsigma12 * grad_rho12_grad_phi,
                                           aux)
            return vals


        # Pre-calculate the phi-indedendent wave function parts/products
        Rnl12 = {}
        if xc.add_gradient_corrections:
            grad_phi_x_nophi, grad_phi_y_nophi = {}, {}
            dr1dx = x/r1
            dc1dx = -y*dr1dx / r1**2
            ds1dx = (r1 - x*dr1dx) / r1**2
            dr2dx = x/r2
            dc2dx = -(y - R)*dr2dx / r2**2
            ds2dx = (r2 - x*dr2dx) / r2**2
            dr1dy = y/r1
            dc1dy = (r1 - y*dr1dy) / r1**2
            ds1dy = -x*dr1dy / r1**2
            dr2dy = (y - R)/r2
            dc2dy = (r2 - (y - R)*dr2dy) / r2**2
            ds2dy = -x*dr2dy / r2**2

        for key in selected:
            integral, nl1, nl2 = key
            Rnl1 = e1.Rnl(r1, nl1)
            Rnl2 = e2.Rnl(r2, nl2)
            lm1, lm2 = integral.split('_')
            Theta1 = sph_nophi(lm1, c1, s1)
            Theta2 = sph_nophi(lm2, c2, s2)
            Rnl12[key] = Rnl1 * Rnl2 * Theta1 * Theta2

            if xc.add_gradient_corrections:
                dRnl1 = e1.Rnl(r1, nl1, der=1)
                dRnl2 = e2.Rnl(r2, nl2, der=1)
                dTheta1 = sph_nophi_der(lm1, c1, s1)
                dTheta2 = sph_nophi_der(lm2, c2, s2)
                grad_phi_x_nophi[key] = (dRnl1*s1*Rnl2 + Rnl1*dRnl2*s2) \
                                        * Theta1 * Theta2
                grad_phi_x_nophi[key] += Rnl1 * Rnl2 \
                         * ((dTheta1[0]*dc1dx + dTheta1[1]*ds1dx) * Theta2 \
                            + Theta1 * (dTheta2[0]*dc2dx + dTheta2[1]*ds2dx))
                grad_phi_y_nophi[key] = (dRnl1*c1*Rnl2 + Rnl1*dRnl2*c2) \
                                        * Theta1 * Theta2
                grad_phi_y_nophi[key] += Rnl1 * Rnl2 \
                         * ((dTheta1[0]*dc1dy + dTheta1[1]*ds1dy) * Theta2 \
                            + Theta1 * (dTheta2[0]*dc2dy + dTheta2[1]*ds2dy))


        # Breakpoints and precision thresholds for the integration
        break_points = 2 * np.pi * np.linspace(0., 1., num=5, endpoint=True)
        epsrel = 1e-2
        epsabs = 1e-5

        sym1, sym2, sym3 = e1.get_symbol(), e2.get_symbol(), e3.get_symbol()

        # First the values for rCM = 0
        x0 = 0.
        y0 = 0.5 * R
        vals, err = quad_vec(integrands, 0., 2*np.pi,
                             epsrel=epsrel, epsabs=epsabs,
                             points=break_points)

        results = {key: [] for key in selected}
        for i, key in enumerate(selected):
            integral, nl1, nl2 = key
            lm1, lm2 = integral.split('_')
            vals[i] += e3.pp.get_nonlocal_integral(sym1, sym2, sym3, x0, y0, R,
                                                   nl1, nl2, lm1, lm2)
            results[key].append(vals[i])

        # Now the actual grid
        rmin = 1e-2

        for r in Sgrid:
            for a in Tgrid:
                x0 = r * np.sin(a)
                y0 = 0.5 * R + r * np.cos(a)
                if ((x0**2 + y0**2) < rmin or (x0**2 + (y0-R)**2) < rmin):
                    # Third atom too close to one of the first two atoms
                    vals = np.zeros(len(selected))
                else:
                    vals, err = quad_vec(integrands, 0., 2*np.pi,
                                         epsrel=epsrel, epsabs=epsabs,
                                         points=break_points)

                    for i, key in enumerate(selected):
                        integral, nl1, nl2 = key
                        lm1, lm2 = integral.split('_')
                        vals[i] += e3.pp.get_nonlocal_integral(sym1, sym2, sym3,
                                                               x0, y0, R, nl1,
                                                               nl2, lm1, lm2)

                for i, key in enumerate(selected):
                    results[key].append(vals[i])

        self.timer.stop('calculate_offsite3c')
        return results

    def run_repulsion(self, e3, Rgrid, Sgrid, Tgrid, ntheta=150, nr=50,
                      wflimit=1e-7, xc='LDA', write=True, filename=None):
        """
        Calculates the three-center repulsive energy contribution.

        Parameters
        ----------
        e3 : AtomicBase-like object
            Object with atomic properties for the third atom.
        Rgrid, Sgrid, Tgrid : list or array
            Lists with distances defining the three-atom geometries.
        write : bool, optional
            Whether to write the integrals to file (the default)
            or return them as a dictionary instead.
        filename : str, optional
            File name to use in case write=True. The default (None)
            implies that a '<el1>-<el2>_repulsion3c_<el3>.3cf'
            template is used.
        ntheta, nr, wflimit, xc :
            See SlaterKosterTable.run().

        Returns
        -------
        output : dict of dict of np.ndarray, optional
            Dictionary with the values for each el1-el2 pair
            and integral type (only 's_s' in this case).
            Only returned if write=False.
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Three-center repulsion with %s' % e3.get_symbol(), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        self.timer.start('run_repulsion3c')
        wf_range = self.get_range(wflimit)
        numST = len(Sgrid) * len(Tgrid)
        output = {}

        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2, sym3 = e1.get_symbol(), e2.get_symbol(), e3.get_symbol()

            output[(sym1, sym2)] = {'s_s': []}

            for i, R in enumerate(Rgrid):
                print('Starting for R=%.3f' % R, file=self.txt, flush=True)

                d = None
                if R < 2 * wf_range:
                    grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)
                    if len(grid) > 0:
                        d = self.calculate_repulsion(e1, e2, e3, R, grid, area,
                                                Sgrid=Sgrid, Tgrid=Tgrid, xc=xc)

                if d is None:
                    d = np.zeros(1 + numST)

                output[(sym1, sym2)]['s_s'].append(d)

            if write:
                if filename is None:
                    fname = '%s-%s_repulsion3c_%s.3cf' % (sym1, sym2, sym3)
                else:
                    fname = filename
                print('Writing to %s' % fname, file=self.txt, flush=True)
                write_3cf(fname, Rgrid, Sgrid, Tgrid, output[(sym1, sym2)])

        self.timer.stop('run_repulsion3c')
        if not write:
            return output

    def calculate_repulsion(self, e1, e2, e3, R, grid, area, Sgrid,
                            Tgrid, xc='LDA'):
        self.timer.start('calculate_repulsion3c')

        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (R - y)**2)
        aux = area * x

        rho1 = e1.electron_density(r1, only_valence=True)
        rho2 = e2.electron_density(r2, only_valence=True)
        rho12 = rho1 + rho2

        self.timer.start('vxc')
        if xc in ['LDA', 'PW92']:
            exc_spl = EXC_PW92_Spline()
            exc1 = np.sum(rho1 * exc_spl(rho1) * aux)
            exc2 = np.sum(rho2 * exc_spl(rho2) * aux)
            exc12 = np.sum(rho12 * exc_spl(rho12) * aux)

            vxc_spl = VXC_PW92_Spline()
            evxc1 = np.sum(rho1 * vxc_spl(rho1) * aux)
            evxc2 = np.sum(rho2 * vxc_spl(rho2) * aux)
            evxc12 = np.sum(rho12 * vxc_spl(rho12) * aux)
        else:
            xc = LibXC(xc)

            sigma = e1.electron_density(r1, der=1, only_valence=True)**2
            out = xc.compute_all(rho1, sigma)
            exc1 = np.sum(rho1 * out['zk'] * aux)
            evxc1 = np.sum(rho1 * out['vrho'] * aux)
            if xc.add_gradient_corrections:
                evxc1 += 2. * np.sum(out['vsigma'] * sigma * aux)

            sigma = e2.electron_density(r2, der=1, only_valence=True)**2
            out = xc.compute_all(rho2, sigma)
            exc2 = np.sum(rho2 * out['zk'] * aux)
            evxc2 = np.sum(rho2 * out['vrho'] * aux)
            if xc.add_gradient_corrections:
                evxc2 += 2. * np.sum(out['vsigma'] * sigma * aux)

            c1 = y / r1  # cosine of theta_1
            c2 = (y - R) / r2  # cosine of theta_2
            s1 = x / r1  # sine of theta_1
            s2 = x / r2  # sine of theta_2
            drho1 = e1.electron_density(r1, der=1)
            drho2 = e2.electron_density(r2, der=1)
            grad_rho1_x, grad_rho1_y = drho1 * s1, drho1 * c1
            grad_rho2_x, grad_rho2_y = drho2 * s2, drho2 * c2
            grad_rho12_x = grad_rho1_x + grad_rho2_x
            grad_rho12_y = grad_rho1_y + grad_rho2_y
            sigma = grad_rho12_x**2 + grad_rho12_y**2
            out = xc.compute_all(rho12, sigma)
            exc12 = np.sum(rho12 * out['zk'] * aux)
            evxc12 = np.sum(rho12 * out['vrho'] * aux)
            if xc.add_gradient_corrections:
                evxc12 += 2. * np.sum(out['vsigma'] * sigma * aux)

        self.timer.stop('vxc')
        self.timer.stop('prelude')

        def integrands(phi):
            rA = np.sqrt((x - x0*np.cos(phi))**2 + (y - y0)**2 \
                         + (x0*np.sin(phi))**2)

            rho3 = e3.electron_density(rA, only_valence=True)
            rho13 = rho1 + rho3
            rho23 = rho2 + rho3
            rho123 = rho12 + rho3

            if xc in ['LDA', 'PW92']:
                exc3 = np.sum(rho3 * exc_spl(rho3) * aux)
                exc13 = np.sum(rho13 * exc_spl(rho13) * aux)
                exc23 = np.sum(rho23 * exc_spl(rho23) * aux)
                exc123 = np.sum(rho123 * exc_spl(rho123) * aux)
                evxc3 = np.sum(rho3 * vxc_spl(rho3) * aux)
                evxc13 = np.sum(rho13 * vxc_spl(rho13) * aux)
                evxc23 = np.sum(rho23 * vxc_spl(rho23) * aux)
                evxc123 = np.sum(rho123 * vxc_spl(rho123) * aux)
            else:
                drdx = (x - x0*np.cos(phi)) / rA
                drdy = (y - y0) / rA
                drdphi = ((x - x0*np.cos(phi))*x0*np.sin(phi) \
                          + x0*np.sin(phi)*x0*np.cos(phi)) / rA

                drho3 = e3.electron_density(rA, der=1, only_valence=True)
                grad_rho3_x = drho3 * drdx
                grad_rho3_y = drho3 * drdy
                grad_rho3_phi = drho3 * drdphi / x

                sigma = drho3**2 * (drdx**2 + drdy**2 + (drdphi/x)**2)
                out = xc.compute_all(rho3, sigma)
                exc3 = np.sum(rho3 * out['zk'] * aux)
                evxc3 = np.sum(rho3 * out['vrho'] * aux)
                if xc.add_gradient_corrections:
                    evxc3 += 2. * np.sum(out['vsigma'] * sigma * aux)

                sigma = (grad_rho1_x + grad_rho3_x)**2 \
                        + (grad_rho1_y + grad_rho3_y)**2 \
                        + grad_rho3_phi**2
                out = xc.compute_all(rho13, sigma)
                exc13 = np.sum(rho13 * out['zk'] * aux)
                evxc13 = np.sum(rho13 * out['vrho'] * aux)
                if xc.add_gradient_corrections:
                    evxc13 += 2. * np.sum(out['vsigma'] * sigma * aux)

                sigma = (grad_rho2_x + grad_rho3_x)**2 \
                        + (grad_rho2_y + grad_rho3_y)**2 \
                        + grad_rho3_phi**2
                out = xc.compute_all(rho23, sigma)
                exc23 = np.sum(rho23 * out['zk'] * aux)
                evxc23 = np.sum(rho23 * out['vrho'] * aux)
                if xc.add_gradient_corrections:
                    evxc23 += 2. * np.sum(out['vsigma'] * sigma * aux)

                sigma = (grad_rho12_x + grad_rho3_x)**2 \
                        + (grad_rho12_y + grad_rho3_y)**2 \
                        + grad_rho3_phi**2
                out = xc.compute_all(rho123, sigma)
                exc123 = np.sum(rho123 * out['zk'] * aux)
                evxc123 = np.sum(rho123 * out['vrho'] * aux)
                if xc.add_gradient_corrections:
                    evxc123 += 2. * np.sum(out['vsigma'] * sigma * aux)

            Exc = exc123 - exc12 - exc13 - exc23 + exc1 + exc2 + exc3
            Evxc = evxc123 - evxc12 - evxc13 - evxc23 + evxc1 + evxc2 + evxc3
            vals = np.array([Exc, -Evxc])
            return vals

        # Breakpoints and precision thresholds for the integration
        break_points = 2 * np.pi * np.linspace(0., 1., num=5, endpoint=True)
        epsrel = 1e-2
        epsabs = 1e-5

        # First the values for rCM = 0
        x0 = 0.
        y0 = 0.5 * R
        vals, err = quad_vec(integrands, 0., 2*np.pi,
                             epsrel=epsrel, epsabs=epsabs,
                             points=break_points)
        results = [sum(vals)]

        # Now the actual grid
        rmin = 1e-2

        for r in Sgrid:
            for a in Tgrid:
                x0 = r * np.sin(a)
                y0 = 0.5 * R + r * np.cos(a)
                if ((x0**2 + y0**2) < rmin or (x0**2 + (y0-R)**2) < rmin):
                    # Third atom too close to one of the first two atoms
                    vals = [0.]*2
                else:
                    vals, err = quad_vec(integrands, 0., 2*np.pi,
                                         epsrel=epsrel, epsabs=epsabs,
                                         points=break_points)

                results.append(sum(vals))

        self.timer.stop('calculate_repulsion3c')
        return results
