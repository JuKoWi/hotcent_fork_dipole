import numpy as np
from scipy.integrate import quad_vec
from scipy.interpolate import SmoothBivariateSpline
from hotcent.interpolation import CubicSplineFunction
from hotcent.orbitals import (calculate_slako_coeff, ANGULAR_MOMENTUM,
                              ORBITAL_LABELS, ORBITALS)
from hotcent.slako import SlaterKosterTable
from hotcent.slako import INTEGRALS as INTEGRALS_2c
from hotcent.threecenter import (INTEGRALS, INTEGRAL_PAIRS, select_integrals,
                                 select_orbitals, sph_nophi, sph_phi)
from hotcent.xc import LibXC, VXC_PW92_Spline, XC_PW92
import _hotcent


class Offsite3cTable(SlaterKosterTable):
    def __init__(self, *args, **kwargs):
        SlaterKosterTable.__init__(self, *args, **kwargs)
        self.overlap_fct = {}  # dictionary with core-valence overlap functions

    def run(self, e3, Rgrid, Sgrid, Tgrid, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA'):
        """ Calculates off-site three-center integrals.

        parameters:
        ------------
        e3: AtomicDFT object for the third atom
        Rgrid, Sgrid, Tgrid: lists defining the three-atom geometries

        other parameters:
        -----------------
        see SlaterKosterTable.run()
        """
        assert xc == 'LDA', 'Functionals other than LDA are not yet implemented'

        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Off-site three-center calculations with %s' % e3.get_symbol(),
              file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        self.timer.start('run_offsite3c')
        self.wf_range = self.get_range(wflimit)
        numST = len(Sgrid) * len(Tgrid)

        for p, (e1, e2) in enumerate(self.pairs):
            filename = '%s-%s_offsite3c_%s.3cf' % \
                       (e1.get_symbol(), e2.get_symbol(), e3.get_symbol())
            print('Writing to %s' % filename, file=self.txt, flush=True)

            selected = select_integrals(e1, e2)

            print('Integrals:', end=' ', file=self.txt)
            for s in selected:
                print(s[0], end=' ', file=self.txt)
            print(file=self.txt, flush=True)

            rmin, rmax = 0., np.max(Sgrid) + 0.5*np.max(Rgrid)
            self.build_core_valence_overlap(e1, e3, rmin, rmax, numr=100)
            self.build_core_valence_overlap(e2, e3, rmin, rmax, numr=100)

            with open(filename, 'w') as f:
                f.write('%.6f %.6f %d\n' % (Rgrid[0], Rgrid[-1], len(Rgrid)))
                f.write('%.6f %.6f %d\n' % (Sgrid[0], Sgrid[-1], len(Sgrid)))
                f.write('%d\n' % len(Tgrid))
                f.write(' '.join([key[0] for key in selected]) + '\n')

            for i, R in enumerate(Rgrid):
                print('Starting for R=%.3f' % R, file=self.txt, flush=True)

                d = None
                if R < 2 * self.wf_range:
                    grid, area = self.make_grid(R, nt=ntheta, nr=nr)
                    if len(grid) > 0:
                        d = self.calculate(selected, e1, e2, e3, R, grid, area,
                                           Sgrid=Sgrid, Tgrid=Tgrid, xc=xc)

                if d is None:
                    d = {key: np.zeros(1 + numST) for key in selected}

                with open(filename, 'a') as f:
                    for j in range(1 + numST):
                        f.write(' '.join(['%.6e' % d[key][j]
                                          for key in selected]))
                        f.write('\n')

        self.timer.stop('run_offsite3c')

    def calculate(self, selected, e1, e2, e3, R, grid, area, Sgrid, Tgrid,
                  xc='LDA'):
        self.timer.start('calculate_offsite3c')
        assert xc == 'LDA', 'Functionals other than LDA are not yet implemented'

        # TODO: boilerplate
        # common for all integrals (not wf-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (R - y)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y - R) / r2  # cosine of theta_2
        s1 = np.sqrt(1. - c1**2)  # sine of theta_1
        s2 = np.sqrt(1. - c2**2)  # sine of theta_2

        self.timer.start('vrho')
        rho = e1.electron_density(r1) + e2.electron_density(r2)
        if xc in ['LDA', 'PW92']:
            xc = VXC_PW92_Spline()
            vxc = xc(rho)
            self.timer.stop('vrho')
        else:
            xc = LibXC(xc)
            drho1 = e1.electron_density(r1, der=1)
            drho2 = e2.electron_density(r2, der=1)
            grad_x = drho1 * s1
            grad_x += drho2 * s2
            grad_y = drho1 * c1
            grad_y += drho2 * c2
            sigma = np.sqrt(grad_x**2 + grad_y**2)**2
            out = xc.compute_all(rho, sigma)
            vxc = out['vrho']
            self.timer.stop('vrho')
            self.timer.start('vsigma')
            # add gradient corrections to vxc
            # provided that we have enough points
            # (otherwise we get "dfitpack.error:
            # (m>=(kx+1)*(ky+1)) failed for hidden m")
            if out['vsigma'] is not None and len(x) > 16:
                splx = SmoothBivariateSpline(x, y, out['vsigma'] * grad_x)
                sply = SmoothBivariateSpline(x, y, out['vsigma'] * grad_y)
                vxc += -2. * splx(x, y, dx=1, dy=0, grid=False)
                vxc += -2. * sply(x, y, dx=0, dy=1, grid=False)
            self.timer.stop('vsigma')

        assert np.shape(vxc) == (len(grid),)
        self.timer.stop('prelude')

        vxc123 = np.zeros_like(r1)

        def integrands(phi):
            rA = np.sqrt((x0*np.cos(phi) - x)**2 + (y0 - y)**2 \
                         + (x0*np.sin(phi))**2)

            V = e3.nuclear_potential(rA) + e3.hartree_potential(rA) - vxc
            rho123 = rho + e3.electron_density(rA)
            if isinstance(xc, XC_PW92):
                _hotcent.vxc_lda(rho123, vxc123)
                V += vxc123
            elif isinstance(xc, VXC_PW92_Spline):
                V += xc(rho123)
            V *= area * x

            vals = np.zeros(len(selected))
            for i, (integral, nl1, nl2) in enumerate(selected):
                lm1, lm2 = integral.split('_')
                vals[i] = np.dot(Rnl12[(integral, nl1, nl2)], V)
                vals[i] *= sph_phi(lm1, phi) * sph_phi(lm2, phi)
            return vals


        def get_pseudopotential_term(integral, nl1, nl2, phi):
            x3 = x0 * np.cos(phi)
            y3 = x0 * np.sin(phi)
            z3 = y0
            v13 = np.array([x3, y3, z3])
            r13 = np.linalg.norm(v13)
            v13 /= r13
            v23 = np.array([x3, y3, z3-R])
            r23 = np.linalg.norm(v23)
            v23 /= r23
            pseudo = 0.
            lm1, lm2 = INTEGRAL_PAIRS[integral]
            l1 = 'spdf'.index(lm1[0])
            l2 = 'spdf'.index(lm2[0])
            sym1 = e1.get_symbol()
            sym2 = e2.get_symbol()
            sym3 = e3.get_symbol()

            for n3, l3, nl3 in e3.list_states():
                if nl3 in e3.valence:
                    continue

                for lm3 in ORBITALS[l3]:
                    term = e3.get_eigenvalue(nl3)
                    ilm3 = ORBITAL_LABELS.index(lm3)

                    # first atom
                    S3 = 0.
                    x, y, z = v13
                    ilm = ORBITAL_LABELS.index(lm1)
                    minl = min(l1, l3)
                    for tau in range(minl+1):
                        skint = self.get_overlap(sym1, sym3, nl1, nl3, tau, r13)
                        if ilm3 >= ilm:
                            coef = calculate_slako_coeff(x, y, z, ilm+1,
                                                         ilm3+1, tau+1)
                        else:
                            coef = calculate_slako_coeff(x, y, z, ilm3+1,
                                                         ilm+1, tau+1)
                            coef *= (-1)**(l1 + l3)
                        S3 += coef * skint
                    term *= S3

                    # second atom
                    S3 = 0.
                    x, y, z = v23
                    ilm = ORBITAL_LABELS.index(lm2)
                    minl = min(l2, l3)
                    for tau in range(minl+1):
                        skint = self.get_overlap(sym2, sym3, nl2, nl3, tau, r23)
                        if ilm3 >= ilm:
                            coef = calculate_slako_coeff(x, y, z, ilm+1,
                                                         ilm3+1, tau+1)
                        else:
                            coef = calculate_slako_coeff(x, y, z, ilm3+1,
                                                         ilm+1, tau+1)
                            coef *= (-1)**(l2 + l3)
                        S3 += coef * skint
                    term *= S3

                    pseudo += term
            return pseudo

        # Pre-calculate the phi-indedendent wave function products
        Rnl12 = {}
        for key in selected:
            integral, nl1, nl2 = key
            lm1, lm2 = integral.split('_')
            Rnl12[key] = e1.Rnl(r1, nl1) * e2.Rnl(r2, nl2) \
                         * sph_nophi(lm1, c1, s1) * sph_nophi(lm2, c2, s2)

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

        results = {key: [] for key in selected}
        for i, key in enumerate(selected):
            pseudo = get_pseudopotential_term(*key, phi=0.)
            vals[i] -= pseudo
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
                        pseudo = get_pseudopotential_term(*key, phi=0.)
                        vals[i] -= pseudo

                for i, key in enumerate(selected):
                    results[key].append(vals[i])

        self.timer.stop('calculate_offsite3c')
        return results

    def build_core_valence_overlap(self, e1, e3, rmin, rmax, numr, xc='LDA'):
        """ Builds the core-valence overlap integral interpolators. """
        for nl1 in e1.valence:
            l1 = 'spdf'.index(nl1[1])

            for n3, l3, nl3 in e3.list_states():
                if nl3 in e3.valence:
                    continue

                for tau in range(min(l1, l3) + 1):
                    key = (e1.get_symbol(), e3.get_symbol(), nl1, nl3, tau)
                    if key in self.overlap_fct:
                        continue

                    print('Calculating overlaps for ', key)
                    rval = np.linspace(rmin, rmax, num=numr, endpoint=True)

                    if l1 < l3:
                        sk_integral = nl1[1] + nl3[1] + 'spdf'[tau]
                        sk_selected = [(sk_integral, nl1, nl3)]
                    else:
                        sk_integral = nl3[1] + nl1[1] + 'spdf'[tau]
                        sk_selected = [(sk_integral, nl3, nl1)]

                    iint = INTEGRALS_2c.index(sk_integral)

                    sval = []
                    for r13 in rval:
                        grid, area = self.make_grid(r13, nt=150, nr=50)
                        # TODO: H integrals are not needed here
                        if l1 < l3:
                            s, h, h2 = self.calculate_mels(sk_selected, e1, e3,
                                                        r13, grid, area, xc=xc)
                        else:
                            s, h, h2 = self.calculate_mels(sk_selected, e3, e1,
                                                        r13, grid, area, xc=xc)
                        if len(grid) == 0:
                            assert abs(s[iint]) < 1e-24
                        sval.append(s[iint])

                    self.overlap_fct[key] = CubicSplineFunction(rval, sval)

    def get_overlap(self, sym1, sym2, nl1, nl2, tau, r):
        """ Returns the orbital overlap, evaluated by interpolation. """
        key = (sym1, sym2, nl1, nl2, tau)
        s = self.overlap_fct[key](r, der=0)
        return s
