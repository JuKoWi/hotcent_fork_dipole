import numpy as np
from scipy.integrate import quad_vec
from scipy.interpolate import SmoothBivariateSpline
from hotcent.orbitals import (calculate_slako_coeff, ANGULAR_MOMENTUM,
                              ORBITAL_LABELS, ORBITALS)
from hotcent.slako import SlaterKosterTable
from hotcent.slako import INTEGRALS as INTEGRALS_2c
from hotcent.threecenter import (INTEGRALS, INTEGRAL_PAIRS, select_integrals,
                                 select_orbitals, sph_nophi, sph_phi)
from hotcent.xc import XC_PW92, LibXC
import _hotcent


class Onsite3cTable(SlaterKosterTable):
    def run(self, e2, e3, Rgrid, Sgrid, Tgrid, ntheta=150, nr=50, wflimit=1e-7,
            xc='LDA'):
        """ Calculates on-site three-center integrals.

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
        print('On-site three-center calculations with %s' % e3.get_symbol(),
              file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        self.timer.start('run_onsite3c')
        self.wf_range = self.get_range(wflimit)
        numST = len(Sgrid) * len(Tgrid)

        selected = select_integrals(self.ela, self.elb)
        print('Integrals:', end=' ', file=self.txt)
        for s in selected:
            print(s[0], end=' ', file=self.txt)
        print(file=self.txt, flush=True)

        if e2.get_symbol() == e3.get_symbol():
            pairs = [(e2, e3)]
        else:
            pairs = [(e2, e3), (e3, e2)]

        for elc, eld in pairs:
            filename = '%s-%s_onsite3c_%s-%s.3cf' % (self.ela.get_symbol(),
                   self.elb.get_symbol(), elc.get_symbol(), eld.get_symbol())
            print('Writing to %s' % filename, file=self.txt, flush=True)

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
                        d = self.calculate(selected, self.ela, self.elb,
                                           elc, eld, R, grid, area,
                                           Sgrid=Sgrid, Tgrid=Tgrid, xc=xc)

                if d is None:
                    d = {key: np.zeros(1 + numST) for key in selected}

                with open(filename, 'a') as f:
                    for j in range(1 + numST):
                        f.write(' '.join(['%.6e' % d[key][j]
                                          for key in selected]))
                        f.write('\n')

        self.timer.stop('run_onsite3c')

    def calculate(self, selected, e1a, e1b, e2, e3, R, grid, area, Sgrid, Tgrid,
                  xc='LDA'):
        self.timer.start('calculate_onsite3c')
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
        rho = e1a.electron_density(r1) + e2.electron_density(r2)
        if xc in ['LDA', 'PW92']:
            xc = XC_PW92()
            vxc = xc.vxc(rho)
            self.timer.stop('vrho')
        else:
            xc = LibXC(xc)
            drho1 = e1a.electron_density(r1, der=1)
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

        rho1 = e1a.electron_density(r1)
        if isinstance(xc, XC_PW92):
            vxc1 = xc.vxc(rho1)
        vxc13 = np.zeros_like(r1)
        vxc123 = np.zeros_like(r1)

        def integrands(phi):
            rA = np.sqrt((x0*np.cos(phi) - x)**2 + (y0 - y)**2 \
                         + (x0*np.sin(phi))**2)
            rho3 = e3.electron_density(rA)
            rho13 = rho1 + rho3
            rho123 = rho + rho3

            if isinstance(xc, XC_PW92):
                _hotcent.vxc_lda(rho123, vxc123)
                _hotcent.vxc_lda(rho13, vxc13)
            V = area * x * (vxc123 - vxc13 - vxc + vxc1)

            vals = np.zeros(len(selected))
            for i, (integral, nl1, nl2) in enumerate(selected):
                lm1, lm2 = integral.split('_')
                vals[i] = np.dot(Rnl12[(integral, nl1, nl2)], V)
                vals[i] *= sph_phi(lm1, phi) * sph_phi(lm2, phi)
            return vals

        # Pre-calculate the phi-indedendent wave function products
        Rnl12 = {}
        for key in selected:
            integral, nl1, nl2 = key
            lm1, lm2 = integral.split('_')
            Rnl12[key] = e1a.Rnl(r1, nl1) * e1b.Rnl(r1, nl2) \
                         * sph_nophi(lm1, c1, s1) * sph_nophi(lm2, c1, s1)

        # Breakpoints and precision thresholds for the integration
        break_points = 2 * np.pi * np.linspace(0., 1., num=5, endpoint=True)
        epsrel = 1e-2
        epsabs = 1e-5

        # First the values for rCM = 0
        x0 = 0.
        y0 = 0.5 * R
        vals, err = quad_vec(integrands, 0., 2*np.pi, epsrel=epsrel,
                             epsabs=epsabs, points=break_points)

        results = {key: [] for key in selected}
        for i, key in enumerate(selected):
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
                    results[key].append(vals[i])

        self.timer.stop('calculate_onsite3c')
        return results
