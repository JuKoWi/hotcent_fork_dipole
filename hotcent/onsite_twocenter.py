#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from scipy.interpolate import SmoothBivariateSpline
from ase.data import atomic_numbers, atomic_masses
from hotcent.xc import XC_PW92, LibXC
from hotcent.slako import (g, INTEGRAL_PAIRS, INTEGRALS, NUMSK,
                           select_integrals, SlaterKosterTable,
                           tail_smoothening)


class Onsite2cTable(SlaterKosterTable):
    def _write_skf(self, handle, pair):
        """ Write to SKF file format; this function
        is an adaptation of hotbit.io.hbskf

        By default the 'simple' format is chosen, and the 'extended'
        format is only used when necessary (i.e. when there are f-electrons
        included in the valence of one of the elements).
        """
        # TODO: boilerplate (similar to SlaterKosterTable.write_skf)

        symbols = (self.ela.get_symbol(), self.elb.get_symbol())
        if pair == symbols:
             index = 0
        elif pair == symbols[::-1]:
             index = 1
        else:
             msg = 'Requested ' + str(pair) + ' pair, but this calculator '
             msg += 'is restricted to the %s-%s pair.' % symbols
             raise ValueError(msg)

        extended_format = any(['f' in nl
                               for nl in (self.ela.valence + self.elb.valence)])
        if extended_format:
            print('@', file=handle)

        grid_dist = self.Rgrid[1] - self.Rgrid[0]
        grid_npts = len(self.tables[index])
        grid_npts += int(self.Rgrid[0] / (self.Rgrid[1] - self.Rgrid[0])) - 1
        print("%.12f, %d" % (grid_dist, grid_npts), file=handle)

        el1, el2 = self.ela.get_symbol(), self.elb.get_symbol()

        m = atomic_masses[atomic_numbers[symbols[index]]]
        print("%.3f, 19*0.0" % m, file=handle)

        # Table containing the Slater-Koster integrals
        if extended_format:
            indices = range(NUMSK)
        else:
            indices = [INTEGRALS.index(name) for name in INTEGRALS
                       if 'f' not in name[:2]]

        if self.Rgrid[0] != 0:
            n = int(np.round(self.Rgrid[0] / (self.Rgrid[1] - self.Rgrid[0])))
            for i in range(n-1):
                print('%d*0.0,' % len(indices), file=handle)

        ct, theader = 0, ''
        for i in range(len(self.tables[index])):
            line = ''
            for j in indices:
                if self.tables[index][i, j] == 0:
                    ct += 1
                    theader = str(ct) + '*0.0 '
                else:
                    ct = 0
                    line += theader
                    theader = ''
                    line += '{0: 1.12e}  '.format(self.tables[index][i, j])

            if theader != '':
                ct = 0
                line += theader

            print(line, file=handle)

    def run(self, e2, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50,
            wflimit=1e-7, smoothen_tails=True, superposition='density',
            xc='LDA'):
        """ Calculates on-site two-center integrals.

        parameters:
        ------------
        e2: AtomicDFT object for the second atom

        other parameters:
        -----------------
        see SlaterKosterTable.run()
        """
        assert superposition in ['density', 'potential']

        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('On-site two-center calculations with %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        self.timer.start('run_onsite2c')
        self.wf_range = self.get_range(wflimit)
        self.Rgrid = rmin + dr * np.arange(N)
        self.tables = [np.zeros((len(self.Rgrid), NUMSK))
                       for i in range(self.nel)]

        for p, (e1a, e1b) in enumerate(self.pairs):
            selected = select_integrals(e1a, e1b)

            print('Integrals:', end=' ', file=self.txt)
            for s in selected:
                print(s[0], end=' ', file=self.txt)
            print(file=self.txt, flush=True)

            data = {key: [] for key in selected}

            for i, R in enumerate(self.Rgrid):
                d = None
                if R < 2 * self.wf_range:
                    grid, area = self.make_grid(R, nt=ntheta, nr=nr)
                    if i % 10 == 0:
                        print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                              file=self.txt, flush=True)

                    if len(grid) > 0:
                        d = self.calculate(selected, e1a, e1b, e2, R, grid,
                                    area, superposition=superposition, xc=xc)

                if d is None:
                    d = {key: 0. for key in selected}

                for key in selected:
                    data[key].append(d[key])

            for key in selected:
                data[key] = np.array(data[key])
                integral, nl1, nl2 = key
                index = INTEGRALS.index(integral)
                if smoothen_tails:
                    # Smooth the curves near the cutoff
                    self.tables[p][:, index] = tail_smoothening(self.Rgrid,
                                                                data[key])
                else:
                    self.tables[p][:, index] = data[key]

            pair = (e1a.get_symbol(), e1b.get_symbol())
            fn = '%s-%s_onsite2c_%s.skf' % (pair[0], pair[1], e2.get_symbol())
            with open(fn, 'w') as handle:
                self._write_skf(handle, pair)

        self.timer.stop('run_onsite2c')

    def calculate(self, selected, e1a, e1b, e2, R, grid, area,
                  superposition='potential', xc='LDA'):
        self.timer.start('calculate_onsite2c')

        assert superposition == 'density'

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
        veff = e1a.nuclear_potential(r1) + e1a.hartree_potential(r1)
        veff += e2.nuclear_potential(r2) + e2.hartree_potential(r2)
        if xc in ['LDA', 'PW92']:
            xc = XC_PW92()
            veff += xc.vxc(rho)
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
            veff += out['vrho']
            self.timer.stop('vrho')
            self.timer.start('vsigma')
            # add gradient corrections to vxc
            # provided that we have enough points
            # (otherwise we get "dfitpack.error:
            # (m>=(kx+1)*(ky+1)) failed for hidden m")
            if out['vsigma'] is not None and len(x) > 16:
                splx = SmoothBivariateSpline(x, y, out['vsigma'] * grad_x)
                sply = SmoothBivariateSpline(x, y, out['vsigma'] * grad_y)
                veff += -2. * splx(x, y, dx=1, dy=0, grid=False)
                veff += -2. * sply(x, y, dx=0, dy=1, grid=False)
            self.timer.stop('vsigma')

        assert np.shape(veff) == (len(grid),)
        self.timer.stop('prelude')

        V = veff - e1a.effective_potential(r1)
        sym1a, sym1b, sym2 = e1a.get_symbol(), e1b.get_symbol(), e2.get_symbol()

        results = {}
        for key in selected:
            integral, nl1, nl2 = key

            Rnl1 = e1a.Rnl(r1, nl1)
            Rnl2 = e1b.Rnl(r1, nl2)
            gphi = g(c1, c1, s1, s1, integral)
            aux = gphi * area * x
            val = np.sum(Rnl1 * Rnl2 * V * aux)

            lm1, lm2 = INTEGRAL_PAIRS[integral]
            val += e2.pp.get_nonlocal_integral(sym1a, sym1b, sym2, 0., R, 0.,
                                               nl1, nl2, lm1, lm2)
            results[key] = val

        self.timer.stop('calculate_onsite2c')
        return results
