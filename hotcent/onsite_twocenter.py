#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from ase.data import atomic_numbers, atomic_masses
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.slako import (dg, g, INTEGRAL_PAIRS, INTEGRALS, NUMSK,
                           select_integrals, tail_smoothening)
from hotcent.xc import XC_PW92, LibXC


class Onsite2cTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

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
        nzeros = int(np.round(self.Rgrid[0] / grid_dist)) - 1
        assert nzeros >= 0
        grid_npts += nzeros
        print("%.12f, %d" % (grid_dist, grid_npts), file=handle)

        m = atomic_masses[atomic_numbers[symbols[index]]]
        print("%.3f, 19*0.0" % m, file=handle)

        # Table containing the Slater-Koster integrals
        if extended_format:
            indices = range(NUMSK)
        else:
            indices = [INTEGRALS.index(name) for name in INTEGRALS
                       if 'f' not in name[:2]]

        for i in range(nzeros):
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
            xc='LDA', write=True, filename=None):
        """
        Calculates on-site two-center Hamiltonian integrals.

        Parameters
        ----------
        e2 : AtomicBase-like object
            Object with atomic properties for the second atom.
        write : bool, optional
            Whether to write the integrals to file (the default)
            or return them as a dictionary instead.
        filename : str, optional
            File name to use in case write=True. The default (None)
            implies that a '<el1a>-<el1b>_onsite2c_<el2>.skf'
            template is used.
        rmin, dr, N, ntheta, nr, wflimit, smoothen_tails, superposition, xc :
            See SlaterKosterTable.run().

        Returns
        -------
        output : dict of dict of np.ndarray, optional
            Dictionary with the values for each el1a-el1b pair
            and integral type. Only returned if write=False.
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('On-site two-center calculations with %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'
        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert superposition == 'density'

        self.timer.start('run_onsite2c')
        wf_range = self.get_range(wflimit)
        self.Rgrid = rmin + dr * np.arange(N)
        self.tables = [np.zeros((len(self.Rgrid), NUMSK))
                       for i in range(self.nel)]
        output = {}

        for p, (e1a, e1b) in enumerate(self.pairs):
            sym1a, sym1b = e1a.get_symbol(), e1b.get_symbol()

            print('Integrals for %s-%s pair:' % (sym1a, sym1b), end=' ',
                  file=self.txt)
            selected = select_integrals(e1a, e1b)
            for s in selected:
                print(s[0], end=' ', file=self.txt)
            print(file=self.txt, flush=True)

            output[(sym1a, sym1b)] = {integral: []
                                      for (integral, nl1, nl2) in selected}

            for i, R in enumerate(self.Rgrid):
                d = None
                if R < 2 * wf_range:
                    grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)
                    if i % 10 == 0:
                        print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                              file=self.txt, flush=True)

                    if len(grid) > 0:
                        d = self.calculate(selected, e1a, e1b, e2, R, grid,
                                    area, superposition=superposition, xc=xc)

                if d is None:
                    d = {key: 0. for key in selected}

                for key in selected:
                    integral, nl1, nl2 = key
                    output[(sym1a, sym1b)][integral].append(d[key])

            for key in selected:
                integral, nl1, nl2 = key
                output[(sym1a, sym1b)][integral] = np.array(
                                            output[(sym1a, sym1b)][integral])
                if smoothen_tails:
                    output[(sym1a, sym1b)][integral] = tail_smoothening(
                                self.Rgrid, output[(sym1a, sym1b)][integral])

                index = INTEGRALS.index(integral)
                self.tables[p][:, index] = output[(sym1a, sym1b)][integral]

            if write:
                if filename is None:
                    items = (sym1a, sym1b, e2.get_symbol())
                    fname = '%s-%s_onsite2c_%s.3cf' % items
                else:
                    fname = filename
                print('Writing to %s' % fname, file=self.txt, flush=True)
                with open(fname, 'w') as f:
                    self._write_skf(f, (sym1a, sym1b))

        self.timer.stop('run_onsite2c')
        if not write:
            return output

    def calculate(self, selected, e1a, e1b, e2, R, grid, area,
                  superposition='density', xc='LDA'):
        self.timer.start('calculate_onsite2c')

        assert superposition == 'density'

        # TODO: boilerplate
        # common for all integrals (not wf-dependent parts)
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (y - R)**2)
        c1 = y / r1  # cosine of theta_1
        c2 = (y - R) / r2  # cosine of theta_2
        s1 = x / r1  # sine of theta_1
        s2 = x / r2  # sine of theta_2

        rho = e1a.electron_density(r1) + e2.electron_density(r2)
        veff = e1a.neutral_atom_potential(r1)
        veff += e2.neutral_atom_potential(r2)

        self.timer.start('vxc')
        if xc in ['LDA', 'PW92']:
            xc = XC_PW92()
            veff += xc.vxc(rho)
        else:
            xc = LibXC(xc)
            drho1 = e1a.electron_density(r1, der=1)
            drho2 = e2.electron_density(r2, der=1)
            # TODO: boilerplate
            grad_rho_x = drho1 * s1 + drho2 * s2
            grad_rho_y = drho1 * c1 + drho2 * c2
            sigma = grad_rho_x**2 + grad_rho_y**2
            out = xc.compute_vxc(rho, sigma)
            veff += out['vrho']
            if xc.add_gradient_corrections:
                vsigma = out['vsigma']
                dr1dx = x/r1
                dc1dx = -y*dr1dx / r1**2
                ds1dx = (r1 - x*dr1dx) / r1**2
                dr1dy = y/r1
                dc1dy = (r1 - y*dr1dy) / r1**2
                ds1dy = -x*dr1dy / r1**2
        self.timer.stop('vxc')

        assert np.shape(veff) == (len(grid),)
        V = veff - e1a.effective_potential(r1)
        sym1a, sym1b, sym2 = e1a.get_symbol(), e1b.get_symbol(), e2.get_symbol()
        self.timer.stop('prelude')

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

            if xc.add_gradient_corrections:
                self.timer.start('vsigma')
                dRnl1 = e1a.Rnl(r1, nl1, der=1)
                dRnl2 = e1b.Rnl(r1, nl2, der=1)
                dgphi = dg(c1, c1, s1, s1, integral)
                dgphidx = (dgphi[0] + dgphi[1]) * dc1dx \
                          + (dgphi[2] + dgphi[3]) * ds1dx
                dgphidy = (dgphi[0] + dgphi[1]) * dc1dy \
                          + (dgphi[2] + dgphi[3]) * ds1dy
                grad_phi_x = (dRnl1 * Rnl2 + Rnl1 * dRnl2) * s1 * gphi
                grad_phi_x += Rnl1 * Rnl2 * dgphidx
                grad_phi_y = (dRnl1 * Rnl2 + Rnl1 * dRnl2) * c1 * gphi
                grad_phi_y += Rnl1 * Rnl2 * dgphidy
                grad_rho_grad_phi = grad_rho_x * grad_phi_x \
                                    + grad_rho_y * grad_phi_y
                val += 2. * np.sum(vsigma * grad_rho_grad_phi * area * x)
                self.timer.stop('vsigma')

            results[key] = val

        self.timer.stop('calculate_onsite2c')
        return results
