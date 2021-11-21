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
                           print_integral_overview, select_integrals,
                           tail_smoothening)
from hotcent.xc import XC_PW92, LibXC


class Onsite2cTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='monopolar',
                                     **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            smoothen_tails=True, superposition='density', xc='LDA'):
        """
        Calculates on-site two-center Hamiltonian integrals.

        Parameters
        ----------
        rmin, dr, N, ntheta, nr, wflimit, smoothen_tails, superposition, xc :
            See Offsite2cTable.run().
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('On-site two-center calculations with %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()
        self.timer.start('run_onsite2c')

        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'
        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert superposition == 'density'

        wf_range = self.get_range(wflimit)
        grid, area = self.make_grid(wf_range, nt=ntheta, nr=nr)

        self.Rgrid = rmin + dr * np.arange(N)
        self.tables = {}

        e1, e2 = self.ela, self.elb
        selected = select_integrals(e1, e1)
        print_integral_overview(e1, e1, selected, file=self.txt)

        for bas1a in range(len(e1.basis_sets)):
            for bas1b in range(len(e1.basis_sets)):
                self.tables[(bas1a, bas1b)] = np.zeros((N, NUMSK))

        for i, R in enumerate(self.Rgrid):
            if R < 2 * wf_range:
                if i % 10 == 0:
                    print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                          file=self.txt, flush=True)

                H = self.calculate(selected, e1, e2, R, grid, area,
                                   superposition=superposition, xc=xc)
                for key in selected:
                    integral, nl1a, nl1b = key
                    bas1a = e1.get_basis_set_index(nl1a)
                    bas1b = e1.get_basis_set_index(nl1b)
                    index = INTEGRALS.index(integral)
                    self.tables[(bas1a, bas1b)][i, index] = H[key]

        if smoothen_tails:
            for key in self.tables:
                for i in range(NUMSK):
                    self.tables[key][:, i] = tail_smoothening(self.Rgrid,
                                                        self.tables[key][:, i])
        self.timer.stop('run_onsite2c')

    def calculate(self, selected, e1, e2, R, grid, area,
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

        rho = e1.electron_density(r1) + e2.electron_density(r2)
        veff = e1.neutral_atom_potential(r1)
        veff += e2.neutral_atom_potential(r2)

        self.timer.start('vxc')
        if xc in ['LDA', 'PW92']:
            xc = XC_PW92()
            veff += xc.vxc(rho)
        else:
            xc = LibXC(xc)
            drho1 = e1.electron_density(r1, der=1)
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
        V = veff - e1.effective_potential(r1)
        sym1, sym2 = e1.get_symbol(), e2.get_symbol()
        self.timer.stop('prelude')

        results = {}
        for key in selected:
            integral, nl1a, nl1b = key

            Rnl1a = e1.Rnl(r1, nl1a)
            Rnl1b = e1.Rnl(r1, nl1b)
            gphi = g(c1, c1, s1, s1, integral)
            aux = gphi * area * x
            val = np.sum(Rnl1a * Rnl1b * V * aux)

            lm1a, lm1b = INTEGRAL_PAIRS[integral]
            val += e2.pp.get_nonlocal_integral(sym1, sym1, sym2, 0., R, 0.,
                                               nl1a, nl1b, lm1a, lm1b)

            if xc.add_gradient_corrections:
                self.timer.start('vsigma')
                dRnl1a = e1.Rnl(r1, nl1a, der=1)
                dRnl1b = e1.Rnl(r1, nl1b, der=1)
                dgphi = dg(c1, c1, s1, s1, integral)
                dgphidx = (dgphi[0] + dgphi[1]) * dc1dx \
                          + (dgphi[2] + dgphi[3]) * ds1dx
                dgphidy = (dgphi[0] + dgphi[1]) * dc1dy \
                          + (dgphi[2] + dgphi[3]) * ds1dy
                grad_phi_x = (dRnl1a * Rnl1b + Rnl1a * dRnl1b) * s1 * gphi
                grad_phi_x += Rnl1a * Rnl1b * dgphidx
                grad_phi_y = (dRnl1a * Rnl1b + Rnl1a * dRnl1b) * c1 * gphi
                grad_phi_y += Rnl1a * Rnl1b * dgphidy
                grad_rho_grad_phi = grad_rho_x * grad_phi_x \
                                    + grad_rho_y * grad_phi_y
                val += 2. * np.sum(vsigma * grad_rho_grad_phi * area * x)
                self.timer.stop('vsigma')

            results[key] = val

        self.timer.stop('calculate_onsite2c')
        return results

    def write(self):
        """
        Writes all Slater-Koster integral tables to file.

        The filename template corresponds to
        '<el1>-<el1>_onsite2c_<el2>.skf'.

        By default the 'simple' format is chosen, and the 'extended'
        format is only used when necessary (i.e. when there are f-electrons
        included in the valence).
        """
        template = '%s-%s_onsite2c_%s.skf'
        sym1, sym2 = self.ela.get_symbol(), self.elb.get_symbol()

        for bas1, valence1 in enumerate(self.ela.basis_sets):
            for bas2, valence2 in enumerate(self.ela.basis_sets):
                filename = template % (sym1 + '+'*bas1, sym1 + '+'*bas2, sym2)
                print('Writing to %s' % filename, file=self.txt, flush=True)

                is_extended = any([nl[1] == 'f' for nl in valence1+valence2])
                mass = atomic_masses[atomic_numbers[self.ela.get_symbol()]]

                key = (bas1, bas2)
                with open(filename, 'w') as f:
                    self._write_skf(f, key, is_extended, mass)

    def _write_skf(self, handle, key, is_extended, mass):
        """ Write to SKF file format. """
        # TODO: boilerplate (similar to Offsite2cTable._write_skf)
        if is_extended:
            print('@', file=handle)

        grid_dist = self.Rgrid[1] - self.Rgrid[0]
        grid_npts = len(self.tables[key])
        nzeros = int(np.round(self.Rgrid[0] / grid_dist)) - 1
        assert nzeros >= 0
        grid_npts += nzeros
        print("%.12f, %d" % (grid_dist, grid_npts), file=handle)

        print("%.3f, 19*0.0" % mass, file=handle)

        # Table containing the Slater-Koster integrals
        if is_extended:
            indices = range(NUMSK)
        else:
            indices = [INTEGRALS.index(name) for name in INTEGRALS
                       if 'f' not in name[:2]]

        for i in range(nzeros):
            print('%d*0.0,' % len(indices), file=handle)

        ct, theader = 0, ''
        for i in range(len(self.tables[key])):
            line = ''
            for j in indices:
                if self.tables[key][i, j] == 0:
                    ct += 1
                    theader = str(ct) + '*0.0 '
                else:
                    ct = 0
                    line += theader
                    theader = ''
                    line += '{0: 1.12e}  '.format(self.tables[key][i, j])

            if theader != '':
                ct = 0
                line += theader

            print(line, file=handle)
