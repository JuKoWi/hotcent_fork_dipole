#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
""" Defintion of the SlaterKosterTable class (and
supporting methods) for calculating Slater-Koster
integrals.

The code below draws heavily from the Hotbit code
written by Pekka Koskinen (https://github.com/pekkosk/
hotbit/blob/master/hotbit/parametrization/slako.py).
"""
import os
import sys
import numpy as np
from scipy.interpolate import CubicSpline, SmoothBivariateSpline
from ase.units import Bohr
from ase.data import atomic_numbers, atomic_masses, covalent_radii
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.timing import Timer
from hotcent.xc import XC_PW92, LibXC
from hotcent.interpolation import CubicSplineFunction
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import _hotcent
except ModuleNotFoundError:
    print('Warning: C-extensions not available')
    _hotcent = None


class SlaterKosterTable:
    def __init__(self, ela, elb, txt='-', timing=False):
        """ Construct Slater-Koster table for given elements.

        Parameters:
        -----------
        ela:    AtomicDFT object
        elb:    AtomicDFT object
        txt:    where output should be printed
                use '-' for stdout (default), None for /dev/null,
                any other string for a text file, or a file handle
        timing: output of timing summary after calculation
        """
        self.ela = ela
        self.elb = elb

        if ela.get_symbol() != elb.get_symbol():
            self.nel = 2
            self.pairs = [(ela, elb), (elb, ela)]
            self.elements = [ela, elb]
        else:
            self.nel = 1
            self.pairs = [(ela, elb)]
            self.elements = [ela]

        if txt is None:
            self.txt = open(os.devnull, 'w')
        elif isinstance(txt, str):
            if txt == '-':
                self.txt = sys.stdout
            else:
                self.txt = open(txt, 'a')
        else:
            self.txt = txt

        self.timer = Timer('SlaterKosterTable', txt=self.txt, enabled=timing)

    def __del__(self):
        self.timer.summary()

    def write(self, filename=None, pair=None, eigenvalues={},
              hubbardvalues={}, occupations={}, spe=0.):
        """ Write SK tables to a file

        filename: str with name of file to write to.
                  The file format is selected by the extension
                  (.par or .skf).
                  Defaults to self.ela-self.elb_no_repulsion.skf

        pair: either (symbol_a, symbol_b) or (symbol_b, symbol_a)
              to select which of the two SK tables to write

        other kwargs: {nl: value}-dictionaries with eigenvalues,
              hubbardvalues and valence orbital occupations, as well
              as the spin-polarization error (all typically calculated
              on the basis of atomic DFT calculations). These will be
              written to the second line of a homo-nuclear .skf file.
              Examples: hubbardvalues={'2s': 0.5}, spe=0.2,
                        occupations={'3d':10, '4s': 1}, etc.
        """
        if pair is None:
            pair = (self.ela.get_symbol(), self.elb.get_symbol())

        fn = '%s-%s_no_repulsion.skf' % pair if filename is None else filename

        ext = fn[-4:]

        assert ext in ['.par', '.skf'], \
               "Unknown format: %s (-> choose .par or .skf)" % ext

        with open(fn, 'w') as handle:
            if ext == '.par':
                self._write_par(handle)
            elif ext == '.skf':
                self._write_skf(handle, pair, eigenvalues, hubbardvalues,
                                occupations, spe)

    def _write_skf(self, handle, pair, eigval, hubval, occup, spe):
        """ Write to SKF file format; this function
        is an adaptation of hotbit.io.hbskf

        By default the 'simple' format is chosen, and the 'extended'
        format is only used when necessary (i.e. when there are f-electrons
        included in the valence of one of the elements).
        """
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

        el1, el2 = self.ela.get_symbol(), self.elb.get_symbol()
        if el1 == el2:
            fields = ['E_f', 'E_d', 'E_p', 'E_s', 'SPE', 'U_f', 'U_d',
                      'U_p', 'U_s', 'f_f', 'f_d', 'f_p', 'f_s']

            if not extended_format:
                fields = [field for field in fields if field[-1] != 'f']

            labels = {'SPE': spe}
            for prefix, d in zip(['E', 'U', 'f'], [eigval, hubval, occup]):
                keys = list(d.keys())
                for l in ['s', 'p', 'd', 'f']:
                    check = [key[-1] == l for key in keys]
                    assert sum(check) in [0, 1], (keys, l)
                    if sum(check) == 1:
                        key = keys[check.index(True)]
                        labels['%s_%s' % (prefix, l)] = d[key]

            line = ' '.join(fields)
            for field in fields:
                val = labels[field] if field in labels else 0
                s = '%d' % val if isinstance(val, int) else '%.6f' % val
                line = line.replace(field, s)
            print(line, file=handle)

        m = atomic_masses[atomic_numbers[symbols[index]]]
        print("%.3f, 19*0.0" % m, file=handle)

        # Table containing the Slater-Koster integrals
        if extended_format:
            indices = range(2*NUMSK)
        else:
            indices = [INTEGRALS.index(name) for name in INTEGRALS
                       if 'f' not in name[:2]]
            indices.extend([j+NUMSK for j in indices])

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

    def _write_par(self, handle):
        for p, (e1, e2) in enumerate(self.pairs):
            line = '%s-%s_table=' % (e1.get_symbol(), e2.get_symbol())
            print(line, file=handle)

            for i, R in enumerate(self.Rgrid):
                print('%.6e' % R, end=' ', file=handle)

                for t in range(2*NUMSK):
                    x = self.tables[p][i, t]
                    if abs(x) < 1e-90:
                        print('0.', end=' ', file=handle)
                    else:
                        print('%.6e' % x, end=' ', file=handle)
                print(file=handle)

            print('\n\n', file=handle)

    def plot(self, filename=None):
        """ Plot the Slater-Koster table with matplotlib.

        parameters:
        ===========
        filename:     name for the figure
        """
        self.timer.start('plotting')
        assert plt is not None, 'Matplotlib could not be imported!'

        fig = plt.figure()
        fig.subplots_adjust(hspace=1e-4, wspace=1e-4)

        el1 = self.ela.get_symbol()
        rmax = 6 * covalent_radii[atomic_numbers[el1]] / Bohr
        ymax = max(1, self.tables[0].max())
        if self.nel == 2:
            el2 = self.elb.get_symbol()
            rmax = max(rmax, 6 * covalent_radii[atomic_numbers[el2]] / Bohr)
            ymax = max(ymax, self.tables[1].max())

        for i in range(NUMSK):
            name = INTEGRALS[i]
            ax = plt.subplot(NUMSK//2, 2, i + 1)

            for p, (e1, e2) in enumerate(self.pairs):
                s1, s2 = e1.get_symbol(), e2.get_symbol()

                if p == 0:
                    s = '-'
                    lw = 1
                    alpha = 1.0
                else:
                    s = '--'
                    lw = 4
                    alpha = 0.2

                if np.all(abs(self.tables[p][:, i]) < 1e-10):
                    ax.text(0.03, 0.5 + p * 0.15,
                            'No %s integrals for <%s|%s>' % (name, s1, s2),
                            transform=ax.transAxes, size=10, va='center')

                    if not ax.get_subplotspec().is_last_row():
                        plt.xticks([], [])
                    if not ax.get_subplotspec().is_first_col():
                        plt.yticks([], [])
                else:
                    plt.plot(self.Rgrid, self.tables[p][:, i] , c='r',
                             ls=s, lw=lw, alpha=alpha)
                    plt.plot(self.Rgrid, self.tables[p][:, i+NUMSK], c='b',
                             ls=s, lw=lw, alpha=alpha)
                    plt.axhline(0, c='k', ls='--')
                    ax.text(0.8, 0.1 + p * 0.15, name, size=10,
                            transform=ax.transAxes)

                    if ax.get_subplotspec().is_last_row():
                        plt.xlabel('r (Bohr)')
                    else:
                        plt.xticks([], [])
                    if not ax.get_subplotspec().is_first_col():
                        plt.yticks([],[])

                plt.xlim([0, rmax])
                plt.ylim(-ymax, ymax)

        plt.figtext(0.3, 0.95, 'H', color='r', size=20)
        plt.figtext(0.34, 0.95, 'S', color='b', size=20)
        plt.figtext(0.38, 0.95, ' Slater-Koster tables', size=20)
        e1, e2 = self.ela.get_symbol(), self.elb.get_symbol()
        plt.figtext(0.3, 0.92, '(thin solid: <%s|%s>, wide dashed: <%s|%s>)' \
                    % (e1, e2, e2, e1), size=10)

        if filename is None:
            filename = '%s-%s_slako.pdf' % (e1, e2)
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
        self.timer.stop('plotting')

    def get_range(self, fractional_limit):
        """ Define ranges for the atoms: largest r such that Rnl(r)<limit. """
        wf_range = 0.

        for el in self.elements:
            r = max([el.get_wf_range(nl, fractional_limit)
                     for nl in el.get_valence_orbitals()])
            print('Wave function range for %s = %.5f a0' % (el.get_symbol(), r),
                  file=self.txt)
            wf_range = max(r, wf_range)

        assert wf_range < 20, 'Wave function range exceeds 20 Bohr radii. ' \
                              'Decrease wflimit?'
        return wf_range

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            superposition='potential', xc='LDA', stride=1, smoothen_tails=True):
        """ Calculate the Slater-Koster table.

        parameters:
        ------------
        rmin, dr, N: parameters defining the equidistant grid of interatomic
                separations: the shortest distance rmin and grid spacing dr
                (both in Bohr radii) and the number of grid points N.
        ntheta: number of angular divisions in polar grid
                (more dense towards bonding region).
        nr:     number of radial divisions in polar grid
                (more dense towards origins).
                with p=q=2 (powers in polar grid) ntheta~3*nr is
                optimal (with fixed grid size)
                with ntheta=150, nr=50 you get~1E-4 accuracy for H-elements
                (beyond that, gain is slow with increasing grid size)
        wflimit: value below which the radial wave functions are considered
                to be negligible. This determines how far the polar grids
                around the atomic centers extend in space.
        superposition: 'density' or 'potential': whether to use the density
                superposition or potential superposition approach for the
                Hamiltonian integrals.
        xc:     name of the exchange-correlation functional to be used
                in calculating the effective potential in the density
                superposition scheme. If the PyLibXC module is available,
                any LDA or GGA (but not hybrid or MGGA) functional available
                via LibXC can be specified. E.g. for using the N12
                functional, set xc='XC_GGA_X_N12+XC_GGA_C_N12'.
                If PyLibXC is not available, only the local density
                approximation xc='PW92' (alias: 'LDA') can be chosen.
        stride: the desired SK-table typically has quite a large number
                of points (N=500-1000), even though the integrals
                themselves are comparatively smooth. To speed up the
                construction of the SK-table, one can therefore restrict
                the expensive integrations to a subset N' = N // stride,
                and map the resulting curves on the N-grid afterwards.
                The default stride = 1 means that N' = N (no shortcut).
        smoothen_tails: whether to modify the 'tails' of the Slater-Koster
                integrals so that they smoothly decay to zero.
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Slater-Koster table construction for %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'
        assert superposition in ['density', 'potential']

        self.timer.start('calculate_tables')
        self.wf_range = self.get_range(wflimit)
        Nsub = N // stride
        Rgrid = rmin + stride * dr * np.arange(Nsub)
        tables = [np.zeros((Nsub, 2*NUMSK)) for i in range(self.nel)]
        dH = 0.
        Hmax = 0.

        for p, (e1, e2) in enumerate(self.pairs):
            sym1, sym2 = e1.get_symbol(), e2.get_symbol()
            print('Integrals for %s-%s pair:' % (sym1, sym2), end=' ',
                  file=self.txt)
            selected = select_integrals(e1, e2)
            for s in selected:
                print(s[0], end=' ', file=self.txt)
            print(file=self.txt, flush=True)

        for i, R in enumerate(Rgrid):
            if R > 2 * self.wf_range:
                break

            grid, area = self.make_grid(R, nt=ntheta, nr=nr)

            if  i == Nsub - 1 or Nsub // 10 == 0 or i % (Nsub // 10) == 0:
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            for p, (e1, e2) in enumerate(self.pairs):
                selected = select_integrals(e1, e2)
                S, H, H2 = 0., 0., 0.
                if len(grid) > 0:
                    S, H, H2 = self.calculate_mels(selected, e1, e2, R, grid,
                                                   area, xc=xc,
                                                   superposition=superposition)
                    Hmax = max(Hmax, max(abs(H)))
                    dH = max(dH, max(abs(H - H2)))
                tables[p][i, :NUMSK] = H
                tables[p][i, NUMSK:] = S

        if superposition == 'potential':
            print('Maximum value for H=%.2g' % Hmax, file=self.txt)
            print('Maximum error for H=%.2g' % dH, file=self.txt)
            print('     Relative error=%.2g %%' % (dH / Hmax * 100),
                  file=self.txt)

        self.Rgrid = rmin + dr * np.arange(N)

        if stride > 1:
            self.tables = [np.zeros((N, 2*NUMSK)) for i in range(self.nel)]
            for p in range(self.nel):
                for i in range(2*NUMSK):
                    spl = CubicSplineFunction(Rgrid, tables[p][:, i])
                    self.tables[p][:, i] = spl(self.Rgrid)
        else:
            self.tables = tables

        if smoothen_tails:
            # Smooth the curves near the cutoff
            for p in range(self.nel):
                for i in range(2*NUMSK):
                    self.tables[p][:, i] = tail_smoothening(self.Rgrid,
                                                        self.tables[p][:, i])

        self.timer.stop('calculate_tables')

    def calculate_mels(self, selected, e1, e2, R, grid, area,
                       superposition='potential', xc='LDA',
                       only_overlap=False):
        """ Perform integration for selected H and S integrals.

        parameters:
        -----------
        selected: list of [('dds', '3d', '4d'), (...)]
        e1: <bra| element
        e2: |ket> element
        R: e1 is at origin, e2 at z=R
        grid: list of grid points on (d, z)-plane
        area: d-z areas of the grid points.
        superposition: 'density' or 'potential' superposition scheme
        xc: exchange-correlation functional (see description in self.run())
        only_overlap: whether to only evaluate the overlap integrals
                      (in which case the 'xc' and 'superposition' keywords
                      are ignored)

        return:
        -------
        List of H,S and H2 for selected integrals. In the potential
        superposition scheme, H2 is calculated using a different technique
        and can be used for error estimation. This is not available
        for the density superposition scheme, where simply H2=0 is returned.

        S: simply R1 * R2 * angle_part

        H: operate (derivate) R2 <R1 | t + Veff - Conf1 - Conf2 | R2>.
           With potential superposition: Veff = Veff1 + Veff2
           With density superposition: Veff = Vxc(n1 + n2)

        H2: operate with full h2 and hence use eigenvalue of | R2>
            with full Veff2:
              <R1 | (t1 + Veff1) + Veff2 - Conf1 - Conf2 | R2>
            = <R1 | h1 + Veff2 - Conf1 - Conf2 | R2> (operate with h1 on left)
            = <R1 | e1 + Veff2 - Conf1 - Conf2 | R2>
            = e1 * S + <R1 | Veff2 - Conf1 - Conf2 | R2>
            -> H and H2 can be compared and error estimated
        """
        self.timer.start('calculate_mels')

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

        if not only_overlap and superposition == 'potential':
            self.timer.start('vrho')
            v1 = e1.effective_potential(r1) - e1.confinement(r1)
            v2 = e2.effective_potential(r2) - e2.confinement(r2)
            veff = v1 + v2
            self.timer.stop('vrho')
        elif not only_overlap and superposition == 'density':
            self.timer.start('vrho')
            rho = e1.electron_density(r1) + e2.electron_density(r2)
            veff = e1.neutral_atom_potential(r1)
            veff += e2.neutral_atom_potential(r2)
            if xc in ['LDA', 'PW92']:
                xc = XC_PW92()
                veff += xc.vxc(rho)
                self.timer.stop('vrho')
            else:
                xc = LibXC(xc)
                drho1 = e1.electron_density(r1, der=1)
                drho2 = e2.electron_density(r2, der=1)
                grad_x = drho1 * s1
                grad_x += drho2 * s2
                grad_y = drho1 * c1
                grad_y += drho2 * c2
                sigma = grad_x**2 + grad_y**2
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

        if not only_overlap:
            assert np.shape(veff) == (len(grid),)
        self.timer.stop('prelude')

        # calculate all selected integrals
        Sl = np.zeros(NUMSK)
        if not only_overlap:
            Hl, H2l = np.zeros(NUMSK), np.zeros(NUMSK)

        for integral, nl1, nl2 in selected:
            index = INTEGRALS.index(integral)
            gphi = g(c1, c2, s1, s2, integral)
            aux = gphi * area * x

            Rnl1 = e1.Rnl(r1, nl1)
            Rnl2 = e2.Rnl(r2, nl2)
            S = np.sum(Rnl1 * Rnl2 * aux)
            Sl[index] = S

            if not only_overlap:
                l2 = ANGULAR_MOMENTUM[nl2[1]]
                ddunl2 = e2.unl(r2, nl2, der=2)

                H = np.sum(Rnl1 * (-0.5 * ddunl2 / r2 + (veff + \
                           l2 * (l2 + 1) / (2 * r2 ** 2)) * Rnl2) * aux)

                sym1, sym2 = e1.get_symbol(), e2.get_symbol()
                lm1, lm2 = INTEGRAL_PAIRS[integral]
                H += e1.pp.get_nonlocal_integral(sym1, sym2, sym1, 0., 0., R,
                                                 nl1, nl2, lm1, lm2)
                H += e2.pp.get_nonlocal_integral(sym1, sym2, sym2, 0., R, R,
                                                 nl1, nl2, lm1, lm2)

                if superposition == 'potential':
                    H2 = np.sum(Rnl1 * Rnl2 * aux * (v2 - e1.confinement(r1)))
                    H2 += e1.get_epsilon(nl1) * S
                elif superposition == 'density':
                    H2 = 0

                Hl[index] = H
                H2l[index] = H2

        self.timer.stop('calculate_mels')
        if only_overlap:
            return Sl
        else:
            return Sl, Hl, H2l

    def run_repulsion(self, rmin=0.4, dr=0.02, N=None, ntheta=600, nr=100,
                      wflimit=1e-7, smoothen_tails=True, xc='LDA'):
        """ Calculates the 'repulsive' contributions to the total energy
        (i.e. the double-counting and ion-ion interaction terms), which
        are stored in self.erep.

        parameters:
        ------------
        Same as in the run() method.
        """
        print('\n\n', file=self.txt)
        print('***********************************************', file=self.txt)
        print('Repulsion calculation for %s and %s' % \
              (self.ela.get_symbol(), self.elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert xc == 'LDA', 'Functionals other than LDA are not yet implemented'

        self.timer.start('run_repulsion')
        wf_range = self.get_range(wflimit)
        self.Rgrid = rmin + dr * np.arange(N)
        self.erep = np.zeros(N)

        for i, R in enumerate(self.Rgrid):
            self.wf_range = wf_range + R
            grid, area = self.make_grid(R, nt=ntheta, nr=nr)

            if  i == N - 1 or N // 10 == 0 or i % (N // 10) == 0:
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            self.erep[i] = self.calculate_repulsion(self.ela, self.elb, R, grid,
                                                    area, xc=xc)

        if smoothen_tails:
            # Smooth the curves near the cutoff
            self.erep = tail_smoothening(self.Rgrid, self.erep)

        self.timer.stop('run_repulsion')

    def get_repulsion_spline_block(self):
        """ Returns a string with the Spline block for the repulsive energy
        (in SKF format). """
        lines = 'Spline\n'

        dr = self.Rgrid[1] - self.Rgrid[0]
        n = len(self.Rgrid) - 1
        lines += '%d %.6f\n' % (n, self.Rgrid[-1])

        # Fit the exponential function for radii below self.Rgrid[0]
        r0 = self.Rgrid[0]
        f0 = self.erep[0]
        f1 = (self.erep[1] - self.erep[0]) / dr
        f2 = (self.erep[2] - 2.*self.erep[1] + self.erep[0]) / dr**2
        assert f1 < 0, 'Cannot fit exponential repulsion when derivative > 0'
        assert f2 > 0, 'Cannot fit exponential repulsion when curvature < 0'

        a1 = -f2 / f1
        a2 = np.log(-f1) - np.log(a1) + a1 * r0
        a3 = f0 - np.exp(-a1*r0 + a2)
        assert np.abs(-a1 * np.exp(-a1*r0 + a2) - f1) < 1e-8
        assert np.abs(a1**2 * np.exp(-a1*r0 + a2) - f2) < 1e-8
        lines += '%.6f %.6f %.6f\n' % (a1, a2, a3)

        # Set up the cubic spline function spanning self.Rgrid
        # and matching the exponential function on its left
        spl = CubicSpline(self.Rgrid, self.erep, bc_type=((1, f1), 'natural'))
        assert np.abs(spl(r0, nu=0) - f0) < 1e-8
        assert np.abs(spl(r0, nu=1) - f1) < 1e-8

        # Fit the additional coefficients for the 5th-order spline at the end
        r0 = self.Rgrid[-1]
        f0 = spl(r0, nu=0)
        f1 = spl(r0, nu=1)
        v = -f0 / dr**4
        w = -f1 / dr**3
        c4 = 5.*v - w
        c5 = (v - c4) / dr
        assert np.abs(spl.c[3][-1] + spl.c[2][-1]*dr + spl.c[1][-1]*dr**2 \
                      + spl.c[0][-1]*dr**3 + c4*dr**4 + c5*dr**5) < 1e-8
        assert np.abs(spl.c[2][-1] + 2.*spl.c[1][-1]*dr + 3.*spl.c[0][-1]*dr**2\
                      + 4.*c4*dr**3 + 5.*c5*dr**4) < 1e-8

        # Now add all the spline lines
        for i in range(n):
            items = [self.Rgrid[i], self.Rgrid[i]+dr, spl.c[3][i],
                     spl.c[2][i], spl.c[1][i], spl.c[0][i]]
            if i == n-1:
                items += [c4, c5]
            lines += ' '.join(map(lambda x: '%.9f' % x, items)) + '\n'

        return lines

    def calculate_repulsion(self, e1, e2, R, grid, area, xc='LDA'):
        """ Returns the 'repulsive' contribution to the total energy for the
        given interatomic distance, with a two-center approximation for the
        exchange-correlation terms.

        NOTE: one-center terms are substracted (as these only shift the
        atom energies). The repulsion should hence decay to 0 for large R.

        parameters:
        -----------
        e1: <bra| element
        e2: |ket> element
        R: e1 is at origin, e2 at z=R
        grid: list of grid points on (d, z)-plane
        area: d-z areas of the grid points.
        xc: exchange-correlation functional (see description in self.run())
        """
        self.timer.start('calculate_repulsion')

        # TODO: boilerplate
        self.timer.start('prelude')
        x = grid[:, 0]
        y = grid[:, 1]
        r1 = np.sqrt(x**2 + y**2)
        r2 = np.sqrt(x**2 + (R - y)**2)

        rho1 = e1.electron_density(r1, only_valence=True)
        rho2 = e2.electron_density(r2, only_valence=True)
        rho12 = rho1 + rho2

        if xc in ['LDA', 'PW92']:
            self.timer.start('vrho')
            xc = XC_PW92()
            exc1 = xc.exc(rho1)
            exc2 = xc.exc(rho2)
            exc12 = xc.exc(rho12)
            vxc1 = xc.vxc(rho1)
            vxc2 = xc.vxc(rho2)
            vxc12 = xc.vxc(rho12)
            self.timer.stop('vrho')
        else:
            pass  # TODO
        self.timer.stop('prelude')

        aux = 2 * np.pi * area * x

        vhar1 = e1.hartree_potential(r1, only_valence=True)
        vhar2 = e2.hartree_potential(r2, only_valence=True)
        Ehar = np.sum(vhar1 * rho2 * aux)
        Ehar += np.sum(vhar2 * rho1 * aux)

        Z1 = e1.get_number_of_electrons(only_valence=True)
        Z2 = e2.get_number_of_electrons(only_valence=True)
        Enuc = Z1 * Z2 / R

        Exc = np.sum((exc12 * rho12 - exc1 * rho1 - exc2 * rho2) * aux)
        Evxc = np.sum((vxc12 * rho12 - vxc1 * rho1 - vxc2 * rho2) * aux)

        Erep = Enuc - 0.5*Ehar + Exc - Evxc
        self.timer.stop('calculate_repulsion')
        return Erep

    def make_grid(self, Rz, nt, nr, p=2, q=2, view=False):
        """ Construct a double-polar grid.

        Parameters:
        -----------
        Rz: element 1 is at origin, element 2 at z=Rz
        nt: number of theta grid points
        nr: number of radial grid points
        p: power describing the angular distribution of grid points
           (larger puts more weight towards theta=0)
        q: power describing the radial disribution of grid points
           (larger puts more weight towards centers)
        view: view the distribution of grid points with matplotlib

        Plane at R/2 divides two polar grids.


         ^ (z-axis)
         |--------_____               phi_j
         |       /     ----__         *
         |      /            \       /  *
         |     /               \    /  X *                X=coordinates of the center of area element(z,d),
         |    /                  \  \-----* phi_(j+1)     area=(r_(i+1)**2-r_i**2)*(phi_(j+1)-phi_j)/2
         |   /                    \  r_i   r_(i+1)
         |  /                      \
         | /                       |
         *2------------------------|           polar centered on atom 2
         | \                       |
         |  \                     /                                                     1
         |   \                   /                                                     /  \
         |-------------------------- z=h -line         ordering of sector slice       /     \
         |   /                   \                                      points:      /        \
         |  /                     \                                                 /          \
         | /                       |                                               /     0       4
         *1------------------------|--->      polar centered on atom 1            2            /
         | \                       |    (r_perpendicular (xy-plane) = 'd-axis')    \        /
         |  \                      /                                                 \   /
         |   \                    /                                                    3
         |    \                  /
         |     \               /
         |      \           /
         |       \ ___ ---
         |---------
        """
        self.timer.start('make_grid')
        rmin, rmax = 1e-7, self.wf_range
        h = Rz / 2

        if _hotcent is not None:
            grid, area = _hotcent.make_grid(Rz, rmin, rmax, nt, nr, p, q)
        else:
            max_range = self.wf_range
            T = np.linspace(0, 1, nt) ** p * np.pi
            R = rmin + np.linspace(0, 1, nr) ** q * (rmax - rmin)

            area = np.array([])
            d = np.array([])
            z = np.array([])

            # first calculate grid for polar centered on atom 1:
            # the z=h-like starts cutting full elements starting from point (1)
            Tj0 = T[:nt - 1]
            Tj1 = T[1: nt]

            for i in range(nr - 1):
                # corners of area element
                z1 = R[i + 1] * np.cos(Tj0)
                z2 = R[i] * np.cos(Tj0)
                z3 = R[i] * np.cos(Tj1)
                z4 = R[i + 1] * np.cos(Tj1)

                cond_list = [z1 <= h,  # area fully inside region
                     (z1 > h) * (z2 <= h) * (z4 <= h),  # corner 1 outside region
                     (z1 > h) * (z2 > h) * (z3 <= h) * (z4 <= h),  # 1 & 2 outside
                     (z1 > h) * (z2 > h) * (z3 <= h) * (z4 > h),  # only 3 inside
                     (z1 > h) * (z2 <= h) * (z3 <= h) * (z4 > h),  # 1 & 4 outside
                     (z1 > h) * (z3 > h) * ~((z2 <= h) * (z4 > h))]

                r0_list = [0.5 * (R[i] + R[i + 1]),
                           0.5 * (R[i] + R[i + 1]),
                           0.5 * (R[i] + R[i + 1]),
                           lambda x: 0.5 * (R[i] + h / np.cos(x)),
                           lambda x: 0.5 * (R[i] + h / np.cos(x)),
                           0,
                           np.nan]
                r0 = np.piecewise(Tj1, cond_list, r0_list)

                Th0 = np.piecewise(h / R[i], [np.abs(h / R[i]) > 1],
                                   [np.nan, lambda x: np.arccos(x)])
                Th1 = np.piecewise(h / R[i + 1], [np.abs(h / R[i + 1]) > 1],
                                   [np.nan, lambda x: np.arccos(x)])

                t0_list = [lambda x: 0.5 * x,
                           0.5 * Th1,
                           0.5 * Th1,
                           0.5 * Th0,
                           lambda x: 0.5 * x,
                           0,
                           np.nan]
                t0 = 0.5 * Tj1
                t0 += np.piecewise(Tj0, cond_list, t0_list)

                rr = 0.5 * (R[i + 1] ** 2 - R[i] ** 2)
                A_list0 = [lambda x: rr * -x,
                           lambda x: rr * -x - 0.5 * R[i + 1] ** 2 * (Th1 - x) \
                                     + 0.5 * h ** 2 * (np.tan(Th1) - np.tan(x)),
                           lambda x: rr * -x - (rr * -x + 0.5 * R[i + 1] ** 2 \
                                     * (Th1 - Th0)),
                           0.,
                           lambda x: 0.5 * h ** 2 * -np.tan(x) \
                                     - 0.5 * R[i] ** 2 * -x,
                           -1,
                           np.nan]
                A = np.piecewise(Tj0, cond_list, A_list0)

                A_list1 = [lambda x: rr * x,
                           lambda x: rr * x,
                           lambda x: rr * x - (rr * Th0 - 0.5 * h ** 2 \
                                     * (np.tan(Th1) - np.tan(Th0))),
                           lambda x: 0.5 * h ** 2 * (np.tan(x) - np.tan(Th0)) \
                                     - 0.5 * R[i] ** 2 * (x - Th0),
                           lambda x: 0.5 * h ** 2 * np.tan(x) \
                                     - 0.5 * R[i] ** 2 * x,
                           0,
                           np.nan]
                A += np.piecewise(Tj1, cond_list, A_list1)

                dd = r0 * np.sin(t0)
                zz = r0 * np.cos(t0)
                select = np.sqrt(dd ** 2 + zz ** 2) < max_range
                select *= np.sqrt(dd ** 2 + (Rz - zz) ** 2) < max_range
                select *= A > 0
                area = np.hstack((area, A[select]))
                d = np.hstack((d, dd[select]))
                z = np.hstack((z, zz[select]))
            grid = np.array([d, z]).T

        self.timer.start('symmetrize')
        # calculate the polar centered on atom 2 by mirroring the other grid
        grid2 = grid.copy()
        grid2[:, 1] = -grid[:, 1]
        shift = np.zeros_like(grid)
        shift[:, 1] = 2 * h
        grid = np.concatenate((grid, grid2 + shift))
        area = np.concatenate((area, area))
        self.timer.stop('symmetrize')

        if view:
            assert plt is not None, 'Matplotlib could not be imported!'
            plt.plot([h, h ,h])
            plt.scatter(grid[:, 0], grid[:, 1], s=10 * area / np.max(area))
            plt.show()
            plt.clf()

        self.timer.stop('make_grid')
        return grid, area


INTEGRALS = ['ffs', 'ffp', 'ffd', 'fff',
             'dfs', 'dfp', 'dfd', 'dds', 'ddp', 'ddd',
             'pfs', 'pfp', 'pds', 'pdp', 'pps', 'ppp',
             'sfs', 'sds', 'sps', 'sss']

NUMSK = len(INTEGRALS)

INTEGRAL_PAIRS = {
    'sss': ('s', 's'),
    'sps': ('s', 'pz'),
    'sds': ('s', 'dz2'),
    'sfs': ('s', 'fz3'),
    'pps': ('pz', 'pz'),
    'ppp': ('px', 'px'),
    'pds': ('pz', 'dz2'),
    'pdp': ('px', 'dxz'),
    'pfs': ('pz', 'fz3'),
    'pfp': ('px', 'fxz2'),
    'dds': ('dz2', 'dz2'),
    'ddp': ('dxz', 'dxz'),
    'ddd': ('dxy', 'dxy'),
    'dfs': ('dz2', 'fz3'),
    'dfp': ('dxz', 'fxz2'),
    'dfd': ('dxy', 'fxyz'),
    'ffs': ('fz3', 'fz3'),
    'ffp': ('fxz2', 'fxz2'),
    'ffd': ('fxyz', 'fxyz'),
    'fff': ('fx(x2-3y2)', 'fx(x2-3y2)'),
}


def search_integrals(lm1, lm2):
    """ Returns the sigma/pi/... integrals to be considered for the
    given pair of orbitals. """
    integrals, ordered = [], []

    for integral, pair in INTEGRAL_PAIRS.items():
        if pair == (lm1, lm2):
            integrals.append(integral)
            ordered.append(True)
        elif pair == (lm2, lm1):
            integrals.append(integral)
            ordered.append(False)

    return integrals, ordered


def select_orbitals(val1, val2, integral):
    """ Select orbitals from given valences to calculate given integral.
    e.g. ['2s', '2p'], ['4s', '3d'], 'sds' --> ('2s', '3d')
    """
    nl1 = None
    for nl in val1:
        if nl[1] == integral[0]:
            nl1 = nl

    nl2 = None
    for nl in val2:
        if nl[1] == integral[1]:
            nl2 = nl

    return nl1, nl2


def select_integrals(e1, e2):
    """ Return list of integrals (integral, nl1, nl2)
    to be done for element pair e1, e2. """
    selected = []
    val1, val2 = e1.get_valence_orbitals(), e2.get_valence_orbitals()

    for integral in INTEGRALS:
        nl1, nl2 = select_orbitals(val1, val2, integral)
        if nl1 is None or nl2 is None:
            continue
        else:
            selected.append((integral, nl1, nl2))

    return selected


def g(c1, c2, s1, s2, integral):
    """ Returns the angle-dependent part of the given two-center integral,
    with c1 and s1 (c2 and s2) the sine and cosine of theta_1 (theta_2)
    for the atom at origin (atom at z=Rz). These expressions are obtained
    by integrating analytically over phi.
    """
    if integral == 'ffs':
        return 7. / 8 * c1 * (5 * c1**2 - 3) * c2 * (5 * c2**2 - 3)
    elif integral == 'ffp':
        return 21. / 32 * (5 * c1**2 - 1) * s1 * (5 * c2**2 - 1) * s2
    elif integral == 'ffd':
        return 105. / 16 * s1**2 * c1 * s2**2 * c2
    elif integral == 'fff':
        return 35. / 32 * s1**3 * s2**3
    elif integral == 'dfs':
        return np.sqrt(35.) / 8 * (3 * c1**2 - 1) * c2 * (5 * c2**2 - 3)
    elif integral == 'dfp':
        return 3. / 8 * np.sqrt(17.5) * s1 * c1 * s2 * (5 * c2**2 - 1)
    elif integral == 'dfd':
        return 15. / 16 * np.sqrt(7.) * s1**2 * s2**2 * c2
    elif integral == 'dds':
        return 5. / 8 * (3 * c1**2 - 1) * (3 * c2**2 - 1)
    elif integral == 'ddp':
        return 15. / 4 * s1 * c1 * s2 * c2
    elif integral == 'ddd':
        return 15. / 16 * s1**2 * s2**2
    elif integral == 'pfs':
        return np.sqrt(21.) / 4 * c1 * c2 * (5 * c2**2 - 3)
    elif integral == 'pfp':
        return 3. / 8 * np.sqrt(3.5) * s1 * s2 * (5 * c2**2 - 1)
    elif integral == 'pds':
        return np.sqrt(15.) / 4 * c1 * (3 * c2**2 - 1)
    elif integral == 'pdp':
        return np.sqrt(45.) / 4 * s1 * s2 * c2
    elif integral == 'pps':
        return 3. / 2 * c1 * c2
    elif integral == 'ppp':
        return 3. / 4 * s1 * s2
    elif integral == 'sfs':
        return np.sqrt(7.) / 4 * c2 * (5 * c2**2 - 3)
    elif integral == 'sds':
        return np.sqrt(5.) / 4 * (3 * c2**2 - 1)
    elif integral == 'sps':
        return np.sqrt(3.) / 2 * c2
    elif integral == 'sss':
        return 0.5 * np.ones_like(c1)


def tail_smoothening(x, y):
    """ For given grid-function y(x), make smooth tail.

    Aim is to get (e.g. for Slater-Koster tables and repulsions) smoothly
    behaving energies and forces near cutoff region.

    Make is such that y and y' go smoothly exactly to zero at last point.
    Method: take largest neighboring points y_k and y_(k+1) (k<N-3) such
    that line through them passes zero below x_(N-1). Then fit
    third-order polynomial through points y_k, y_k+1 and y_N-1.

    Return:
    smoothed y-function on same grid.
    """
    if np.all(abs(y) < 1e-10):
        return y

    Nzero = 0
    for i in range(len(y) - 1, 1, -1):
        if abs(y[i]) < 1e-60:
            Nzero += 1
        else:
            break

    N = len(y) - Nzero
    y = y[:N]
    xmax = x[:N][-1]

    for i in range(N - 3, 1, -1):
        x0i = x[i] - y[i] / ((y[i + 1] - y[i]) /(x[i + 1] - x[i]))
        if x0i < xmax:
            k = i
            break
    else:
        print('N:', N, 'len(y):', len(y))
        for i in range(len(y)):
            print(x[i], y[i])
        raise RuntimeError('Problem with tail smoothening')

    if k < N / 4:
        for i in range(N):
            print(x[i], y[i])
        msg = 'Problem with tail smoothening: requires too large tail.'
        raise RuntimeError(msg)

    if k == N - 3:
        y[-1] = 0.
        y = np.append(y, np.zeros(Nzero))
        return y
    else:
        # g(x)=c2*(xmax-x)**m + c3*(xmax-x)**(m+1) goes through
        # (xk,yk),(xk+1,yk+1) and (xmax,0)
        # Try different m if g(x) should change sign (this we do not want)
        sgn = np.sign(y[k])
        for m in range(2, 10):
            a1, a2 = (xmax - x[k]) ** m, (xmax - x[k]) ** (m + 1)
            b1, b2 = (xmax-  x[k + 1]) ** m, (xmax - x[k + 1]) ** (m + 1)
            c3 = (y[k] - a1 * y[k + 1] / b1) / (a2 - a1 * b2 / b1)
            c2 = (y[k] - a2 * c3) / a1

            for i in range(k + 2,N):
                y[i] = c2 * (xmax - x[i]) ** 2 + c3 * (xmax - x[i]) ** 3

            y[-1] = 0.  # once more explicitly

            if np.all(y[k:] * sgn >= 0):
                y = np.append(y, np.zeros(Nzero))
                break

            if m == 9:
                msg = 'Problems with smoothening; need for new algorithm?'
                raise RuntimeError(msg)

    return y
