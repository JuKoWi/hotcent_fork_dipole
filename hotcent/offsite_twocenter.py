#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from ase.units import Bohr
from ase.data import atomic_numbers, atomic_masses, covalent_radii
from hotcent.interpolation import CubicSplineFunction
from hotcent.multiatom_integrator import MultiAtomIntegrator
from hotcent.orbitals import ANGULAR_MOMENTUM
from hotcent.slako import (dg, g, INTEGRAL_PAIRS, INTEGRALS, NUMSK,
                           select_integrals, tail_smoothening)
from hotcent.xc import XC_PW92, LibXC
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import _hotcent
except ModuleNotFoundError:
    print('Warning: C-extensions not available')
    _hotcent = None


class Offsite2cTable(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, ntheta=150, nr=50, wflimit=1e-7,
            superposition='density', xc='LDA', stride=1, smoothen_tails=True):
        """
        Calculates off-site two-center Hamiltonian and overlap integrals.

        Parameters
        ----------
        rmin : float, optional
            Shortest interatomic separation to consider.
        dr : float, optional
            Grid spacing for the interatomic separations.
        N : int
            Number of grid points for the interatomic separations.
        ntheta : int, optional
            Number of angular divisions in polar grid (more dense towards
            the bonding region).
        nr : int, optional
            Number of radial divisions in polar grid (more dense towards
            the atomic centers). With p=q=2 (default powers in polar grid)
            ntheta ~ 3*nr is optimal (for a given grid size).
            With ntheta=150, nr=50 you get ~1e-4 accuracy for H integrals
            (beyond that, gain is slow with increasing grid size).
        wflimit : float, optional
            Value below which the radial wave functions are considered
            to be negligible. This determines how far the polar grids
            around the atomic centers extend in space.
        superposition : str, optional
            Whether to use the density superposition ('density', default)
            or potential superposition ('potential') scheme for the
            Hamiltonian integrals.
        xc : str, optional
            Name of the exchange-correlation functional to be used
            in calculating the effective potential in the density
            superposition scheme (default: 'LDA').
            If the PyLibXC module is available, any LDA or GGA (but not
            hybrid or MGGA) functional available via LibXC can be specified.
            To e.g. use the N12 functional, set 'XC_GGA_X_N12+XC_GGA_C_N12'.
            If PyLibXC is not available, only the local density approximation
            xc='LDA' (alias: 'PW92') can be chosen.
        stride : int, optional
            The desired Skater-Koster table typically has quite a large
            number of points (N=500-1000), even though the integrals
            themselves are comparatively smooth. To speed up the
            construction of the SK-table, one can therefore restrict the
            expensive integrations to a subset N' = N // stride and map
            the resulting curves on the N-grid afterwards. The default
            stride (1) means that N' = N (no shortcut).
        smoothen_tails : bool, optional
            Whether to modify the 'tails' of the Slater-Koster integrals
            so that they smoothly decay to zero (default: True).
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

        self.timer.start('run_offsite2c')
        wf_range = self.get_range(wflimit)
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
            if R > 2 * wf_range:
                break

            grid, area = self.make_grid(R, wf_range, nt=ntheta, nr=nr)

            if  i == Nsub - 1 or Nsub // 10 == 0 or i % (Nsub // 10) == 0:
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt, flush=True)

            for p, (e1, e2) in enumerate(self.pairs):
                selected = select_integrals(e1, e2)
                S, H, H2 = 0., 0., 0.
                if len(grid) > 0:
                    S, H, H2 = self.calculate(selected, e1, e2, R, grid,
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

        self.timer.stop('run_offsite2c')

    def calculate(self, selected, e1, e2, R, grid, area,
                  superposition='potential', xc='LDA', only_overlap=False):
        """
        Calculates the selected Hamiltonian and overlap integrals.

        Parameters
        ----------
        selected : list of (integral, nl1, nl2) tuples
            Selected integrals to evaluate (for example
            [('dds', '3d', '4d'), (...)]).
        e1 : AtomicBase-like object
            <bra| element.
        e2 : AtomicBase-like object
            |ket> element.
        R : float
            Interatomic distance (e1 is at the origin, e2 at z=R).
        grid : np.ndarray
            2D arrray of grid points in the (x, z)-plane.
        area : np.ndarray
            1D array of areas associated with the grid points.
        superposition : str, optional
            Superposition scheme ('density' or 'potential').
        xc : str, optional
            Exchange-correlation functional (see self.run()).
            Defaults to LDA.
        only_overlap : bool, optional
            Whether to only evaluate the overlap integrals
            (default: False). If True, the 'xc' and 'superposition'
            have no influence.

        Returns
        -------
        S: np.ndarray
            Overlap integrals (R1 * R2 * angle_part).

        H: np.ndarray
            Hamiltonian integrals (<R1 | T + Veff - Vconf1 - Vconf2 | R2>).
            With potential superposition: Veff = Veff1 + Veff2.
            With density superposition: Veff = Vna1 + Vna2 + Vxc(rho1 + rho2).

        H2: np.ndarray
            Hamiltonian integrals calculated in an alternative way:
            H2 =  <R1 | (T1 + Veff1) + Veff2 - Vconf1 - Vconf2 | R2>
            = <R1 | H1 + Veff2 - Vconf1 - Vconf2 | R2> (operate with H1 on left)
            = <R1 | e1 + Veff2 - VConf1 - Vconf2 | R2>
            = e1 * S + <R1 | Veff2 - Vconf1 - Vconf2 | R2>
            By comparing H and H2, a numerical error can be estimated.
            This is only performend in the potential superposition scheme.
            In the density superposition scheme H2 only contains zeros.
        """
        self.timer.start('calculate_offsite2c')

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

        self.timer.start('veff')
        if not only_overlap and superposition == 'potential':
            v1 = e1.effective_potential(r1) - e1.confinement(r1)
            v2 = e2.effective_potential(r2) - e2.confinement(r2)
            veff = v1 + v2
        elif not only_overlap and superposition == 'density':
            rho = e1.electron_density(r1) + e2.electron_density(r2)
            veff = e1.neutral_atom_potential(r1)
            veff += e2.neutral_atom_potential(r2)
            if xc in ['LDA', 'PW92']:
                xc = XC_PW92()
                veff += xc.vxc(rho)
                vsigma = None
            else:
                xc = LibXC(xc)
                drho1 = e1.electron_density(r1, der=1)
                drho2 = e2.electron_density(r2, der=1)
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
                    dr2dx = x/r2
                    dc2dx = -(y - R)*dr2dx / r2**2
                    ds2dx = (r2 - x*dr2dx) / r2**2
                    dr1dy = y/r1
                    dc1dy = (r1 - y*dr1dy) / r1**2
                    ds1dy = -x*dr1dy / r1**2
                    dr2dy = (y - R)/r2
                    dc2dy = (r2 - (y - R)*dr2dy) / r2**2
                    ds2dy = -x*dr2dy / r2**2
        self.timer.stop('veff')

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

                if superposition == 'density' and xc.add_gradient_corrections:
                    self.timer.start('vsigma')
                    dRnl1 = e1.Rnl(r1, nl1, der=1)
                    dRnl2 = e2.Rnl(r2, nl2, der=1)
                    dgphi = dg(c1, c2, s1, s2, integral)
                    dgphidx = dgphi[0]*dc1dx + dgphi[1]*dc2dx \
                              + dgphi[2]*ds1dx + dgphi[3]*ds2dx
                    dgphidy = dgphi[0]*dc1dy + dgphi[1]*dc2dy \
                              + dgphi[2]*ds1dy + dgphi[3]*ds2dy
                    grad_phi_x = (dRnl1 * s1 * Rnl2 + Rnl1 * dRnl2 * s2) * gphi
                    grad_phi_x += Rnl1 * Rnl2 * dgphidx
                    grad_phi_y = (dRnl1 * c1 * Rnl2 + Rnl1 * dRnl2 * c2) * gphi
                    grad_phi_y += Rnl1 * Rnl2 * dgphidy
                    grad_rho_grad_phi = grad_rho_x * grad_phi_x \
                                        + grad_rho_y * grad_phi_y
                    H += 2. * np.sum(vsigma * grad_rho_grad_phi * area * x)
                    self.timer.stop('vsigma')

                Hl[index] = H
                H2l[index] = H2

        self.timer.stop('calculate_offsite2c')
        if only_overlap:
            return Sl
        else:
            return Sl, Hl, H2l

    def write(self, filename=None, pair=None, eigenvalues={},
              hubbardvalues={}, occupations={}, spe=0.):
        """
        Writes a Slater-Koster table to a file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to write to. The format is selected based
            on the extension (.par or .skf). Defaults to the
            '<el1>-<el2>_offsite2c.skf' template.
        pair : (str, str) tuple, optional
            Selects which of the two Slater-Koster tables to write,
            to be used in the heteronuclear case. Defaults to
            the symbol pair of (self.ela, self.elb).
        eigenvalues : dict, optional
            {nl: value} dictionary with valence orbital eigenvalues
            (or one-center onsite Hamiltonian integrals, if you will).
            Only written in the homonuclear case.
        hubbardvalues : dict, optional
            {nl: value} dictionary with valence orbital Hubbard values
            Only written in the homonuclear case.
        occupations : dict, optional
            {nl: value} dictionary with valence orbital occupations.
            Only written in the homonuclear case.
        """
        if pair is None:
            pair = (self.ela.get_symbol(), self.elb.get_symbol())

        fn = '%s-%s_offsite2c.skf' % pair if filename is None else filename

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
