""" Defintion of the SlaterKosterTable class (and
supporting methods) for calculating Slater-Koster
integrals.

The code below draws heavily from the Hotbit code 
written by Pekka Koskinen (https://github.com/pekkosk/
hotbit/blob/master/hotbit/parametrization/slako.py).
"""
from __future__ import division, print_function
import sys
from math import sin, cos, tan, sqrt
import numpy as np
from scipy.interpolate import SmoothBivariateSpline
from ase.data import atomic_numbers, atomic_masses
from hotcent.timing import Timer
from hotcent.atom_hotcent import XC_PW92
try:
    from gpaw.xc import XC
    has_gpaw = True
except:
    print('Warning: could not import from GPAW')
    has_gpaw = False


class SlaterKosterTable:
    def __init__(self, ela, elb, txt=None, timing=False):
        """ Construct Slater-Koster table for given elements.
                
        parameters:
        -----------
        ela:    element objects (KSAllElectron or Element)
        elb:    element objects (KSAllElectron or Element)    
        txt:    output file object or file name
        timing: output of timing summary after calculation
        """
        self.ela = ela
        self.elb = elb
        self.timing = timing

        if txt == None:
            self.txt = sys.stdout
        else:
            if type(txt) == type(''):
                self.txt = open(txt, 'a')
            else:
                self.txt = txt                

        #self.comment = self.ela.get_comment()

        if ela.get_symbol() != elb.get_symbol():
            self.nel = 2
            self.pairs = [(ela, elb), (elb, ela)]
            self.elements = [ela, elb]
            #self.comment += '\n' + self.elb.get_comment()
        else:
            self.nel = 1
            self.pairs = [(ela, elb)]
            self.elements = [ela]

        self.timer = Timer('SlaterKosterTable', txt=self.txt, enabled=timing)
                                        
        print('\n\n\n\n', file=self.txt)                                        
        print('***********************************************', file=self.txt)
        print('Slater-Koster table construction for %2s and %2s' % \
              (ela.get_symbol(), elb.get_symbol()), file=self.txt)
        print('***********************************************', file=self.txt)
        self.txt.flush()

    def __del__(self):
        self.timer.summary()
        
    def get_table(self):
        """ Return tables. """
        return self.Rgrid, self.tables        
                
    def smooth_tails(self):
        """ Smooth the behaviour of tables near cutoff. """
        for p in range(self.nel):
            for i in range(20):
                self.tables[p][:, i] = tail_smoothening(self.Rgrid,
                                                        self.tables[p][:, i])                
        
    def write(self, filename=None, pair=None):
        """ Use symbol1-symbol2.par as default.

        filename: str with name of file to write to

        pair: either (symbol_a, symbol_b) or (symbol_b, symbol_a)
              to select which table to write in the SKF format
        """
        if pair is None:
            pair = (self.ela.get_symbol(), self.elb.get_symbol())

        fn = '%s-%s.skf' % pair if filename is None else filename

        ext = fn[-4:]

        assert ext in ['.par', '.skf'], \
               "Unknown format: %s (-> choose .par or .skf)" % ext

        with open(fn, 'w') as handle:
            if ext == '.par':
                self._write_par(handle)
            elif ext == '.skf':
                self._write_skf(handle, pair=pair)

    def _write_skf(self, handle, pair):
        """ Write to SKF file format; this function
        is an adaptation of hotbit.io.hbskf 
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

        grid_dist = self.Rgrid[1] - self.Rgrid[0]
        grid_npts = len(self.tables[index])
        grid_npts += int(self.Rgrid[0] / (self.Rgrid[1] - self.Rgrid[0]))
        print("%.12f, %d" % (grid_dist, grid_npts), file=handle)

        el1, el2 = self.ela.get_symbol(), self.elb.get_symbol()
        if el1 == el2:
            print("E_d   E_p   E_s   SPE   U_d   U_p   U_s   f_d    f_p   f_s",
                  file=handle)

        m = atomic_masses[atomic_numbers[symbols[index]]]
        print("%.3f, 19*0.0" % m, file=handle)

        # Integral table containing the DFTB Hamiltonian

        if self.Rgrid[0] != 0:
            n = int(self.Rgrid[0] / (self.Rgrid[1] - self.Rgrid[0]))
            for i in range(n):
                print('%d*0.0,' % len(self.tables[0][0]), file=handle)

        ct, theader = 0, ''
        for i in range(len(self.tables[index])):
            line = ''

            for j in range(len(self.tables[index][i])):
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
                theader = ''
                line += '{0: 1.12e}  '.format(self.tables[index][i, j])

            print(line, file=handle)

    def _write_par(self, handle):
        #print('slako_comment=', file=f)
        #print(self.get_comment(), '\n\n', file=f)
        for p, (e1, e2) in enumerate(self.pairs):
            line = '%s-%s_table=' % (e1.get_symbol(), e2.get_symbol())
            print(line, file=handle)

            for i, R in enumerate(self.Rgrid):
                print('%.6e' % R, end=' ', file=handle)

                for t in range(20):
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
        filename:     for graphics file
        """
        try:
            import pylab as pl
        except:
            raise AssertionError('pylab could not be imported')

        fig = pl.figure()
        fig.subplots_adjust(hspace=1e-4, wspace=1e-4)

        mx = max(1, self.tables[0].max())
        if self.nel == 2:
            mx = max(mx, self.tables[1].max())

        for i in range(10):
            name = integrals[i]
            ax = pl.subplot(5, 2, i + 1)

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
                    ax.text(0.03, 0.02 + p * 0.15, 
                            'No %s integrals for <%s|%s>' % (name, s1, s2), 
                            transform=ax.transAxes, size=10)
                    if not ax.is_last_row():
                        pl.xticks([], [])
                    if not ax.is_first_col():
                        pl.yticks([], [])
                else:
                    pl.plot(self.Rgrid, self.tables[p][:, i] , c='r', 
                            ls=s, lw=lw, alpha=alpha)
                    pl.plot(self.Rgrid, self.tables[p][:, i + 10], c='b', 
                            ls=s, lw=lw, alpha=alpha)
                    pl.axhline(0, c='k', ls='--')
                    pl.title(name, position=(0.9, 0.8)) 

                    if ax.is_last_row():
                        pl.xlabel('r (Bohr)')                                        
                    else:
                        pl.xticks([], [])

                    if not ax.is_first_col():                   
                        pl.yticks([],[])

                    pl.ylim(-mx, mx)
                    pl.xlim(0)
        
        pl.figtext(0.3, 0.95, 'H', color='r', size=20)
        pl.figtext(0.34, 0.95, 'S', color='b', size=20)
        pl.figtext(0.38, 0.95, ' Slater-Koster tables', size=20)
        e1, e2 = self.ela.get_symbol(), self.elb.get_symbol()
        pl.figtext(0.3, 0.92, '(thin solid: <%s|%s>, wide dashed: <%s|%s>)' \
                   % (e1, e2, e2, e1), size=10)
        
        if filename is None:
            filename = '%s-%s_slako.pdf' % (e1, e2)
        pl.savefig(filename)            
    
    def get_range(self, fractional_limit):
        """ Define ranges for the atoms: largest r such that Rnl(r)<limit. """
        self.timer.start('define ranges')
        wf_range = 0.

        for el in self.elements:
            r = max([el.get_wf_range(nl, fractional_limit) 
                     for nl in el.get_valence_orbitals()])
            print('wf range for %s=%10.5f' % (el.get_symbol(), r), 
                  file=self.txt)
            wf_range = max(r, wf_range)

        assert wf_range < 20, 'Wave function range >20 Bohr. Decrease wflimit?'
        self.timer.stop('define ranges')        
        return wf_range
        
    def run(self, R1, R2, N, ntheta=150, nr=50, wflimit=1e-7, 
            superposition='potential', xc='LDA'):
        """ Calculate the Slater-Koster table. 
         
        parameters:
        ------------
        R1, R2, N: make table from R1 to R2 with N points
        ntheta: number of angular divisions in polar grid
                (more dense towards bonding region).
        nr:     number of radial divisions in polar grid
                (more dense towards origins).
                with p=q=2 (powers in polar grid) ntheta~3*nr is 
                optimal (with fixed grid size)
                with ntheta=150, nr=50 you get~1E-4 accuracy for H-elements
                (beyond that, gain is slow with increasing grid size)
        wflimit: use max range for wfs such that at R(rmax)<wflimit*max(R(r))
        superposition: 'density' or 'potential': whether to use the density
                superposition or potential superposition approach for the 
                Hamiltonian integrals. 
        xc:     name of the exchange-correlation functional to be used
                in calculating the effective potential in the density
                superposition scheme. If the GPAW module is available,
                any LDA or GGA (but not hybrid or MGGA) functional available
                via GPAW can be specified, in the same syntax as the GPAW
                calculator. Native GGA functionals in GPAW include, at present,
                PBE, revPBE, RPBE, and PW91 (see gpaw/xc/kernel.py) and some
                pure-python implementations of them ('pyPBE', 'pyPBEsol',
                'pyRPBE'). Many more (LDA/GGA) functionals can be used if
                GPAW has been linked to LibXC (e.g. for using the N12
                functional, set xc='XC_GGA_X_N12+XC_GGA_C_N12').
                Also the PyLibXC module can be used (see pylibxc_interface.py).
                If GPAW or PyLibXC is not available, only the local density
                approximation 'PW92' (alias: 'LDA') can be chosen.
        """
        assert R1 >= 1e-3, 'For stability; use R1 >~ 1e-3'
        assert superposition in ['density', 'potential']
        if not has_gpaw:
            assert xc in ['LDA', 'PW92']

        self.timer.start('calculate tables')   
        self.wf_range = self.get_range(wflimit)        
        Rgrid = np.linspace(R1, R2, N)
        self.N = N
        self.Rgrid = Rgrid
        self.dH = 0.
        self.Hmax = 0.

        if self.nel == 1: 
            self.tables = [np.zeros((N, 20))]
        else: 
            self.tables = [np.zeros((N, 20)), np.zeros((N, 20))]
        
        print('Start making table...', file=self.txt)
        self.txt.flush()

        for p, (e1, e2) in enumerate(self.pairs):
            print('Integrals:', end=' ', file=self.txt)
            selected = select_integrals(e1, e2)
            for s in selected:
                print(s[0], end=' ', file=self.txt)
            print(file=self.txt)
            self.txt.flush()

        for Ri, R in enumerate(Rgrid):
            if R > 2 * self.wf_range: 
                break

            grid, areas = self.make_grid(R, nt=ntheta, nr=nr)

            if  Ri == N - 1 or N // 10 == 0 or np.mod(Ri, N // 10) == 0:                    
                print('R=%8.2f, %i grid points ...' % (R, len(grid)),
                      file=self.txt)
                self.txt.flush()
 
            for p, (e1, e2) in enumerate(self.pairs):
                selected = select_integrals(e1, e2) 
                if len(grid) == 0:
                    S, H, H2 = 0., 0., 0.
                else:
                    S, H, H2 = self.calculate_mels(selected, e1, e2, R, grid,
                                                   areas, xc=xc,
                                                   superposition=superposition)
                    self.Hmax = max(self.Hmax, max(abs(H)))
                    self.dH = max(self.dH, max(abs(H - H2)))
                self.tables[p][Ri, :10] = H
                self.tables[p][Ri, 10:] = S

        if superposition == 'potential':        
            print('Maximum value for H=%.2g' % self.Hmax, file=self.txt)
            print('Maximum error for H=%.2g' % self.dH, file=self.txt)        
            print('     Relative error=%.2g %%' % (self.dH / self.Hmax * 100),
                  file=self.txt)

        self.smooth_tails()
        self.timer.stop('calculate tables')  
        #self.comment+='\n'+asctime()
        self.txt.flush()
               
    def calculate_mels(self, selected, e1, e2, R, grid, area, 
                       superposition='potential', xc='LDA'):
        """ 
        Perform integration for selected H and S integrals.
         
        parameters:
        -----------
        selected: list of [('dds','3d','4d'),(...)]
        e1: <bra| element
        e2: |ket> element
        R: e1 is at origin, e2 at z=R
        grid: list of grid points on (d,z)-plane
        area: d-z areas of the grid points.
        superposition: 'density' or 'potential' superposition scheme
        xc: exchange-correlation functional (see description in self.run())

        return:
        -------
        List of H,S and H2 for selected integrals. In the potential
        superposition scheme, H2 is calculated using a different technique
        and can be used for error estimation. This is not available
        for the density superposition scheme, where simply H2=0 is returned.
        
        S: simply R1*R2*angle-part

        H: operate (derivate) R2 <R1|t+Veff-Conf1-Conf2|R2>.
           With potential superposition: Veff = Veff1 + Veff2
           With density superposition: Veff = Vxc(n1 + n2)  (XC=PW92)

        H2: operate with full h2 and hence use eigenvalue of |R2> 
            with full Veff2:
              <R1|(t1+Veff1)+Veff2-Conf1-Conf2|R2> 
            = <R1|h1+Veff2-Conf1-Conf2|R2> (operate with h1 on left)
            = <R1|e1+Veff2-Conf1-Conf2|R2> 
            = e1*S + <R1|Veff2-Conf1-Conf2|R2> 
            -> H and H2 can be compared and error estimated
        """
        self.timer.start('calculate_mels')
        Sl, Hl, H2l = np.zeros(10), np.zeros(10), np.zeros(10)
        
        # common for all integrals (not wf-dependent parts)
        self.timer.start('prelude')

        N = len(grid)
        x = grid[:N, 0]
        y = grid[:N, 1]
        r1 = np.sqrt(x ** 2 + y ** 2)
        r2 = np.sqrt(x ** 2 + (R - y) ** 2)
        t1 = np.arccos(y / r1)
        t2 = np.arccos((y - R) / r2)
        radii = np.array([r1, r2]).T
        gphi = g(t1, t2).T

        if superposition == 'potential':
            v1 = e1.effective_potential(r1) - e1.confinement(r1)
            v2 = e2.effective_potential(r2) - e2.confinement(r2)
            veff = v1 + v2
        elif superposition == 'density':
            n = e2.electron_density(r1) + e2.electron_density(r2)
            veff = e1.V_nuclear(r1) + e1.hartree_potential(r1)
            veff += e2.V_nuclear(r2) + e2.hartree_potential(r2)
            if not has_gpaw:
                assert xc in ['LDA', 'PW92']
                func = XC_PW92()
                veff += func.vxc(n)
            else:
                grad_x = e1.electron_density(r1, der=1) * np.sin(t1)
                grad_x += e2.electron_density(r2, der=1) * np.sin(t2)
                grad_y = e1.electron_density(r1, der=1) * np.cos(t1)
                grad_y += e2.electron_density(r2, der=1) * np.cos(t2)
                sigma = np.sqrt([grad_x ** 2 + grad_y ** 2]) ** 2
                dens = np.array([n])
                dedn = np.zeros_like(dens)
                exc = np.zeros_like(dens)
                dedsigma = np.zeros_like(dens)
                if type(xc) == str:
                    func = XC(xc)
                else:
                    func = xc
                func.kernel.calculate(exc, dens, dedn, sigma, dedsigma)
                veff += dedn[0]
                # add gradient corrections to vxc
                splx = SmoothBivariateSpline(x, y, dedsigma * grad_x)
                sply = SmoothBivariateSpline(x, y, dedsigma * grad_y)
                veff += -2. * splx(x, y, dx=1, dy=0, grid=False)
                veff += -2. * sply(x, y, dx=0, dy=1, grid=False)

        assert np.shape(gphi) == (N, 10)
        assert np.shape(radii) == (N, 2)
        assert np.shape(veff) == (N,)

        self.timer.stop('prelude')                             
        
        # calculate all selected integrals
        for integral, nl1, nl2 in selected:           
            index = integrals.index(integral)
            S, H, H2 = 0., 0., 0.
            l2 = angular_momentum[nl2[1]]

            nA = len(area)
            r1 = radii[:nA, 0]
            r2 = radii[:nA, 1]
            d, z = grid[:nA, 0], grid[:nA, 1]
            aux = gphi[:nA, index] * area * d
            Rnl1 = e1.Rnl(r1, nl1)
            Rnl2 = e2.Rnl(r2, nl2)
            ddunl2 = e2.unl(r2, nl2, der=2)

            S = np.sum(Rnl1 * Rnl2 * aux)
            
            H = np.sum(Rnl1 * (-0.5* ddunl2 / r2 + (veff + \
                       l2 * (l2 + 1) / (2 * r2 ** 2)) * Rnl2) * aux)
            
            if superposition == 'potential':
                H2 = np.sum(Rnl1 * Rnl2 * aux * (v2 - e1.confinement(r1)))
                H2 += e1.get_epsilon(nl1) * S
            elif superposition == 'density':
                H2 = 0

            Sl[index] = S
            Hl[index] = H
            H2l[index] = H2
            
        self.timer.stop('calculate_mels')
        return Sl, Hl, H2l
        
    def make_grid(self, Rz, nt, nr, p=2, q=2, view=False):
        """
        Construct a double-polar grid.
        
        Parameters:
        -----------
        Rz: element 1 is at origin, element 2 at z=Rz
        nt: number of theta grid points
        nr: number of radial grid points
        p: power describing the angular distribution of grid points 
           (larger puts more weight towards theta=0)
        q: power describing the radial disribution of grid points 
           (larger puts more weight towards centers)   
        view: view the distribution of grid points with pylab.
          
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
        self.timer.start('make grid')
        rmin, rmax = (1e-7, self.wf_range)
        max_range = self.wf_range
        h = Rz / 2

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
            d1 = R[i + 1] * np.sin(Tj0)
            z1 = R[i + 1] * np.cos(Tj0)
            d2 = R[i] * np.sin(Tj0)
            z2 = R[i] * np.cos(Tj0)
            d3 = R[i] * np.sin(Tj1)
            z3 = R[i] * np.cos(Tj1)
            d4 = R[i + 1] * np.sin(Tj1)
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
                                 + 0.5 * h ** 2 * (tan(Th1) - np.tan(x)),
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
                                 * (tan(Th1) - tan(Th0))),
                       lambda x: 0.5 * h ** 2 * (np.tan(x) - tan(Th0)) \
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
            import pylab as pl
            pl.plot([h, h ,h])        
            pl.scatter(grid[:, 0], grid[:, 1], s=10 * area / max(area))
            pl.show()
            
        self.timer.stop('make grid')            
        return grid, area
        
        
integrals = ['dds', 'ddp', 'ddd', 'pds', 'pdp', 'pps', 'ppp', 
             'sds', 'sps', 'sss']
angular_momentum = {'s':0, 'p':1, 'd':2}


def select_orbitals(val1, val2, integral):
    """ 
    Select orbitals from given valences to calculate given integral. 
    e.g. ['2s','2p'],['4s','3d'],'sds' --> '2s' & '3d'
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
    """ Return list of integrals (integral,nl1,nl2) 
    to be done for element pair e1, e2. """
    selected = []
    val1, val2 = e1.get_valence_orbitals(), e2.get_valence_orbitals()

    for ii, integral in enumerate(integrals):
        nl1, nl2 = select_orbitals(val1 , val2 , integral)

        if nl1 == None or nl2 == None:
            continue
        else:
            selected.append((integral, nl1, nl2))

    return selected 
            
        
def g(t1, t2):
    """
    Return the angle-dependent part of the two-center 
    integral (it) with t1=theta_1 (atom at origin)
    and t2=theta2 (atom at z=Rz). These dependencies
    come after integrating analytically over phi.
    """
    c1, c2, s1, s2 = np.cos(t1), np.cos(t2), np.sin(t1), np.sin(t2)   
    return np.array([5. / 8 * (3 * c1 ** 2 - 1) * (3 * c2 ** 2 - 1),\
                     15. / 4 * s1 * c1 * s2 * c2,\
                     15. / 16 * s1 ** 2 * s2 ** 2,\
                     np.sqrt(15.) / 4 * c1 * (3 * c2 ** 2 - 1),\
                     np.sqrt(45.) / 4 * s1 * s2 * c2,\
                     3. / 2 * c1 * c2,\
                     3. / 4 * s1 * s2,\
                     np.sqrt(5.) / 4 * (3 * c2 ** 2 - 1),\
                     np.sqrt(3.) / 2 * c2,\
                     0.5*np.ones_like(t1)])


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

    if k < N/4:
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
