""" Defintion of the GPAWAllElectron class for 
calculations with the GPAW atomic DFT code.
"""
from __future__ import print_function
import os
import pickle
from math import sqrt, pi, log
import numpy as np
from hotcent.atom import AllElectron
from hotcent.interpolation import Function
from hotcent.confinement import ZeroConfinement, SoftConfinement
from gpaw.xc import XC
from gpaw.utilities import hartree
from gpaw.atom.radialgd import AERadialGridDescriptor
from gpaw.atom.all_electron import AllElectron as GPAWAllElectron
from gpaw.atom.all_electron import shoot, tempdir


class GPAWAE(AllElectron, GPAWAllElectron):
    def __init__(self, symbol, **kwargs):
        """ 
        Run Kohn-Sham all-electron calculation for a given atom 
        using the atomic DFT calculator in GPAW.

        Parameters: see parent AllElectron class
        """
        AllElectron.__init__(self, symbol, **kwargs)

        config = kwargs['configuration']
        config = config.replace('[', '').replace(']', '')
        config = ','.join(config.split())
 
        GPAWAllElectron.__init__(self, symbol, xcname=self.xcname,
                                 configuration=config, 
                                 scalarrel=self.scalarrel,
                                 gpernode=self.nodegpts,
                                 txt=self.txt, nofiles=True) 

        self.timer.stop('init')

    def get_orbital_index(self, nl):
        for index, (n, l) in enumerate(zip(self.n_j, self.l_j)):
            if str(n) + 'spdfgh'[l] == nl:
                break
        else:
            msg = "GPAW atom ain't got %s orbitals for %s" % (nl, self.symbol)
            raise ValueError(msg)
        return index 

    def run(self, use_restart_file=False, wf_confinement_scheme='old'):
        """
        Parameters:

        use_restart_file: possibility to use restart file (untested)

        wf_confinement_scheme: determines how to apply the orbital
             confinement potentials for getting the confined orbitals:

            'old' = by applying the confinement for the chosen nl to
                    all states and reconverging all states (but only
                    saving the confined nl orbital),

            'new' = by applying the confinement only to the chosen nl and
                    reconverging only that state, while keeping the others
                    equal to those in the free (nonconfined) atom.

            The 'new' scheme is how basis sets are generated in e.g. GPAW.
            This option is also significantly faster, especially for
            heavier atoms.
            The choice of scheme does not affect the calculation of the
            confined density (which always happens the 'old' way).
        """
        self.timer.start('run')
        assert wf_confinement_scheme in ['old', 'new']

        valence = self.get_valence_orbitals()
        self.enl = {}
        self.Rnlg = {}
        self.unlg = {}
        bar = '=' * 50

        print(bar, file=self.txt)
        print('Initial run without any confinement', file=self.txt)
        print('for pre-converging orbitals and eigenvalues', file=self.txt)
        nl_0 = [nl for nl in valence
                if isinstance(self.wf_confinement[nl], ZeroConfinement)]
        if len(nl_0) > 0:
            print('and to get the (nonconfined) %s orbital%s' % \
                  (' and '.join(nl_0), 's' if len(nl_0) > 1 else ''),
                  file=self.txt)
        print(bar, file=self.txt)

        GPAWAllElectron.run(self, use_restart_file=use_restart_file)

        u_j = self.u_j.copy()
        e_j = [e for e in self.e_j]
        self.rgrid = self.r.copy()
        # Note: since the grid in GPAW-AllElectron starts at exactly 0,
        # there are cases where we will need to take limits for r->0.
        delta = (0. - self.rgrid[1]) / (self.rgrid[2] - self.rgrid[1])

        for n, l, nl in self.list_states():
            if nl in valence:
                assert nl in self.wf_confinement
                self.u_j = u_j.copy()
                self.e_j = [e for e in e_j]

                vconf = self.wf_confinement[nl]
                if not isinstance(vconf, ZeroConfinement):
                    print('%s\nApplying %s' % (bar, vconf), file=self.txt)
                    print('to get a confined %s orbital' % nl, file=self.txt)
                    print(bar, file=self.txt)
                    if wf_confinement_scheme == 'old':
                        self.run_confined(vconf,
                                          use_restart_file=use_restart_file)
                    elif wf_confinement_scheme == 'new':
                        j = self.get_orbital_index(nl)
                        if isinstance(vconf, SoftConfinement):
                            rc = vconf.rc
                        else:
                            rc = np.max(self.rgrid)
                        v = vconf(self.rgrid)
                        u, e  = GPAWAllElectron.solve_confined(self, j, rc, v)
                        self.u_j[j], self.e_j[j] = u, e
                        print('Confined %s eigenvalue: %.6f' % (nl, e),
                               file=self.txt)

            index = self.get_orbital_index(nl)
            self.rgrid[0] = 1.
            self.unlg[nl] = self.u_j[index].copy()
            self.enl[nl] = self.e_j[index]
            self.Rnlg[nl] = self.unlg[nl] / self.rgrid
            self.Rnlg[nl][0] = self.Rnlg[nl][1]
            self.Rnlg[nl][0] += delta * (self.Rnlg[nl][2] - self.Rnlg[nl][1])
            self.rgrid[0] = 0.
            self.Rnl_fct[nl] = Function('spline', self.rgrid, self.Rnlg[nl])
            self.unl_fct[nl] = Function('spline', self.rgrid, self.unlg[nl])

        vconf = self.confinement
        print(bar, file=self.txt)
        print('Applying %s' % vconf, file=self.txt)
        print('to get the confined electron density', file=self.txt)
        print(bar, file=self.txt)

        self.u_j = u_j.copy()
        self.e_j = [e for e in e_j]
        self.run_confined(vconf, use_restart_file=use_restart_file)

        self.rgrid[0] = 1.
        self.veff = self.vr.copy() / self.rgrid
        self.veff[0] = self.veff[1] + delta * (self.veff[2] - self.veff[1])
        self.dens = self.calculate_density()
        self.total_energy = self.ETotal
        self.Hartree = self.vHr.copy() / self.rgrid
        self.Hartree[0] = self.Hartree[1]
        self.Hartree[0] += delta * (self.Hartree[2] - self.Hartree[1])
        self.rgrid[0] = 0.

        self.solved = True
        self.timer.stop('run')
        self.timer.summary()
        self.txt.flush()

    def run_confined(self, vconf, use_restart_file=True):
        """ Minor modification of the parent run()
        method to do full SCF for a confined atom.

        Note: vconf should be a callable function
              (not something array-like)
        """
        t = self.text
        N = self.N
        beta = self.beta
        t(N, 'radial gridpoints.')
        self.rgd = AERadialGridDescriptor(beta / N, 1.0 / N, N)
        g = np.arange(N, dtype=float)
        self.r = self.rgd.r_g
        self.dr = self.rgd.dr_g
        self.d2gdr2 = self.rgd.d2gdr2()

        # Number of orbitals:
        nj = len(self.n_j)

        # Radial wave functions multiplied by radius:
        self.u_j = np.zeros((nj, self.N))

        # Effective potential multiplied by radius:
        self.vr = np.zeros(N)

        # Add confinement potential:
        self.vr += vconf(self.r) * self.r  # mod

        # Electron density:
        self.n = np.zeros(N)

        # Always spinpaired nspins=1
        if type(self.xcname) == str:
            self.xc = XC(self.xcname)
        else:
            self.xc = self.xcname

        # Initialize for non-local functionals
        if self.xc.type == 'GLLB':
            self.xc.pass_stuff_1d(self)
            self.xc.initialize_1d()
            
        n_j = self.n_j
        l_j = self.l_j
        f_j = self.f_j
        e_j = self.e_j
        
        Z = self.Z    # nuclear charge
        r = self.r    # radial coordinate
        dr = self.dr  # dr/dg
        n = self.n    # electron density
        vr = self.vr  # effective potential multiplied by r

        vHr = np.zeros(self.N)
        self.vXC = np.zeros(self.N)

        restartfile = '%s/%s.restart' % (tempdir, self.symbol)
        if self.xc.type == 'GLLB' or not use_restart_file:
            # Do not start from initial guess when doing
            # non local XC!
            # This is because we need wavefunctions as well
            # on the first iteration.
            fd = None
        else:
            try:
                fd = open(restartfile, 'rb')
            except IOError:
                fd = None
            else:
                try:
                    n[:] = pickle.load(fd)
                except (ValueError, IndexError):
                    fd = None
                else:
                    norm = np.dot(n * r**2, dr) * 4 * pi
                    if abs(norm - sum(f_j)) > 0.01:
                        fd = None
                    else:
                        t('Using old density for initial guess.')
                        n *= sum(f_j) / norm

        if fd is None:
            self.initialize_wave_functions()
            n[:] = self.calculate_density()

        bar = '|------------------------------------------------|'
        t(bar)
        
        niter = 0
        qOK = log(1e-10)
        mix = self.mix  # mod
        nitermax = self.maxiter  # mod
        
        # orbital_free needs more iterations and coefficient
        if self.orbital_free:
            mix = 0.01
            nitermax = 2000
            e_j[0] /= self.tw_coeff
            if Z > 10 : #help convergence for third row elements
                mix = 0.002
                nitermax = 10000
            
        vrold = None
        
        while True:
            # calculate hartree potential
            hartree(0, n * r * dr, r, vHr)

            # add potential from nuclear point charge (v = -Z / r)
            vHr -= Z

            # calculated exchange correlation potential and energy
            self.vXC[:] = 0.0

            if self.xc.type == 'GLLB':
                # Update the potential to self.vXC an the energy to self.Exc
                Exc = self.xc.get_xc_potential_and_energy_1d(self.vXC)
            else:
                Exc = self.xc.calculate_spherical(self.rgd,
                                                  n.reshape((1, -1)),
                                                  self.vXC.reshape((1, -1)))

            # calculate new total Kohn-Sham effective potential and
            # admix with old version

            vr[:] = (vHr + self.vXC * r + vconf(r) * r)  # mod

            if self.orbital_free:
                vr /= self.tw_coeff

            if niter > 0:
                vr[:] = mix * vr + (1 - mix) * vrold
            vrold = vr.copy()

            # solve Kohn-Sham equation and determine the density change
            self.solve_confined()
            dn = self.calculate_density() - n
            n += dn

            # estimate error from the square of the density change integrated
            q = log(np.sum((r * dn)**2))

            # print progress bar
            if niter == 0:
                q0 = q
                b0 = 0
            else:
                b = int((q0 - q) / (q0 - qOK) * 50)
                if b > b0:
                    self.txt.write(bar[b0:min(b, 50)])
                    self.txt.flush()
                    b0 = b

            # check if converged and break loop if so
            if q < qOK:
                self.txt.write(bar[b0:])
                self.txt.flush()
                break

            niter += 1
            if niter > nitermax:
                raise RuntimeError('Did not converge!')

        tau = self.calculate_kinetic_energy_density()

        t()
        t('Converged in %d iteration%s.' % (niter, 's'[:niter != 1]))

        try:
            fd = open(restartfile, 'wb')
        except IOError:
            pass
        else:
            pickle.dump(n, fd)
            try:
                os.chmod(restartfile, 0o666)
            except OSError:
                pass

        # <mods>
        Eeig = 0        
        for f, e in zip(f_j, e_j):
            Eeig += f * e

        hartree(0, n * r * dr, r, vHr)
        Ehar = 2 * pi * np.dot(n * r * vHr, dr)
        self.vHr = vHr
        
        Evxc = 4 * pi * np.dot(n * self.vXC * r * r, dr)

        Etot = Eeig - Ehar + Exc - Evxc

        t()
        t('Energy contributions:')
        t('-------------------------')
        t('E_eig:  %+13.6f' % Eeig)
        t('E_har:  %+13.6f' % Ehar)
        t('E_xc:   %+13.6f' % Exc)
        t('E_vxc:  %+13.6f' % Evxc)
        t('-------------------------')
        t('Total:     %+13.6f' % Etot)
        self.ETotal = Etot
        t()
        # </mods>

        t('state      eigenvalue         ekin         rmax')
        t('-----------------------------------------------')
        for m, l, f, e, u in zip(n_j, l_j, f_j, e_j, self.u_j):
            # Find kinetic energy:
            k = e - np.sum((np.where(abs(u) < 1e-160, 0, u)**2 *  # XXXNumeric!
                            vr * dr)[1:] / r[1:])

            # Find outermost maximum:
            g = self.N - 4
            while u[g - 1] >= u[g]:
                g -= 1
            x = r[g - 1:g + 2]
            y = u[g - 1:g + 2]
            A = np.transpose(np.array([x**i for i in range(3)]))
            c, b, a = np.linalg.solve(A, y)
            assert a < 0.0
            rmax = -0.5 * b / a

            s = 'spdf'[l]
            t('%d%s^%-4.1f: %12.6f %12.6f %12.3f' % (m, s, f, e, k, rmax))
        t('-----------------------------------------------')
        t('(units: Bohr and Hartree)')

        for m, l, u in zip(n_j, l_j, self.u_j):
            self.write(u, 'ae', n=m, l=l)

        self.write(n, 'n')
        self.write(vr, 'vr')
        self.write(vHr, 'vHr')
        self.write(self.vXC, 'vXC')
        self.write(tau, 'tau')

    def solve_confined(self):
        """ Minor modification of Parent solve() method
        to deal with confined atoms 
        """
        r = self.r
        dr = self.dr
        vr = self.vr

        c2 = -(r / dr)**2
        c10 = -self.d2gdr2 * r**2  # first part of c1 vector

        if self.scalarrel:
            self.r2dvdr = np.zeros(self.N)
            self.rgd.derivative(vr, self.r2dvdr)
            self.r2dvdr *= r
            self.r2dvdr -= vr
        else:
            self.r2dvdr = None

        # solve for each quantum state separately
        for j, (n, l, e, u) in enumerate(zip(self.n_j, self.l_j,
                                             self.e_j, self.u_j)):
            nodes = n - l - 1  # analytically expected number of nodes
            delta = -0.2 * e
            nn, A = shoot(u, l, vr, e, self.r2dvdr, r, dr, c10, c2,
                          self.scalarrel)

            # adjust eigenenergy until u has the correct number of nodes
            while nn != nodes:
                diff = np.sign(nn - nodes)
                while diff == np.sign(nn - nodes):
                    e -= diff * delta
                    nn, A = shoot(u, l, vr, e, self.r2dvdr, r, dr, c10, c2,
                                  self.scalarrel)
                delta /= 2

            # adjust eigenenergy until u is smooth at the turning point
            de = 1.0
            while abs(de) > 1e-9:
                norm = np.dot(np.where(abs(u) < 1e-160, 0, u)**2, dr)
                u *= 1.0 / sqrt(norm)
                de = 0.5 * A / norm
                e -= de  # mod 
                nn, A = shoot(u, l, vr, e, self.r2dvdr, r, dr, c10, c2,
                              self.scalarrel)
            self.e_j[j] = e
            u *= 1.0 / sqrt(np.dot(np.where(abs(u) < 1e-160, 0, u)**2, dr))
