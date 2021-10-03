#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import pickle
import numpy as np
from hotcent.atomic_base import NOT_SOLVED_MESSAGE
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import ZeroConfinement
from hotcent.interpolation import CubicSplineFunction
from hotcent.orbitals import ANGULAR_MOMENTUM
try:
    import matplotlib.pyplot as plt
except:
    plt = None


class PseudoAtomicDFT(AtomicDFT):
    def __init__(self, symbol, pp, *args, **kwargs):
        """ Create an atomic DFT calculator with a pseudopotential.

        Parameters:
        -----------
        symbol: chemical symbol
        pp: a pseudopotential instance

        The remaining (keyword) arguments are as in the AtomicDFT class.
        """
        AtomicDFT.__init__(self, symbol, *args, **kwargs)
        self.pp = pp

    def electron_density(self, r, der=0, only_valence=True):
        """ Return the all-electron density at r. """
        assert self.solved, NOT_SOLVED_MESSAGE
        assert only_valence

        if self.densval_fct is None:
            rcmax = 0
            nlmax = None
            for nl in self.wf_confinement:
                if hasattr(self.wf_confinement[nl], 'rc'):
                    rc = self.wf_confinement[nl].rc
                    if rc > rcmax:
                        rcmax = rc
                        nlmax = nl

            if nlmax is None:
                self.densval_fct = CubicSplineFunction(self.rgrid, self.dens)
            else:
                self.densval_fct = self.construct_wfn_interpolator(self.rgrid,
                                                                   self.dens,
                                                                   nlmax)
        return self.densval_fct(r, der=der)

    def calculate_veff(self, dens):
        """ Returns the effective potential (without nonlocal
        pseudopotential contributions) from the given density.
        """
        vhar = self.calculate_hartree_potential(dens, only_valence=True)
        exc, vxc = self.xc.evaluate(dens, self.grid)
        vconf = self.confinement(self.rgrid)
        vloc = self.pp.local_potential(self.rgrid)
        return vhar + vxc + vconf + vloc

    def guess_density(self):
        """ Returns a guess for the initial (valence) electron density. """
        dens = self.pp.get_valence_density(self.rgrid)
        return dens

    def calculate_energies(self, enl, unlg, dens, echo='valence'):
        """ Calculate the different contributions to the total energy. """
        self.timer.start('energies')
        assert echo in [None, 'valence']

        self.bs_energy = 0.0
        self.pp_energy = 0.0
        for nl in self.valence:
            self.bs_energy += self.configuration[nl] * enl[nl]
            l = ANGULAR_MOMENTUM[nl[1]]
            vpseudo = self.pp.local_potential(self.rgrid) \
                      + self.pp.nonlocal_potential(self.rgrid, l)
            self.pp_energy += self.grid.integrate(vpseudo * unlg[nl]**2,
                                                  use_dV=True)

        vhar = self.calculate_hartree_potential(dens, only_valence=True)
        self.vhar_energy = 0.5 * self.grid.integrate(vhar * dens, use_dV=True)

        exc, vxc = self.xc.evaluate(dens, self.grid)
        self.vxc_energy = self.grid.integrate(vxc * dens, use_dV=True)
        self.exc_energy = self.grid.integrate(exc * dens, use_dV=True)

        vconf = self.confinement(self.rgrid)
        self.confinement_energy = self.grid.integrate(vconf * dens,
                                                      use_dV=True)

        self.total_energy = self.bs_energy - self.vhar_energy + self.pp_energy
        self.total_energy += -self.vxc_energy + self.exc_energy

        if echo is not None:
            line = '%s orbital eigenvalues:' % echo
            print('\n'+line, file=self.txt)
            print('-' * len(line), file=self.txt)
            for n, l, nl in self.list_states():
                if nl in self.valence:
                    print('  %s:   %.12f' % (nl, enl[nl]), file=self.txt)

            print('\nenergy contributions:', file=self.txt)
            print('----------------------------------------', file=self.txt)
            print('sum of eigenvalues:     %.12f' % self.bs_energy,
                  file=self.txt)
            print('Hartree energy:         %.12f' % self.vhar_energy,
                  file=self.txt)
            print('pseudopotential energy: %.12f' % self.pp_energy,
                  file=self.txt)
            print('vxc correction:         %.12f' % self.vxc_energy,
                  file=self.txt)
            print('exchange + corr energy: %.12f' % self.exc_energy,
                  file=self.txt)
            print('----------------------------------------', file=self.txt)
            print('total energy:           %.12f\n' % self.total_energy,
                  file=self.txt)
        self.timer.stop('energies')

    def run(self, write=None):
        """ Execute the required atomic DFT calculations

        Parameters:

        write: None or a filename for saving the rgrid, effective
               potential and electron density.
        """
        def header(*args):
            print('=' * 50, file=self.txt)
            print('\n'.join(args), file=self.txt)
            print('=' * 50, file=self.txt)

        val = self.get_valence_orbitals()

        assert self.perturbative_confinement
        assert all([nl in val for nl in self.wf_confinement])
        nl_x = [nl for nl in val if nl not in self.wf_confinement]
        assert len(nl_x) == 0 or len(nl_x) == len(val), nl_x

        self.enl = {}
        self.unlg = {}
        self.Rnlg = {}
        self.unl_fct = {nl: None for nl in val}
        self.Rnl_fct = {nl: None for nl in val}

        header('Initial run without any confinement',
               'for pre-converging orbitals and eigenvalues')
        self.confinement = ZeroConfinement()
        dens_free, veff_free, enl_free, unlg_free, Rnlg_free = self.outer_scf()

        for nl, wf_confinement in self.wf_confinement.items():
            self.confinement = wf_confinement
            if self.confinement is None:
                self.confinement = ZeroConfinement()
            header('Applying %s' % self.confinement,
                   'to get a confined %s orbital' % nl)

            l = ANGULAR_MOMENTUM[nl[1]]
            veff = veff_free + self.confinement(self.rgrid) \
                   + self.pp.nonlocal_potential(self.rgrid, l)

            enl = {nl: enl_free[nl]}
            itmax, enl, d_enl, unlg, Rnlg = self.inner_scf(0, veff, enl, {},
                                                           solve=[nl], ae=False)
            print('Confined %s eigenvalue: %.6f' % (nl, enl[nl]), file=self.txt)

            self.enl.update(enl)
            self.unlg.update(unlg)
            self.Rnlg.update(Rnlg)

        for nl in val:
            if nl not in self.wf_confinement:
                self.enl[nl] = enl_free[nl]
                self.unlg[nl] = unlg_free[nl]
                self.Rnlg[nl] = Rnlg_free[nl]

        self.dens = self.calculate_density(self.unlg, only_valence=True)
        self.vhar = self.calculate_hartree_potential(self.dens,
                                                     only_valence=True)
        self.densval = self.dens
        self.vharval = self.vhar
        exc, self.vxc = self.xc.evaluate(self.dens, self.grid)
        self.confinement = ZeroConfinement()
        self.veff = self.calculate_veff(self.dens)

        if write is not None:
            with open(write, 'w') as f:
                pickle.dump(self.rgrid, f)
                pickle.dump(self.veff, f)
                pickle.dump(self.dens, f)

        self.solved = True
        self.timer.summary()
        self.txt.flush()

    def get_onecenter_integral(self, nl):
        """ Returns the chosen one-center integral (<phi|H|phi>). """
        assert self.solved, NOT_SOLVED_MESSAGE
        l = ANGULAR_MOMENTUM[nl[1]]
        veff = np.copy(self.veff)
        self.veff += self.pp.nonlocal_potential(self.rgrid, l)
        e = AtomicDFT.get_onecenter_integral(self, nl)
        self.veff = veff
        return e

    def outer_scf(self):
        """ Solve the self-consistent potential. """
        self.timer.start('outer_scf')
        print('\nStart iteration...', file=self.txt)
        enl = {nl: 0. for nl in self.valence}
        d_enl = {nl: 0. for nl in self.valence}
        unlg = {nl: None for nl in self.valence}
        Rnlg = {nl: None for nl in self.valence}

        dens = self.guess_density()
        veff = self.calculate_veff(dens)

        vnonloc = {}
        for nl in self.valence:
            l = 'spdf'.index(nl[1])
            vnonloc[nl] = self.pp.nonlocal_potential(self.rgrid, l)

        for it in range(self.maxiter):
            veff *= 1. - self.mix
            veff += self.mix * self.calculate_veff(dens)

            for nl in self.valence:
                itmax, e, d_e, ug, Rg = self.inner_scf(it, veff + vnonloc[nl],
                                                       enl, d_enl, solve=[nl],
                                                       ae=False)
                enl.update(e)
                d_enl.update(d_e)
                unlg.update(ug)
                Rnlg.update(Rg)

            dens0 = dens.copy()
            dens = self.calculate_density(unlg, only_valence=True)
            diff = self.grid.integrate(np.abs(dens - dens0), use_dV=True)

            if diff < self.convergence['density'] and it > 5:
                d_enl_max = max(d_enl.values())
                if d_enl_max < self.convergence['energies']:
                    break

            if np.mod(it, 10) == 0:
                line = 'iter %3i, dn=%.1e>%.1e, max %i sp-iter' % \
                       (it, diff, self.convergence['density'], itmax)
                print(line, file=self.txt, flush=True)

            if it == self.maxiter - 1:
                if self.timing:
                    self.timer.summary()
                err = 'Density not converged in %i iterations' % (it + 1)
                raise RuntimeError(err)

        self.calculate_energies(enl, unlg, dens, echo='valence')

        print('converged in %i iterations' % it, file=self.txt)
        nel = self.get_number_of_electrons(only_valence=True)
        line = '%9.4f electrons, should be %9.4f' % \
               (self.grid.integrate(dens, use_dV=True), nel)
        print(line, file=self.txt, flush=True)

        self.timer.stop('outer_scf')
        return dens, veff, enl, unlg, Rnlg
