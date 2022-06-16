#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from scipy import linalg
import becke
from ase.units import Bohr
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.orbitals import ANGULAR_MOMENTUM, ORBITAL_LABELS, ORBITALS
from hotcent.phillips_kleinman import PhillipsKleinmanPP
from hotcent.slako import INTEGRAL_PAIRS as INTEGRAL_PAIRS_2c
from hotcent.spherical_harmonics import (sph_cartesian, sph_cartesian_der,
                                         sph_cartesian_der2)
from hotcent.threecenter import (INTEGRAL_PAIRS as INTEGRAL_PAIRS_3c,
                                 XZ_ANTISYMMETRIC_ORBITALS,
                                 XZ_SYMMETRIC_ORBITALS)
from hotcent.xc import LibXC


def ase2becke(atoms_ase):
    atoms_becke = [(atom.number, tuple(atom.position / Bohr))
                   for atom in atoms_ase]
    return atoms_becke


def set_becke_settings(settings):
    becke.settings.radial_grid_factor, becke.settings.lebedev_order = settings


class BeckeHarris:
    """
    Calculator for Hamiltonian and overlap matrix elements and
    repulsive energies using Becke integration grids. The electron
    density is taken as the sum of the atomic densities, which is
    similar to how the Harris functional is typically applied.

    Note: this calculator is primarily meant for checking Hotcent's
    integration procedures on few-atom, nonperiodic structures.
    As such, it is not optimized for speed.

    Note: this requires the [becke](
    https://github.com/humeniuka/becke_multicenter_integration)
    Python package written by Alexander Humeniuk.

    Parameters
    ----------
    elements : dict of AtomicBase-like
        Atomic calculators for every anticipated chemical element.
    xc : str, optional
        Exchange-correlation functional from LibXC.
        Default: LDA_X+LDA_C_PW, i.e. LDA in the PW92 parametrization.
    settings_T, settings_V : tuple of 2 ints
        Settings for the Becke integration grids for integrals involving
        kinetic operators and potential operators, respectively.
        The first item is the radial grid factor, the second the
        Lebedev order for the angular grid.
    """
    def __init__(self, elements, xc='LDA_X+LDA_C_PW', settings_T=(13, 47),
                 settings_V=(13, 47)):
        self.elements = elements
        self.xc = LibXC(xc)
        self.settings_T = settings_T
        self.settings_V = settings_V
        set_becke_settings(settings_V)

        # Find the pseudopotential type
        all_pk_pp = all([isinstance(el.pp, PhillipsKleinmanPP)
                         for sym, el in self.elements.items()])
        all_kb_pp = all([isinstance(el.pp, KleinmanBylanderPP)
                         for sym, el in self.elements.items()])
        assert all_pk_pp or all_kb_pp, \
               'Cannot handle mixture of PK and KB pseudopotentials'
        self.pp_type = 'PK' if all_pk_pp else 'KB'

    def get_symbol(self, index):
        return self.atoms_ase.symbols[index]

    def get_position(self, index):
        return self.atoms_ase.positions[index, :] / Bohr

    def get_valence_orbitals(self, index):
        sym = self.get_symbol(index)
        nls, lms = [], []
        for valence in self.elements[sym].basis_sets:
            for nl in valence:
                l = ANGULAR_MOMENTUM[nl[1]]
                for lm in ORBITALS[l]:
                    nls.append(nl)
                    lms.append(lm)
        return (nls, lms)

    def run_selected_HS(self, atoms_ase, mode, check_mc=True):
        self.atoms_ase = atoms_ase
        self.atoms_becke = ase2becke(atoms_ase)
        assert mode in ['onsite', 'offsite']

        N = len(atoms_ase)
        assert N in [1, 2, 3]

        self.iA = 0
        is_onsite = mode == 'onsite'
        self.iB = 0 if is_onsite else 1

        symA = self.get_symbol(self.iA)
        self.elA = self.elements[symA]
        self.xA, self.yA, self.zA = self.get_position(self.iA)

        symB = self.get_symbol(self.iB)
        self.elB = self.elements[symB]
        self.xB, self.yB, self.zB = self.get_position(self.iB)

        # Find out which integrals need to be calculated
        if N == 1:
            assert is_onsite
            pairs = [('%s_%s' % (lmA, lmA), (lmA, lmA))
                     for lmA in ORBITAL_LABELS]
            labels = 'on1c,overlap'
        elif N == 2:
            pairs = [(k, v) for k, v in INTEGRAL_PAIRS_2c.items()]
            labels = 'on2c' if is_onsite else 'off2c,overlap'
        elif N == 3:
            pairs = []
            for integral, (lmA, lmB) in INTEGRAL_PAIRS_3c.items():
                if lmA[0] == 'f' or lmB[0] == 'f':
                    continue
                if lmA in XZ_ANTISYMMETRIC_ORBITALS:
                    is_nonzero = lmB in XZ_ANTISYMMETRIC_ORBITALS
                elif lmA in XZ_SYMMETRIC_ORBITALS:
                    is_nonzero = lmB not in XZ_ANTISYMMETRIC_ORBITALS
                else:
                    raise NotImplementedError(lmA)
                if is_nonzero:
                    pairs.append((integral, (lmA, lmB)))
            labels = 'on3c' if is_onsite else 'off3c'

        # Evaluate the integrals
        for ibasisA, valenceA in enumerate(self.elA.basis_sets):
            for ibasisB, valenceB in enumerate(self.elB.basis_sets):

                results = {}
                for integral, (lmA, lmB) in pairs:
                    for nlA in valenceA:
                        if nlA[1] == lmA[0]:
                            break
                    else:
                        continue

                    for nlB in valenceB:
                        if nlB[1] == lmB[0]:
                            break
                    else:
                        continue

                    self.nlA, self.nlB = nlA, nlB
                    self.lmA, self.lmB = lmA, lmB
                    nlmA = nlA[0] + lmA + nlA[2:]
                    nlmB = nlB[0] + lmB + nlB[2:]

                    print('Running %s_%s %s_%s' % (symA, nlmA, symB, nlmB),
                          flush=True)
                    if is_onsite:
                        out = self.calculate_onsite_integral_approximations()
                    else:
                        out = self.calculate_offsite_integral_approximations()

                    if check_mc:
                        out_mc = self.calculate_multicenter_integral()
                        tol = 1e-8
                    print()

                    if N == 1:
                        if check_mc:
                            assert abs(out_mc['H_mc'] - out['H_1c']) < tol
                        results[integral] = (out['H_1c'], out['S'])
                    elif N == 2:
                        if check_mc:
                            assert abs(out_mc['H_mc'] - out['H_2c']) < tol
                        if is_onsite:
                            results[integral] = out['H_2c'] - out['H_1c']
                        else:
                            results[integral] = (out['H_2c'], out['S'])
                    elif N == 3:
                        if check_mc:
                            assert abs(out_mc['H_mc'] - out['H_3c']) < tol
                        results[integral] = out['H_3c'] - out['H_2c']

                # Print summary
                print('=== {0} [{1}, {2}] ==='.format(labels, ibasisA, ibasisB))
                for integral, values in results.items():
                    print("'%s': " % integral, end='')
                    fmt = lambda x: '%.8f' % x
                    if isinstance(values, tuple):
                        print("(%s)," % ', '.join(map(fmt, values)))
                    else:
                        print(fmt(values) + ",")
                print()

        return

    def run_all_HS(self, atoms_ase, indices_A=None, indices_B=None,
                   print_matrices=True, print_eigenvalues=True):
        self.atoms_ase = atoms_ase
        self.atoms_becke = ase2becke(atoms_ase)

        # Collect atom indices and valence orbital labels
        if indices_A is None:
            indices_A = list(range(len(self.atoms_becke)))
        else:
            assert len(indices_A) == len(set(indices_A)), indices_A

        if indices_B is None:
            indices_B = list(range(len(self.atoms_becke)))
        else:
            assert len(indices_B) == len(set(indices_B)), indices_B

        iAs, nlAs, lmAs = [], [], []
        for index in indices_A:
            nls, lms = self.get_valence_orbitals(index)
            iAs.extend([index] * len(lms))
            nlAs.extend(nls)
            lmAs.extend(lms)

        iBs, nlBs, lmBs = [], [], []
        for index in indices_B:
            nls, lms = self.get_valence_orbitals(index)
            iBs.extend([index] * len(lms))
            nlBs.extend(nls)
            lmBs.extend(lms)

        # Calculate (parts of the) Hamiltonian and overlap matrices
        shape = (len(lmAs), len(lmBs))
        S = np.zeros(shape)
        H_22 = np.zeros(shape)
        H_23 = np.zeros(shape)
        H_32 = np.zeros(shape)
        H_33 = np.zeros(shape)
        H_mc = np.zeros(shape)

        for indexA, (iA, nlA, lmA) in enumerate(zip(iAs, nlAs, lmAs)):
            self.iA, self.nlA, self.lmA = iA, nlA, lmA
            nlmA = nlA[0] + lmA + nlA[2:]
            self.xA, self.yA, self.zA = self.get_position(self.iA)
            symA = self.get_symbol(self.iA)
            self.elA = self.elements[symA]

            for indexB, (iB, nlB, lmB) in enumerate(zip(iBs, nlBs, lmBs)):
                self.iB, self.nlB, self.lmB = iB, nlB, lmB
                nlmB = nlB[0] + lmB + nlB[2:]
                self.xB, self.yB, self.zB = self.get_position(self.iB)
                symB = self.get_symbol(self.iB)
                self.elB = self.elements[symB]

                print('\nRunning iA %d iB %d nlmA %s nlmB %s' % \
                      (self.iA, self.iB, nlmA, nlmB))
                is_onsite = self.iA == self.iB

                if is_onsite:
                    out = self.calculate_onsite_integral_approximations()
                else:
                    out = self.calculate_offsite_integral_approximations()
                out.update(self.calculate_multicenter_integral())
                print(out, flush=True)

                H_mc[indexA, indexB] = out['H_mc']
                H_22[indexA, indexB] = out['H_2c']
                if is_onsite:
                    H_23[indexA, indexB] = out['H_2c']
                    H_32[indexA, indexB] = out['H_3c']
                else:
                    H_23[indexA, indexB] = out['H_3c']
                    H_32[indexA, indexB] = out['H_2c']
                H_33[indexA, indexB] = out['H_3c']
                S[indexA, indexB] = out['S']

        if print_matrices:
            def print_matrix(M, nperline=4):
                print('np.array([')
                for row in M:
                    print('[', end='')
                    i = 0
                    while i < len(row):
                        part = row[i:min(i+nperline, len(row))]
                        items = list(map(lambda x: '%.6e' % x, part))
                        print(', '.join(items), end='')
                        i += nperline
                        if i >= len(row):
                            print('],')
                        else:
                            print(',')
                print('])\n')

            print('=== Overlap matrix S ===')
            print_matrix(S)

            for name, H in zip(['H_22', 'H_23', 'H_32', 'H_33', 'H_mc'],
                               [H_22, H_23, H_32, H_33, H_mc]):
                print('=== Hamiltonian matrix %s ===' % name)
                print_matrix(H, nperline=4)

                if print_eigenvalues:
                    eigE, eigV = linalg.eig(H, b=S)
                    print('bands:')
                    items = list(map(lambda x: '%.6e' % x.real, sorted(eigE)))
                    print('[' + ', '.join(items) + '],')
                    print()

        results = {'S': S, 'H_22': H_22, 'H_23': H_23, 'H_32': H_32,
                   'H_33': H_33, 'H_mc': H_mc}
        return results

    def calculate_onsite_integral_approximations(self):
        assert self.iA == self.iB
        assert self.elA.get_symbol() == self.elB.get_symbol()

        results = {}

        atoms_1c = [self.atoms_becke[self.iA]]
        Sab = becke.overlap(atoms_1c, self.phiA, self.phiB)
        print("<a|b> =", Sab)
        results['S'] = Sab

        if self.lmA == self.lmB:
            set_becke_settings(self.settings_T)
            T = lambda x, y, z: self.phiA(x, y, z) * self.lap_phiB(x, y, z)
            Tab = -0.5 * becke.integral(atoms_1c, T)
            set_becke_settings(self.settings_V)

            V1c = lambda x, y, z: self.phiA(x,y,z) * self.Veff_on1c(x, y, z) \
                                  * self.phiB(x, y, z)
            Vab1c = becke.integral(atoms_1c, V1c)
            nlPP1c = self.get_nonlocal_pseudopotential_term('on1c')
        else:
            Tab = 0.
            Vab1c = 0.
            nlPP1c = 0.
        print("<a|T|b> =", Tab)
        print("<a|V1c|b> =", Vab1c)
        print("<a|nlPP1c|b> =", nlPP1c)
        print("<a|T+V1c+nlPP1c|b> =", Tab+Vab1c+nlPP1c)
        results['H_1c'] = Tab + Vab1c + nlPP1c

        Vcf2c = lambda x, y, z: self.phiA(x, y, z) * self.phiB(x,y,z) \
                                * self.Veff_on2c(x, y, z)
        Vabcf2c = becke.integral(self.atoms_becke, Vcf2c)
        print("<a|Vcf2c|b> =", Vabcf2c)
        nlPP2c = self.get_nonlocal_pseudopotential_term('on2c')
        print("<a|nlPP2c|b> =", nlPP2c)
        print("<a|Vcf2c+nlPP2c|b> =", Vabcf2c+nlPP2c)
        print("<a|T+V1c+Vcf2c+nlPP1c+nlPP2c|b> =",
              Tab+Vab1c+Vabcf2c+nlPP1c+nlPP2c)
        results['H_2c'] = Tab + Vab1c + Vabcf2c + nlPP1c + nlPP2c

        Vcf3c = lambda x, y, z: self.phiA(x,y,z) * self.phiB(x,y,z) \
                                * self.Veff_on3c(x, y, z)
        Vabcf3c = becke.integral(self.atoms_becke, Vcf3c)
        print("<a|Vcf3c|b> =", Vabcf3c)
        print("<a|T+V1c+Vcf2c+Vcf3c+nlPP1c+nlPP2c|b> =",
              Tab+Vab1c+Vabcf2c+Vabcf3c+nlPP1c+nlPP2c)
        results['H_3c'] = Tab + Vab1c + Vabcf2c + Vabcf3c + nlPP1c + nlPP2c
        return results

    def calculate_offsite_integral_approximations(self):
        assert self.iA != self.iB

        results = {}

        atoms_2c = [self.atoms_becke[self.iA], self.atoms_becke[self.iB]]
        Sab = becke.overlap(atoms_2c, self.phiA, self.phiB)
        print("<a|b> =", Sab)
        results['S'] = Sab

        set_becke_settings(self.settings_T)
        T = lambda x, y, z: self.phiA(x, y, z) * self.lap_phiB(x, y, z)
        Tab = -0.5 * becke.integral(atoms_2c, T)
        set_becke_settings(self.settings_V)
        print("<a|T|b> =", Tab)

        V2c = lambda x, y, z: self.phiA(x, y, z) * self.phiB(x, y, z) \
                              * self.Veff_off2c(x, y, z)
        Vab2c = becke.integral(self.atoms_becke, V2c)
        print("<a|V2c|b> =", Vab2c)
        nlPP2c = self.get_nonlocal_pseudopotential_term('off2c')
        print("<a|nlPP2c|b> =", nlPP2c)
        print("<a|T+V2c+nlPP2c|b> =", Tab+Vab2c+nlPP2c)
        results['H_2c'] = Tab + Vab2c + nlPP2c

        V3c = lambda x, y, z: self.phiA(x,y,z) * self.phiB(x, y, z) \
                              * self.Veff_off3c(x, y, z)
        Vab3c = becke.integral(self.atoms_becke, V3c)
        nlPP3c = self.get_nonlocal_pseudopotential_term('off3c')
        print("<a|V3c|b> =", Vab3c)
        print("<a|nlPP3c|b> =", nlPP3c)
        print("<a|V3c+nlPP3c|b> =", Vab3c+nlPP3c)
        print("<a|T+V2c+V3c+nlPP2c+nlPP3c|b> =", Tab+Vab2c+Vab3c+nlPP2c+nlPP3c)
        results['H_3c'] = Tab + Vab2c + Vab3c + nlPP2c + nlPP3c
        return results

    def calculate_multicenter_integral(self):
        results = {}

        if self.iA == self.iB:
            if self.lmA == self.lmB:
                atoms_1c = [self.atoms_becke[self.iA]]
                set_becke_settings(self.settings_T)
                T = lambda x, y, z: self.phiA(x, y, z) * self.lap_phiB(x, y, z)
                Tab = -0.5 * becke.integral(atoms_1c, T)
                set_becke_settings(self.settings_V)
            else:
                Tab = 0.
        else:
            atoms_2c = [self.atoms_becke[self.iA], self.atoms_becke[self.iB]]
            set_becke_settings(self.settings_T)
            T = lambda x, y, z: self.phiA(x, y, z) * self.lap_phiB(x, y, z)
            Tab = -0.5 * becke.integral(atoms_2c, T)
            set_becke_settings(self.settings_V)
        print("<a|T|b> =", Tab)

        nlPPmc = self.get_nonlocal_pseudopotential_term('mc')
        Vmc = lambda x, y, z: self.phiA(x, y, z) * self.phiB(x, y, z) \
                              * self.Veff_mc(x, y, z)
        Vabmc = becke.integral(self.atoms_becke, Vmc)
        print("<a|nlPPmc|b> =", nlPPmc)
        print("<a|Vmc|b> =", Vabmc)
        print("<a|T+Vmc|b> =", Tab+Vabmc)
        print("<a|T+Vmc+nlPPmc|b> =", Tab+Vabmc+nlPPmc)
        results['H_mc'] = Tab+Vabmc+nlPPmc
        return results

    def phiA(self, x, y, z):
        dx, dy, dz = x - self.xA, y - self.yA, z - self.zA
        rA = np.sqrt(dx**2 + dy**2 + dz**2)
        return self.elA.Rnl(rA, self.nlA) \
               * sph_cartesian(dx, dy, dz, rA, self.lmA)

    def phiB(self, x, y, z):
        dx, dy, dz = x - self.xB, y - self.yB, z - self.zB
        rB = np.sqrt(dx**2 + dy**2 + dz**2)
        return self.elB.Rnl(rB, self.nlB) \
               * sph_cartesian(dx, dy, dz, rB, self.lmB)

    def lap_phiA(self, x, y, z):
        dx, dy, dz = x - self.xA, y - self.yA, z - self.zA
        rA = np.sqrt(dx**2 + dy**2 + dz**2)

        Rnl = self.elA.Rnl(rA, self.nlA)
        dRnldr = self.elA.Rnl(rA, self.nlA, der=1)
        d2Rnldr2 = self.elA.Rnl(rA, self.nlA, der=2)

        lap = 0.
        for i, d in enumerate([dx, dy, dz]):
            drdx = d / rA
            d2rdx2 = (rA - d * drdx) / rA**2
            dYlmdx = sph_cartesian_der(dx, dy, dz, rA, self.lmA, der='xyz'[i])
            d2Ylmdx2 = sph_cartesian_der2(dx, dy, dz, rA, self.lmA, der='xyz'[i])
            d2Rnldx2 = d2Rnldr2 * drdx**2 + dRnldr * d2rdx2
            lap += d2Rnldx2 * sph_cartesian(dx, dy, dz, rA, self.lmA) \
                   + 2. * dRnldr * drdx * dYlmdx + Rnl * d2Ylmdx2
        return lap

    def lap_phiB(self, x, y, z):
        dx, dy, dz = x - self.xB, y - self.yB, z - self.zB
        rB = np.sqrt(dx**2 + dy**2 + dz**2)

        Rnl = self.elB.Rnl(rB, self.nlB)
        dRnldr = self.elB.Rnl(rB, self.nlB, der=1)
        d2Rnldr2 = self.elB.Rnl(rB, self.nlB, der=2)

        lap = 0.
        for i, d in enumerate([dx, dy, dz]):
            drdx = d / rB
            d2rdx2 = (rB - d * drdx) / rB**2
            dYlmdx = sph_cartesian_der(dx, dy, dz, rB, self.lmB, der='xyz'[i])
            d2Ylmdx2 = sph_cartesian_der2(dx, dy, dz, rB, self.lmB, der='xyz'[i])
            d2Rnldx2 = d2Rnldr2 * drdx**2 + dRnldr * d2rdx2
            lap += d2Rnldx2 * sph_cartesian(dx, dy, dz, rB, self.lmB) \
                   + 2. * dRnldr * drdx * dYlmdx + Rnl * d2Ylmdx2
        return lap

    def get_rho(self, x, y, z, indices, **kwargs):
        N = np.size(x)
        rho = np.zeros(N)
        for iC in range(len(self.atoms_becke)):
            if iC not in indices: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            rho += self.elements[symC].electron_density(rC, **kwargs)
        return rho

    def get_grad_rho(self, x, y, z, indices, **kwargs):
        N = np.size(x)
        grad_rho = np.zeros((3, N))
        for iC in range(len(self.atoms_becke)):
            if iC not in indices: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            drdx, drdy, drdz = (x - xC)/rC, (y - yC)/rC, (z - zC)/rC
            drhodr = self.elements[symC].electron_density(rC, der=1, **kwargs)
            grad_rho += drhodr * np.array([drdx, drdy, drdz])
        return grad_rho

    def get_grad_rho_grad_sigma(self, x, y, z, indices, **kwargs):
        grad_rho = self.get_grad_rho(x, y, z, indices, **kwargs)
        N = np.size(x)
        grad_sigma = np.zeros((3, N))

        for iC in range(len(self.atoms_becke)):
            if iC not in indices: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)

            drdx, drdy, drdz = (x - xC)/rC, (y - yC)/rC, (z - zC)/rC
            drhodr = self.elements[symC].electron_density(rC, der=1, **kwargs)
            d2rdx2, d2rdy2, d2rdz2 = (1. - drdx**2)/rC, (1. - drdy**2)/rC, \
                                     (1. - drdz**2)/rC
            d2rdxy, d2rdxz, d2rdyz = -drdx*drdy/rC, -drdx*drdz/rC, -drdy*drdz/rC
            d2rhodr2 = self.elements[symC].electron_density(rC, der=2, **kwargs)

            grad_sigma[0, :] += 2. * grad_rho[0, :] \
                                * (d2rhodr2 * drdx * drdx + drhodr * d2rdx2)
            grad_sigma[0, :] += 2. * grad_rho[1, :] \
                                * (d2rhodr2 * drdx * drdy + drhodr * d2rdxy)
            grad_sigma[0, :] += 2. * grad_rho[2, :] \
                                * (d2rhodr2 * drdx * drdz + drhodr * d2rdxz)
            grad_sigma[1, :] += 2. * grad_rho[0, :] \
                                * (d2rhodr2 * drdy * drdx + drhodr * d2rdxy)
            grad_sigma[1, :] += 2. * grad_rho[1, :] \
                                * (d2rhodr2 * drdy * drdy + drhodr * d2rdy2)
            grad_sigma[1, :] += 2. * grad_rho[2, :] \
                                * (d2rhodr2 * drdy * drdz + drhodr * d2rdyz)
            grad_sigma[2, :] += 2. * grad_rho[0, :] \
                                * (d2rhodr2 * drdz * drdx + drhodr * d2rdxz)
            grad_sigma[2, :] += 2. * grad_rho[1, :] \
                                * (d2rhodr2 * drdz * drdy + drhodr * d2rdyz)
            grad_sigma[2, :] += 2. * grad_rho[2, :] \
                                * (d2rhodr2 * drdz * drdz + drhodr * d2rdz2)

        grad_rho_grad_sigma = np.sum(grad_rho * grad_sigma, axis=0)
        return grad_rho_grad_sigma

    def get_lap_rho(self, x, y, z, indices, **kwargs):
        N = np.size(x)
        lap_rho = np.zeros(N)

        for iC in range(len(self.atoms_becke)):
            if iC not in indices: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            drdx, drdy, drdz = (x - xC)/rC, (y - yC)/rC, (z - zC)/rC
            drhodr = self.elements[symC].electron_density(rC, der=1, **kwargs)
            d2rdx2, d2rdy2, d2rdz2 = (1. - drdx**2)/rC, (1. - drdy**2)/rC, \
                                     (1. - drdz**2)/rC
            d2rhodr2 = self.elements[symC].electron_density(rC, der=2, **kwargs)
            lap_rho += d2rhodr2 * drdx**2 + drhodr * d2rdx2
            lap_rho += d2rhodr2 * drdy**2 + drhodr * d2rdy2
            lap_rho += d2rhodr2 * drdz**2 + drhodr * d2rdz2
        return lap_rho

    def get_rho_deriv1(self, x, y, z, indices, **kwargs):
        """ Returns a dict with selected first derivatives of the density. """
        grad_rho = self.get_grad_rho(x, y, z, indices, **kwargs)
        results = {
            'sigma': np.sum(grad_rho**2, axis=0),
        }
        return results

    def get_rho_deriv2(self, x, y, z, indices, **kwargs):
        """ Returns a dict with selected second derivatives of the density. """
        results = {
            'lap_rho': self.get_lap_rho(x, y, z, indices, **kwargs),
            'grad_rho_grad_sigma': self.get_grad_rho_grad_sigma(x, y, z,
                                                                indices,
                                                                **kwargs),
        }
        return results

    def get_exc(self, rho, sigma=None):
        return self.xc.compute_exc(rho, sigma=sigma)

    def get_vxc(self, rho, sigma=None, lap_rho=None, grad_rho_grad_sigma=None):
        out = self.xc.compute_vxc(rho, sigma=sigma, fxc=True)
        vxc = out['vrho']
        if self.xc.add_gradient_corrections:
           vxc -= 2. * out['vsigma'] * lap_rho
           vxc -= 2. * out['v2rhosigma'] * sigma
           vxc -= 2. * out['v2sigma2'] * grad_rho_grad_sigma
        return vxc

    def Veff_on1c(self, x, y, z):
        dx, dy, dz = x - self.xA, y - self.yA, z - self.zA
        rA = np.sqrt(dx**2 + dy**2 + dz**2)
        v = self.elA.neutral_atom_potential(rA)
        rho = self.elA.electron_density(rA)
        drho = {}
        if self.xc.add_gradient_corrections:
            drho.update(self.get_rho_deriv1(x, y, z, [self.iA]))
            drho.update(self.get_rho_deriv2(x, y, z, [self.iA]))
        v += self.get_vxc(rho, **drho)
        return v

    def Veff_off2c(self, x, y, z):
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        rB = np.sqrt((x - self.xB)**2 + (y - self.yB)**2 + (z - self.zB)**2)
        v = self.elA.neutral_atom_potential(rA)
        v += self.elB.neutral_atom_potential(rB)
        rho = self.elA.electron_density(rA) + self.elB.electron_density(rB)
        drho = {}
        if self.xc.add_gradient_corrections:
            drho.update(self.get_rho_deriv1(x, y, z, [self.iA, self.iB]))
            drho.update(self.get_rho_deriv2(x, y, z, [self.iA, self.iB]))
        v += self.get_vxc(rho, **drho)
        return v

    def Veff_on2c(self, x, y, z):
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        rhoA = self.elA.electron_density(rA)
        drho = {}
        if self.xc.add_gradient_corrections:
            drho.update(self.get_rho_deriv1(x, y, z, [self.iA]))
            drho.update(self.get_rho_deriv2(x, y, z, [self.iA]))
        vxcA = self.get_vxc(rhoA, **drho)

        v = 0.
        for iC in range(len(self.atoms_becke)):
            if iC in [self.iA, self.iB]: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            v += self.elements[symC].neutral_atom_potential(rC)
            rhoC = self.elements[symC].electron_density(rC)
            drho = {}
            if self.xc.add_gradient_corrections:
                drho.update(self.get_rho_deriv1(x, y, z, [self.iA, iC]))
                drho.update(self.get_rho_deriv2(x, y, z, [self.iA, iC]))
            v += self.get_vxc(rhoA+rhoC, **drho) - vxcA
        return v

    def Veff_on3c(self, x, y, z):
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        rhoA = self.elA.electron_density(rA)
        drho = {}
        if self.xc.add_gradient_corrections:
            drho.update(self.get_rho_deriv1(x, y, z, [self.iA]))
            drho.update(self.get_rho_deriv2(x, y, z, [self.iA]))
        vxcA = self.get_vxc(rhoA, **drho)

        v = 0.
        for iC in range(len(self.atoms_becke)):
            if iC in [self.iA, self.iB]: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            rhoC = self.elements[symC].electron_density(rC)
            drho = {}
            if self.xc.add_gradient_corrections:
                drho.update(self.get_rho_deriv1(x, y, z, [self.iA, iC]))
                drho.update(self.get_rho_deriv2(x, y, z, [self.iA, iC]))
            vxcAC = self.get_vxc(rhoA+rhoC, **drho)

            for iD in range(len(self.atoms_becke)):
                if iD in [self.iA, self.iB, iC]: continue
                symD = self.get_symbol(iD)
                xD, yD, zD = self.get_position(iD)
                rD = np.sqrt((x - xD)**2 + (y - yD)**2 + (z - zD)**2)
                rhoD = self.elements[symD].electron_density(rD)
                drho = {}
                if self.xc.add_gradient_corrections:
                    drho.update(self.get_rho_deriv1(x, y, z, [self.iA, iD]))
                    drho.update(self.get_rho_deriv2(x, y, z, [self.iA, iD]))
                vxcAD = self.get_vxc(rhoA+rhoD, **drho)
                drho = {}
                if self.xc.add_gradient_corrections:
                    drho.update(self.get_rho_deriv1(x, y, z, [self.iA, iC, iD]))
                    drho.update(self.get_rho_deriv2(x, y, z, [self.iA, iC, iD]))
                v += self.get_vxc(rhoA+rhoC+rhoD, **drho) - vxcAC - vxcAD + vxcA
        v *= 0.5
        return v

    def Veff_off3c(self, x, y, z):
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        rB = np.sqrt((x - self.xB)**2 + (y - self.yB)**2 + (z - self.zB)**2)
        rhoAB = self.elA.electron_density(rA) + self.elB.electron_density(rB)
        drho = {}
        if self.xc.add_gradient_corrections:
            drho.update(self.get_rho_deriv1(x, y, z, [self.iA, self.iB]))
            drho.update(self.get_rho_deriv2(x, y, z, [self.iA, self.iB]))
        vxcAB = self.get_vxc(rhoAB, **drho)

        v = 0.
        for iC in range(len(self.atoms_becke)):
            if iC in [self.iA, self.iB]: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            v += self.elements[symC].neutral_atom_potential(rC)
            rhoABC = rhoAB + self.elements[symC].electron_density(rC)
            drho = {}
            if self.xc.add_gradient_corrections:
                drho.update(self.get_rho_deriv1(x, y, z,
                                                [self.iA, self.iB, iC]))
                drho.update(self.get_rho_deriv2(x, y, z,
                                                [self.iA, self.iB, iC]))
            v += self.get_vxc(rhoABC, **drho) - vxcAB
        return v

    def Veff_mc(self, x, y, z):
        rho = 0.
        v = 0.
        for iC in range(len(self.atoms_becke)):
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            v += self.elements[symC].neutral_atom_potential(rC)
            rho += self.elements[symC].electron_density(rC)
        drho = {}
        if self.xc.add_gradient_corrections:
            all_indices = list(range(len(self.atoms_ase)))
            drho.update(self.get_rho_deriv1(x, y, z, all_indices))
            drho.update(self.get_rho_deriv2(x, y, z, all_indices))
        v += self.get_vxc(rho, **drho)
        return v

    def get_nonlocal_pseudopotential_term(self, expansion):
        assert expansion in ['on1c', 'on2c', 'off2c', 'off3c', 'mc']
        if expansion in ['on1c', 'on2c']:
            assert self.iA == self.iB
        elif expansion in ['off2c', 'off3c']:
            assert self.iA != self.iB

        if self.pp_type == 'PK':
            nlpp = self.get_pseudo_phillips_kleinman(expansion)
        elif self.pp_type == 'KB':
            nlpp = self.get_pseudo_kleinman_bylander(expansion)
        return nlpp

    def get_pseudo_phillips_kleinman(self, expansion):
        if expansion == 'on1c':
            indices = []
        elif expansion == 'on2c':
            indices = []
            for index in range(len(self.atoms_becke)):
                if index != self.iA:
                    indices.append(index)
        elif expansion == 'off2c':
            indices = []
        elif expansion in ['off3c', 'mc']:
            indices = []
            for index in range(len(self.atoms_becke)):
                if index not in [self.iA, self.iB]:
                    indices.append(index)

        nlpp = 0.
        for iC in range(len(self.atoms_becke)):
            if iC not in indices: continue

            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            atoms_AC = [self.atoms_becke[self.iA], self.atoms_becke[iC]]
            atoms_CB = [self.atoms_becke[iC], self.atoms_becke[self.iB]]

            for nC, lC, nlC in self.elements[symC].list_states():
                if nlC in self.elements[symC].valence:
                    continue

                for lmC in ORBITALS[lC]:
                    def phiC(x, y, z):
                        rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                        return self.elements[symC].Rnl(rC, nlC) \
                               * sph_cartesian(x - xC, y - yC, z - zC, rC, lmC)

                    eps = -self.elements[symC].get_eigenvalue(nlC)
                    sAC = becke.overlap(atoms_AC, self.phiA, phiC)
                    sCB = becke.overlap(atoms_CB, phiC, self.phiB)
                    nlpp += sAC * eps * sCB
        return nlpp

    def get_pseudo_kleinman_bylander(self, expansion):
        if expansion == 'on1c':
            indices = [self.iA]
        elif expansion == 'on2c':
            indices = []
            for index in range(len(self.atoms_becke)):
                if index != self.iA:
                    indices.append(index)
        elif expansion == 'off2c':
            indices = [self.iA, self.iB]
        elif expansion == 'off3c':
            indices = []
            for index in range(len(self.atoms_becke)):
                if index not in [self.iA, self.iB]:
                    indices.append(index)
        elif expansion == 'mc':
            indices = list(range(len(self.atoms_becke)))

        nlpp = 0.
        for iC in range(len(self.atoms_becke)):
            if iC not in indices: continue

            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            atoms_AC = [self.atoms_becke[self.iA], self.atoms_becke[iC]]
            atoms_CB = [self.atoms_becke[iC], self.atoms_becke[self.iB]]

            rAC = np.linalg.norm(self.get_position(iC) \
                                 - self.get_position(self.iA))
            rCB = np.linalg.norm(self.get_position(self.iB) \
                                 - self.get_position(iC))

            for nlC in self.elements[symC].pp.energies.keys():
                lC = ANGULAR_MOMENTUM[nlC[1]]

                for lmC in ORBITALS[lC]:
                    def phiC(x, y, z):
                        rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                        return self.elements[symC].pp.projectors[nlC](rC) \
                               * sph_cartesian(x - xC, y - yC, z - zC, rC, lmC)

                    eps = self.elements[symC].pp.energies[nlC]

                    if rAC < 1e-3:
                        if self.lmA == lmC:
                            sAC = self.elements[symC].pp.overlap_onsite[ \
                                                                (nlC, self.nlA)]
                        else:
                            sAC = 0.
                    else:
                        sAC = becke.overlap(atoms_AC, self.phiA, phiC)

                    if rCB < 1e-3:
                        if self.lmB == lmC:
                            sCB = self.elements[symC].pp.overlap_onsite[ \
                                                                (nlC, self.nlB)]
                        else:
                            sCB = 0.
                    else:
                        sCB = becke.overlap(atoms_CB, phiC, self.phiB)

                    nlpp += sAC * eps * sCB
        return nlpp

    def run_repulsion(self, atoms_ase):
        self.atoms_ase = atoms_ase
        self.atoms_becke = ase2becke(atoms_ase)
        print('=== Repulsion ===')

        def print_energies(results, prefix):
            print(prefix, 'energies:')
            for key, val in results.items():
                if isinstance(val, list):
                    print(key, '=', ' '.join(map(lambda x: '%.8f' % x, val)))
                else:
                    print(key, '=', '%.8f' % val)

            Etot = np.sum(results['Enuc']) - 0.5 * np.sum(results['Evhar'])
            Etot += np.sum(results['Exc']) - np.sum(results['Evxc'])
            print('Etotal = %.8f' % Etot)
            print()
            return

        results_mc = self.calculate_multicenter_repulsion()
        print_energies(results_mc, 'Multi-center')

        results_1c = self.calculate_onecenter_repulsion()
        print_energies(results_1c, 'One-center')

        for key, values in results_1c.items():
            results_mc[key] -= np.sum(values)
        print_energies(results_mc, 'Multi-center minus one-center')

        results_2c = self.calculate_twocenter_repulsion()
        print_energies(results_2c, 'Two-center')

        for key, values in results_2c.items():
            results_mc[key] -= np.sum(values)
        print_energies(results_mc, 'Multi-center minus one- & two-center')
        return

    def calculate_multicenter_repulsion(self):
        all_indices = list(range(len(self.atoms_becke)))
        only_val = dict(only_valence=True)

        def get_drho(x, y, z, indices, deriv2=True):
            drho = {}
            if self.xc.add_gradient_corrections:
                drho.update(self.get_rho_deriv1(x, y, z, indices))
                if deriv2:
                    drho.update(self.get_rho_deriv2(x, y, z, indices))
            return drho

        def Vhar_rho_mc(x, y, z):
            rho = self.get_rho(x, y, z, all_indices, **only_val)
            vhar = 0.
            for iC in all_indices:
                symC = self.get_symbol(iC)
                xC, yC, zC = self.get_position(iC)
                rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                vhar += self.elements[symC].hartree_potential(rC, **only_val)
            return vhar * rho

        def Exc_rho_mc(x, y, z):
            rho = self.get_rho(x, y, z, all_indices)
            drho = get_drho(x, y, z, all_indices, deriv2=False)
            exc = self.get_exc(rho, **drho)
            return exc * rho

        def Vxc_rho_mc(x, y, z):
            rho = self.get_rho(x, y, z, all_indices)
            drho = get_drho(x, y, z, all_indices, deriv2=True)
            vxc = self.get_vxc(rho, **drho)
            return vxc * rho

        def Enuc_mc():
            Enuc = 0.
            for iC in all_indices:
                symC = self.get_symbol(iC)
                xC, yC, zC = self.get_position(iC)
                ZC = self.elements[symC].get_number_of_electrons(**only_val)

                for iD in all_indices:
                    if iD == iC: continue
                    symD = self.get_symbol(iD)
                    xD, yD, zD = self.get_position(iD)
                    ZD = self.elements[symD].get_number_of_electrons(**only_val)
                    rCD = np.sqrt((xD - xC)**2 + (yD - yC)**2 + (zD - zC)**2)
                    Enuc += 0.5 * ZC * ZD / rCD
            return Enuc

        results = {
            'Enuc': Enuc_mc(),
            'Evhar': becke.integral(self.atoms_becke, Vhar_rho_mc),
            'Evxc': becke.integral(self.atoms_becke, Vxc_rho_mc),
            'Exc': becke.integral(self.atoms_becke, Exc_rho_mc),
        }
        return results

    def calculate_onecenter_repulsion(self):
        results = {key: [] for key in ['Enuc', 'Evhar', 'Evxc', 'Exc']}
        only_val = dict(only_valence=True)

        def get_drho(x, y, z, indices, deriv2=True):
            drho = {}
            if self.xc.add_gradient_corrections:
                drho.update(self.get_rho_deriv1(x, y, z, indices))
                if deriv2:
                    drho.update(self.get_rho_deriv2(x, y, z, indices))
            return drho

        for iC in range(len(self.atoms_becke)):
            atoms_1c = [self.atoms_becke[iC]]
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)

            def Vhar_rho_1c(x, y, z):
                rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                rho = self.elements[symC].electron_density(rC, **only_val)
                vhar = self.elements[symC].hartree_potential(rC, **only_val)
                return vhar * rho

            def Exc_rho_1c(x, y, z):
                rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                rho = self.elements[symC].electron_density(rC)
                drho = get_drho(x, y, z, [iC], deriv2=False)
                return self.get_exc(rho, **drho) * rho

            def Vxc_rho_1c(x, y, z):
                rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                rho = self.elements[symC].electron_density(rC)
                drho = get_drho(x, y, z, [iC], deriv2=True)
                return self.get_vxc(rho, **drho) * rho

            results['Enuc'].append(0.)
            results['Evhar'].append(becke.integral(atoms_1c, Vhar_rho_1c))
            results['Exc'].append(becke.integral(atoms_1c, Exc_rho_1c))
            results['Evxc'].append(becke.integral(atoms_1c, Vxc_rho_1c))

        return results

    def calculate_twocenter_repulsion(self):
        all_indices = list(range(len(self.atoms_becke)))
        only_val = dict(only_valence=True)

        def get_drho(x, y, z, indices, deriv2=True):
            drho = {}
            if self.xc.add_gradient_corrections:
                drho.update(self.get_rho_deriv1(x, y, z, indices))
                if deriv2:
                    drho.update(self.get_rho_deriv2(x, y, z, indices))
            return drho

        results = {key: [] for key in ['Enuc', 'Evhar', 'Evxc', 'Exc']}

        for iC in all_indices:
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)

            for iD in all_indices:
                if iD == iC: continue
                symD = self.get_symbol(iD)
                xD, yD, zD = self.get_position(iD)
                atoms_2c = [self.atoms_becke[iC], self.atoms_becke[iD]]

                def Vhar_rho_2c(x, y, z):
                    rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                    rhoC = self.elements[symC].electron_density(rC, **only_val)
                    vharC = self.elements[symC].hartree_potential(rC, **only_val)
                    rD = np.sqrt((x - xD)**2 + (y - yD)**2 + (z - zD)**2)
                    rhoD = self.elements[symD].electron_density(rD, **only_val)
                    vharD = self.elements[symD].hartree_potential(rD, **only_val)
                    vhar_rho = (vharC + vharD) * (rhoC + rhoD)
                    vhar_rho -= rhoC * vharC + rhoD * vharD
                    return 0.5 * vhar_rho

                def Exc_rho_2c(x, y, z):
                    rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                    rhoC = self.elements[symC].electron_density(rC)
                    drho = get_drho(x, y, z, [iC], deriv2=False)
                    exc_rho = -self.get_exc(rhoC, **drho) * rhoC

                    rD = np.sqrt((x - xD)**2 + (y - yD)**2 + (z - zD)**2)
                    rhoD = self.elements[symD].electron_density(rD)
                    drho = get_drho(x, y, z, [iD], deriv2=False)
                    exc_rho -= self.get_exc(rhoD, **drho) * rhoD

                    drho = get_drho(x, y, z, [iC, iD], deriv2=False)
                    exc_rho += self.get_exc(rhoC+rhoD, **drho) * (rhoC+rhoD)
                    return 0.5 * exc_rho

                def Vxc_rho_2c(x, y, z):
                    rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
                    rhoC = self.elements[symC].electron_density(rC)
                    drho = get_drho(x, y, z, [iC], deriv2=True)
                    vxc_rho = -self.get_vxc(rhoC, **drho) * rhoC

                    rD = np.sqrt((x - xD)**2 + (y - yD)**2 + (z - zD)**2)
                    rhoD = self.elements[symD].electron_density(rD)
                    drho = get_drho(x, y, z, [iD], deriv2=True)
                    vxc_rho -= self.get_vxc(rhoD, **drho) * rhoD

                    drho = get_drho(x, y, z, [iC, iD], deriv2=True)
                    vxc_rho += self.get_vxc(rhoC+rhoD, **drho) * (rhoC + rhoD)
                    return 0.5 * vxc_rho

                def Enuc_2c():
                    ZC = self.elements[symC].get_number_of_electrons(**only_val)
                    ZD = self.elements[symD].get_number_of_electrons(**only_val)
                    rCD = np.sqrt((xD - xC)**2 + (yD - yC)**2 + (zD - zC)**2)
                    return 0.5 * ZC * ZD / rCD

                results['Enuc'].append(Enuc_2c())
                results['Evhar'].append(becke.integral(atoms_2c, Vhar_rho_2c))
                results['Exc'].append(becke.integral(atoms_2c, Exc_rho_2c))
                results['Evxc'].append(becke.integral(atoms_2c, Vxc_rho_2c))

        return results


class BeckeHarrisKernels(BeckeHarris):
    """
    Abstract calculator for "kernel" matrix elements involving
    second derivatives of the Hartree and XC energies.

    Parameters
    ----------
    See BeckeHarris.__init__()
    """
    def __init__(self, *args, xc='LDA_X+LDA_C_PW', **kwargs):
        BeckeHarris.__init__(self, *args, xc=xc, **kwargs)
        self.xc_polarized = LibXC(xc, spin_polarized=True)

    def get_fxc(self, rho, spin):
        assert not self.xc.add_gradient_corrections, 'Not yet implemented'

        if spin:
            out = self.xc_polarized.compute_vxc_polarized(rho / 2., rho / 2.,
                                                          fxc=True)
            fxc = (out['v2rho2_up'] - out['v2rho2_updown']) / 2.
        else:
            out = self.xc.compute_vxc(rho, fxc=True)
            fxc = out['v2rho2']
        return fxc

    def fxc_on1c(self, x, y, z, spin):
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        rho = self.elA.electron_density(rA)
        fxc = self.get_fxc(rho, spin)
        return fxc

    def fxc_off2c(self, x, y, z, spin):
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        rB = np.sqrt((x - self.xB)**2 + (y - self.yB)**2 + (z - self.zB)**2)
        rho = self.elA.electron_density(rA) + self.elB.electron_density(rB)
        fxc = self.get_fxc(rho, spin)
        return fxc

    def fxc_on2c(self, x, y, z, spin):
        all_indices = list(range(len(self.atoms_becke)))
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        rhoA = self.elA.electron_density(rA)
        fxcA = self.get_fxc(rhoA, spin)

        fxc = 0.
        for iC in all_indices:
            if iC in [self.iA, self.iB]: continue
            symC = self.get_symbol(iC)
            xC, yC, zC = self.get_position(iC)
            rC = np.sqrt((x - xC)**2 + (y - yC)**2 + (z - zC)**2)
            rhoC = self.elements[symC].electron_density(rC)
            fxc += self.get_fxc(rhoA+rhoC, spin) - fxcA
        return fxc

    def fxc_mc(self, x, y, z, spin):
        all_indices = list(range(len(self.atoms_becke)))
        rho = self.get_rho(x, y, z, all_indices, only_valence=True)
        fxc = self.get_fxc(rho, spin)
        return fxc


class BeckeHarrisSubshellKernels(BeckeHarrisKernels):
    """
    Calculator for "kernel" matrix elements involving subshell-dependent
    second derivatives of the Hartree and XC energies.

    Parameters
    ----------
    See BeckeHarris.__init__()
    """
    def get_valence_subshells(self, index):
        sym = self.get_symbol(index)
        nls = []
        for valence in self.elements[sym].basis_sets:
            nls.extend(valence)
        return nls

    def run_all_kernels(self, atoms_ase, indices_A=None, indices_B=None,
                        print_matrices=True, spin=False,
                        substract_pointcharge=False):
        self.atoms_ase = atoms_ase
        self.atoms_becke = ase2becke(atoms_ase)

        # Collect atom indices and subshell labels
        if indices_A is None:
            indices_A = list(range(len(self.atoms_becke)))
        else:
            assert len(indices_A) == len(set(indices_A)), indices_A

        if indices_B is None:
            indices_B = list(range(len(self.atoms_becke)))
        else:
            assert len(indices_B) == len(set(indices_B)), indices_B

        iAs, nlAs = [], []
        for index in indices_A:
            nls = self.get_valence_subshells(index)
            iAs.extend([index] * len(nls))
            nlAs.extend(nls)

        iBs, nlBs = [], []
        for index in indices_B:
            nls = self.get_valence_subshells(index)
            iBs.extend([index] * len(nls))
            nlBs.extend(nls)

        # Calculate kernels
        shape = (len(nlAs), len(nlBs))
        K_10 = np.zeros(shape)
        K_20 = np.zeros(shape)
        K_12 = np.zeros(shape)
        K_22 = np.zeros(shape)
        K_mc = np.zeros(shape)

        for indexA, (iA, nlA) in enumerate(zip(iAs, nlAs)):
            self.iA, self.nlA = iA, nlA
            self.xA, self.yA, self.zA = self.get_position(self.iA)
            symA = self.get_symbol(self.iA)
            self.elA = self.elements[symA]

            vharA = self.get_subshell_vharA()

            for indexB, (iB, nlB) in enumerate(zip(iBs, nlBs)):
                self.iB, self.nlB = iB, nlB
                self.xB, self.yB, self.zB = self.get_position(self.iB)
                symB = self.get_symbol(self.iB)
                self.elB = self.elements[symB]

                print('\nRunning iA %d iB %d nlA %s nlB %s %s' % \
                      (self.iA, self.iB, nlA, nlB, spin))
                is_onsite = self.iA == self.iB

                # Hartree contribution
                if spin:
                    KharAB = 0.
                else:
                    KharAB = self.evaluate_KharAB(vharA)
                    if substract_pointcharge and not is_onsite:
                        R = np.linalg.norm(self.get_position(self.iA) \
                                           - self.get_position(self.iB))
                        KharAB -= 1. / R
                print("<a|fhartree|b> =", KharAB)

                # XC contribution
                if is_onsite:
                    out = self.calculate_onsite_kernel_approximations(spin)
                else:
                    out = self.calculate_offsite_kernel_approximations(spin)
                out.update(self.calculate_multicenter_kernel(spin))
                print(out, flush=True)

                K_mc[indexA, indexB] = out['K_mc'] + KharAB
                if is_onsite:
                    K_10[indexA, indexB] = out['K_1c'] + KharAB
                    K_20[indexA, indexB] = out['K_2c'] + KharAB
                    K_12[indexA, indexB] = out['K_1c'] + KharAB
                else:
                    K_12[indexA, indexB] = out['K_2c'] + KharAB
                K_22[indexA, indexB] = out['K_2c'] + KharAB

        if print_matrices:
            def print_matrix(M, nperline=4):
                print('np.array([')
                for row in M:
                    print('[', end='')
                    i = 0
                    while i < len(row):
                        part = row[i:min(i+nperline, len(row))]
                        items = list(map(lambda x: '%.6e' % x, part))
                        print(', '.join(items), end='')
                        i += nperline
                        if i >= len(row):
                            print('],')
                        else:
                            print(',')
                print('])\n')

            for name, K in zip(['K_10', 'K_20', 'K_12', 'K_22', 'K_mc'],
                               [K_10, K_20, K_12, K_22, K_mc]):
                template = '=== Kernel matrix {0} (spin-polarized = {1}) ==='
                print(template.format(name, spin))
                print_matrix(K, nperline=4)

        results = {'K_10': K_10, 'K_20': K_20, 'K_12': K_12, 'K_22': K_22,
                   'K_mc': K_mc}
        return results

    def calculate_onsite_kernel_approximations(self, spin):
        assert self.iA == self.iB
        assert self.elA.get_symbol() == self.elB.get_symbol()

        results = {}
        atoms_1c = [self.atoms_becke[self.iA]]

        Kxc1c = lambda x, y, z: self.subshellA(x, y, z) \
                                * self.fxc_on1c(x, y, z, spin) \
                                * self.subshellB(x, y, z)
        Kxcab1c = becke.integral(atoms_1c, Kxc1c)
        print("<a|fxc1c|b> =", Kxcab1c)
        results['K_1c'] = Kxcab1c

        Kxc2c = lambda x, y, z: self.subshellA(x, y, z) \
                                * self.fxc_on2c(x, y, z, spin) \
                                * self.subshellB(x, y, z)
        Kxcab2c = becke.integral(self.atoms_becke, Kxc2c)
        print("<a|fxc2c|b> =", Kxcab2c)
        results['K_2c'] = Kxcab1c + Kxcab2c
        return results

    def calculate_offsite_kernel_approximations(self, spin):
        assert self.iA != self.iB

        results = {}

        Kxc2c = lambda x, y, z: self.subshellA(x, y, z) \
                                * self.fxc_off2c(x, y, z, spin) \
                                * self.subshellB(x, y, z)
        Kxcab2c = becke.integral(self.atoms_becke, Kxc2c)
        print("<a|fxc2c|b> =", Kxcab2c)

        results['K_2c'] = Kxcab2c
        return results

    def calculate_multicenter_kernel(self, spin):
        results = {}

        Kxcmc = lambda x, y, z: self.subshellA(x, y, z) \
                                * self.fxc_mc(x, y, z, spin) \
                                * self.subshellB(x, y, z)
        Kxcabmc = becke.integral(self.atoms_becke, Kxcmc)
        print("<a|fxcmc|b> =", Kxcabmc)
        results['K_mc'] = Kxcabmc
        return results

    def subshellA(self, x, y, z):
        rA = np.sqrt((x - self.xA)**2 + (y - self.yA)**2 + (z - self.zA)**2)
        return self.elA.Rnl(rA, self.nlA)**2 / (4. * np.pi)

    def subshellB(self, x, y, z):
        rB = np.sqrt((x - self.xB)**2 + (y - self.yB)**2 + (z - self.zB)**2)
        return self.elB.Rnl(rB, self.nlB)**2 / (4. * np.pi)

    def get_subshell_vharA(self):
        vharA = becke.poisson(self.atoms_becke, self.subshellA)
        return vharA

    def evaluate_KharAB(self, vharA):
        atoms_B = [self.atoms_becke[self.iB]]

        def integrand(x, y, z):
            return vharA(x, y, z) * self.subshellB(x, y, z)

        KharAB = becke.integral(atoms_B, integrand)
        return KharAB
