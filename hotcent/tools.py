''' Tools for tuning the confinement potentials to
fit band structures calculated with e.g. DFT. '''
from __future__ import print_function
import os
import sys
import numpy as np
from scipy.optimize import minimize
from ase.io import read
from ase.units import Bohr
from ase.data import covalent_radii, atomic_numbers
try:
    import matplotlib
    matplotlib.use('agg')
except ImportError:
    print('Warning: could not import matplotlib')
try:
    from hotcent.atom_gpaw import GPAWAE as AE
except ImportError:
    from hotcent.atom_hotcent import HotcentAE as AE
from hotcent.slako import SlaterKosterTable
from hotcent.confinement import PowerConfinement 

class Element:
    def __init__(self, symbol, configuration=None, valence=[], eigenvalues_spd=[],
                 spe=0., hubbardvalues_spd=[], occupations_spd=[]):
        self.symbol = symbol
        self.configuration = configuration
        self.valence = valence
        self.eigenvalues_spd = eigenvalues_spd
        self.spe = spe  # spin polarization error
        self.hubbardvalues_spd = hubbardvalues_spd
        self.occupations_spd = occupations_spd


class BandStructure:
    # make this inherit from ase.dft.band_structure.BandStructure?
    def __init__(self, atoms=None, eigenvalues=None, kpts_path={}, kpts=None,
                 nsemicore=0, weight=1., kBT=1.5, nspin=1):
        if isinstance(atoms, str):
            self.atoms = read(atoms)
        else:
            self.atoms = atoms

        if isinstance(eigenvalues, str):
            if eigenvalues.endswith('.npy'):
                self.eigenvalues = np.load(eigenvalues)
            else:
                self.eigenvalues = np.loadtxt(eigenvalues)
        else:
            self.eigenvalues = eigenvalues

        self.kpts_path = kpts_path
        self.kpts = kpts
        self.nsemicore = nsemicore
        self.weight = weight
        self.kBT = kBT
        self.nspin = nspin


class SlaterKosterGenerator:
    def __init__(self, elements, bandstructures, DftbPlusCalc=None, 
                 xc='PBE', superposition='density', rmin=0.4, dr=0.02, 
                 N=900, verbose=True):
        self.elements = elements
        self.bandstructures = bandstructures
        self.DftbPlusCalc = DftbPlusCalc
        self.xc = xc
        self.superposition = superposition
        self.rmin = rmin
        self.dr = dr
        self.N = N
        self.verbose = verbose

    def run(self, initial_guess={}, rhobeg=0.2, tol=1e-2, maxiter=1000):
        # maybe add "autofill" option here later
        if not initial_guess:
            initial_guess = self.make_initial_guess()
        self.keys = sorted(initial_guess)
        rconf0 = [initial_guess[k] for k in self.keys]
        result = minimize(self._residual, rconf0, method='COBYLA', tol=tol, 
                          options={'rhobeg':rhobeg, 'maxiter':maxiter})
        return {self.keys[i]:result.x[i] for i in range(len(self.keys))}

    def make_initial_guess(self, factor=1.85):
        initial_guess = {}
        for el in self.elements:
            rcov = covalent_radii[atomic_numbers[el.symbol]]
            for nl in el.valence + ['n']:
                key = '%s_%s' % (el.symbol, nl)
                initial_guess[key] = factor * rcov / Bohr
        return initial_guess

    def _residual(self, rconf):
        if self.verbose:
            print('RCONF:', rconf)
            sys.stdout.flush()

        try:
            self.generate_skf(rconf)
        except (ValueError, AssertionError, RuntimeError) as err:
            if self.verbose:
                print(err.message)
                sys.stdout.flush()
            return 1e23

        residual = 0.
        for bs_dft in self.bandstructures:
            bs_dftb = self.calculate_bandstructure(atoms=bs_dft.atoms,
                                                   kpts_path=bs_dft.kpts_path,
                                                   kpts=bs_dft.kpts)
            shape_dft = np.shape(bs_dft.eigenvalues)
            shape_dftb = np.shape(bs_dftb.eigenvalues)

            assert shape_dft[0] == shape_dftb[0], [shape_dft, shape_dftb]
            assert shape_dft[1] == shape_dftb[1], [shape_dft, shape_dftb]
            
            nskip = bs_dft.nsemicore / 2
            imin = min(shape_dft[2] - nskip, shape_dftb[2])

            diffs = bs_dft.eigenvalues[:, :, nskip:nskip + imin] - \
                    bs_dftb.eigenvalues[:, :, :imin]
            logw = -1. * np.abs(bs_dft.eigenvalues[:, :, nskip:nskip + imin])
            logw /= bs_dft.kBT
            residual += ((bs_dft.weight * np.exp(logw) * diffs) ** 2).sum()

        if self.verbose:
            print('RESIDUAL:', residual)
            sys.stdout.flush()

        return residual

    def calculate_bandstructure(self, atoms=None, kpts_path={}, kpts=None, 
                                plot=False, filename='bandstructure.png',
                                emax=10., emin=-10.):
        calc = self.DftbPlusCalc(atoms=atoms, kpts=kpts)
        atoms.set_calculator(calc)
        etot = atoms.get_potential_energy()
        efermi = calc.get_fermi_level()

        calc = self.DftbPlusCalc(atoms=atoms, kpts=kpts_path,
                                 Hamiltonian_MaxSCCIterations=1, 
                                 Hamiltonian_ReadInitialCharges='Yes',
                                 Hamiltonian_SCCTolerance='1e3')
        atoms.set_calculator(calc)
        etot = atoms.get_potential_energy()

        calc.results['fermi_levels'] = np.array([efermi])

        bs = calc.band_structure()
        if plot:
            bs.plot(filename=filename, show=False, emax=emax, emin=emin)

        eigenvalues = bs.energies.copy() - efermi
        return BandStructure(atoms=atoms, kpts=kpts, kpts_path=kpts_path,
                             eigenvalues=eigenvalues)

    def generate_skf(self, rconf):
        atoms = []

        for i, el in enumerate(self.elements):
            key = el.symbol + '_n'
            if isinstance(rconf, dict):
                 r0 = rconf[key]
            else:
                 index = self.keys.index(key)
                 r0 = rconf[index] 
            conf = PowerConfinement(r0=r0, s=2)

            wf_conf = {}
            for nl in el.valence:
                key = el.symbol + '_%s' % nl
                if isinstance(rconf, dict):
                    r0 = rconf[key]
                else:
                    index = self.keys.index(key)
                    r0 = rconf[index]
                wf_conf[nl] = PowerConfinement(r0=r0, s=2)

            atom = AE(el.symbol,
                      confinement=conf,
                      wf_confinement=wf_conf,
                      xcname=self.xc,
                      configuration=el.configuration,
                      valence=el.valence,
                      scalarrel=True,
                      timing=False,
                      nodegpts=150,
                      mix=0.2,
                      txt='hotcent.out',
                      )
            atom.run()
            atoms.append(atom)

        for i, el1 in enumerate(self.elements):
            for j, el2 in enumerate(self.elements):
                rmax = self.rmin + (self.N - 1) * self.dr
                sk = SlaterKosterTable(atoms[i], atoms[j], timing=False,
                                       txt='hotcent.out')
                sk.run(self.rmin, rmax, self.N, xc=self.xc,
                       superposition=self.superposition)

                filename = '%s-%s.skf' % (el1.symbol, el2.symbol)
                sk.write(filename)

                if i == j:
                    values = el1.eigenvalues_spd[::-1]
                    values += [el1.spe]
                    values += el1.hubbardvalues_spd[::-1]
                    values += el1.occupations_spd[::-1]
                    os.system('sed -i "2s/.*/%s/" %s' % \
                              (' '.join(map(str, values)), filename))
        return
