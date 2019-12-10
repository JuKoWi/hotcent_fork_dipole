''' Tools for tuning the confinement potentials to
fit band structures calculated with e.g. DFT. '''
import os
import copy
import numpy as np
from scipy.optimize import minimize
from ase.io import read
from ase.units import Bohr
from ase.data import covalent_radii, atomic_numbers
from ase.dft.band_structure import BandStructure as BS
from ase.calculators.calculator import kpts2ndarray
try:
    import matplotlib
    matplotlib.use('agg')
except ImportError:
    print('Warning: could not import matplotlib')
from hotcent.atomic_dft import AtomicDFT
from hotcent.slako import SlaterKosterTable
from hotcent.confinement import (PowerConfinement, WoodsSaxonConfinement,
                                 SoftConfinement)


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


class BandStructure(BS):
    def __init__(self, atoms=None, energies=None, kpts={}, kpts_scf=None,
                 nsemicore=0, weight=1., kBT=1.5, nspin=1, xcoord=None,
                 xticks=None, xticklabels=None):
        # Eigenenergies assumed to be already
        # appropriately referenced
        if isinstance(atoms, str):
            self.atoms = read(atoms)
        else:
            self.atoms = atoms

        if isinstance(energies, str):
            if energies.endswith('.npy'):
                energies = np.load(energies)
            else:
                energies = np.loadtxt(energies)
        else:
            energies = energies

        path = self.atoms.cell.bandpath(**kpts)
        BS.__init__(self, path, energies, reference=0.0)

        self.kpts = kpts
        self.kpts_scf = kpts_scf
        self.nsemicore = nsemicore
        self.weight = weight
        self.kBT = kBT
        self.nspin = nspin
        self.xcoord = xcoord
        self.xticks = xticks
        self.xticklabels = xticklabels

    def get_labels(self):
        try:
            return BS.get_labels(self)
        except ValueError:
            # ASE doesn't recognize the crystal structure,
            # in which case you should have passed on the
            # appropriate coordinates, ticks, and labels
            assert self.xcoord is not None
            assert self.xticks is not None
            assert self.xticklabels is not None
            return self.xcoord, self.xticks, self.xticklabels


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
        self.N = N  # eiter int or dict
        self.verbose = verbose

    def parse_confinement_dict(self, conf):
        self.var_param = []
        self.fix_param = []
        for targets in sorted(conf):
            for param in sorted(conf[targets]):
                label = targets + '.' + param
                item = (label, conf[targets][param])
                if label.endswith('_guess'):
                    self.var_param.append(item)
                else:
                    self.fix_param.append(item)

    def get_vpar_dict(self, opt_param, keep_suffix=True):
        vpar = {}
        assert len(opt_param) == len(self.var_param)
        for i, (label, value) in enumerate(self.var_param + self.fix_param):
            targets = label.split('.')[0].split(',')
            par = label.split('.')[1]
            if not keep_suffix:
                par = par.split('_')[0]
            val = opt_param[i] if i < len(opt_param) else value
            for x in targets:
                if x in vpar:
                    vpar[x][par] = val
                else:
                    vpar[x] = {par: val}
        return vpar

    def get_vconf_dict(self, vpar):
        vconf = {}
        for targets, pardict in vpar.items():
            kwargs = {k.split('_guess')[0]:v for k, v in pardict.items()}
            is_pow = np.all([kw in ['s', 'r0'] for kw in kwargs])
            is_sax = np.all([kw in ['w', 'a', 'r0'] for kw in kwargs])
            is_soft = np.all([kw in ['amp', 'rc', 'x_ri'] for kw in kwargs])
            if is_pow and not is_sax and not is_soft:
                conf = PowerConfinement(**kwargs)
            elif is_sax and not is_pow and not is_soft:
                conf = WoodsSaxonConfinement(**kwargs)
            elif is_soft and not is_pow and not is_sax:
                conf = SoftConfinement(**kwargs)
            else:
                msg = 'Bad Vconf keywords: ' + ' '.join(sorted(kwargs))
                raise ValueError(msg)

            for key in targets.split(','):
                assert key not in vconf, 'Multiple definition of %s conf' % key
                vconf[key] = conf
        return vconf

    def run(self, initial_guess={}, rhobeg=0.2, tol=1e-2, maxiter=1000,
            callback=None):
        if not initial_guess:
            initial_guess = self.make_initial_guess()
        self.parse_confinement_dict(initial_guess)

        opt_param = [val for label, val in self.var_param]
        if self.verbose:
            print('PARAM labels:')
            print(' '.join([label.split('_guess')[0]
                            for label, val in self.var_param]), flush=True)

        result = minimize(self._residual, opt_param, args=(callback,),
                          method='COBYLA', tol=tol, options={'rhobeg':rhobeg,
                          'maxiter':maxiter})

        return self.get_vpar_dict(result.x, keep_suffix=True)

    def make_initial_guess(self, factor=1.85, orbital_dependent=True):
        initial_guess = {}
        for el in self.elements:
            rcov = covalent_radii[atomic_numbers[el.symbol]]
            r0 = factor * rcov / Bohr
            if orbital_dependent:
                for nl in el.valence + ['n']:
                    key = '%s_%s' % (el.symbol, nl)
                    initial_guess[key] = {'s_guess':2, 'r0_guess':r0}
            else:
                key = ','.join([el.symbol + '_' + nl for nl in el.valence])
                initial_guess[key] = {'s_guess':2, 'r0_guess':r0}
                initial_guess[el.symbol + '_n'] = {'s_guess':2, 'r0_guess':r0}
        return initial_guess

    def _residual(self, opt_param, callback):
        """
        Returns the total residual for the generated band structures
        compared to the reference band structures.

        args:

        opt_param: the (internally defined) list of parameters used
                   in setting up the confinement potentials (and hence
                   Slater-Koster tables).
                   If the relevant set of SKF files has already been
                   created, setting opt_param to None will result in
                   just recalculating the band structures and the total
                   residual, without re-generating the SK tables.
                   In this way, one can then also quickly optimize the
                   eigenvalues and Hubbard values.
        """
        if self.verbose:
            print('PARAM:', opt_param, flush=True)

        if opt_param is not None:
            try:
                self.generate_skf(opt_param)
            except (ValueError, AssertionError, RuntimeError, IndexError,
                    TypeError) as err:
                if self.verbose:
                    print(err.message, flush=True)
                return 1e23

        residual = 0.
        for bs_dft in self.bandstructures:
            bs_dftb = self.calculate_bandstructure(bs_dft)
            shape_dft = np.shape(bs_dft.energies)
            shape_dftb = np.shape(bs_dftb.energies)

            assert shape_dft[0] == shape_dftb[0], [shape_dft, shape_dftb]
            assert shape_dft[1] == shape_dftb[1], [shape_dft, shape_dftb]
            
            nskip = bs_dft.nsemicore // 2
            imin = min(shape_dft[2] - nskip, shape_dftb[2])

            diffs = bs_dft.energies[:, :, nskip:nskip + imin] - \
                    bs_dftb.energies[:, :, :imin]
            logw = -1. * np.abs(bs_dft.energies[:, :, nskip:nskip + imin])
            logw /= bs_dft.kBT
            residual += ((bs_dft.weight * np.exp(logw) * diffs) ** 2).sum()

        if callback is not None:
            residual = callback()

        if self.verbose:
            print('RESIDUAL:', residual, flush=True)

        return residual

    def calculate_bandstructure(self, bs, ref='vbm'):
        atoms = bs.atoms.copy()
        calc = self.DftbPlusCalc(atoms=atoms, kpts=bs.kpts_scf)
        atoms.set_calculator(calc)
        etot = atoms.get_potential_energy()

        efermi = calc.get_fermi_level()
        if ref.lower() == 'vbm':
            eig = np.array(calc.results['eigenvalues'])
            eref = np.max(eig[eig < efermi])
        elif ref.lower() == 'fermi':
            eref = efermi
        else:
            msg = 'Reference "%s" should be either "vbm" or "fermi"' % ref
            raise ValueError(msg)

        calc = self.DftbPlusCalc(atoms=atoms, kpts=bs.kpts,
                                 Hamiltonian_MaxSCCIterations=1,
                                 Hamiltonian_ReadInitialCharges='Yes',
                                 Hamiltonian_SCCTolerance='1e3')
        atoms.set_calculator(calc)
        etot = atoms.get_potential_energy()

        bs_new = copy.deepcopy(bs)
        bs_new.kpts = calc.get_ibz_k_points()
        bs_new.energies = []
        for s in range(calc.get_number_of_spins()):
            bs_new.energies.append([calc.get_eigenvalues(kpt=k, spin=s)
                                    for k in range(len(bs_new.kpts))])
        bs_new.energies = np.array(bs_new.energies)
        bs_new.energies -= eref
        return bs_new

    def generate_skf(self, arg):
        if isinstance(arg, dict):
            vpar = arg
        else:
            vpar = self.get_vpar_dict(arg, keep_suffix=False)

        if self.verbose:
            print('VPAR:')
            for key in sorted(vpar):
                print('\'' + key + '\':' + str(vpar[key]))
        vconf = self.get_vconf_dict(vpar)

        atoms = []
        keys = sorted(vconf)

        for i, el in enumerate(self.elements):
            conf = vconf['%s_n' % el.symbol]
            wf_conf = {nl: vconf['%s_%s' % (el.symbol, nl)]
                       for nl in el.valence}
            atom = AtomicDFT(el.symbol,
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

                if isinstance(self.N, dict):
                    keys = [el1.symbol + '-' + el2.symbol,
                            el2.symbol + '-' + el1.symbol,
                            'default']
                    for key in keys:
                        if key in self.N:
                            N = self.N[key]
                            break
                else:
                    N = self.N

                rmax = self.rmin + (N - 1) * self.dr
                sk = SlaterKosterTable(atoms[i], atoms[j], timing=False,
                                       txt='hotcent.out')
                sk.run(self.rmin, rmax, N, xc=self.xc,
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
