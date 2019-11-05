import os
import sys
from hotcent.slako import SlaterKosterTable
from hotcent.confinement import PowerConfinement
from ase.data import covalent_radii, atomic_numbers
from ase.units import Bohr, Hartree

code = sys.argv[1].lower()

if code == 'hotcent':
    from hotcent.atom_hotcent import HotcentAE as AE
elif code == 'gpaw':
    from hotcent.atom_gpaw import GPAWAE as AE

element = 'C'

# Get KS all-electron ground state of confined atom:
r0 = 1.85 * covalent_radii[atomic_numbers[element]] / Bohr
atom = AE(element,
          confinement=PowerConfinement(r0=r0, s=2),
          wf_confinement=PowerConfinement(r0=r0, s=2),
          configuration='[He] 2s2 2p2',
          valence=['2s', '2p'],
          timing=True,
          )
atom.run()
atom.plot_Rnl(only_valence=False)
atom.plot_density()

# Compute Slater-Koster integrals:
rmin, dr, N = 0.5, 0.05, 250
rmax = rmin + (N - 1) * dr
sk = SlaterKosterTable(atom, atom, timing=True)
sk.run(rmin, rmax, N, superposition='potential')
sk.write('%s-%s_no_repulsion.par' % (element, element))
sk.write('%s-%s_no_repulsion.skf' % (element, element))
sk.plot()
