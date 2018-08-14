import os
from hotcent.atom import KSAllElectron
from hotcent.slako import SlaterKosterTable
from hotcent.confinement import PowerConfinement
from ase.data import covalent_radii, atomic_numbers
from ase.units import Bohr, Hartree

element = 'C'

# Get KS all-electron ground state of confined atom:
elmfile = '%s.elm' % element

if os.path.exists(elmfile):
    atom.read(elmfile)
else:
    r0 = 1.85 * covalent_radii[atomic_numbers[element]] / Bohr
    atom = KSAllElectron(element,
                         confinement=PowerConfinement(r0=r0, s=2),
                         configuration='[He] 2s2 2p2',
                         valence=['2s', '2p'],
                         )
    atom.run()
    #atom.write(elmfile)

# Compute Slater-Koster integrals:
rmin, dr, N = 0.5, 0.05, 250
rmax = rmin + (N - 1) * dr
sk = SlaterKosterTable(atom, atom)
sk.run(rmin, rmax, N)
sk.write('%s-%s_no_repulsion.par' % (element, element))
sk.write('%s-%s_no_repulsion.skf' % (element, element))
sk.plot()
