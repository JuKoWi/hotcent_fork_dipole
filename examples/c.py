import os
from hotcent.atom import KSAllElectron
from hotcent.slako import SlaterKosterTable
from hotcent.confinement import PowerConfinement #, WoodsSaxonConfinement
from ase.data import covalent_radii, atomic_numbers

# Get KS all-electron ground state of confined atom:
elmfile = 'C.elm'
if os.path.exists(elmfile):
    atom.read(elmfile)
else:
    r0 = 1.85 * covalent_radii[atomic_numbers['C']]
    atom = KSAllElectron('C',
                         confinement=PowerConfinement(r0=2.0, s=2),
                         configuration={'2s':2, '2p':2},
                         valence=['2s', '2p'],
                         )
    atom.run()
    #atom.write(elmfile)

exit()
# Compute Slater-Koster integrals:
rmin, dr, N = 0.5, 0.05, 300
rmax = rmin + (N - 1) * dr
sk = SlaterKosterTable(atom, atom)
sk.run(rmin=rmin, rmax=rmax, N=N)
sk.write('Au-Au_no_repulsion.par')
sk.write('Au-Au_no_repulsion.skf')
sk.plot()
