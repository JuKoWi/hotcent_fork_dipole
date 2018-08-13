import os
from hotcent.atom import KSAllElectron
from hotcent.slako import SlaterKosterTable
from hotcent.confinement import PowerConfinement #, WoodSaxonConfinement

# Get KS all-electron ground state of confined atom:
elmfile = 'Au.elm'
if os.path.exists(elmfile):
    atom.read(elmfile)
else:
    confinements = {'s':PowerConfinement(),
                    'p':PowerConfinement(),
                    'd':PowerConfinement(),
                    'n':PowerConfinement(),
                    }
    atom = KSAllElectron('Au',
                         confinements=confinements,
                         configuration={'5s':2,'2p':2},
                         valence=['2s','2p'],
    atom.run()
    atom.write(elmfile)

# Compute Slater-Koster integrals:
rmin, dr, N = 0.5, 0.05, 300
rmax = rmin + (N - 1) * dr
sk = SlaterKosterTable(atom, atom)
sk.run(rmin=rmin, rmax=rmax, N=N)
sk.write('Au-Au_no_repulsion.par')
sk.write('Au-Au_no_repulsion.skf')
sk.plot()
