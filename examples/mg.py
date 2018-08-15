import os
import sys
from hotcent.atom import KSAllElectron
from hotcent.slako import SlaterKosterTable
from hotcent.confinement import PowerConfinement 

element = 'Mg'

# Get KS all-electron ground state of confined atom:
elmfile = '%s.elm' % element

if os.path.exists(elmfile):
    atom.read(elmfile)
else:
    conf = PowerConfinement(r0=14.0, s=2)
    wf_conf = {'3s': PowerConfinement(r0=5.5, s=2),
               '3p': PowerConfinement(r0=5.5, s=2),
              }
    atom = KSAllElectron(element,
                         confinement=conf,
                         wf_confinement=wf_conf,
                         configuration='[Ne] 3s2 3p0',
                         valence=['3s', '3p'],
                         )
    atom.run()
    #atom.write(elmfile)

for nl in ['3s', '3p']:
    print nl, 'eigenvalue: %.5f' % atom.get_epsilon(nl)

sys.stdout.flush()

# Compute Slater-Koster integrals:
rmin, dr, N = 0.5, 0.05, 300
rmax = rmin + (N - 1) * dr
sk = SlaterKosterTable(atom, atom)
sk.run(rmin, rmax, N)
sk.write('%s-%s_no_repulsion.par' % (element, element))
sk.write('%s-%s_no_repulsion.skf' % (element, element))
sk.plot()
