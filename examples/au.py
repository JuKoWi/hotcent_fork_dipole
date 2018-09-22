""" This example aims to reproduce the Au-Au 
Slater-Koster table generation procedure by
Fihey and coworkers (doi:10.1002/jcc.24046). """
import os
from hotcent.atom_gpaw import GPAWAE
from hotcent.slako import SlaterKosterTable
from hotcent.confinement import PowerConfinement 

element = 'Au'

# Get KS all-electron ground state of confined atom:
elmfile = '%s.elm' % element

if os.path.exists(elmfile):
    atom.read(elmfile)
else:
    conf = PowerConfinement(r0=9.41, s=2)
    wf_conf = {'5d': PowerConfinement(r0=6.50, s=2),
               '6s': PowerConfinement(r0=6.50, s=2),
               '6p': PowerConfinement(r0=4.51, s=2),
               }
    atom = GPAWAE(element,
                  xcname='PBE',
                  confinement=conf,
                  wf_confinement=wf_conf,
                  configuration='[Xe] 4f14 5d10 6s1 6p0',
                  valence=['5d', '6s', '6p'],
                  scalarrel=True,
                  timing=True,
                  nodegpts=150,
                  mix=0.2,
                  txt='-',
                  )
    atom.run()
    atom.plot_density()
    #atom.write(elmfile)

for nl in ['5d', '6s', '6p']:
    print nl, 'eigenvalue: %.5f' % atom.get_epsilon(nl)

# Compute Slater-Koster integrals:
rmin, dr, N = 0.4, 0.02, 900  #0.5, 0.05, 380
rmax = rmin + (N - 1) * dr
sk = SlaterKosterTable(atom, atom, timing=True)
sk.run(rmin, rmax, N, superposition='density')
sk.write('Au-Au_no_repulsion.par')
sk.write('Au-Au_no_repulsion.skf')
sk.plot()
