import sys
from hotcent.confinement import SoftConfinement

code = sys.argv[1].lower()

if 'hotcent' in code:
    from hotcent.atom_hotcent import HotcentAE as AE
elif 'gpaw' in code:
    from hotcent.atom_gpaw import GPAWAE as AE

assert 'gpaw' in code, 'New wf_confinement_scheme not yet in HotcentAE'
 
atom = AE('Si',
          xcname='LDA',
          confinement=None,
          wf_confinement={'3s':SoftConfinement(amp=12., rc=6.85, x_ri=0.6),
                          '3p':SoftConfinement(amp=12., rc=8.70, x_ri=0.6)},
          configuration='[Ne] 3s2 3p2',
          valence=['3s', '3p'],
          scalarrel=False,
          timing=False,
          txt='-',
          )
atom.run(wf_confinement_scheme='new')
