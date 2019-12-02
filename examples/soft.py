import sys
from ase.units import Ha
from hotcent.confinement import SoftConfinement

code = sys.argv[1].lower()

if 'hotcent' in code:
    from hotcent.atom_hotcent import HotcentAE as AE
elif 'gpaw' in code:
    from hotcent.atom_gpaw import GPAWAE as AE

atom = AE('Si',
          xcname='LDA',
          confinement=None,
          wf_confinement=None,
          configuration='[Ne] 3s2 3p2',
          valence=['3s', '3p'],
          scalarrel=False,
          timing=False,
          txt='-',
          )
atom.run()
eps_free = {nl: atom.get_eigenvalue(nl) for nl in atom.valence}

atom = AE('Si',
          xcname='LDA',
          confinement=None,
          wf_confinement={'3s':SoftConfinement(amp=12., rc=6.74, x_ri=0.6),
                          '3p':SoftConfinement(amp=12., rc=8.70, x_ri=0.6)},
          configuration='[Ne] 3s2 3p2',
          valence=['3s', '3p'],
          scalarrel=False,
          timing=False,
          txt='-',
          )
atom.run(wf_confinement_scheme='perturbative')
eps_conf = {nl: atom.get_eigenvalue(nl) for nl in atom.valence}

print('\nChecking eigenvalue shifts:')
diff_ref = {'3s': 0.104 / Ha, '3p': 0.103 / Ha}  # from GPAW's gpaw-basis tool
for nl in atom.valence:
    diff = eps_conf[nl] - eps_free[nl]
    ok = abs(diff - diff_ref[nl]) < 1e-4
    items = (nl, diff_ref[nl], diff, 'OK' if ok else 'FAIL')
    print('  %s  %.6f  %.6f  | %s' % items)
