from ase.units import Ha
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT

valence = ['3s', '3p']

wf_confinement = {'3s': SoftConfinement(rc=5.550736),
                  '3p': SoftConfinement(rc=7.046078)}

pp = KleinmanBylanderPP('S.pbe.psf', valence)
pp.plot_Vl(filename='S_Vl.png')
pp.plot_valence_density(filename='S_dens.png')

atom = PseudoAtomicDFT('S', pp,
                       xc='GGA_X_PBE+GGA_C_PBE',
                       valence=valence,
                       configuration='[Ne] 3s2 3p4',
                       wf_confinement=wf_confinement,
                       perturbative_confinement=True,
                       scalarrel=False,
                       timing=False,
                       )
atom.run()

# Compare the total energy with the one from Siesta v4.1.5
e = atom.get_energy()
e_ref = -273.865791 / Ha
e_tol = 1e-3
e_diff = e - e_ref
items = (e, e_ref, e_diff, 'OK' if abs(e_diff) < e_tol else 'FAIL')
print('E_tot [Ha] = %.6f   ref = %.6f    diff = %.6f\t| %s' % items)

# Compare the individual eigenvalues with those from Siesta v4.1.5
e_ref = {'3s': -0.163276829e+02 / Ha,
         '3p': -0.620202160e+01 / Ha}
e_tol = 1e-4
for nl in valence:
    e = atom.get_onecenter_integral(nl)
    e_diff = e - e_ref[nl]
    items = (nl, e, e_ref[nl], e_diff, 'OK' if abs(e_diff) < e_tol else 'FAIL')
    print('E_%s  [Ha] = %.6f    ref = %.6f     diff = %.6f\t| %s' % items)
