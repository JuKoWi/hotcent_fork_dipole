from ase.units import Ha
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT

valence = ['3s', '3p']

wf_confinement = {'3s': SoftConfinement(rc=5.550736),
                  '3p': SoftConfinement(rc=7.046078)}

pp = KleinmanBylanderPP('../tests/pseudos/S.psf', valence)
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
is_ok = abs(e_diff) < e_tol
items = (e, e_ref, e_diff, 'OK' if is_ok else 'FAIL')
print('E_tot [Ha] = %.6f   ref = %.6f    diff = %.6f\t| %s' % items)
assert is_ok

# Compare the individual eigenvalues with those from Siesta v4.1.5
e_ref = {'3s': -0.163276829e+02 / Ha,
         '3p': -0.620202160e+01 / Ha}
e_tol = 1e-4
for nl in valence:
    e = atom.get_onecenter_integral(nl)
    e_diff = e - e_ref[nl]
    is_ok = abs(e_diff) < e_tol
    items = (nl, e, e_ref[nl], e_diff, 'OK' if is_ok else 'FAIL')
    print('E_%s  [Ha] = %.6f    ref = %.6f     diff = %.6f\t| %s' % items)
    assert is_ok

# Compare the 3p Hubbard parameter with one obtained with Siesta v4.1.5
U = atom.get_hubbard_value('3p', scheme='central', maxstep=1.)
U_ref = (2*273.865791 - 261.846966 - 274.229224) / Ha
U_tol = 1e-3
U_diff = U - U_ref
is_ok = abs(U_diff) < U_tol
items = (U, U_ref, U_diff, 'OK' if is_ok else 'FAIL')
print('U_3p  [Ha] = %.6f   ref = %.6f    diff = %.6f\t| %s' % items)
assert is_ok
