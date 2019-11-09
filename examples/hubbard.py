from ase.units import Ha
from hotcent.atom_hotcent import HotcentAE
from hotcent.confinement import PowerConfinement

energies = {}
for occ2p, label in zip([2, 1, 3], ['neutral', 'cation', 'anion']):
    atom = HotcentAE('Si',
                     confinement=PowerConfinement(r0=40., s=4),
                     configuration='[Ne] 3s2 3p%d' % occ2p,
                     valence=['3s', '3p'],
                     scalarrel=False,
                     )
    atom.run()
    energies[label] = atom.total_energy

EA = energies['neutral'] - energies['anion']
IE = energies['cation'] - energies['neutral']
U = IE - EA

labels = ['EA', 'IE', 'U']
values = [EA, IE, U]
references = [0.044065, 0.288307, 0.244242]
eps = 1e-4

print('==========================================')
print('# Property\t[Ha]\t\t[eV]')
for l, v, r in zip(labels, values, references):
    print('%s\t\t%.6f\t%.3f' % (l, v, v * Ha))
    assert abs(v - r) < eps, (l, v, r, eps) 
