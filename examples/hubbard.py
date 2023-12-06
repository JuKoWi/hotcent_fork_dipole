from ase.units import Ha
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import PowerConfinement

atom = AtomicDFT('C',
                 xc='LDA',
                 confinement=PowerConfinement(r0=40., s=4),
                 perturbative_confinement=False,
                 configuration='[He] 2s2 2p2',
                 valence=['2s', '2p'],
                 scalarrel=False,
                 timing=False,
                 )

values = []
schemes = ['central', 'forward', 'backward']
for scheme in schemes:
    u = atom.get_hubbard_value('2p', scheme=scheme, maxstep=1.)
    values.append(u)

references = [0.346391, 0.383819, 0.368047]
eps = 1e-4

print('\n========================================================')
print('# Scheme\tU_ref [Ha]\tU [Ha]\t\tU [eV]')
for s, v, r in zip(schemes, values, references):
    print('%s\t%.6f\t%.6f\t%.3f' % (s.ljust(8), r, v, v * Ha))
    assert abs(v - r) < eps, (s, v, r, eps)
