from __future__ import print_function
import os
from hotcent.atom import KSAllElectron
from hotcent.slako import SlaterKosterTable
from hotcent.confinement import PowerConfinement

def check(x, y, eps):
    line = 'Result: {: .8f} | Ref.: {: .8f} | '.format(x, y)
    line += 'OK' if abs(x - y) < eps else 'FAIL'
    return line

txt = os.devnull
eps = 1e-7

# Check confined boron atom
atom1 = KSAllElectron('B',
                      confinement=PowerConfinement(r0=2.9, s=2),
                      configuration='[He] 2s2 2p1',
                      valence=['2s', '2p'],
                      txt=txt,
                      )
atom1.run()
ener = atom1.get_energy()
print('B -- Etot  | %s' % check(ener, -23.079723850586106, eps))
e_2s = atom1.get_epsilon('2s')
print('B -- E_2s  | %s' % check(e_2s, 0.24990807910273996, eps))
e_2p = atom1.get_epsilon('2p')
print('B -- E_2p  | %s' % check(e_2p, 0.47362603289831301, eps))

# Check confined hydrogen atom
atom2 = KSAllElectron('H',
                      confinement=PowerConfinement(r0=1.1, s=2),
                      configuration='1s1',
                      valence=['1s'],
                      txt=txt,
                      )
atom2.run()
ener = atom2.get_energy()
print('H -- Etot  | %s' % check(ener, 0.58885808402033557, eps))
e_1s = atom2.get_epsilon('1s')
print('H -- E_1s  | %s' % check(e_1s, 1.00949638195278960, eps))

# Check B-H Slater-Koster integrals
sk = SlaterKosterTable(atom1, atom2, txt=txt)

R, nt, nr = 2.0, 150, 50
sk.wf_range = sk.get_range(1e-7)
grid, areas = sk.make_grid(R, nt=nt, nr=nr)

S, H, H2 = sk.calculate_mels([('sss', '2s', '1s')], atom1, atom2,
                             R, grid, areas)
print('B-H sps S  | %s' % check(S[9], -0.34627316, eps))
print('B-H sps H  | %s' % check(H[9], 0.29069921, eps))
print('B-H sps H2 | %s' % check(H2[9], 0.29077550, eps))

S, H, H2 = sk.calculate_mels([('sps', '1s', '2p')], atom2, atom1, 
                             R, grid, areas)
print('H-B sps S  | %s' % check(S[8], -0.47147340, eps))
print('H-B sps H  | %s' % check(H[8], 0.33033299, eps))
print('H-B sps H2 | %s' % check(H2[8], 0.33028900, eps))
