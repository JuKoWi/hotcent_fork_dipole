from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.new_dipole.offsite_twocenter_new import Offsite2cTable
from hotcent.new_dipole.offsite_twocenter_dipole import Offsite2cTableDipole
from hotcent.new_dipole.utils import bohr_to_angstrom
import matplotlib.pyplot as plt
import numpy as np
import sys

plt.rcParams.update({'font.size': 19})
plt.rcParams['savefig.bbox'] = 'tight'

def plot_radial_parts(atoms:list, orbs:list, rmax_au=4):
    x_bohr = np.linspace(start=0, stop=rmax_au, num=1000)[1:]
    x_angstrom = bohr_to_angstrom(x_bohr)
    fig, ax = plt.subplots(figsize=(16,9))
    for i,atom in enumerate(atoms):
        for j,orb in enumerate(orbs[i]):
            R = atom.Rnl(x_bohr, nl=orb)
            ax.plot(x_angstrom, R, label=f"{atom.symbol}, {orb}")
    ax.set_xlabel(r'r / $\AA$')
    ax.set_ylabel('R(r)')
    ax.set_xlim(left=0)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.legend()
    plt.savefig('Radial_parts.pdf')
    plt.show()

def find_similar_zeta(zeta, atoms:list, orbs:list, rmax_au=4):
    x_bohr = np.linspace(start=0, stop=rmax_au, num=1000)[1:]
    x_angstrom = bohr_to_angstrom(x_bohr)
    fig, ax = plt.subplots(figsize=(16,9))
    for i,atom in enumerate(atoms):
        for j,orb in enumerate(orbs[i]):
            R = atom.Rnl(x_bohr, nl=orb)
            ax.plot(x_angstrom, R, label=f"{atom.symbol}, {orb}")
    for i in range(3):
        N1 = (2 * zeta[i]/np.pi)**(3/4)*5
        R = N1*x_bohr**(i+1) * np.exp(-zeta[i]*x_bohr**2) #overwrite with gaussian for testing
        ax.plot(x_angstrom, R, label=f"l = {i}, zeta = {zeta[i]}")
    ax.set_xlabel(r'r / $\AA$')
    ax.set_ylabel('R(r)')
    ax.set_xlim(left=0)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.legend()
    plt.savefig('Radial_parts.pdf')
    plt.show()


# Carbon
# element = 'C'
# xc = 'GGA_X_PBE+GGA_C_PBE'
# r0 = 3.2 # Bohr
# conf = PowerConfinement(r0=50.0, s=4)
# wf_conf = {'2s': PowerConfinement(r0=r0, s=8.2),
#            '2p': PowerConfinement(r0=r0, s=8.2),
#            }

# atom = AtomicDFT(element,
#                 xc = xc,
#                  confinement=conf,
#                  perturbative_confinement=False,
#                  configuration='[He] 2s2 2p2',
#                  valence=['2s', '2p'],
#                  scalarrel=True,
#                  maxiter=2500,
#                  timing=False,
#                  nodegpts=150,
#                  mix=0.2,
#                  txt='-',
#                  )
# atom.run()
# eigenvalues=atom.enl

# atom.set_confinement(conf)
# atom.set_wf_confinement(wf_confinement=wf_conf)
# atom.run()

#Mo and S
xc='GGA_X_PBE+GGA_C_PBE'

# atomS = AtomicDFT('S',
#                 xc = xc,
#                 perturbative_confinement=False,
#                 confinement=PowerConfinement(r0=50, s=4),
#                 configuration='[Ne] 3s2 3p4 3d0',
#                 valence=['3s', '3p', '3d'], 
#                 scalarrel=True,
#                 maxiter=2500,
#                 timing=False,
#                 nodegpts=2500,
#                 mix=0.2,
#                 txt='-',
#                 rmax=500,
#                 )
# atomS.run()
# print(atomS.enl)
# eigenvaluesS = atomS.enl


atomMo = AtomicDFT('Mo',
                xc = xc,
                perturbative_confinement=False,
                configuration='[Kr] 4d4 5s2 5p0',
                valence=['4d', '5s', '5p'],
                confinement=PowerConfinement(r0=40, s=4),
                scalarrel=True,
                maxiter=2500,
                timing=False,
                # nodegpts=150,
                mix=0.2,
                txt='-',
                rmax=100,
                )
atomMo.run()
print(atomMo.enl)
eigenvaluesMo = atomMo.enl



#Use parameters from 10.1021/ct4004959 (Heine 2013)
rcovS = 3.9
rcovMo = 4.3
confS = PowerConfinement(r0=50, s=4)
confMo = PowerConfinement(r0=50, s=4)

wf_confS = {'3s': PowerConfinement(r0=rcovS, s=4.6),
           '3p': PowerConfinement(r0=rcovS, s=4.6),
           '3d': PowerConfinement(r0=rcovS, s=4.6),
           }

wf_confMo = {'4d': PowerConfinement(r0=rcovMo, s=11.6),
           '5s': PowerConfinement(r0=rcovMo, s=11.6),
           '5p': PowerConfinement(r0=rcovMo, s=11.6),
           }

# atomS.set_confinement(confS)
# atomS.set_wf_confinement(wf_confinement=wf_confS)
# atomS.run()

atomMo.set_confinement(confMo)
atomMo.set_wf_confinement(wf_confinement=wf_confMo)
atomMo.run()

atom_list = [
            # atom, 
            #  atomS, 
            atomMo
             ]
orbital_list = [
                # ['1s', '2s', '2p'], 
                # ['3s', '3p'], 
                ['4d', '5s']
                ]
    
# plot_radial_parts(atoms=atom_list, orbs=orbital_list)
zeta = 1
zeta = [zeta, zeta, zeta, zeta]
find_similar_zeta(atoms=atom_list, orbs=orbital_list, zeta=zeta)
