from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.new_dipole.offsite_twocenter_new import Offsite2cTable
from hotcent.new_dipole.offsite_twocenter_dipole import Offsite2cTableDipole
from hotcent.new_dipole.utils import bohr_to_angstrom
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle

plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.bbox'] = 'tight'

def plot_radial_parts(atoms:list, orbs:list, rmax_au=4, rmax_log=8):
    """Plot radial parts from NAOs"""
    x_bohr = np.linspace(start=0, stop=rmax_au, num=1000)[1:]
    x_bohr_log = np.linspace(start=0, stop=rmax_log, num=1000)[1:]
    x_angstrom = bohr_to_angstrom(x_bohr)
    x_angstrom_log = bohr_to_angstrom(x_bohr_log)
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(15.5,4.5))
    for i,atom in enumerate(atoms):
        for j,orb in enumerate(orbs[i]):
            R = atom.Rnl(x_bohr, nl=orb)
            R_log = atom.Rnl(x_bohr_log, nl=orb)
            axs[1].semilogy(x_angstrom_log, np.abs(R_log), label=f"{atom.symbol}, {orb}")
            axs[0].plot(x_angstrom, R, label=f"{atom.symbol}, {orb}")
    axs[0].set_title('a', loc='left', fontweight='bold')
    axs[1].set_title('b', loc='left', fontweight='bold')
    axs[0].set_xlabel(r'$r$ $[\mathrm{\AA}]$')
    axs[1].set_xlabel(r'$r$ $[\mathrm{\AA}]$')
    axs[0].set_ylabel(r'$R_{nl}$')
    axs[1].set_ylabel(r'$\left| R_{nl} \right|$')
    axs[0].set_xlim(left=0)
    axs[1].set_xlim(left=0)
    axs[1].set_ylim(bottom=1e-15)
    axs[0].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    axs[0].legend()
    axs[1].legend()
    plt.savefig('Radial_parts.pdf')
    plt.show()

def find_similar_zeta(zeta, atoms:list, orbs:list,rmax_log, rmax_au=12):
    """Plot GTO radial parts alongside NAO radial parts"""
    x_bohr = np.linspace(start=0, stop=rmax_au, num=1000)[1:]
    x_bohr_log = np.linspace(start=0, stop=rmax_log, num=1000)[1:]
    x_angstrom = bohr_to_angstrom(x_bohr)
    x_angstrom_log = bohr_to_angstrom(x_bohr_log)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15.5, 4.5))
    for i,atom in enumerate(atoms):
        for j,orb in enumerate(orbs[i]):
            R = atom.Rnl(x_bohr, nl=orb)
            R_log = atom.Rnl(x_bohr_log, nl=orb)
            axs[0].plot(x_angstrom, R, label=f"{atom.symbol}, {orb}")
            axs[1].semilogy(x_angstrom_log, np.abs(R_log), label=f"{atom.symbol}, {orb}")
    for i in range(3):
        if i==1:
            continue
        else:
            N1 = (2 * zeta[i]/np.pi)**(3/4)*5
            R = N1*x_bohr**(i+1) * np.exp(-zeta[i]*x_bohr**2) #overwrite with gaussian for testing
            R_log = N1*x_bohr_log**(i+1) * np.exp(-zeta[i]*x_bohr_log**2) #overwrite with gaussian for testing
            axs[0].plot(x_angstrom, R, label=rf"$l$ = {i}, $\zeta$ = {zeta[i]}")
            axs[1].plot(x_angstrom_log, np.abs(R_log), label=f"$l$ = {i}, $\zeta$ = {zeta[i]}")
    axs[0].set_xlabel(r'r $[\mathrm{\AA}]$')
    axs[1].set_xlabel(r'r $[\mathrm{\AA}]$')
    axs[0].set_ylabel(r'$R(r)$')
    axs[1].set_ylabel(r'$|R(r)|$')
    axs[0].set_xlim(left=0)
    axs[1].set_xlim(left=0)
    axs[0].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    axs[0].legend()
    axs[1].legend()
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
                confinement=PowerConfinement(r0=50, s=4),
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
                # ['3s', '3p', '3d'], 
                ['4p', 
                 '4d',
                   '5s']
                ]
    
# plot_radial_parts(atoms=atom_list, orbs=orbital_list)
zeta = 0.5
zeta = [zeta, zeta, zeta, zeta]

# find_similar_zeta(atoms=atom_list, orbs=orbital_list, zeta=zeta, rmax_au=3, rmax_log=7)
plot_radial_parts(atoms=atom_list, orbs=orbital_list, rmax_au=4, rmax_log=12)
