from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.new_dipole.offsite_twocenter_new import Offsite2cTable
from hotcent.new_dipole.offsite_twocenter_dipole import Offsite2cTableDipole
from ase.data import covalent_radii, atomic_numbers
from ase.units import Bohr
import sys

# Get KS all-electron ground state of confined atom:
xc = 'GGA_X_PBE+GGA_C_PBE'

#Use parameters from 10.1021/ct4004959 (Heine 2013)
rcovS = 3.9
rcovMo = 4.3
confS = PowerConfinement(r0=rcovS, s=4.6)
confMo = PowerConfinement(r0=rcovMo, s=11.6)

wf_confS = {'3s': PowerConfinement(r0=rcovS, s=4.6),
           '3p': PowerConfinement(r0=rcovS, s=4.6),
           '3d': PowerConfinement(r0=rcovS, s=4.6),
           }

wf_confMo = {'4d': PowerConfinement(r0=rcovMo, s=11.6),
           '5s': PowerConfinement(r0=rcovMo, s=11.6),
           '5p': PowerConfinement(r0=rcovMo, s=11.6),
           }


atomS = AtomicDFT('S',
                xc = xc,
                 perturbative_confinement=False,
                 configuration='[Ne] 3s2 3p4',
                 valence=['3s', '3p'],
                 scalarrel=True,
                 maxiter=3500,
                 timing=False,
                 nodegpts=150,
                 mix=0.2,
                 txt='-',
                 )
atomS.run()

atomMo = AtomicDFT('Mo',
                xc = xc,
                 perturbative_confinement=False,
                 configuration='[Kr] 4d5 5s1 5p0',
                 valence=['4d', '5s', '5p'],
                 scalarrel=True,
                 maxiter=2500,
                 timing=False,
                 nodegpts=150,
                 mix=0.2,
                 txt='-',
                 )
atomMo.run()


eigenvaluesS = atomS.enl
eigenvaluesMo = atomMo.enl
print(eigenvaluesS)
print(eigenvaluesMo)
sys.exit()

atomS.set_confinement(confS)
atomS.set_wf_confinement(wf_confinement=wf_confS)
atomS.run()

atomMo.set_confinement(confMo)
atomMo.set_wf_confinement(wf_confinement=wf_confMo)
atomMo.run()

# Compute Slater-Koster integrals:
rmin, dr, N = 0.4, 0.02, 900

off2cMoS = Offsite2cTable(atomS, atomMo, timing=True)
off2cMoS.run(rmin, dr, N)
off2cMoS.write(dftbplus_format=False)  # writes to default C-C_offsite2c.skf filename

off2cS = Offsite2cTable(atomS, atomS, timing=True)
off2cS.run(rmin, dr, N)
off2cS.write(dftbplus_format=False, eigenvalues=eigenvaluesS)  # writes to default C-C_offsite2c.skf filename

off2cMo = Offsite2cTable(atomMo, atomMo, timing=True)
off2cMo.run(rmin, dr, N)
off2cMo.write(dftbplus_format=False, eigenvalues=eigenvaluesMo)  # writes to default C-C_offsite2c.skf filename


# # Compute Integrals for dipole
off2c_dipoleS = Offsite2cTableDipole(atomS, atomS, timing=False)
off2c_dipoleS.run(rmin, dr, N)
off2c_dipoleS.write_dipole()

off2c_dipoleMo = Offsite2cTableDipole(atomMo, atomMo, timing=False)
off2c_dipoleMo.run(rmin, dr, N)
off2c_dipoleMo.write_dipole()

off2c_dipoleMoS = Offsite2cTableDipole(atomMo, atomS, timing=False)
off2c_dipoleMoS.run(rmin, dr, N)
off2c_dipoleMoS.write_dipole()





