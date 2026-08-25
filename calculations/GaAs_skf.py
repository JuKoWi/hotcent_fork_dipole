from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.pos_op.offsite_twocenter_new import Offsite2cTable
from hotcent.pos_op.offsite_twocenter_posop import Offsite2cTablePosOp
from ase.data import covalent_radii, atomic_numbers
from ase.units import Bohr
import sys

xc='GGA_X_PBE+GGA_C_PBE'

atomGa = AtomicDFT('Ga',
                xc = xc,
                perturbative_confinement=False,
                confinement=PowerConfinement(r0=50, s=4),
                configuration='[Ar] 3d10 4s2 4p1 4d0',
                valence=['4s', '4p', '4d'], 
                scalarrel=True,
                maxiter=2500,
                timing=False,
                nodegpts=2500,
                mix=0.2,
                txt='-',
                rmax=500,
                )
atomGa.run()
eigenvaluesGa = atomGa.enl


atomAs = AtomicDFT('As',
                xc = xc,
                perturbative_confinement=False,
                configuration='[Ar] 3d10 4s2 4p3 4d0',
                valence=['4s', '4p', '4d'],
                confinement=PowerConfinement(r0=50, s=4),
                scalarrel=True,
                maxiter=2500,
                timing=False,
                # nodegpts=150,
                mix=0.2,
                txt='-',
                rmax=100,
                )
atomAs.run()
print(atomAs.enl)
eigenvaluesAs = atomAs.enl



#Use parameters from 10.1021/ct4004959 (Heine 2013)
rcovGa = 5.9
rcovMo = 4.4
# confS = PowerConfinement(r0=50, s=4)
# confMo = PowerConfinement(r0=50, s=4)

wf_confGa = {'4s': PowerConfinement(r0=rcovGa, s=8.8),
           '4p': PowerConfinement(r0=rcovGa, s=8.8),
           '4d': PowerConfinement(r0=rcovGa, s=8.8),
           }

wf_confAs = {'4d': PowerConfinement(r0=rcovMo, s=5.6),
           '4s': PowerConfinement(r0=rcovMo, s=5.6),
           '4p': PowerConfinement(r0=rcovMo, s=5.6),
           }

# atomS.set_confinement(confS)
atomGa.set_wf_confinement(wf_confinement=wf_confGa)
atomGa.run()

# atomMo.set_confinement(confMo)
atomAs.set_wf_confinement(wf_confinement=wf_confAs)
atomAs.run()

# Compute Slater-Koster integrals:
rmin, dr, N = 0.4, 0.02, 900

off2cMoS = Offsite2cTable(atomGa, atomAs, timing=True)
off2cMoS.run(rmin, dr, N, xc=xc, nr=200, ntheta=400, wflimit=1e-9)
off2cMoS.write(dftbplus_format=False) 
off2cMoS.write(dftbplus_format=True, filename_template='{el1}-{el2}dftb.skf')  

off2cS = Offsite2cTable(atomGa, atomGa, timing=True)
off2cS.run(rmin, dr, N, xc=xc, nr=200, ntheta=400, wflimit=1e-9)
off2cS.write(dftbplus_format=False, eigenvalues=eigenvaluesGa) 
off2cS.write(dftbplus_format=True, eigenvalues=eigenvaluesGa, filename_template='{el1}-{el2}dftb.skf')  


off2cMo = Offsite2cTable(atomAs, atomAs, timing=True)
off2cMo.run(rmin, dr, N, xc=xc, nr=200, ntheta=400, wflimit=1e-9)
off2cMo.write(dftbplus_format=False, eigenvalues=eigenvaluesAs) 
off2cMo.write(dftbplus_format=True, eigenvalues=eigenvaluesAs, filename_template='{el1}-{el2}dftb.skf')  


# # Compute Integrals for dipole
off2c_dipoleS = Offsite2cTablePosOp(atomGa, atomGa, timing=False)
off2c_dipoleS.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-9)
off2c_dipoleS.write_dipole()

off2c_dipoleMo = Offsite2cTablePosOp(atomAs, atomAs, timing=False)
off2c_dipoleMo.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-9)
off2c_dipoleMo.write_dipole()

off2c_dipoleMoS = Offsite2cTablePosOp(atomAs, atomGa, timing=False)
off2c_dipoleMoS.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-9)
off2c_dipoleMoS.write_dipole()

