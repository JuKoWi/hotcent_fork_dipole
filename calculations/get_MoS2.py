from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.new_dipole.offsite_twocenter_new import Offsite2cTable
from hotcent.new_dipole.offsite_twocenter_dipole import Offsite2cTableDipole
from ase.data import covalent_radii, atomic_numbers
from ase.units import Bohr
import sys

# Get KS all-electron ground state of confined atom:
def run_atomic_dft(element, configuration, valence, xc='GGA_X_PBE+GGA_C_PBE', scalarrel=False):
    """Function by Alex to find optimal values for r0 and rmax"""
    r0 = 50.
    retry = True
    while (r0 > 10) and retry:
        rmax = 100.
        while (rmax < 700) and retry:
            try:
                atom = AtomicDFT(element,
                             xc=xc,
                             configuration=configuration,
                             perturbative_confinement=False,
                             valence=valence,
                             scalarrel=scalarrel,
                             confinement=PowerConfinement(r0=r0, s=4),
                            # nodegpts=2500,
                             rmax=rmax,
                             mix=0.2,
                             maxiter=2500,
                             txt=None,
                             timing=None,
                             )

                atom.run()

                atom.info = {}
                atom.info['eigenvalues'] = {nl: atom.get_eigenvalue(nl) for nl in atom.valence}
                atom.info['hubbardvalues'] = {nl: atom.get_hubbard_value(nl, scheme='central', maxstep=1.) for nl in atom.valence}
                retry = False
            except (RuntimeError,AssertionError):
                rmax = rmax + 100.
                retry = True
        if retry:
            r0 = r0 - 10.
    if retry:
        raise(RuntimeError('No convergence achieved.'))
    print('--- success for rmax=', rmax, ', r0=', r0)
    atom.info['occupations'] = valence
    return atom


# atomS = run_atomic_dft(element='S', configuration='[Ne] 3s2 3p4 3d0', valence=['3s', '3p', '3d'],  xc='GGA_X_PBE+GGA_C_PBE', scalarrel=True)
# print(atomS.info['eigenvalues'])
# atomMo= run_atomic_dft(element='Mo', configuration='[Kr] 4d4 5s2 5p0', valence=['4d', '5s', '5p'],  xc='GGA_X_PBE+GGA_C_PBE', scalarrel=True)
# print(atomMo.info['eigenvalues'])

xc='GGA_X_PBE+GGA_C_PBE'

atomS = AtomicDFT('S',
                xc = xc,
                perturbative_confinement=False,
                confinement=PowerConfinement(r0=50, s=4),
                configuration='[Ne] 3s2 3p4 3d0',
                valence=['3s', '3p', '3d'], 
                scalarrel=True,
                maxiter=2500,
                timing=False,
                nodegpts=2500,
                mix=0.2,
                txt='-',
                rmax=500,
                )
atomS.run()
print(atomS.enl)
eigenvaluesS = atomS.enl


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
# confS = PowerConfinement(r0=50, s=4)
# confMo = PowerConfinement(r0=50, s=4)

wf_confS = {'3s': PowerConfinement(r0=rcovS, s=4.6),
           '3p': PowerConfinement(r0=rcovS, s=4.6),
           '3d': PowerConfinement(r0=rcovS, s=4.6),
           }

wf_confMo = {'4d': PowerConfinement(r0=rcovMo, s=11.6),
           '5s': PowerConfinement(r0=rcovMo, s=11.6),
           '5p': PowerConfinement(r0=rcovMo, s=11.6),
           }

# atomS.set_confinement(confS)
atomS.set_wf_confinement(wf_confinement=wf_confS)
atomS.run()

# atomMo.set_confinement(confMo)
atomMo.set_wf_confinement(wf_confinement=wf_confMo)
atomMo.run()

# Compute Slater-Koster integrals:
rmin, dr, N = 0.4, 0.02, 900

off2cMoS = Offsite2cTable(atomS, atomMo, timing=True)
off2cMoS.run(rmin, dr, N, xc=xc, nr=200, ntheta=400, wflimit=1e-9)
off2cMoS.write(dftbplus_format=False) 
off2cMoS.write(dftbplus_format=True, filename_template='{el1}-{el2}dftb.skf')  

off2cS = Offsite2cTable(atomS, atomS, timing=True)
off2cS.run(rmin, dr, N, xc=xc, nr=200, ntheta=400, wflimit=1e-9)
off2cS.write(dftbplus_format=False, eigenvalues=eigenvaluesS) 
off2cS.write(dftbplus_format=True, eigenvalues=eigenvaluesS, filename_template='{el1}-{el2}dftb.skf')  


off2cMo = Offsite2cTable(atomMo, atomMo, timing=True)
off2cMo.run(rmin, dr, N, xc=xc, nr=200, ntheta=400, wflimit=1e-9)
off2cMo.write(dftbplus_format=False, eigenvalues=eigenvaluesMo) 
off2cMo.write(dftbplus_format=True, eigenvalues=eigenvaluesMo, filename_template='{el1}-{el2}dftb.skf')  


# # Compute Integrals for dipole
off2c_dipoleS = Offsite2cTableDipole(atomS, atomS, timing=False)
off2c_dipoleS.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-9)
off2c_dipoleS.write_dipole()

off2c_dipoleMo = Offsite2cTableDipole(atomMo, atomMo, timing=False)
off2c_dipoleMo.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-9)
off2c_dipoleMo.write_dipole()

off2c_dipoleMoS = Offsite2cTableDipole(atomMo, atomS, timing=False)
off2c_dipoleMoS.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-9)
off2c_dipoleMoS.write_dipole()





