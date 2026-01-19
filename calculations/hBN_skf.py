from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.new_dipole.offsite_twocenter_new import Offsite2cTable
from hotcent.new_dipole.offsite_twocenter_dipole import Offsite2cTableDipole


xc = 'GGA_X_PBE+GGA_C_PBE'

atomN = AtomicDFT('N',
                xc = xc,
                perturbative_confinement=False,
                confinement=PowerConfinement(r0=50, s=4),
                configuration='[He] 2s2 2p3 ',
                valence=['2s', '2p'], 
                scalarrel=True,
                maxiter=2500,
                timing=False,
                nodegpts=2500,
                mix=0.2,
                txt='-',
                rmax=500,
                )
atomN.run()
print(atomN.enl)
eigenvaluesN = atomN.enl

atomB = AtomicDFT('B',
                xc = xc,
                perturbative_confinement=False,
                confinement=PowerConfinement(r0=50, s=4),
                configuration='[He] 2s2 2p1 ',
                valence=['2s', '2p'], 
                scalarrel=True,
                maxiter=2500,
                timing=False,
                nodegpts=2500,
                mix=0.2,
                txt='-',
                rmax=500,
                )
atomB.run()
print(atomB.enl)
eigenvaluesB = atomB.enl

rcovB = 3.0
rcovN = 3.4

wfconfN = {'2s': PowerConfinement(r0=rcovN, s=13.4),
           '2p': PowerConfinement(r0=rcovN, s=13.4)
           }

wfconfB = {'2s': PowerConfinement(r0=rcovB, s=10.4),
           '2p': PowerConfinement(r0=rcovB, s=10.4)
            }

atomB.set_wf_confinement(wf_confinement=wfconfB)
atomB.run()

atomN.set_wf_confinement(wf_confinement=wfconfN)
atomN.run()

rmin, dr, N = 0.4, 0.02, 600

off2chBN = Offsite2cTable(atomN, atomB, timing=True)
off2chBN.run(rmin, dr, N, xc=xc)
off2chBN.write(dftbplus_format=False) 
off2chBN.write(dftbplus_format=True, filename_template='{el1}-{el2}dftb.skf')  

off2cN = Offsite2cTable(atomN, atomN, timing=True)
off2cN.run(rmin, dr, N, xc=xc)
off2cN.write(dftbplus_format=False, eigenvalues=eigenvaluesN) 
off2cN.write(dftbplus_format=True, eigenvalues=eigenvaluesN, filename_template='{el1}-{el2}dftb.skf')  


off2cB = Offsite2cTable(atomB, atomB, timing=True)
off2cB.run(rmin, dr, N, xc=xc)
off2cB.write(dftbplus_format=False, eigenvalues=eigenvaluesB) 
off2cB.write(dftbplus_format=True, eigenvalues=eigenvaluesB, filename_template='{el1}-{el2}dftb.skf')  


# # Compute Integrals for dipole
off2c_dipoleN = Offsite2cTableDipole(atomN, atomN, timing=False)
off2c_dipoleN.run(rmin, dr, N)
off2c_dipoleN.write_dipole()

off2c_dipoleB = Offsite2cTableDipole(atomB, atomB, timing=False)
off2c_dipoleB.run(rmin, dr, N)
off2c_dipoleB.write_dipole()

off2c_dipolehBN = Offsite2cTableDipole(atomB, atomN, timing=False)
off2c_dipolehBN.run(rmin, dr, N)
off2c_dipolehBN.write_dipole()
