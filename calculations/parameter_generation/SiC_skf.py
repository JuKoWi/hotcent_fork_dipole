from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.pos_op.offsite_twocenter_new import Offsite2cTable
from hotcent.pos_op.offsite_twocenter_posop import Offsite2cTablePosOp
from hotcent.pos_op.onsite_momentum import onsite_momentum


xc = "GGA_X_PBE+GGA_C_PBE"

atomSi = AtomicDFT(
    "Si",
    xc=xc,
    perturbative_confinement=False,
    confinement=PowerConfinement(r0=50, s=4),
    configuration="[Ne] 3s2 3p2 ",
    valence=["3s", "3p"],
    scalarrel=True,
    maxiter=2500,
    timing=False,
    nodegpts=2500,
    mix=0.2,
    txt="-",
    rmax=500,
)
atomSi.run()
eigenvaluesSi = dict(atomSi.enl)
hubvals_Si = {}
hubvals_Si['3s'] = atomSi.get_hubbard_value(nl='3s', maxstep=0.1)
hubvals_Si['3p'] = atomSi.get_hubbard_value(nl='3p', maxstep=0.1)

atomC = AtomicDFT(
    "C",
    xc=xc,
    perturbative_confinement=False,
    confinement=PowerConfinement(r0=50, s=4),
    configuration="[He] 2s2 2p2 ",
    valence=["2s", "2p"],
    scalarrel=True,
    maxiter=2500,
    timing=False,
    nodegpts=2500,
    mix=0.2,
    txt="-",
    rmax=500,
)
atomC.run()
eigenvaluesC = dict(atomC.enl)
hubvals_C = {}
hubvals_C['2s'] = atomC.get_hubbard_value(nl='2s', maxstep=0.1)
hubvals_C['2p'] = atomC.get_hubbard_value(nl='2p', maxstep=0.1)

rcovC = 3.2
rcovSi = 4.4

wfconfSi = {
    "3s": PowerConfinement(r0=rcovSi, s=8.2),
    "3p": PowerConfinement(r0=rcovSi, s=8.2),
}

wfconfC = {
    "2s": PowerConfinement(r0=rcovC, s=12.8),
    "2p": PowerConfinement(r0=rcovC, s=12.8),
}

atomC.set_wf_confinement(wf_confinement=wfconfC)
atomC.run()

atomSi.set_wf_confinement(wf_confinement=wfconfSi)
atomSi.run()

rmin, dr, N = 0.4, 0.02, 600

off2cSiC = Offsite2cTable(atomSi, atomC, timing=True)
off2cSiC.run(rmin, dr, N, xc=xc,superposition='potential')
off2cSiC.write(format="DFTB+", filename_template="{el1}-{el2}.skf")
off2cSiC.write_hdf5()


off2cSi = Offsite2cTable(atomSi, atomSi, timing=True)
off2cSi.run(rmin, dr, N, xc=xc, superposition='potential')
off2cSi.write(format="DFTB+",eigenvalues=eigenvaluesSi, hubbardvalues=hubvals_Si, filename_template="{el1}-{el2}.skf")
off2cSi.write_hdf5(eigenvalues=eigenvaluesSi)
onsite_momentum(atom=atomSi, fname="N-Nskf.hdf5", to_h5py=True)


off2cC = Offsite2cTable(atomC, atomC, timing=True)
off2cC.run(rmin, dr, N, xc=xc, superposition='potential')
off2cC.write(format="DFTB+",eigenvalues=eigenvaluesC, hubbardvalues=hubvals_C, filename_template="{el1}-{el2}.skf")
off2cC.write_hdf5(eigenvalues=eigenvaluesC)
onsite_momentum(atom=atomC, fname='B-Bskf.hdf5', to_h5py=True)


# # Compute Integrals for dipole
off2c_dipoleSi = Offsite2cTablePosOp(atomSi, atomSi, timing=False)
off2c_dipoleSi.run(rmin, dr, N)
off2c_dipoleSi.write_dipole_hdf5()

off2c_dipoleC = Offsite2cTablePosOp(atomC, atomC, timing=False)
off2c_dipoleC.run(rmin, dr, N)
off2c_dipoleC.write_dipole_hdf5()

off2c_dipoleSiC = Offsite2cTablePosOp(atomC, atomSi, timing=False)
off2c_dipoleSiC.run(rmin, dr, N)
off2c_dipoleSiC.write_dipole_hdf5()
