from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.pos_op.offsite_twocenter_new import Offsite2cTable
from hotcent.pos_op.offsite_twocenter_posop import Offsite2cTablePosOp
from hotcent.pos_op.onsite_momentum import onsite_momentum


xc = "GGA_X_PBE+GGA_C_PBE"

atomN = AtomicDFT(
    "N",
    xc=xc,
    perturbative_confinement=False,
    confinement=PowerConfinement(r0=50, s=4),
    configuration="[He] 2s2 2p3 ",
    valence=["2s", "2p"],
    scalarrel=True,
    maxiter=2500,
    timing=False,
    nodegpts=2500,
    mix=0.2,
    txt="-",
    rmax=500,
)
atomN.run()
eigenvaluesN = atomN.enl

atomB = AtomicDFT(
    "B",
    xc=xc,
    perturbative_confinement=False,
    confinement=PowerConfinement(r0=50, s=4),
    configuration="[He] 2s2 2p1 ",
    valence=["2s", "2p"],
    scalarrel=True,
    maxiter=2500,
    timing=False,
    nodegpts=2500,
    mix=0.2,
    txt="-",
    rmax=500,
)
atomB.run()
eigenvaluesB = atomB.enl

rcovB = 3.0
rcovN = 3.4

wfconfN = {
    "2s": PowerConfinement(r0=rcovN, s=13.4),
    "2p": PowerConfinement(r0=rcovN, s=13.4),
}

wfconfB = {
    "2s": PowerConfinement(r0=rcovB, s=10.4),
    "2p": PowerConfinement(r0=rcovB, s=10.4),
}

atomB.set_wf_confinement(wf_confinement=wfconfB)
atomB.run()

atomN.set_wf_confinement(wf_confinement=wfconfN)
atomN.run()

rmin, dr, N = 0.4, 0.02, 600

off2chBN = Offsite2cTable(atomN, atomB, timing=True)
off2chBN.run(rmin, dr, N, xc=xc)
off2chBN.write_hdf5()


off2cN = Offsite2cTable(atomN, atomN, timing=True)
off2cN.run(rmin, dr, N, xc=xc)
off2cN.write_hdf5(eigenvalues=eigenvaluesN)
onsite_momentum(atom=atomN, fname="N-Nskf.hdf5", to_h5py=True)


off2cB = Offsite2cTable(atomB, atomB, timing=True)
off2cB.run(rmin, dr, N, xc=xc)
off2cB.write_hdf5(eigenvalues=eigenvaluesB)
onsite_momentum(atom=atomB, fname='B-Bskf.hdf5', to_h5py=True)


# # Compute Integrals for dipole
off2c_dipoleN = Offsite2cTablePosOp(atomN, atomN, timing=False)
off2c_dipoleN.run(rmin, dr, N)
off2c_dipoleN.write_dipole_hdf5()

off2c_dipoleB = Offsite2cTablePosOp(atomB, atomB, timing=False)
off2c_dipoleB.run(rmin, dr, N)
off2c_dipoleB.write_dipole_hdf5()

off2c_dipolehBN = Offsite2cTablePosOp(atomB, atomN, timing=False)
off2c_dipolehBN.run(rmin, dr, N)
off2c_dipolehBN.write_dipole_hdf5()
