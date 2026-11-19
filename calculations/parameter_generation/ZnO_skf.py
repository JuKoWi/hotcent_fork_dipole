from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.pos_op.offsite_twocenter_new import Offsite2cTable
from hotcent.pos_op.offsite_twocenter_posop import Offsite2cTablePosOp
from hotcent.pos_op.onsite_momentum import onsite_momentum

xc = "GGA_X_PBE+GGA_C_PBE"

atomZn = AtomicDFT(
    "Zn",
    xc=xc,
    perturbative_confinement=False,
    confinement=PowerConfinement(r0=50, s=4),
    configuration="[Ar] 3d10 4s2 4p0",
    valence=["4s", "4p", "3d"],
    scalarrel=True,
    maxiter=2500,
    timing=False,
    nodegpts=2500,
    mix=0.2,
    txt="-",
    rmax=500,
)
atomZn.run()
eigenvaluesZn = dict(atomZn.enl)
hubvals_Zn = {}
hubvals_Zn['4s'] = atomZn.get_hubbard_value(nl='4s', maxstep=0.1)
hubvals_Zn['4p'] = atomZn.get_hubbard_value(nl='4p', maxstep=0.1)
hubvals_Zn['3d'] = atomZn.get_hubbard_value(nl='3d', maxstep=0.1)


atomO = AtomicDFT(
    "O",
    xc=xc,
    perturbative_confinement=False,
    configuration="[He] 2s2 2p4",
    valence=["2s", "2p"],
    confinement=PowerConfinement(r0=50, s=4),
    scalarrel=True,
    maxiter=2500,
    timing=False,
    # nodegpts=150,
    mix=0.2,
    txt="-",
    rmax=100,
)
atomO.run()
print(atomO.enl)
eigenvaluesO = dict(atomO.enl)
hubvals_O = {}
hubvals_O['2s'] = atomO.get_hubbard_value(nl='2s', maxstep=0.1)
hubvals_O['2p'] = atomO.get_hubbard_value(nl='2p', maxstep=0.1)


# Use parameters from 10.1021/ct4004959 (Heine 2013)
rcovZn = 4.6
rcovO = 3.1
# confS = PowerConfinement(r0=50, s=4)
# confMo = PowerConfinement(r0=50, s=4)

wf_confZn = {
    "4s": PowerConfinement(r0=rcovZn, s=2.2),
    "4p": PowerConfinement(r0=rcovZn, s=2.2),
    "3d": PowerConfinement(r0=rcovZn, s=2.2),
}

wf_confO = {
    "2s": PowerConfinement(r0=rcovO, s=12.4),
    "2p": PowerConfinement(r0=rcovO, s=12.4),
}

# atomS.set_confinement(confS)
atomZn.set_wf_confinement(wf_confinement=wf_confZn)
atomZn.run()

# atomMo.set_confinement(confMo)
atomO.set_wf_confinement(wf_confinement=wf_confO)
atomO.run()

# Compute Slater-Koster integrals:
rmin, dr, N = 0.4, 0.02, 900

off2cZnO = Offsite2cTable(atomZn, atomO, timing=True)
off2cZnO.run(rmin, dr, N, xc=xc, nr=200, ntheta=400, wflimit=1e-8, superposition='potential')
off2cZnO.write(format="DFTB+", filename_template="{el1}-{el2}.skf")
off2cZnO.write_hdf5()

off2cZn = Offsite2cTable(atomZn, atomZn, timing=True)
off2cZn.run(rmin, dr, N, xc=xc, nr=200, ntheta=400, wflimit=1e-8, superposition='potential')
off2cZn.write_hdf5(eigenvalues=eigenvaluesZn)
off2cZn.write(format="DFTB+", eigenvalues=eigenvaluesZn, hubbardvalues=hubvals_Zn, filename_template="{el1}-{el2}.skf")
onsite_momentum(atom=atomZn, fname="Zn-Znskf.hdf5", to_h5py=True)

off2cO = Offsite2cTable(atomO, atomO, timing=True)
off2cO.run(rmin, dr, N, xc=xc, nr=200, ntheta=400, wflimit=1e-8, superposition='potential')
off2cO.write_hdf5(eigenvalues=eigenvaluesO)
off2cO.write(format="DFTB+", eigenvalues=eigenvaluesO, hubbardvalues=hubvals_O, filename_template="{el1}-{el2}.skf")
onsite_momentum(atom=atomO, fname="O-Oskf.hdf5", to_h5py=True)


# # Compute Integrals for dipole
off2c_dipoleZn = Offsite2cTablePosOp(atomZn, atomZn, timing=False)
off2c_dipoleZn.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-8)
off2c_dipoleZn.write_dipole_hdf5()

off2c_dipoleO = Offsite2cTablePosOp(atomO, atomO, timing=False)
off2c_dipoleO.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-8)
off2c_dipoleO.write_dipole_hdf5()

off2c_dipoleZnO = Offsite2cTablePosOp(atomO, atomZn, timing=False)
off2c_dipoleZnO.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-8)
off2c_dipoleZnO.write_dipole_hdf5()
