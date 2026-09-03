from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.pos_op.offsite_twocenter_new import Offsite2cTable
from hotcent.pos_op.offsite_twocenter_posop import Offsite2cTablePosOp
import sys


configuration = "[Ne] 3s2 3p1 3d0"
valence = ["3s", "3p", "3d"]
xc = "GGA_X_PBE+GGA_C_PBE"
element = "Al"
scalarrel = True
atom = AtomicDFT(
    element,
    xc=xc,
    configuration=configuration,
    perturbative_confinement=False,
    valence=valence,
    scalarrel=scalarrel,
    confinement=PowerConfinement(r0=60, s=4),
    rmax=600,
)
atom.run()
eigenvalues = atom.enl

conf = PowerConfinement(r0=60.0, s=4)
r0 = 5.9  # Bohr
wf_conf = {
    "3s": PowerConfinement(r0=r0, s=12.4),
    "3p": PowerConfinement(r0=r0, s=12.4),
    "3d": PowerConfinement(r0=r0, s=12.4),
}

atom.set_confinement(conf)
atom.set_wf_confinement(wf_confinement=wf_conf)
atom.run()
eigenvalues_confined = atom.enl

# Compute Slater-Koster integrals:
rmin, dr, N = 0.4, 0.02, 900
off2c = Offsite2cTable(atom, atom, timing=True)
off2c.run(rmin, dr, N, xc=xc, nr=200, ntheta=400, wflimit=1e-9)
off2c.write(
    format=False, eigenvalues=eigenvalues
)  # writes to default Al-Al.skf filename
off2c.write(
    format=True, eigenvalues=eigenvalues, filename_template="{el1}-{el2}dftb.skf"
)

# Compute Integrals for dipole
rmin, dr, N = 0.4, 0.02, 900
off2c = Offsite2cTablePosOp(atom, atom, timing=False)
off2c.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-9)
off2c.write_dipole()

print(eigenvalues)
print(eigenvalues_confined)
