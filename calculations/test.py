from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.pos_op.offsite_twocenter_new import Offsite2cTable
from hotcent.pos_op.offsite_twocenter_posop import Offsite2cTablePosOp
from hotcent.pos_op.onsite_momentum import onsite_momentum
import sys
import numpy as np
from hotcent.pos_op.rotation_transform import transform_to_real

element = "C"
xc = "GGA_X_PBE+GGA_C_PBE"
conf = PowerConfinement(r0=50.0, s=4)
r0 = 3.2  # Bohr
wf_conf = {
    "2s": PowerConfinement(r0=r0, s=8.2),
    "2p": PowerConfinement(r0=r0, s=8.2),
}

atom = AtomicDFT(
    element,
    xc=xc,
    confinement=conf,
    perturbative_confinement=False,
    configuration="[He] 2s2 2p2 3d0",
    valence=["2p", "2s", "3d"],
    scalarrel=True,
    maxiter=2500,
    timing=False,
    nodegpts=150,
    mix=0.2,
    txt="-",
)
atom.run()

onsite_momentum(atom, 'testfile.txt')

