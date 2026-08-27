from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.pos_op.offsite_twocenter_new import Offsite2cTable
from hotcent.pos_op.offsite_twocenter_posop import Offsite2cTablePosOp

element = 'Au'
xc = 'GGA_X_PBE+GGA_C_PBE'

# Get KS all-electron ground state of confined atom
conf = PowerConfinement(r0=9.41, s=2)
atom = AtomicDFT(element,
                 xc=xc,
                 confinement=None,
                 wf_confinement=None,
                 perturbative_confinement=False,
                 configuration='[Xe] 4f14 5d10 6s1 6p0',
                 valence=['5d', '6s', '6p'],
                 scalarrel=True,
                 timing=True,
                 nodegpts=150,
                 mix=0.2,
                 txt='-',
                 )
atom.run()
eigenvalues=atom.enl

wf_conf = {'5d': PowerConfinement(r0=4.8, s=2),
           '6s': PowerConfinement(r0=4.8, s=2),
           '6p': PowerConfinement(r0=4.8, s=2),
           }

atom.set_wf_confinement(wf_confinement=wf_conf)
atom.run()
eigenvalues_confined = atom.enl

# Compute Slater-Koster integrals:
rmin, dr, N = 0.4, 0.02, 900
off2c = Offsite2cTable(atom, atom, timing=True)
off2c.run(rmin, dr, N, xc=xc, nr=200, ntheta=400)
off2c.write(format=False, eigenvalues=eigenvalues)  # writes to default Au-Au.skf filename
off2c.write(format=True, eigenvalues=eigenvalues, filename_template='{el1}-{el2}dftb.skf')  

# Compute Integrals for dipole
rmin, dr, N = 0.4, 0.02, 900
off2c = Offsite2cTablePosOp(atom, atom, timing=False)
off2c.run(rmin, dr, N, nr=200, ntheta=400)
off2c.write_dipole()

print(eigenvalues)
print(eigenvalues_confined)
