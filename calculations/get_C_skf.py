from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.new_dipole.offsite_twocenter_new import Offsite2cTable
from hotcent.new_dipole.offsite_twocenter_dipole import Offsite2cTableDipole


# Get KS all-electron ground state of confined atom:
element = 'C'
xc = 'GGA_X_PBE+GGA_C_PBE'
r0 = 3.2 # Bohr
conf = PowerConfinement(r0=50.0, s=4)
wf_conf = {'2s': PowerConfinement(r0=r0, s=8.2),
           '2p': PowerConfinement(r0=r0, s=8.2),
           }

atom = AtomicDFT(element,
                xc = xc,
                 confinement=conf,
                 perturbative_confinement=False,
                 configuration='[He] 2s2 2p2',
                 valence=['2s', '2p'],
                 scalarrel=True,
                 maxiter=2500,
                 timing=False,
                 nodegpts=150,
                 mix=0.2,
                 txt='-',
                 )
atom.run()
eigenvalues=atom.enl

atom.set_confinement(conf)
atom.set_wf_confinement(wf_confinement=wf_conf)
atom.run()

# Compute Slater-Koster integrals:
rmin, dr, N = 0.4, 0.02, 900
off2c = Offsite2cTable(atom, atom, timing=True)
off2c.run(rmin, dr, N, xc=xc, nr=200, ntheta=400, wflimit=1e-9)
off2c.write(dftbplus_format=False, eigenvalues=eigenvalues)  # writes to default C-C_offsite2c.skf filename
off2c.write(dftbplus_format=True, eigenvalues=eigenvalues, filename_template='{el1}-{el2}dftb.skf')  

# Compute Integrals for dipole
rmin, dr, N = 0.4, 0.02, 900
off2c = Offsite2cTableDipole(atom, atom, timing=False)
off2c.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-9)
off2c.write_dipole()



