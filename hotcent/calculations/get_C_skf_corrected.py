from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.new_dipole.offsite_twocenter_new import Offsite2cTable
from hotcent.new_dipole.offsite_twocenter_dipole import Offsite2cTableDipole

element = 'C'
xc = 'GGA_X_PBE+GGA_C_PBE'
xc = 'LDA'

r0 = 2.65 # Bohr

# Get KS all-electron ground state of (nearly) unconfined atom
conf = PowerConfinement(r0=30.0, s=4)
wf_conf = {'2s': PowerConfinement(r0=r0, s=2),
           '2p': PowerConfinement(r0=r0, s=2),
           }
atom = AtomicDFT(element,
                 xc=xc,
                 confinement=conf,
#                 wf_confinement=wf_conf,
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

atom.info = {}
atom.info['occupations'] = {'2s': 2, '2p': 2}
atom.info['eigenvalues'] = {nl: atom.get_eigenvalue(nl) for nl in atom.valence}
atom.info['hubbardvalues'] = {nl: atom.get_hubbard_value(nl, scheme='central', maxstep=1.) for nl in atom.valence}

# Get KS all-electron ground state of confined atom
conf = PowerConfinement(r0=7., s=2)
atom.set_confinement(conf)
atom.set_wf_confinement(wf_conf)
atom.run()


print(atom.info['eigenvalues'])