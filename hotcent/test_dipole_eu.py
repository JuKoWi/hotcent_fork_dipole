
""" This example aims to reproduce the Eu-Eu Slater-Koster
table in the rare-0-2 dataset from Sanna and coworkers
(doi:10.1103/PhysRevB.76.155128). """
from offsite_twocenter_dipole import Offsite2cTableDipole
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT

element = 'Eu'
xc = 'LDA'

# Get KS all-electron ground state of confined atom
conf = PowerConfinement(r0=6., s=2)
wf_conf = {'6s': PowerConfinement(r0=5., s=2),
           '6p': PowerConfinement(r0=5., s=2),
           '5d': PowerConfinement(r0=6., s=2),
           '4f': PowerConfinement(r0=6., s=2),
           }
atom = AtomicDFT(element,
                 xc=xc,
                 confinement=conf,
                 wf_confinement=wf_conf,
                 perturbative_confinement=False,
                 configuration='[Xe] 4f7 6s2 6p0 5d0',
                 valence=['5d', '6s', '6p', '4f'],
                 scalarrel=True,
                 timing=True,
                 nodegpts=150,
                 mix=0.2,
                 txt='-',
                 )
atom.run()
atom.plot_Rnl()
atom.plot_density()

# Compute Slater-Koster integrals:
rmin, dr, N = 0.56, 0.04, 420
off2c = Offsite2cTableDipole(atom, atom, timing=True)
off2c.run(rmin, dr, N, superposition='potential', xc=xc)
off2c.write()
off2c.plot_minimal()
