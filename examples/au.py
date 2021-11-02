""" This example aims to reproduce the Au-Au
Slater-Koster table generation procedure by
Fihey and coworkers (doi:10.1002/jcc.24046). """
from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT

element = 'Au'
xc = 'GGA_X_PBE+GGA_C_PBE'

# Get KS all-electron ground state of confined atom
conf = PowerConfinement(r0=9.41, s=2)
wf_conf = {'5d': PowerConfinement(r0=6.50, s=2),
           '6s': PowerConfinement(r0=6.50, s=2),
           '6p': PowerConfinement(r0=4.51, s=2),
           }
atom = AtomicDFT(element,
                 xc=xc,
                 confinement=conf,
                 wf_confinement=wf_conf,
                 configuration='[Xe] 4f14 5d10 6s1 6p0',
                 valence=['5d', '6s', '6p'],
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
rmin, dr, N = 0.4, 0.02, 900
off2c = Offsite2cTable(atom, atom, timing=True)
off2c.run(rmin, dr, N, superposition='density', xc=xc)
off2c.write('Au-Au_offsite2c.par')
off2c.write('Au-Au_offsite2c.skf')
off2c.plot()
