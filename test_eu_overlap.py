
""" This example aims to reproduce the Eu-Eu Slater-Koster
table in the rare-0-2 dataset from Sanna and coworkers
(doi:10.1103/PhysRevB.76.155128). """
from hotcent.new_dipole.offsite_twocenter_new import Offsite2cTable
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
import numpy as np
from ase import Atoms
from ase.io import write
from hotcent.new_dipole.assemble_integrals import SK_Integral_Overlap


USE_EXISTING_SKF = True
element = 'Eu'
xc = 'LDA'

if not USE_EXISTING_SKF:
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

    # Compute Slater-Koster integrals:
    rmin, dr, N = 0.56, 0.04, 420
    off2c = Offsite2cTable(atom, atom, timing=True)
    off2c.run(rmin, dr, N, superposition='potential', xc=xc)
    off2c.write()


vec = np.random.normal(size=3)
vec = vec/np.linalg.norm(vec)

atoms = Atoms('Eu2', positions=[
    [0.0, 0.0, 0.0],
    [vec[0], vec[1], vec[2]]
])

#assemble actual matrix elements
write('Eu2.xyz', atoms)
method1 = SK_Integral_Overlap('Eu', 'Eu')
method1.load_atom_pair('Eu2.xyz')
method1.set_euler_angles()
method1.load_SK_file('Eu-Eu_offsite2c.skf')
method1.set_rotation_matrix()
res1 = method1.calculate_overlap()
# method1.check_rotation_implementation()
