import numpy as np
from ase import Atoms
from ase.io import write
from hotcent.new_dipole.assemble_integrals import SK_With_Shift

vec = np.random.normal(size=3)
vec = vec/np.linalg.norm(vec)

atoms = Atoms('Eu2', positions=[
    [0.0, 0.0, 0.0],
    [vec[0], vec[1], vec[2]]
])
write('Eu2.xyz', atoms)

integrals = SK_With_Shift()
integrals.load_atom_pair('Eu2.xyz')
integrals.set_euler_angles()
integrals.choose_relevant_matrix()
integrals.load_sk_files(path='Eu-Eu_offsite2c.skf', path_dp='Eu-Eu_offsite2c-dipole.skf')
res = integrals.calculate_dipole_elements()
print(res)


