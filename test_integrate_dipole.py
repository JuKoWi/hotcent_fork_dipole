from hotcent.new_dipole.assemble_integrals import SK_Integral_Dipole
from ase import Atoms
from ase.io import write

atoms = Atoms('C2', positions=[
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.54]
])
write('C2.xyz', atoms)
basis = ['2s', '2p']


integral = SK_Integral_Dipole('C', 'C')
integral.load_atom_pair('C2.xyz')
integral.set_rotation_matrix()
integral.load_SK_dipole_file('C-C_offsite2c-dipole.skf')
print(integral.calculate_dipole())
