from ase import atom
from ase.build import mx2 

atoms = mx2()
print(atoms.get_cell())
print(atoms.get_positions())

