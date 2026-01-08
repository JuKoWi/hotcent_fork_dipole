from hotcent.new_dipole.files_for_comparison import Seedname_TB 
from ase.visualize import view
from ase import Atoms
from ase.io import read

MoS2 = read(filename="mos2.gen")
print(MoS2.get_chemical_symbols())
print(MoS2.get_positions())
print(MoS2.get_cell())
print(MoS2.get_pbc())
# view(MoS2)


max_l = {'C':1, 'H':0, 'S':2, 'Mo':2}
seedname_mos2 = Seedname_TB(MoS2, skpath="skfiles/self_made", maxl_dict=max_l, skpath_dipole="skfiles/self_made_dipole")
seedname_mos2.write_seedname()