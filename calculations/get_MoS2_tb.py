from hotcent.pos_op.tight_binding import SlaterKosterIntegrator 
from ase.visualize import view
from ase.build import mx2
from ase import Atoms
from ase.io import read, write
import sys

# MoS2 = read(filename="mos2.gen")
# MoS2.pbc = (True, True, False)
# print(MoS2.get_chemical_symbols())
# print(MoS2.get_positions())
# print(MoS2.get_cell())
# print(MoS2.get_pbc())
# view(MoS2)

MoS2 = mx2(vacuum=20)
MoS2.pbc = (True, True, True)
# print(MoS2.get_chemical_symbols())
print(MoS2.get_positions())
# print(MoS2.get_cell())
# print(MoS2.get_pbc())
# view(MoS2)


max_l = {'C':1, 'H':0, 'S':2, 'Mo':2}
seedname_mos2 = SlaterKosterIntegrator(MoS2, skpath="skfiles/self_made", maxl_dict=max_l, skpath_dipole="skfiles/self_made_dipole")
seedname_mos2.write_seedname()
seedname_mos2.write_seedname_momentum()
MoS2.write('mos2.cif')

