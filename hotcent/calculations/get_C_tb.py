from ase import Atoms
from ase.build import graphene
from ase.visualize import view
from ase.build import molecule
from hotcent.new_dipole.files_for_comparison import Seedname_TB


max_l = {'C':1, 'H':0, 'S':2, 'Mo':2}
graphene = graphene('CC', size=(1,1,1), vacuum=10)
lcao_graphene = Seedname_TB(graphene, skpath="skfiles/self_made_fine_grid", maxl_dict=max_l, skpath_dipole="skfiles/self_made_fine_grid_dipole")
lcao_graphene.write_seedname()