from ase import Atoms
from ase.io import write
from ase.build import graphene
from ase.visualize import view
from ase.build import molecule
from hotcent.pos_op.tight_binding import SlaterKosterIntegrator


max_l = {'C':1, 'H':0, 'S':2, 'Mo':2}
graphene = graphene('CC', size=(1,1,1), vacuum=10)
print(graphene.get_cell())
# view(graphene)
lcao_graphene = SlaterKosterIntegrator(graphene, 
                                       skpath="sk_experimental/sk_unique", 
                                       maxl_dict=max_l, 
                                       skpath_posop="sk_experimental/sk_posop_unique", 
                                       format='unique')
lcao_graphene.write_seedname()
# lcao_graphene.write_seedname_momentum()
#graphene.write('graphene.cif')