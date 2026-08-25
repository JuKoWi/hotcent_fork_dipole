from ase import Atoms
from ase.visualize import view
from hotcent.pos_op.tight_binding import SlaterKosterIntegrator

pseudo_carbon = Atoms('C', positions=[(0,0,0)],
                      cell=[2, 2, 20],
                      pbc=[1,1,1])
# view(pseudo_carbon)
max_l = {'C':1, 'H':0, 'S':2, 'Mo':2}
lcao_pseudocarbon = SlaterKosterIntegrator(pseudo_carbon, skpath="skfiles/sk_conventional", maxl_dict=max_l, skpath_posop="skfiles/sk_posop", conventional_skf=True)
lcao_pseudocarbon.write_seedname()