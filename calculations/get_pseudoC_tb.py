from ase import Atoms
from ase.visualize import view
from hotcent.new_dipole.files_for_comparison import Seedname_TB

pseudo_carbon = Atoms('C', positions=[(0,0,0)],
                      cell=[2, 2, 20],
                      pbc=[1,1,1])
# view(pseudo_carbon)
max_l = {'C':1, 'H':0, 'S':2, 'Mo':2}
lcao_pseudocarbon = Seedname_TB(pseudo_carbon, skpath="skfiles/self_made", maxl_dict=max_l, skpath_dipole="skfiles/self_made_dipole")
lcao_pseudocarbon.write_seedname()
lcao_pseudocarbon.write_seedname_momentum()