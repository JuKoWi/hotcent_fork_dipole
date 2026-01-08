from ase import Atoms 
from ase.build import graphene
from ase.build import molecule
from ase.visualize import view
from hotcent.new_dipole.dftb_matrices import write_atom_matrix



max_l_dict = {'H':0, 'C':1, 'Fe':2}

# g = graphene(size=(2,2,1), vacuum=10.0)
# g.pbc = [False, False, False]
# cutoffs = natural_cutoffs(structure)
# print(neighbor_list(quantities='ijD', a=g, cutoff=3))
# atoms = Atoms('HC', positions=[[0,0,0], [0, 0, bohr_to_angstrom(0.4)]])
atoms = molecule('C6H6')
write_atom_matrix(structure=atoms, maxl_dict=max_l_dict, skdir="./skfiles/skfiles_pbc")



    
