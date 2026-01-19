from hotcent.new_dipole.dftb_matrices import write_atom_matrix
from ase.build import graphene 

a = 2.58
c = 20.0
hBN = graphene('BN', a=a
                )
    
maxl_dict = {'N':1, 'B':1}
write_atom_matrix(structure=hBN, maxl_dict=maxl_dict, skdir='skfiles/self_made')