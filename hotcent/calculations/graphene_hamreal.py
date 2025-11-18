import numpy as np
from ase import Atom
from ase.build import graphene
from ase.neighborlist import *
from ase.visualize import view

def write_overreal(structure, orb_list):
    cutoffs = np.ones((len(structure))) * 10
    nl = NeighborList(cutoffs=cutoffs, self_interaction=True, bothways=True)
    nl.update(structure)
    for i, pos in enumerate(structure.get_positions()):
        Nneighbors = len(nl.get_neighbors(i)[0])
        print(f"{i}\t{Nneighbors}")

n_basis_atom = 4

g = graphene(size=(2,2,1), vacuum=10.0)
orb_list = np.zeros((len(g),))
for i, pos in enumerate(g.get_positions()):
    orb_list[i] = n_basis_atom
g.pbc = [False, False, False]
# cutoffs = natural_cutoffs(structure)
# print(neighbor_list(quantities='ijD', a=g, cutoff=3))
write_overreal(structure=g, orb_list=orb_list)



    
