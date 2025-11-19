import numpy as np
from ase import atoms
from ase.build import graphene
from ase.neighborlist import *
from ase.visualize import view

def write_hamsquare(structure):
    

def write_overreal(structure, orb_list):
    cutoffs = np.ones((len(structure))) * 10
    nl = NeighborList(cutoffs=cutoffs, self_interaction=True, bothways=True)
    nl.update(structure)
    iatom2f = find_central_equivaltens(structure)
    for iatom in range(len(structure)):
        neighbor_idx, offsets = nl.get_neighbors(iatom)
        nneigh = len(neighbor_idx)
        print(f"{iatom}\t{nneigh}\t{orb_list[iatom]}")
    for iatom in range(len(structure)):
        neighbor_idx, offsets = nl.get_neighbors(iatom)
        for ineigh, offset in zip(neighbor_idx, offsets):
            image_ucell = iatom2f[ineigh]
            print(f"{iatom}\t{ineigh}\t{image_ucell}")

def find_central_equivaltens(atoms, tol=1e-12):
    frac = atoms.get_scaled_positions()
    wrapped = frac % 1
    central_index = []
    for i in range(len(atoms)):
        target = wrapped[i]
        diff = np.abs(wrapped-target)
        diff = np.min(diff, 1-diff)
        j = np.where(np.all(diff < tol, axis=1))[0][0]
        central_index.append(j)
    return central_index
        
    


n_basis_atom = 4

g = graphene(size=(2,2,1), vacuum=10.0)
orb_list = np.zeros((len(g),))
for i, pos in enumerate(g.get_positions()):
    orb_list[i] = n_basis_atom
g.pbc = [False, False, False]
# cutoffs = natural_cutoffs(structure)
# print(neighbor_list(quantities='ijD', a=g, cutoff=3))
write_overreal(structure=g, orb_list=orb_list)



    
