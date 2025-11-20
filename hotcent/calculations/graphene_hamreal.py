import numpy as np
from ase import Atoms 
from ase.build import graphene
from ase.neighborlist import *
from ase.visualize import view


def write_overreal(structure, orbdict):
    idx1, idx2 = neighbor_list(quantities='ij', a=structure, self_interaction=True, cutoff=10)
    atomtypes =structure.get_chemical_symbols()
    orbnumbers = [orbdict[key] for key in atomtypes]
    total_orbs = np.sum(orbnumbers)
    table = np.zeros((total_orbs, total_orbs))
    for i, firstatom in enumerate(idx1):
        for j, firstatom in enumerate(idx2):
            pass

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
        
    


g = graphene(size=(2,2,1), vacuum=10.0)
orbdict = {'C':4}
g.pbc = [False, False, False]
# cutoffs = natural_cutoffs(structure)
# print(neighbor_list(quantities='ijD', a=g, cutoff=3))
atoms = Atoms('C2', positions=[[0,0,0], [1,0,0]])
write_overreal(structure=atoms, orbdict=orbdict)



    
