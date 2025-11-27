import numpy as np
import itertools
from ase import Atoms
from ase.build import graphene
from ase.visualize import view
from ase.build import molecule
from ase.neighborlist import *

class Calculation:
    def __init__(self, unit_cell, skpath):
        self.unit_cell = unit_cell
        self.abc = unit_cell.get_cell()
        self.atomtypes = unit_cell.get_chemical_symbols()
        print(self.atomtypes)
        self.skpath = skpath
        self._get_interaction_cutoffs()

    def _get_interaction_cutoffs(self):
        no_repeats_types = list(set(self.atomtypes))
        pairs = list(itertools.combinations_with_replacement(no_repeats_types, 2))
        cutoff_dict = {}
        for pair in pairs:
            pairpath = self.skpath+f'/{pair[0]}-{pair[1]}.skf'
            print(pairpath)
            with open(file=pairpath) as f:
                line1 = f.readline()
                line1 = line1.replace(',', ' ')
                line1 = line1.split()
                dr, Nr = float(line1[0]), int(line1[1])
            max_r = dr * Nr
            cutoff_dict[pair] = max_r
        self.cutoff_dict = cutoff_dict
    
    def calculate_bloch_matrix(self):
        atoms = self.unit_cell
        # print(atoms.neighbor_list('ij'))
                
                


cutoffs = {('C','C'): 1.85, ('H', 'C'): 1, 
           ('C', 'H'): 5,
             ('H', 'H'): 1}
graphene = graphene('CH', vacuum=10)
print(neighbor_list('ijS', a=graphene, cutoff=cutoffs, self_interaction=True))
benzene = molecule('C6H6')
lcao_graphene = Calculation(graphene, skpath="skfiles/skfiles_pbc")
lcao_graphene.calculate_bloch_matrix()