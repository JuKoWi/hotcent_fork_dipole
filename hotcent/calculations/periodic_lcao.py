import numpy as np
import pickle
import os
from hotcent.new_dipole.rotation_transform import Wigner_D_real, to_spherical, PHI, THETA, GAMMA
from pathlib import Path
import itertools
import sympy as sp
# from scipy.interpolate import CubicSpline
from ase import Atoms
from ase.build import graphene
from ase.visualize import view
from ase.build import molecule
from ase.neighborlist import *
from hotcent.new_dipole.utils import *
from hotcent.new_dipole.integrals import get_index_list_dipole, get_index_list_overlap

class Seedname_TB:
    def __init__(self, unit_cell, skpath, maxl_dict):
        self.ucell = unit_cell
        self.abc = unit_cell.get_cell()
        self.atomtypes = unit_cell.get_chemical_symbols()
        self.skpath = skpath
        no_repeats_types = list(set(self.atomtypes))
        self.elem_pairs_unordered = list(itertools.combinations_with_replacement(no_repeats_types, 2))
        self.elem_pairs = list(itertools.product(no_repeats_types, repeat=2))
        self._get_interaction_cutoffs()
        self.maxl_dict = maxl_dict
        orbnumbers = [get_norbs(maxl_dict[key]) for key in self.atomtypes]
        self.total_orbs = np.sum(orbnumbers)

        if os.path.exists("identifier_nonzeros_overlap.pkl"):
            with open("identifier_nonzeros_overlap.pkl", 'rb') as f:
                quant_num_list = pickle.load(f)
                nonzeros = pickle.load(f)
        else:
            quant_num_list, nonzeros = get_index_list_overlap()
        self.quant_nums = quant_num_list
        self.sk_int_idx = nonzeros

        if os.path.exists("symbolic_D_matrix.pkl"):
            pass
            print('Symbolic D matrix exists')
        else: 
            print('Calculate symbolic D-Matrix')            
            Wigner_D_real(euler_phi=PHI, euler_theta=THETA, euler_gamma=GAMMA)
        with open("symbolic_D_matrix.pkl", "rb") as f:
            M = pickle.load(f)
        print(M.__class__)
        M = sp.Matrix(M)
        self.D_symb = sp.lambdify((THETA, PHI, GAMMA), M, 'numpy') 

    def _calculate_lattice_dict(self):
        """create a dictionary with lattice vectors as keys. 
            for every lattice vector the value is a matrix that describes the overlap 
            between the orbitals in the unit cell and the orbitals in the unit cell shifted 
            by the respective lattice vector
        """
        atoms = self.ucell
        S_dict_lattice = {}
        pairA, pairB, R = neighbor_list('ijS', a=atoms, cutoff=self.cutoff_dict) 
        for i, pair in enumerate(pairA):
            R_triple = (int(R[i,0]), int(R[i,1]), int(R[i,2]))
            matrix = S_dict_lattice.setdefault(R_triple, np.zeros((self.total_orbs, self.total_orbs)))
            idxA = pairA[i]
            idxB = pairB[i]
            typeA = atoms.symbols[idxA]
            typeB = atoms.symbols[idxB]
            maxlA = self.maxl_dict[typeA]
            maxlB = self.maxl_dict[typeB]
            n_rows = get_norbs(maxl=maxlA)
            n_cols = get_norbs(maxl=maxlB)
            posA = atoms.positions[idxA]
            posB = atoms.positions[idxB] + np.dot(self.abc, R[i])
            #TODO: Write block between two atoms
        assert np.shape(np.unique(R, axis=0))[0] == len(S_dict_lattice)

    def _calculate_atom_block(self, types, posA, posB, operator):
        """for two atoms, calculate the block of all relevant orbitals"""
        if operator == 'S':
            sk_table = self.S_sk_tables[types]
        elif operator == 'H':
            sk_table = self.H_sk_tables[types]
        elif operator == 'r':
            raise NotImplementedError("Dipole not supported yet")
        

    def _get_interaction_cutoffs(self):
        #TODO: Write assert to make sure that cutoff is direction invariant
        cutoff_dict = {}
        for pair in self.elem_pairs_unordered:
            pairpath = self.skpath+f'/{pair[0]}-{pair[1]}.skf'
            # print(pairpath)
            with open(file=pairpath) as f:
                line1 = f.readline()
                line1 = line1.replace(',', ' ')
                line1 = line1.split()
                dr, Nr = float(line1[0]), int(line1[1])
            max_r = dr * Nr
            cutoff_dict[pair] = max_r
        self.cutoff_dict = cutoff_dict
    
    def _create_SH_dict(self):
        """create dictionaries for Slater Koster tables"""
        S_sk_dict = {}
        H_sk_dict = {}
        for element_comb in self.elem_pairs:
            delta_R, n_points, S, H = self._read_sk_file(elem_pair=element_comb, dipole=False)
            S_sk_dict[element_comb] = SKTable(table=S, deltaR=delta_R, n_points=n_points)
            H_sk_dict[element_comb] = SKTable(table=S, deltaR=delta_R, n_points=n_points)
        self.S_sk_tables = S_sk_dict
        self.H_sk_tables = H_sk_dict

    
    def _read_sk_file(self, elem_pair, dipole=False):
        """returns dr, Nr and the table(s) for a .skf file or the dipole equivalent"""
        path = self.skpath + f"/{elem_pair[0]}-{elem_pair[1]}.skf" #write conditional for dipole
        myfile = Path(path)
        assert myfile.is_file()
        homonuclear = (elem_pair[0] == elem_pair[1])
        with open(path, "r") as f:
            first_line = f.readline().strip()
            extended = 1 if first_line.startswith('@') else 0
            if extended == 0:
                first_line = first_line.replace(',', ' ')
                parts = [p.strip() for p in first_line.split()]
            if extended == 1:
                line2 = f.readline()
                line2 = line2.replace(',', ' ')
                parts = [p.strip() for p in line2.split()]
            delta_R, n_points = float(parts[0]), int(parts[1])
        if not homonuclear:
            extended -= 1
        data = np.loadtxt(path, skiprows=3+extended)
        if dipole:
            pass
        else:
            sk_table_S = data[:, len(self.sk_int_idx):] 
            sk_table_H = data[:, :len(self.sk_int_idx)]
        assert np.shape(data)[0] == n_points
        return delta_R, n_points, sk_table_S, sk_table_H
        
    
    def write_seedname(self):
        filename = 'seedname_tb.dat'
        with open(filename, 'w') as f:
            f.write("Date\n")
            np.savetxt(f, self.abc)

class SKTable:
    """object to store all information from .skf file for one physical quantity 
        (S, H or r)
    """
    def __init__(self, table, deltaR, n_points):
        self.table = table
        self.deltaR = deltaR
        self.n_points = n_points



            


cutoffs = {('C','C'): 1.85, ('H', 'C'): 1, 
           ('C', 'H'): 5,
             ('H', 'H'): 1}
max_l = {'C':1, 'H':0}
graphene = graphene('CH', size=(1,1,1), vacuum=10)
# print(graphene.get_pbc())
# print(neighbor_list('ijS', a=graphene, cutoff=cutoffs, self_interaction=True))
benzene = molecule('C6H6')
lcao_graphene = Seedname_TB(graphene, skpath="skfiles/skfiles_pbc", maxl_dict=max_l)
lcao_graphene._create_SH_dict()
lcao_graphene._calculate_lattice_dict()
lcao_graphene._calculate_atom_block(types=('C','H'), posA=[0,0,0], posB=[1,1,1], operator='H')
print(lcao_graphene.S_sk_tables)