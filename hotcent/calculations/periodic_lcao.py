import numpy as np
import pickle
import os
import sys
sys.path.append("/home/julius/university/uni_jena/master_thesis/hotcent_fork_dipole")
from hotcent.new_dipole.rotation_transform import Wigner_D_real, to_spherical, PHI, THETA, GAMMA
from pathlib import Path
import itertools
import sympy as sym 
from scipy.interpolate import CubicSpline
from ase import Atoms
from ase.build import graphene
from ase.visualize import view
from ase.build import molecule
from ase.neighborlist import *
from hotcent.new_dipole.utils import *
from hotcent.new_dipole.integrals import get_index_list_dipole, get_index_list_overlap
from hotcent.new_dipole.slako_dipole import INTEGRALS_DIPOLE
from hotcent.new_dipole.slako_new import INTEGRALS

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
        self.orbnumbers = [get_norbs(maxl_dict[key]) for key in self.atomtypes]
        self.total_orbs = np.sum(self.orbnumbers)

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
        self.D_symb = sym.lambdify((THETA, PHI, GAMMA), M, 'numpy') 

        self._create_SH_dict()

    def _get_interaction_cutoffs(self):
        #TODO: Write assert to make sure that cutoff is direction invariant
        cutoff_dict = {}
        for pair in self.elem_pairs_unordered:
            pairpath = self.skpath+f'/{pair[0]}-{pair[1]}.skf'
            with open(file=pairpath) as f:
                line1 = f.readline()
                line1 = line1.replace(',', ' ')
                line1 = line1.split()
                dr, Nr = float(line1[0]), int(line1[1])
            max_r = dr * Nr
            cutoff_dict[pair] = max_r
        self.cutoff_dict = cutoff_dict

    def _read_sk_file(self, elem_pair, dipole=False):
        """returns dr, Nr and the table(s) for a .skf file or the dipole equivalent"""
        path = self.skpath + f"/{elem_pair[0]}-{elem_pair[1]}.skf" #write conditional for dipole
        myfile = Path(path)
        assert myfile.is_file()
        homonuclear = (elem_pair[0] == elem_pair[1])
        with open(path, "r") as f:
            first_line = f.readline().strip()
            first_line = first_line.replace(',', ' ')
            line2 = f.readline()
            line2 = line2.replace(',', ' ')
            extended = 1 if first_line.startswith('@') else 0
            if extended == 0:
                parts = [p.strip() for p in first_line.split()]
                eigvals = line2.split()
                eigvals = np.flip(eigvals[:3])
            if extended == 1:
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
        return delta_R, n_points, sk_table_S, sk_table_H, eigvals
        
    def _create_SH_dict(self):
        """create a dictionary where for every element combiantion there is a custom object,
        that contains all the information from the .skf file
        """
        S_sk_dict = {}
        H_sk_dict = {}
        for element_comb in self.elem_pairs:
            delta_R, n_points, S, H, eigvals  = self._read_sk_file(elem_pair=element_comb, dipole=False)
            S_sk_dict[element_comb] = SKTable(table=S, deltaR=delta_R, n_points=n_points, diag_vals=[1,1,1]) #assume the atomic functions to be orthonormal
            H_sk_dict[element_comb] = SKTable(table=H, deltaR=delta_R, n_points=n_points, diag_vals=eigvals)
        self.S_sk_tables = S_sk_dict
        self.H_sk_tables = H_sk_dict

    def _set_euler_angles(self, vec1, vec2):
        """use only two rotations of three possible"""
        R_vec = vec2 - vec1
        if np.all(R_vec == 0):
            euler_theta = 0
            euler_phi = 0
            euler_gamma = 0 
        else:
            R_spherical = to_spherical(R=R_vec)
            euler_theta = - R_spherical[1] # rotate back on z-axis
            euler_phi =  - R_spherical[2] # rotate back on z-axis
            euler_gamma = 0
        return euler_theta, euler_phi, euler_gamma

    def _create_integral_dict(self, sk_table, posA, posB, operator):
        """for one set of atom positions and a certain operator (S,H, r) create a dictionary for with quantum numbers as keys"""
        R_vec = posB - posA
        R = np.linalg.norm(R_vec)
        euler_theta, euler_phi, euler_gamma= self._set_euler_angles(vec1=posA, vec2=posB)
        D_single = np.array(self.D_symb(euler_theta, euler_phi, euler_gamma), dtype=complex)
        D = np.kron(D_single, D_single)
        D = np.real(D)

        deltaR = sk_table.deltaR
        table = sk_table.table
        n_points = sk_table.n_points

        R_grid = deltaR + deltaR * np.arange(n_points)
        cs = CubicSpline(R_grid, table)
        integral_vec = np.zeros((len(self.quant_nums))) # TODO: for dipole moment create if statement
        for i, key in enumerate(sorted(INTEGRALS, key= lambda x: x[0])):
            integral_vec[key[0]] = cs(R)[i]
        integrals = D @ integral_vec
        integral_dict = {}
        for label in self.quant_nums:
            integral_dict[(label[1], label[2], label[3], label[4])] = integrals[label[0]]
        return integral_dict
    
    def _select_matrix_elements(self, max_lA, max_lB, integral_dict):
        pair_matrix = np.zeros((get_norbs(max_lA), get_norbs(max_lB)))
        row_start = 0
        col_start = 0 
        for l1 in range(max_lA + 1):
            size_row = 2 * l1 +1
            for l2 in range(max_lB + 1):
                size_col = 2 * l2 +1
                block = np.zeros((size_row, size_col))
                for mi, m in enumerate(range(-l1, l1 + 1)):
                    for ni, n in enumerate(range(-l2, l2 + 1)):
                        block[mi, ni] = integral_dict[(l1, m, l2, n)]
                pair_matrix[row_start:row_start+size_row, col_start:col_start+size_col] = block
                col_start += size_col
            col_start = 0
            row_start += size_row
        return pair_matrix

    def _calculate_atom_block(self, types, posA, posB, max_lA, max_lB, operator):
        """for two atoms, calculate the block of all relevant orbitals"""
        same_atom = np.allclose(posA, posB)
        if same_atom:
            assert types[0] == types[1]
        if operator == 'S':
            sk_table = self.S_sk_tables[types]
            if same_atom:
                block = np.eye(N=get_norbs(maxl=max_lA))
            else:
                integral_dict = self._create_integral_dict(sk_table=sk_table, posA=posA, posB=posB, operator=operator)
                block = self._select_matrix_elements(max_lA=max_lA, max_lB=max_lB, integral_dict=integral_dict)
        elif operator == 'H':
            sk_table = self.H_sk_tables[types]
            if same_atom:
                eigenvalues = sk_table.diag_vals
                diag = np.eye(get_norbs(maxl=max_lA))
                count = 0
                for i in range(max_lA+1):
                    for j in range(2*i+1):
                        diag[count+j, count+j] = eigenvalues[i]
                    count += 2*i +1
                block = diag
            else:
                integral_dict = self._create_integral_dict(sk_table=sk_table, posA=posA, posB=posB, operator=operator)
                block = self._select_matrix_elements(max_lA=max_lA, max_lB=max_lB, integral_dict=integral_dict)
        elif operator == 'r':
            raise NotImplementedError("Dipole not supported yet")
        return block

    def _calculate_lattice_dict(self, operator):
        """create a dictionary with lattice vectors as keys. 
            for every lattice vector the value is a matrix that describes the overlap 
            between the orbitals in the unit cell and the orbitals in the unit cell shifted 
            by the respective lattice vector
        """
        atoms = self.ucell
        lattice_dict = {}
        pairA, pairB, R = neighbor_list('ijS', a=atoms, cutoff=self.cutoff_dict, self_interaction=True) 
        for i, pair in enumerate(pairA):
            R_triple = (int(R[i,0]), int(R[i,1]), int(R[i,2]))
            matrix = lattice_dict.setdefault(R_triple, np.zeros((self.total_orbs, self.total_orbs)))
            idxA = pairA[i]
            idxB = pairB[i]
            typeA = atoms.symbols[idxA]
            typeB = atoms.symbols[idxB]
            maxlA = self.maxl_dict[typeA]
            maxlB = self.maxl_dict[typeB]
            posA = angstrom_to_bohr(atoms.positions[idxA])
            posB = angstrom_to_bohr(atoms.positions[idxB] + np.dot(self.abc, R[i]))
            block = self._calculate_atom_block(types=(typeA, typeB), posA=posA, posB=posB, max_lA=maxlA, max_lB=maxlB, operator=operator)
            n_rows = get_norbs(maxl=maxlA)
            n_cols = get_norbs(maxl=maxlB)
            start_rows = self._find_block_pos(idx=idxA)
            start_cols = self._find_block_pos(idx=idxB)
            matrix[start_rows:start_rows+n_rows, start_cols:start_cols+n_cols] = block
            lattice_dict[R_triple] = matrix
        assert np.shape(np.unique(R, axis=0))[0] == len(lattice_dict)
        return lattice_dict
    
    def _find_block_pos(self, idx):
        orb_previous = np.sum(self.orbnumbers[:idx])
        return int(orb_previous)
    
    def write_seedname(self):
        filename = 'seedname_tb.dat'
        with open(filename, 'w') as f:
            f.write("Date\n")
            np.savetxt(f, self.abc)
    
    def write_hamoversqr(self):
        matrix = self._calculate_lattice_dict(operator='S')
        matrixH = self._calculate_lattice_dict(operator='H')
        header1 = f"#\tREAL\tNALLORB\tNKPOINT\n"
        header2 = f"\tT\t{self.total_orbs}\t1\n" 
        header3 = "#IKPOINT\n"
        header4 = "\t1\n"
        header5 = "#MATRIX"
        header = header1 + header2 + header3 + header4 + header5
        np.savetxt(fname='oversqr_hotcent.dat', delimiter='\t', fmt='%+.18e', X=matrix, header=header, comments='')
        np.savetxt(fname='hamsqr1_hotcent.dat', delimiter='\t', fmt='%+.18e', X=matrixH, header=header, comments='')

        


class SKTable:
    """object to store all information from .skf file for one physical quantity 
        (S, H or r)
    """
    def __init__(self, table, deltaR, n_points, diag_vals=None):
        self.table = table
        self.deltaR = deltaR
        self.n_points = n_points
        self.diag_vals = diag_vals


cutoffs = {('C','C'): 1.85, ('H', 'C'): 1, 
           ('C', 'H'): 5,
             ('H', 'H'): 1}
max_l = {'C':1, 'H':0}
graphene = graphene('CC', size=(1,1,1), vacuum=10)
benzene = molecule('C6H6')

lcao_graphene = Seedname_TB(graphene, skpath="skfiles/skfiles_pbc", maxl_dict=max_l)
lattice_dict_S = lcao_graphene._calculate_lattice_dict(operator='S')
print(lattice_dict_S[(0,0,0)])