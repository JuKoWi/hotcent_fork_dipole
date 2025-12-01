import numpy as np
import ase as ase
import pickle
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import sympy as sym
from scipy.interpolate import CubicSpline
from hotcent.new_dipole.utils import *
from hotcent.new_dipole.integrals import get_index_list_dipole, get_index_list_overlap
from hotcent.new_dipole.rotation_transform import Wigner_D_real, to_spherical, PHI, THETA, GAMMA
from hotcent.new_dipole.slako_dipole import INTEGRALS_DIPOLE
from hotcent.new_dipole.slako_new import INTEGRALS

"""one single class with methods for all relevant quantities"""

class SK_Integral:
    """calculate S,H or d matrix elements between two atoms 
    based on position and SK-table 
    usage for hamiltonian, overlap:
        Integrals = SK_Integral()
        Integrals.load_atom_pair(path)
        Integrals.load_sk_file(path)
        Integrals.calculate(hamilton->bool)
    usage for dipole:
        Integrals = SK_Integral()
        Integrals.get_list_dipole()
        Integrals.load_atom_pair(path)
        Integrals.load_sk_file_dipole(path)
        Integrals.calculate_dipole()

    """
    def __init__(self):
        if os.path.exists("symbolic_D_matrix.pkl"):
            pass
            # print('Symbolic D matrix exists')
        else: 
            # print('Calculate symbolic D-Matrix')            
            Wigner_D_real(euler_phi=PHI, euler_theta=THETA, euler_gamma=GAMMA)
        with open("symbolic_D_matrix.pkl", "rb") as f:
            M = pickle.load(f)
        self.D_symb = sym.lambdify((THETA, PHI, GAMMA), M, 'numpy') 

        if os.path.exists("identifier_nonzeros_overlap.pkl"):
            with open("identifier_nonzeros_overlap.pkl", 'rb') as f:
                quant_num_list = pickle.load(f)
                nonzeros = pickle.load(f)
        else:
            quant_num_list, nonzeros = get_index_list_overlap()
        self.quant_nums = quant_num_list
        self.sk_int_idx = nonzeros
    
    def get_list_dipole(self):
        """additional list for nonvanishing dipole phi3 integrals"""
        if os.path.exists("identifier_nonzeros_dipole.pkl"):
            with open("identifier_nonzeros_dipole.pkl", 'rb') as f:
                quant_num_list = pickle.load(f)
                nonzeros = pickle.load(f)
        else:
            quant_num_list, nonzeros = get_index_list_dipole()
        self.quant_nums_dipole = quant_num_list
        self.sk_int_idx_dipole = nonzeros

    def load_atom_file(self, path):
        """load position and basis functions to compute the integrals for"""
        atoms = ase.io.read(path)
        self.R_vec = angstrom_to_bohr(atoms.get_distance(0, 1, vector=True)) 
        self.R = angstrom_to_bohr(atoms.get_distance(0,1, vector=False))
        atom1 = angstrom_to_bohr(atoms.get_positions()[0])
        self.atom1_pos = np.array([atom1[1], atom1[2], atom1[0]]) # order consistent with quantum numbers
        self._set_euler_angles()
    
    def load_atom_pair(self, atoms):
        """load a single pair as an ase.atoms object"""
        self.R_vec = angstrom_to_bohr(atoms.get_distance(0, 1, vector=True)) 
        self.R = angstrom_to_bohr(atoms.get_distance(0,1, vector=False))
        atom1 = angstrom_to_bohr(atoms.get_positions()[0])
        self.atom1_pos = np.array([atom1[1], atom1[2], atom1[0]]) # order consistent with quantum numbers
        self._set_euler_angles()
        

    def _set_euler_angles(self):
        """use only two rotations of three possible"""
        if np.all(self.R_vec == 0):
            self.euler_theta = 0
            self.euler_phi = 0
            self.euler_gamma = 0 
        else:
            R_spherical = to_spherical(R=self.R_vec)
            self.euler_theta = - R_spherical[1] # rotate back on z-axis
            self.euler_phi =  - R_spherical[2] # rotate back on z-axis
            self.euler_gamma = 0

    def load_sk_file(self, path, homonuclear):
        """.skf file for H and S"""
        myfile = Path(path)
        assert myfile.is_file()
        with open(path, "r") as f:
            # print('opened file')
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
        self.delta_R = delta_R
        self.n_points = n_points 
        self.sk_table_S = data[:, len(self.sk_int_idx):] 
        self.sk_table_H = data[:, :len(self.sk_int_idx)]

    def load_sk_file_dipole(self, path, path_dipole):
        """.skf file for dipole elements"""
        self.load_sk_file(path=path)
        with open(path_dipole, "r") as f:
            first_line = f.readline().strip()
            extended = 1 if first_line.startswith('@') else 0
            for i,line in enumerate(f, start=1):
                if i-extended == 0:
                    line1 = line.strip()
                    parts = [p.strip() for p in line1.split(', ')]
                    delta_R, n_points = float(parts[0]), int(parts[1])
                    break
        data = np.loadtxt(path_dipole, skiprows=3+extended)
        self.delta_R_dipole = delta_R
        self.n_points_dipole = n_points 
        self.sk_table_dipole = data 

    def _set_rotation_matrix(self):
        # print(f"euler angles: phi={self.euler_phi}, theta={self.euler_theta}, gamma={self.euler_gamma}") 
        self.D_single = np.array(self.D_symb(self.euler_theta, self.euler_phi, self.euler_gamma), dtype=complex)
        D1 = self.D_single
        D2 = self.D_single
        D = np.kron(D1, D2)
        self.D_full = np.real(D)

    def _set_rotation_matrix_dipole(self):
        self.D_single = np.array(self.D_symb(self.euler_theta, self.euler_phi, self.euler_gamma), dtype=complex)
        idx_pstart = 1
        idx_pend = 3
        D1 = self.D_single
        D2 = self.D_single
        D_r = self.D_single[idx_pstart:idx_pend+1, idx_pstart:idx_pend+1] # only take p for operator
        D = np.kron(D1, np.kron(D_r, D2))
        self.D_full_dipole = np.real(D)
        # print(np.all(np.isclose(self.D_full_dipole, np.eye(np.shape(self.D_full_dipole)[0]))))

    def calculate(self, hamilton=False):
        """hamilton decides if hamiltonian is computed instead of overlap"""
        self._set_rotation_matrix()
        R_grid = self.delta_R + self.delta_R * np.arange(self.n_points) 
        if hamilton:
            cs = CubicSpline(R_grid, self.sk_table_H) 
        else:
            cs = CubicSpline(R_grid, self.sk_table_S) 
        integral_vec = np.zeros((len(self.quant_nums)))
        for i, key in enumerate(sorted(INTEGRALS, key= lambda x: x[0])):
            integral_vec[key[0]] = cs(self.R)[i]
        integrals = self.D_full @ integral_vec

        # x = np.arange(stop=np.max(R_grid), step=0.01)
        # plt.plot(x, cs(x)[:,1])
        # if hamilton:
        #     plt.scatter(R_grid, self.sk_table_H[:,1])
        # else:
        #     plt.scatter(R_grid, self.sk_table_S[:,1])
        # plt.show()


        if hamilton:
            self.H_vec = integrals
            H_dict = {}
            for label in self.quant_nums:
                H_dict[(label[1], label[2], label[3], label[4])] = integrals[label[0]]
            self.H_dict = H_dict
        else:
            self.S_vec = integrals
            S_dict = {}
            for label in self.quant_nums:
                S_dict[(label[1],label[2],label[3],label[4])] = integrals[label[0]]
            self.S_dict = S_dict
        return integrals

    def calculate_dipole(self):
        self._set_rotation_matrix_dipole()
        overlap_elements = self.calculate(hamilton=False)
        self.S_vec = overlap_elements
        R_grid_dipole = self.delta_R_dipole + self.delta_R_dipole * np.arange(self.n_points_dipole) 
        cs_dipole = CubicSpline(R_grid_dipole, self.sk_table_dipole) 
        integral_vec_dipole = np.zeros((len(self.quant_nums_dipole)))
        for i, key in enumerate(sorted(INTEGRALS_DIPOLE, key= lambda x: x[0])):
            integral_vec_dipole[key[0]] = cs_dipole(self.R)[i]
        dipole_elements = self.D_full_dipole @ integral_vec_dipole

        overlap_blocks = overlap_elements.reshape(16,16)
        shift_term = np.tile(overlap_blocks, (1,3)).reshape(-1)
        space_factor = np.tile(np.repeat(self.atom1_pos, 16), 16)
        shift_term = shift_term * space_factor
        shifted_dipole = dipole_elements + shift_term
        self.d_vec = shifted_dipole
        return shifted_dipole
    
    def select_matrix_elements(self, max_lA, max_lB):
        """returns 2D array of matrix elements"""
        pair_overlap_matrix = np.zeros((get_norbs(max_lA), get_norbs(max_lB)))        
        pair_hamiltonian_matrix = np.zeros((get_norbs(max_lA), get_norbs(max_lB)))        
        row_start = 0
        col_start = 0
        for l1 in range(max_lA + 1):
            size_row = 2 * l1 +1
            for l2 in range(max_lB + 1):
                size_col = 2 * l2 +1
                block_overlap = np.zeros((size_row, size_col))
                block_hamiltonian = np.zeros((size_row, size_col))
                for mi, m in enumerate(range(-l1, l1 + 1)):
                    for ni, n in enumerate(range(-l2, l2 + 1)):
                        block_overlap[mi, ni] = self.S_dict[(l1, m, l2, n)]
                        block_hamiltonian[mi, ni] = self.H_dict[(l1, m, l2, n)]
                pair_overlap_matrix[row_start:row_start+size_row, col_start:col_start+size_col] = block_overlap
                pair_hamiltonian_matrix[row_start:row_start+size_row, col_start:col_start+size_col] = block_hamiltonian 
                col_start += size_col
            col_start = 0
            row_start += size_row
        return pair_overlap_matrix, pair_hamiltonian_matrix
        




        
