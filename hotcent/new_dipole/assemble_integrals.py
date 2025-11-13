import numpy as np
import ase as ase
import pickle
import os
import sympy as sp
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
        Integrals.load_integral_list()
        Integrals.load_atom_pair(path)
        Integrals.load_sk_file_dipole(path)
        Integrals.calculate_dipole()

    """
    def __init__(self):
        if os.path.exists("symbolic_D_matrix.pkl"):
            print('Symbolic D matrix exists')
        else: 
            print('Calculate symbolic D-Matrix')            
            Wigner_D_real(euler_phi=PHI, euler_theta=THETA, euler_gamma=GAMMA)
        with open("symbolic_D_matrix.pkl", "rb") as f:
            M = pickle.load(f)
        self.D_symb = sp.lambdify((THETA, PHI, GAMMA), M, 'numpy') 

        if os.path.exists("identifier_nonzeros_overlap.pkl"):
            with open("identifier_nonzeros_overlap.pkl", 'rb') as f:
                quant_num_list = pickle.load(f)
                nonzeros = pickle.load(f)
        else:
            quant_num_list, nonzeros = get_index_list_overlap()
        self.quant_nums = quant_num_list
        self.sk_int_idx = nonzeros
    
    def load_integral_list(self):
        """additional list for nonvanishing dipole phi3 integrals"""
        if os.path.exists("identifier_nonzeros_dipole.pkl"):
            with open("identifier_nonzeros_dipole.pkl", 'rb') as f:
                quant_num_list = pickle.load(f)
                nonzeros = pickle.load(f)
        else:
            quant_num_list, nonzeros = get_index_list_dipole()
        self.quant_nums_dipole = quant_num_list
        self.sk_int_idx_dipole = nonzeros

    def load_atom_pair(self, path):
        """load position and basis functions to compute the integrals for"""
        atoms = ase.io.read(path)
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

    def load_sk_file(self, path):
        """.skf file for H and S"""
        with open(path, "r") as f:
            first_line = f.readline().strip()
            extended = 1 if first_line.startswith('@') else 0
            for i,line in enumerate(f, start=1):
                if i - extended == 0:
                    line1 = line.strip()
                    parts = [p.strip() for p in line1.split(', ')]
                    delta_R, n_points = float(parts[0]), int(parts[1])
                    break
        data = np.loadtxt(path, skiprows=3+extended)
        self.delta_R = delta_R
        self.n_points = n_points 
        self.sk_table_S = data[:, len(self.sk_int_idx):] 
        self.sk_table_H = data[:, :len(self.sk_int_idx)]

    def load_sk_file_dipole(self, path):
        """.skf file for dipole elements"""
        with open(path, "r") as f:
            first_line = f.readline().strip()
            extended = 1 if first_line.startswith('@') else 0
            for i,line in enumerate(f, start=1):
                if i-extended == 0:
                    line1 = line.strip()
                    parts = [p.strip() for p in line1.split(', ')]
                    delta_R, n_points = float(parts[0]), int(parts[1])
                    break
        data = np.loadtxt(path, skiprows=3+extended)
        self.delta_R_dipole = delta_R
        self.n_points_dipole = n_points 
        self.sk_table_dipole = data 

    def _set_rotation_matrix(self):
        print(f"euler angles: phi={self.euler_phi}, theta={self.euler_theta}, gamma={self.euler_gamma}") 
        self.D_single = np.array(self.D_symb(self.euler_theta, self.euler_phi, self.euler_gamma), dtype=complex)
        D1 = self.D_single
        D2 = self.D_single
        D = np.kron(D1, D2)
        self.D_full = np.real(D)

    def _set_rotation_matrix_dipole(self):
        print(f"euler angles: phi={self.euler_phi}, theta={self.euler_theta}, gamma={self.euler_gamma}") 
        self.D_single = np.array(self.D_symb(self.euler_theta, self.euler_phi, self.euler_gamma), dtype=complex)
        idx_pstart = 1
        idx_pend = 3
        D1 = self.D_single
        D2 = self.D_single
        D_r = self.D_single[idx_pstart:idx_pend+1, idx_pstart:idx_pend+1] # only take p for operator
        D = np.kron(D1, np.kron(D_r, D2))
        self.D_full_dipole = np.real(D)

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
        dipole_elements = self.D_full @ integral_vec
        return dipole_elements

    def calculate_dipole(self):
        self._set_rotation_matrix_dipole()
        overlap_elements = self.calculate(hamilton=False)

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
        return shifted_dipole
