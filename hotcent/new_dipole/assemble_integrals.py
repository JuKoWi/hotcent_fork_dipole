import sys
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
# np.set_printoptions(threshold=np.inf)

class SK_Integral:
    """calculate S,H or d matrix elements between two atoms 
    only needs position, SK-table and orbital pairs to calculate integrals between
    """
    def __init__(self):
        if os.path.exists("symbolic_D_matrix.pkl"):
            print('Symbolic D matrix exists')
        else: 
            print('Calculate symbolic D-Matrix')            
            Wigner_D_real(euler_phi=PHI, euler_theta=THETA, euler_gamma=GAMMA)
        with open("symbolic_D_matrix.pkl", "rb") as f:
            M = pickle.load(f)
        self.Wigner_D_symb = sp.lambdify((THETA, PHI, GAMMA), M, 'numpy') 

    def load_atom_pair(self, path):
        """load position and basis functions to compute the integrals for"""
        atoms = ase.io.read(path)
        self.R_vec = angstrom_to_bohr(atoms.get_distance(0, 1, vector=True))
        self.r = angstrom_to_bohr(atoms.get_distance(0,1, vector=False))

    def set_euler_angles(self):
        if np.all(self.R_vec == 0):
            self.euler_theta = 0
            self.euler_phi = 0
            self.euler_gamma = 0 
        else:
            R_spherical = to_spherical(R=self.R_vec)
            self.euler_theta = - R_spherical[1] # rotate back on z-axis
            self.euler_phi =  - R_spherical[2] # rotate back on z-axis
            self.euler_gamma = 0

class SK_Integral_Dipole(SK_Integral):
    """subclass for d elements"""
    def __init__(self):
        super().__init__()
        if os.path.exists("identifier_nonzeros_dipole.pkl"):
            with open("identifier_nonzeros_dipole.pkl", 'rb') as f:
                quant_num_list = pickle.load(f)
                nonzeros = pickle.load(f)
        else:
            quant_num_list, nonzeros = get_index_list_dipole()
        self.quant_num_list_dipole = quant_num_list
        self.nonzero_idx_dipole = nonzeros


    def load_SK_dipole_file(self, path):
        with open(path, "r") as f:
            first_line = f.readline().strip()
            extended = 1 if first_line.startswith('@') else 0
        with open(path, 'r') as f:
            for i,line in enumerate(f):
                if i-extended == 0:
                    line1 = line.strip()
                    parts = [p.strip() for p in line1.split(', ')]
                    delta_R, n_points = float(parts[0]), int(parts[1])
        data = np.loadtxt(path, skiprows=3+extended)
        self.delta_R = delta_R
        self.n_points = n_points 
        self.sk_table = data 


    def choose_relevant_matrix(self):
        print(f"euler angles: phi={self.euler_phi}, theta={self.euler_theta}, gamma={self.euler_gamma}") 
        self.Wigner_D_single = np.array(self.Wigner_D_symb(self.euler_theta, self.euler_phi, self.euler_gamma), dtype=complex)
        # self.Wigner_D_single = np.where(np.abs(self.Wigner_D_single) < 1e-15, 0, self.Wigner_D_single) 
        idx_pstart = 1
        idx_pend = 3
        D1 = self.Wigner_D_single
        D2 = self.Wigner_D_single
        D_r = self.Wigner_D_single[idx_pstart:idx_pend+1, idx_pstart:idx_pend+1] # only take p for operator
        D = np.kron(D1, np.kron(D_r, D2))
        self.Wigner_D_full = np.real(D)

    def calculate_dipole(self):
        """for every pair of basis functions calculate three components of dipole function
        for n basis gives (n,n,3) array"""
        R_grid = self.delta_R + self.delta_R * np.arange(self.n_points) 
        cs = CubicSpline(R_grid, self.sk_table) 
        integral_vec = np.zeros((len(self.quant_num_list_dipole)))
        for i, key in enumerate(sorted(INTEGRALS_DIPOLE, key= lambda x: x[0])):
            integral_vec[key[0]] = cs(self.r)[i]
        dipole_elements = self.Wigner_D_full @ integral_vec
        return dipole_elements

    #testing
    def check_rotation_implementation(self):
        #check composition
        idx_pstart = 1
        idx_pend = 3
        Wigner_D_phi = np.real(np.array(self.Wigner_D_symb(0, self.euler_phi, 0), dtype=complex))
        Wigner_D_theta = np.real(np.array(self.Wigner_D_symb(self.euler_theta,0, 0), dtype=complex))
        D1_phi = Wigner_D_phi
        D2_phi = Wigner_D_phi
        D1_theta = Wigner_D_theta
        D2_theta = Wigner_D_theta
        D_r_phi = Wigner_D_phi[idx_pstart:idx_pend+1, idx_pstart:idx_pend+1]
        D_r_theta = Wigner_D_theta[idx_pstart:idx_pend+1, idx_pstart:idx_pend+1]
        D_phi = np.kron(D1_phi, np.kron(D_r_phi, D2_phi))
        # D_phi = np.where(np.abs(D_phi) < 1e-15, 0, D_phi)
        D_theta = np.kron(D1_theta, np.kron(D_r_theta, D2_theta))
        # D_theta = np.where(np.abs(D_theta) < 1e-15, 0, D_theta)
        print("check composition of rotations")
        print(np.allclose(self.Wigner_D_full, D_phi @ D_theta) )
        #check unitary
        D_dagger = np.transpose(self.Wigner_D_full.conj(), (1,0))
        print("check if unitary")
        print(np.allclose(self.Wigner_D_full @ D_dagger, np.eye(np.shape(self.Wigner_D_full)[0])))
    
class SK_Integral_Overlap(SK_Integral):
    """subclass for S or H mat elements"""
    def __init__(self):
        super().__init__()
        if os.path.exists("identifier_nonzeros_overlap.pkl"):
            with open("identifier_nonzeros_overlap.pkl", 'rb') as f:
                quant_num_list = pickle.load(f)
                nonzeros = pickle.load(f)
        else:
            quant_num_list, nonzeros = get_index_list_overlap()
        self.quant_num_list = quant_num_list
        self.nonzero_idx = nonzeros

    def load_SK_file(self, path):
        with open(path, "r") as f:
            first_line = f.readline().strip()
            extended = 1 if first_line.startswith('@') else 0
        with open(path, 'r') as f:
            for i,line in enumerate(f):
                if i - extended == 0:
                    line1 = line.strip()
                    parts = [p.strip() for p in line1.split(', ')]
                    delta_R, n_points = float(parts[0]), int(parts[1])
        data = np.loadtxt(path, skiprows=3+extended)
        self.delta_R = delta_R
        self.n_points = n_points 
        self.sk_table_S = data[:, len(self.nonzero_idx):] 
        print(np.shape(self.sk_table_S))
        self.sk_table_H = data[:, :len(self.nonzero_idx)]
        print(np.shape(self.sk_table_H))

    def choose_relevant_matrix(self):
        print(f"euler angles: phi={self.euler_phi}, theta={self.euler_theta}, gamma={self.euler_gamma}") 
        self.Wigner_D_single = np.array(self.Wigner_D_symb(self.euler_theta, self.euler_phi, self.euler_gamma), dtype=complex)
        # self.Wigner_D_single = np.where(np.abs(self.Wigner_D_single) < 1e-15, 0, self.Wigner_D_single) 
        D1 = self.Wigner_D_single
        D2 = self.Wigner_D_single
        D = np.kron(D1, D2)
        self.Wigner_D_full = np.real(D)

    def calculate_overlap(self):
        R_grid = self.delta_R + self.delta_R * np.arange(self.n_points) 
        cs = CubicSpline(R_grid, self.sk_table_S) 
        integral_vec = np.zeros((len(self.quant_num_list)))
        for i, key in enumerate(sorted(INTEGRALS, key= lambda x: x[0])):
            integral_vec[key[0]] = cs(self.r)[i]
        dipole_elements = self.Wigner_D_full @ integral_vec
        print(np.allclose(self.Wigner_D_full, np.eye(np.shape(self.Wigner_D_full)[0])))
        return dipole_elements

class SK_With_Shift(SK_Integral):
    def __init__(self):
        super().__init__()
        if os.path.exists("identifier_nonzeros_dipole.pkl"):
            with open("identifier_nonzeros_dipole.pkl", 'rb') as f:
                quant_num_list_dipole = pickle.load(f)
                nonzeros_dipole = pickle.load(f)
        else:
            quant_num_list_dipole, nonzeros_dipole= get_index_list_dipole()
        self.quant_num_list_dipole = quant_num_list_dipole
        self.nonzero_idx_dipole = nonzeros_dipole
        if os.path.exists("identifier_nonzeros_overlap.pkl"):
            with open("identifier_nonzeros_overlap.pkl", 'rb') as f:
                quant_num_list = pickle.load(f)
                nonzeros = pickle.load(f)
        else:
            quant_num_list, nonzeros = get_index_list_overlap()
        self.quant_num_list = quant_num_list
        self.nonzero_idx = nonzeros

    def load_sk_files(self, path, path_dp):
        with open(path_dp, "r") as f:
            first_line = f.readline().strip()
            extended = 1 if first_line.startswith('@') else 0
            for i,line in enumerate(f, start=1):
                if i-extended == 0:
                    line1 = line.strip()
                    parts = [p.strip() for p in line1.split(', ')]
                    delta_R_dipole, n_points_dipole = float(parts[0]), int(parts[1])
                    break
        data_dipole = np.loadtxt(path_dp, skiprows=3+extended)
        self.delta_R_dipole = delta_R_dipole
        self.n_points_dipole = n_points_dipole
        self.sk_table_dipole = data_dipole
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
        self.sk_table_S = data[:, len(self.nonzero_idx):] 
        self.sk_table_H = data[:, :len(self.nonzero_idx)]

    def choose_relevant_matrix(self):
        print(f"euler angles: phi={self.euler_phi}, theta={self.euler_theta}, gamma={self.euler_gamma}") 
        self.Wigner_D_single = np.array(self.Wigner_D_symb(self.euler_theta, self.euler_phi, self.euler_gamma), dtype=complex)
        # self.Wigner_D_single = np.where(np.abs(self.Wigner_D_single) < 1e-15, 0, self.Wigner_D_single) 
        idx_pstart = 1
        idx_pend = 3
        D1 = self.Wigner_D_single
        D2 = self.Wigner_D_single
        D_r = self.Wigner_D_single[idx_pstart:idx_pend+1, idx_pstart:idx_pend+1] # only take p for operator
        D_dipole = np.kron(D1, np.kron(D_r, D2))
        self.Wigner_D_full_dipole = np.real(D_dipole)
        D = np.kron(D1, D2)
        self.Wigner_D_full = np.real(D)

    def load_atom_pair(self, path):
        """load position and basis functions to compute the integrals for"""
        atoms = ase.io.read(path)
        atom1 = angstrom_to_bohr(atoms.get_positions()[0])
        self.R_vec = angstrom_to_bohr(atoms.get_distance(0, 1, vector=True))
        self.r = angstrom_to_bohr(atoms.get_distance(0,1, vector=False))
        self.atom1_pos = np.array([atom1[1], atom1[2], atom1[0]])
        print(self.atom1_pos)
        print(angstrom_to_bohr(atoms.get_positions()[1]))

    def calculate_dipole_elements(self):
        R_grid = self.delta_R + self.delta_R * np.arange(self.n_points) 
        cs = CubicSpline(R_grid, self.sk_table_S) 
        integral_vec = np.zeros((len(self.quant_num_list)))
        for i, key in enumerate(sorted(INTEGRALS, key= lambda x: x[0])):
            integral_vec[key[0]] = cs(self.r)[i]
        overlap_elements = self.Wigner_D_full @ integral_vec
        # shift_term = np.kron(overlap_elements, self.atom1_pos) 

        R_grid_dipole = self.delta_R_dipole + self.delta_R_dipole * np.arange(self.n_points_dipole) 
        cs_dipole = CubicSpline(R_grid_dipole, self.sk_table_dipole) 
        integral_vec_dipole = np.zeros((len(self.quant_num_list_dipole)))
        for i, key in enumerate(sorted(INTEGRALS_DIPOLE, key= lambda x: x[0])):
            integral_vec_dipole[key[0]] = cs_dipole(self.r)[i]
        dipole_elements = self.Wigner_D_full_dipole @ integral_vec_dipole

        shift_term = np.zeros_like(dipole_elements)
        for i in range(16):
            first_same = overlap_elements[i*16:(i+1)*16]
            shift_term[i*48: (i+1)*48] = np.kron(first_same, self.atom1_pos)
        shifted_dipole = dipole_elements + shift_term
        return shifted_dipole
