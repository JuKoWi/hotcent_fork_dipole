
import numpy as np
import ase as ase
import pickle
import os
import sympy as sp
from scipy.interpolate import CubicSpline
from hotcent.new_dipole.utils import *
from hotcent.new_dipole.integrals import get_index_list
from hotcent.new_dipole.rotation_transform import Wigner_D_real, to_spherical
from hotcent.new_dipole.slako_dipole import NUMSK, INTEGRALS

class SK_Integral:
    """calculate S,H or d matrix elements between two atoms 
    only needs position, SK-table and orbital pairs to calculate integrals between
    """
    def __init__(self, el1, el2):
        quant_num_list, nonzeros = get_index_list()
        self.quant_num_list = quant_num_list
        self.nonzero_idx = nonzeros
        theta, phi, gamma = sp.symbols('theta phi gamma')
        if os.path.exists("symbolic_D_matrix.pkl"):
            print('Symbolic D matrix exists')
        else: 
            print('Calculate symbolic D-Matrix')            
            Wigner_D_real(euler_phi=phi, euler_theta=theta, euler_gamma=gamma)
        with open("symbolic_D_matrix.pkl", "rb") as f:
            M = pickle.load(f)
        self.Wigner_D_sym = sp.lambdify((theta, phi, gamma), M, 'numpy') 


    def load_atom_pair(self, path):
        """load position and basis functions to compute the integrals for"""
        atoms = ase.io.read(path)
        self.R_vec = angstrom_to_bohr(atoms.get_distance(0, 1, vector=True))
        self.r = angstrom_to_bohr(atoms.get_distance(0,1, vector=False))

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
        self.n_points = n_points -1 # TODO: find out, why this is printed as higher as the actual number of data points
        self.sk_table = data 

    def set_euler_angles(self):
        R_spherical = to_spherical(R=self.R_vec)
        if np.all(self.R_vec == 0):
            self.euler_theta = 0
            self.euler_phi = 0
            self.euler_gamma = 0 
        else:
            self.euler_theta = -R_spherical[1] # rotate back on z-axis
            self.euler_phi = -R_spherical[2] # rotate back on z-axis
            self.euler_gamma = 0

    def choose_relevant_matrix(self):
        print(f"euler angles: phi={self.euler_phi}, theta={self.euler_theta}, gamma={self.euler_gamma}") 
        self.Wigner_D = np.array(self.Wigner_D_sym(theta=self.euler_theta, phi=self.euler_phi, gamma=self.euler_gamma), dtype=complex)
        idx_pstart = 1
        idx_pend = 3
        D1 = self.Wigner_D
        print("complex rotation matrix")
        print(D1)
        D2 = self.Wigner_D
        D_r = self.Wigner_D[idx_pstart:idx_pend+1, idx_pstart:idx_pend+1]
        D = np.kron(D1, np.kron(D_r, D2))
        print("tensorprod rotation matrix")
        print(D)
        print("compare to identity automatically")
        # print(np.allclose(D1, np.eye(np.shape(D1)[0])))
        # print(np.allclose(D, np.eye(np.shape(D)[0])))
        self.Wigner_D = D

    def check_rotation_implementation(self):
        #check composition
        idx_pstart = 1
        idx_pend = 3
        Wigner_D_phi = np.array(self.Wigner_D_sym(0, self.euler_phi, 0), dtype=complex)
        Wigner_D_theta = np.array(self.Wigner_D_sym(self.euler_theta, 0, 0), dtype=complex)
        D1_phi = Wigner_D_phi
        D2_phi = Wigner_D_phi
        D1_theta = Wigner_D_theta
        D2_theta = Wigner_D_theta
        D_r_phi = Wigner_D_phi[idx_pstart:idx_pend+1, idx_pstart:idx_pend+1]
        D_r_theta = Wigner_D_theta[idx_pstart:idx_pend+1, idx_pstart:idx_pend+1]
        D_phi = np.kron(D1_phi, np.kron(D_r_phi, D2_phi))
        D_theta = np.kron(D1_theta, np.kron(D_r_theta, D2_theta))
        print("check composition of rotations")
        print(np.allclose(self.Wigner_D, D_theta @ D_phi))
        #check unitary
        D_dagger = np.transpose(self.Wigner_D.conj(), (1,0))
        print("check if unitary")
        print(np.allclose(self.Wigner_D @ D_dagger, np.eye(np.shape(self.Wigner_D)[0])))
    


    def calculate_dipole(self):
        """for every pair of basis functions calculate three components of dipole function
        for n basis gives (n,n,3) array"""
        R_grid = self.delta_R + self.delta_R * np.arange(self.n_points) 
        cs = CubicSpline(R_grid, self.sk_table, extrapolate=True) 
        integral_vec = np.zeros((len(self.quant_num_list)))
        for i, label in enumerate(INTEGRALS):
            integral_vec[label[0]] = cs(self.r)[i]
        dipole_elements = self.Wigner_D @ integral_vec
        n_functions = int(np.sqrt(len(self.quant_num_list)/3))
        return dipole_elements

class System:
    
    def __init__(self):
        pass

    def choose_pairs(self):
        pass

    
