
import numpy as np
import ase as ase
import pickle
import os
import sympy as sp
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
        if os.path.exists("symbolic_D_matrix.pkl"):
            print('Symbolic D matrix exists')
        else: 
            print('Calculate symbolic D-Matrix')            
            phi = sp.symbols('phi')
            theta = sp.symbols('theta')
            Wigner_D_real(euler_phi=phi, euler_theta=theta)
        with open("symbolic_D_matrix.pkl", "rb") as f:
            M = pickle.load(f)
        theta, phi = sp.symbols('theta phi')
        self.Wigner_D_sym = sp.lambdify((theta, phi), M, 'numpy') 


    def load_atom_pair(self, path):
        """load position and basis functions to compute the integrals for"""
        atoms = ase.io.read(path)
        self.R_vec = atoms.get_distance(0, 1, vector=True) 
        self.r = atoms.get_distance(0,1, vector=False)

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
        R_spherical = to_spherical(R=self.R_vec)
        euler_theta_val = -R_spherical[1] # rotate back on z-axis
        euler_phi_val = -R_spherical[2] #rotate back on z-axis
        print(f"euler angles: phi={euler_phi_val}, theta={euler_theta_val}") 
        self.Wigner_D = np.array(self.Wigner_D_sym(euler_theta_val, euler_phi_val), dtype=complex)
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
        print(np.allclose(D1, np.eye(np.shape(D1)[0])))
        print(np.allclose(D, np.eye(np.shape(D)[0])))
        self.Wigner_D = D

    def calculate_dipole(self):
        """for every pair of basis functions calculate three components of dipole function
        for n basis gives (n,n,3) array"""
        R_grid = self.delta_R + self.delta_R * np.arange(self.n_points) 
        sk_rownumber= np.argmin(np.abs(R_grid-self.r))
        sk_row = self.sk_table[sk_rownumber]
        integral_vec = np.zeros((len(self.quant_num_list)))
        for i, label in enumerate(INTEGRALS):
            integral_vec[label[0]] = sk_row[i]
        dipole_elements = self.Wigner_D @ integral_vec
        n_functions = int(np.sqrt(len(self.quant_num_list)/3))
        dipole_element_components = np.zeros((n_functions, n_functions, 3))
        return dipole_elements

class System:
    
    def __init__(self):
        pass

    def choose_pairs(self):
        pass

    
