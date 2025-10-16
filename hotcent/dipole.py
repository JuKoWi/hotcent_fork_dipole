import numpy as np
import ase

class SK_Integral:
    """calculate S,H or d matrix elements between two atoms 
    only needs position, SK-table and orbital pairs to calculate integrals between
    """
    def __init__(self, el1, el2):
        pass

    def load_atom_pair(self):
        """load position and basis functions to compute the integrals for"""
        pass

    def load_SK_dipole_file(self, path):
        pass

    def choose_relevant_matrix(self):
        pass

    def calculate_dipole(self):
        """for every pair of basis functions calculate three components of dipole function
        for n basis gives (n,n,3) array"""

class System:
    
    def __init__(self):
        pass

    def choose_pairs(self):
        pass

"""How to extract quantum numbers (l,m) from index"""
    