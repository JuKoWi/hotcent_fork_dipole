"""
Script for calculating two center integrals in the Slater-Koster formalism
i.e. applying the Slater-Koster transformation rules
"""
import numpy as np
import os
from hotcent.pos_op.rotation_transform import to_spherical 
from pathlib import Path
import itertools
import sympy as sym 
from scipy.interpolate import CubicSpline
from scipy.linalg import ishermitian
from scipy.constants import physical_constants, angstrom
from ase import Atoms
from ase.neighborlist import *
from hotcent.pos_op.utils import *
from hotcent.pos_op.integrals import get_index_list_dipole, get_index_list_overlap
from hotcent.pos_op.slako_dipole import (INTEGRALS_POSOP, 
                                        UNIQUE_INTEGRALS_POSOP, 
                                        EQUIVALENT_INTEGRALS_POSOP,
                                        ALL_NONZERO_PHI3, 
                                        full_to_unique_posop,
                                        unique_to_full_posop,
                                        EQUIVALENT_ATOMIC_TRANSITIONS,
                                        index_to_quantnum_posop,
                                        UNIQUE_ATOMIC_TRANSITIONS,
)
from hotcent.pos_op.slako_new import (INTEGRALS, 
                                      dftbplus_to_full, 
                                      UNIQUE_INTEGRALS, 
                                      EQUIVALENT_INTEGRALS, 
                                      ALL_NONZERO_PHI2, 
                                      full_to_unique,
                                      unique_to_full,
)
from hotcent.pos_op.generate_sk_rules import D_SYMB, BETA, GAMMA
from hotcent.pos_op.skrules_mini_cse import matrix_elements
from hotcent.pos_op.skrules_posop_mini_cse import matrix_elements_posop

OPERATOR_TYPES = ['S', 'H', 'r']
FILE_FORMAT_OPTIONS = ['unique', 'full', 'DFTB+']
EINSUM = False
VERBOSE = False

class SlaterKosterIntegrator:
    """Takes .skf files in the long (partially redundant) format  used throughout new_dipole/ and calculates the real space matrix elements 
    ordered w.r.t. the lattice shift vectors and saves them in the format of seedname_tb.org from wannier90
    Units ase.Atoms objects:
        geometry: Angstrom
    Units A-B.skf:
        grid distances (header): a0 (a.u. of length)
        H: hartree (a.u. of energy)
        S: no unit
        R: a0 (a.u. of length)
    Units seedname_tb.dat:
        lattice-vectors: Angstrom
        H: eV
        S: no unit
        R: Angstrom
    Units seedname_tb_momentum.dat:
        lattice-vectors: Angstrom
        H: eV
        S: no unit
        p: a.u.
    Internally: 
        atomic units except angstrom
    """

    def __init__(self, atoms_unit_cell, skpath, skpath_posop, maxl_dict, format='unique'):
        """
        atoms_unit_cell: ase.Atoms object containing information about unit cell parameters and unit cell content
        skpath: directory path to .skf files for H and S
        skpath_posop: directory path to .skf files for position operator matrix elements
        maxl_dict: dictionary containing the maximal angular momentum to include in the basis set for each element
        conventional_skf: whether to expect the conventional or the long format for the S/H .skf file
                            the conventional format contains 10 columns for H and S respectively, the long format
                            contains 44 (number of nonzero phi^(2)- integrals) for S and H respectively
        """
        if not (format in FILE_FORMAT_OPTIONS):
            raise ValueError(f"{format} is invalid file format option. Available formats: {FILE_FORMAT_OPTIONS}")
        self.aseAtoms = atoms_unit_cell
        self.abc = atoms_unit_cell.get_cell()
        self.atomtypes = atoms_unit_cell.get_chemical_symbols()
        self.skpath = skpath
        self.skpath_dipole = skpath_posop
        no_repeats_types = list(set(self.atomtypes))
        self.elem_pairs_unordered = list(itertools.combinations_with_replacement(no_repeats_types, 2))
        self.elem_pairs = list(itertools.product(no_repeats_types, repeat=2)) #ordered pairs (e.g. Mo-S and S-Mo are different)
        self._get_interaction_cutoffs()
        self.maxl_dict = maxl_dict
        self.orbnumbers = [dim_atom_basis(maxl_dict[key]) for key in self.atomtypes]
        self.total_orbs = int(np.sum(self.orbnumbers))

        self.D_symb = sym.lambdify((BETA, GAMMA), D_SYMB, 'numpy', cse=True) 

        self.sorted_integrals = sorted(INTEGRALS, key= lambda x: x[0])
        self.sorted_integrals_dipole = sorted(INTEGRALS_POSOP, key=lambda x: x[0])
        self._create_SH_file_dict(format)
        self._create_dipole_file_dict(format)

        self.contraction_path = None
        self.contraction_path_posop = None
        self._scatter_sk = np.array([key[0] for key in self.sorted_integrals]) 


    def _get_interaction_cutoffs(self):
        """read interaction cutoff from .skf files to create neighbor list"""
        #TODO: Write assert to make sure that cutoff is direction invariant
        #TODO: Write assert to make sure that cutoff for dipole is the same
        cutoff_dict = {}
        for pair in self.elem_pairs:
            pairpath = self.skpath+f'/{pair[0]}-{pair[1]}.skf'
            with open(file=pairpath) as f:
                line1 = f.readline()
                line1 = line1.replace(',', ' ')
                line1 = line1.split()
                dr, Nr = float(line1[0]), int(line1[1])
            max_r = dr * Nr
            cutoff_dict[pair] = bohr_to_angstrom(max_r)
        self.cutoff_dict = cutoff_dict

    def _read_sk_file_dftbplus(self, elem_pair):
        """read a Slater-Koster file of the conventional format as used by DFTB+ for example
            elem_pair: ordered 2-tuple of elements
        """
        homonuclear = (elem_pair[0] == elem_pair[1])
        path = self.skpath + f"/{elem_pair[0]}-{elem_pair[1]}.skf" 
        path_interchanged = self.skpath + f"/{elem_pair[1]}-{elem_pair[0]}.skf"
        file = Path(path)
        file_interchanged = Path(path_interchanged)
        assert file.is_file()
        assert file_interchanged.is_file()
        sk_table_H, sk_table_S = dftbplus_to_full(path1=path, path2=path_interchanged) 
        sktable_S_unique = full_to_unique(sk_table_S)
        sktable_H_unique = full_to_unique(sk_table_H)
        with open(path, "r") as f:
            line1 = f.readline().strip()
            line1 = line1.replace(',', ' ')
            line2 = f.readline()
            line2 = line2.replace(',', ' ')
            extended = 1 if line1.startswith('@') else 0
            if extended == 0:
                parts = [p.strip() for p in line1.split()]
                if homonuclear:
                    same_atom = line2.split()
                    same_atom = np.flip(same_atom[:3])
                else:
                    same_atom = None
            if extended == 1:
                parts = [p.strip() for p in line2.split()]
            delta_R, n_points = bohr_to_angstrom(float(parts[0])), int(parts[1])
        if not homonuclear:
            extended -= 1
        assert np.shape(sktable_H_unique)[0] == n_points
        assert np.shape(sktable_S_unique)[0] == n_points
        return delta_R, n_points, sktable_S_unique, sktable_H_unique, same_atom

    def _read_sk_file_full(self, elem_pair, dipole):
        """returns dr, Nr and the table(s) for a .skf file or the dipole equivalent"""
        if dipole:
            path = self.skpath_dipole + f"/{elem_pair[0]}-{elem_pair[1]}.skf" 
        else:
            path = self.skpath + f"/{elem_pair[0]}-{elem_pair[1]}.skf" 
        myfile = Path(path)
        assert myfile.is_file()
        homonuclear = (elem_pair[0] == elem_pair[1])
        with open(path, "r") as f:
            line1 = f.readline().strip()
            line1 = line1.replace(',', ' ')
            line2 = f.readline()
            line2 = line2.replace(',', ' ')
            extended = 1 if line1.startswith('@') else 0
            if extended == 0:
                parts = [p.strip() for p in line1.split()]
                if homonuclear:
                    same_atom = line2.split()
                    if not dipole:
                        same_atom = np.flip(same_atom[:3])
            if extended == 1:
                parts = [p.strip() for p in line2.split()]
            delta_R, n_points = bohr_to_angstrom(float(parts[0])), int(parts[1])
        if not homonuclear:
            extended -= 1
        data = np.loadtxt(path, skiprows=3+extended)
        if dipole:
            sk_table_r = bohr_to_angstrom(data)
            sk_table_r_unique = full_to_unique_posop(sk_table_r)
            for i, integral in enumerate(UNIQUE_INTEGRALS_POSOP): # verify that columns that should be equal are equal
                for equivalent in EQUIVALENT_INTEGRALS_POSOP[integral]:
                    idx = ALL_NONZERO_PHI3.index(abs(equivalent))
                    sign = -1 if equivalent < 0 else 1
                    if not np.allclose(sk_table_r_unique[:,i], sign * sk_table_r[:,idx]):
                        raise ValueError(f"Equivalent columns in {myfile} are not identical")
            sorted_labels = sorted(INTEGRALS_POSOP.keys(), key=lambda x: x[0])
            sorted_labels = [l[1:] for l in sorted_labels]
            if homonuclear:
                if not (len(same_atom) == len(UNIQUE_ATOMIC_TRANSITIONS)):
                    raise ValueError(f"Incorrect number of distinct atomic transition in {myfile}")
                atom_transitions = [bohr_to_angstrom(float(i)) for i in same_atom]
        else:
            sk_table_S = data[:, len(self.sk_int_idx):] 
            sk_table_H = data[:, :len(self.sk_int_idx)]
            sktable_S_unique = full_to_unique(sk_table_S)
            sktable_H_unique = full_to_unique(sk_table_H)
            for i, integral in enumerate(UNIQUE_INTEGRALS):
                for equivalent in EQUIVALENT_INTEGRALS[integral]:
                    idx = ALL_NONZERO_PHI2.index(equivalent)
                    H_is_same = np.allclose(sktable_H_unique[:,i], sk_table_H[:,idx])
                    S_is_same = np.allclose(sktable_S_unique[:,i], sk_table_S[:,idx])
                    if not (H_is_same and S_is_same):
                        raise ValueError(f"Equivalent columns in {myfile} are not identical")
        if not (np.shape(data)[0] == n_points):
            raise ValueError(f"{elem_pair}.skf table block does not match the number of distance points in the header")
        if not homonuclear:
            same_atom = None
            atom_transitions = None
        if dipole:
            return delta_R, n_points, sk_table_r_unique, atom_transitions 
        else:
            return delta_R, n_points, sktable_S_unique, sktable_H_unique, same_atom

    def _read_sk_file_unique(self, elem_pair, dipole):
        if dipole:
            path = self.skpath_dipole + f"/{elem_pair[0]}-{elem_pair[1]}.skf" 
        else:
            path = self.skpath + f"/{elem_pair[0]}-{elem_pair[1]}.skf" 
        myfile = Path(path)
        if not myfile.is_file():
            raise ValueError(f"Could not find file {myfile}")
        homonuclear = (elem_pair[0] == elem_pair[1])
        with open(path, "r") as f:
            line1 = f.readline().strip()
            line1 = line1.replace(',', ' ')
            line2 = f.readline()
            line2 = line2.replace(',', ' ')
            extended = 1 if line1.startswith('@') else 0
            if extended == 0:
                parts = [p.strip() for p in line1.split()]
                if homonuclear:
                    same_atom = line2.split()
                    if not dipole:
                        same_atom = np.flip(same_atom[:3])
            if extended == 1:
                parts = [p.strip() for p in line2.split()]
            delta_R, n_points = bohr_to_angstrom(float(parts[0])), int(parts[1])
        if not homonuclear:
            extended -= 1
        data = np.loadtxt(path, skiprows=3+extended)
        if dipole:
            sk_table_r_unique = bohr_to_angstrom(data)
            sorted_labels = sorted(INTEGRALS_POSOP.keys(), key=lambda x: x[0])
            sorted_labels = [l[1:] for l in sorted_labels]
            if homonuclear:
                if not (len(same_atom) == len(UNIQUE_ATOMIC_TRANSITIONS)):
                    raise ValueError(f"Incorrect number of distinct atomic transition in {myfile}")
                atom_transitions = [bohr_to_angstrom(float(i)) for i in same_atom]
        else:
            sktable_S_unique = data[:, len(UNIQUE_INTEGRALS):] 
            sktable_H_unique = data[:, :len(UNIQUE_INTEGRALS)]
        if not (np.shape(data)[0] == n_points):
            raise ValueError(f"{elem_pair}.skf table block does not match the number of distance points in the header")
        if not homonuclear:
            same_atom = None
            atom_transitions = None
        if dipole:
            return delta_R, n_points, sk_table_r_unique, atom_transitions 
        else:
            return delta_R, n_points, sktable_S_unique, sktable_H_unique, same_atom
        
    def _create_SH_file_dict(self, format):
        """create a dictionary where for every ordered element pair there is a SKTable object,
        that contains all the information from the .skf file
        conventional_skf: whether to use DFTB+ file format for H/S
        """
        S_sk_dict = {}
        H_sk_dict = {}
        for element_comb in self.elem_pairs:
            if format == "DFTB+":
                delta_R, n_points, S, H, eigvals  = self._read_sk_file_dftbplus(elem_pair=element_comb)
            elif format == "full":
                delta_R, n_points, S, H, eigvals  = self._read_sk_file_full(elem_pair=element_comb, dipole=False)
            elif format == "unique":
                delta_R, n_points, S, H, eigvals = self._read_sk_file_unique(elem_pair=element_comb, dipole=False)
            else:
                raise ValueError(f"Reading routine for file format {format} not implemented")
            S_sk_dict[element_comb] = SKTable(table_type='S', table=S, deltaR=delta_R, n_points=n_points, same_atom=[1,1,1]) #assume the atomic functions to be orthonormal
            H_sk_dict[element_comb] = SKTable(table_type='H', table=H, deltaR=delta_R, n_points=n_points, same_atom=eigvals)
        self.S_sk_tables = S_sk_dict
        self.H_sk_tables = H_sk_dict
    
    def _create_dipole_file_dict(self, format):
        """create a dictionary where for every element combination there is a SKTable object,
        that contains all the information from the .skf file
        """
        r_sk_dict = {}
        for element_comb in self.elem_pairs:
            if (format == 'DFTB+' or format == 'unique'):
                delta_R, n_points, r, atom_transitions = self._read_sk_file_unique(elem_pair=element_comb, dipole=True)
            elif format == 'full':
                delta_R, n_points, r, atom_transitions = self._read_sk_file_full(elem_pair=element_comb, dipole=True)
            else:
                raise ValueError(f"Reading routine for file format {format} not implemented for position operator") 
            r_sk_dict[element_comb] = SKTable(table_type='r', table=r, deltaR=delta_R, n_points=n_points, same_atom=atom_transitions)
        self.r_sk_tables = r_sk_dict

    def _set_euler_angles(self, vec1, vec2):
        """
            Find Euler angles for rotation, uses only two rotations of the three possible
            vec1, vec2: position vectors for first and second atom respectively 
        """
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

    def _get_direction_cosines(self, vec1, vec2):
        R_vec = vec2 - vec1
        norm_R_vec = np.linalg.norm(R_vec)
        if np.all(R_vec == 0):
            lmn = np.array([0, 0, 0])
        else:
            lmn = R_vec/norm_R_vec
        return lmn

    def _integrals_atom_pair(self, sk_table, posA, posB, operator, lmaxA, lmaxB, sk_table_dipole=None):
        """
        Evaluate Slater-Koster rules
        For one pair of interacting atoms with fixed positions and certain operator (S,H or r) create the respective 
        dictionary of two center integrals with angular momentum quantum numbers as keys
        sk_table: SKTable object for the respective elements
        posA, posB: positions of selected atoms respectively
        operator: string, choice of operator ('S', 'H' or 'r')
        sk_table_dipole: SKTable object for position operator elements, only required if operator=='r'
        """
        if operator == 'r':
            assert sk_table_dipole != None
            assert sk_table.type == 'S'
        R_vec = posB - posA
        R = np.linalg.norm(R_vec)
        dimA = dim_atom_basis(lmaxA)
        dimB = dim_atom_basis(lmaxB)

        if EINSUM:
            euler_theta, euler_phi, euler_gamma= self._set_euler_angles(vec1=posA, vec2=posB)
            D_single = np.array(self.D_symb(euler_theta, euler_phi), dtype=float)
            integral_vec = np.zeros((16 * 16)) 
            integral_vec[self._scatter_sk] = sk_table.spline_full(R)
            M = np.reshape(integral_vec, (16,16))
            if self.contraction_path is None:
                self.contraction_path = np.einsum_path('ab,bc,dc -> ad', D_single, M, D_single, optimize='optimal')[0]
            integrals = np.einsum('ab,bc,dc -> ad', D_single, M, D_single, optimize=self.contraction_path)[:dimA, :dimB]
            if operator == 'r':
                idx_pstart = 1
                idx_pend = 3
                D_r = D_single[idx_pstart:idx_pend+1, idx_pstart:idx_pend+1]
                spline_eval_posop = sk_table_dipole.spline_full(R)
                integral_vec_dipole = np.zeros((16*3*16))
                for i, key in enumerate(self.sorted_integrals_dipole):
                    integral_vec_dipole[key[0]] = spline_eval_posop[i]
                M_posop = np.reshape(integral_vec_dipole, (16,3,16))
                if self.contraction_path_posop is None:
                    self.contraction_path_posop = np.einsum_path('ai, bj, ck, ijk -> abc', D_single, D_r, D_single, M_posop, optimize='optimal')[0]
                position_elements = np.einsum('ai, bj, ck, ijk -> abc', D_single, D_r, D_single, M_posop, optimize=self.contraction_path_posop)[:dimA,:,:dimB]

        else:
            l, m, n = self._get_direction_cosines(vec1=posA, vec2=posB)
            integral_vec = sk_table.spline(R)
            integrals = matrix_elements(X=integral_vec, l=l, m=m, n=n)[:dimA, :dimB]
            if operator == 'r':
                integral_vec_posop = sk_table_dipole.spline(R)
                position_elements = matrix_elements_posop(X=integral_vec_posop, l=l, m=m, n=n)[:dimA,:,:dimB]

        if VERBOSE:
            print(f"maximal imaginary integral value {np.max(np.abs(np.imag(integrals)))}")
            print(f"maximal imaginary position integral value {np.max(np.abs(np.imag(position_elements)))}")

        if operator == 'r': #consider origin shift
            posA = np.array([posA[1], posA[2], posA[0]])
            shifted_dipole = position_elements + np.einsum('ab, c -> acb', integrals, posA)
            shifted_dipole = np.transpose(shifted_dipole, (1,0,2))[[2,0,1]]
            return shifted_dipole
        else:
            return integrals

    def _create_integral_dict_nablaR(self, sk_table, posA, posB):
        same_atom = np.allclose(posA, posB)
        int_dict_gradR = {}
        if same_atom:
            data = sk_table.table
            zero_rows = np.all(data==0, axis=1)
            index_nonzero = np.argmax(~zero_rows) +1 if (~zero_rows).any() else data.shape[0] +1
            rmin_angst = sk_table.deltaR * index_nonzero
            h = rmin_angst 
        else:
            h = 1e-4 #corresponds to angstrom
        for label in self.quant_nums:
            int_dict_gradR[label[1], label[2], label[3], label[4]] = np.zeros(3)
        for i in range(3):
            unit = np.zeros(3)
            unit[i] = 1
            posBp1 = posB + h * unit 
            posBm1 = posB - h * unit 
            posBp2 = posB + 2 * h * unit
            posBm2 = posB - 2 * h * unit
            int_dictp1 = self._integrals_atom_pair(sk_table=sk_table, posA=posA, posB=posBp1, operator='S')
            int_dictm1 = self._integrals_atom_pair(sk_table=sk_table, posA=posA, posB=posBm1, operator='S')
            int_dictp2 = self._integrals_atom_pair(sk_table=sk_table, posA=posA, posB=posBp2, operator='S')
            int_dictm2 = self._integrals_atom_pair(sk_table=sk_table, posA=posA, posB=posBm2, operator='S')
            for label in self.quant_nums:
                p2 = int_dictp2[label[1], label[2], label[3], label[4]]
                p1 = int_dictp1[label[1], label[2], label[3], label[4]]
                m2 = int_dictm2[label[1], label[2], label[3], label[4]]
                m1 = int_dictm1[label[1], label[2], label[3], label[4]]
                finite_diff = (-p2 + 8 * p1 - 8 * m1 + m2)/(12 *h) #has dimension 1/angstrom
                int_dict_gradR[label[1], label[2], label[3], label[4]][i] = -finite_diff
        return int_dict_gradR

    def _select_momentum_matrix_elements(self, max_lA, max_lB, integral_dict):
        pair_matrix = np.zeros((3, dim_atom_basis(max_lA), dim_atom_basis(max_lB)))
        for i in range(3):
            row_start = 0
            col_start = 0 
            for l1 in range(max_lA + 1):
                size_row = 2 * l1 +1
                for l2 in range(max_lB + 1):
                    size_col = 2 * l2 +1
                    block = np.zeros((size_row, size_col))
                    for mi, m in enumerate(range(-l1, l1 + 1)):
                        for ni, n in enumerate(range(-l2, l2 + 1)):
                            quant_nums = (l1, m, l2, n)
                            block[mi,ni] = integral_dict.get(quant_nums, 0.0)
                    pair_matrix[i, row_start:row_start+size_row, col_start:col_start+size_col] = block
                    col_start += size_col
                col_start = 0
                row_start += size_row
        return pair_matrix

    def _assemble_atom_block(self, types, posA, posB, max_lA, max_lB, operator):
        """for two atoms, calculate the block of all relevant orbitals
        for dipole the block is a 3D array with the first axis for the 3 components x,y,z"""
        same_atom = np.allclose(posA, posB)
        if same_atom:
            assert types[0] == types[1]
        if operator == 'S':
            sk_table = self.S_sk_tables[types]
            if same_atom:
                block = np.eye(N=dim_atom_basis(maxl=max_lA))
            else:
                block = self._integrals_atom_pair(sk_table=sk_table, posA=posA, posB=posB, lmaxA=max_lA, lmaxB=max_lB, operator=operator)
        elif operator == 'H':
            sk_table = self.H_sk_tables[types]
            if same_atom:
                eigenvalues = sk_table.same_atom_vals
                diag = np.zeros((dim_atom_basis(max_lA), dim_atom_basis(max_lA)))
                count = 0
                for i in range(max_lA+1):
                    for j in range(2*i+1):
                        diag[count+j, count+j] = eigenvalues[i]
                    count += 2*i +1
                block = diag
            else:
                block = self._integrals_atom_pair(sk_table=sk_table, posA=posA, posB=posB, lmaxA=max_lA, lmaxB=max_lB, operator=operator)
        elif operator == 'r':
            sk_table = self.S_sk_tables[types]
            sk_table_dipole = self.r_sk_tables[types]
            if same_atom:
                block = np.zeros((16,3,16))
                n_orbs = dim_atom_basis(maxl=max_lA)
                for i,u in enumerate(UNIQUE_ATOMIC_TRANSITIONS):
                    for equivalent in EQUIVALENT_ATOMIC_TRANSITIONS[u]:
                        a, r = divmod(abs(equivalent), 16*3)
                        b, c = divmod(r, 16)
                        sign = -1 if equivalent < 0 else 1
                        block[a,b,c] = sign * sk_table_dipole.same_atom_vals[i]
                block = block[:n_orbs,:,:n_orbs]
                block = np.transpose(block, (1,0,2))[[2,0,1]] # put components as first axis and reorder y,z,x -> x,y,z
                for c in range(3): # origin shift with overlap matrix (identity for intraatomic)
                    block[c] += posA[c] * np.eye(n_orbs)
            else: 
                block = self._integrals_atom_pair(sk_table=sk_table, posA=posA, posB=posB, lmaxA=max_lA, lmaxB=max_lB, operator='r', sk_table_dipole=sk_table_dipole)
        elif operator == 'p':
            sk_table= self.S_sk_tables[types]
            integral_dict = self._create_integral_dict_nablaR(sk_table=sk_table, posA=posA, posB=posB)
            block = self._select_momentum_matrix_elements(max_lA=max_lA, max_lB=max_lB, integral_dict=integral_dict)

        block = np.where(np.abs(block) < 1e-15, 0, block)
        return block

    def _calculate_lattice_dict(self, operator):
        """create a dictionary with lattice vectors as keys. 
            for every lattice vector the value is a matrix that describes the overlap 
            between the orbitals in the unit cell and the orbitals in the unit cell shifted 
            by the respective lattice vector
        """
        atoms = self.aseAtoms
        lattice_dict = {}
        pairA, pairB, R, d = neighbor_list('ijSd', a=atoms, cutoff=self.cutoff_dict, self_interaction=True) 
        self.n_lattice = np.shape(np.unique(R, axis=0))[0]

        #create real space matrix
        for i, pair in enumerate(pairA):
            R_triple = (int(R[i,0]), int(R[i,1]), int(R[i,2]))
            if operator in ('r', 'p'):
                matrix = lattice_dict.setdefault(R_triple, np.zeros((3, self.total_orbs, self.total_orbs)))
            else:
                matrix = lattice_dict.setdefault(R_triple, np.zeros((self.total_orbs, self.total_orbs)))
            idxA = pairA[i]
            idxB = pairB[i]
            typeA = atoms.symbols[idxA]
            typeB = atoms.symbols[idxB]
            maxlA = self.maxl_dict[typeA]
            maxlB = self.maxl_dict[typeB]
            posA = atoms.positions[idxA]
            posB = atoms.positions[idxB] + np.dot(self.abc.T, R[i])
            # assert np.linalg.norm(posB - posA) <= self.cutoff_dict[(typeA, typeB)]
            block = self._assemble_atom_block(types=(typeA, typeB), posA=posA, posB=posB, max_lA=maxlA, max_lB=maxlB, operator=operator)
            n_rows = dim_atom_basis(maxl=maxlA)
            n_cols = dim_atom_basis(maxl=maxlB)
            start_rows = self._find_block_pos(idx=idxA)
            start_cols = self._find_block_pos(idx=idxB)
            if operator in ('r', 'p'):
                matrix[:,start_rows:start_rows+n_rows, start_cols:start_cols+n_cols] = block
            else:
                matrix[start_rows:start_rows+n_rows, start_cols:start_cols+n_cols] = block
            lattice_dict[R_triple] = matrix

        # if operator not in ('r', 'p'):
        #     for lat_vec in np.unique(R, axis=0): 
        #         mat1 = lattice_dict[*lat_vec] 
        #         mat2 = lattice_dict[*(-lat_vec)]
                # symmetry_requirement = np.allclose(mat1, np.linalg.matrix_transpose(mat2), atol=1e-6)
                # assert symmetry_requirement 
        return lattice_dict
    
    def _find_block_pos(self, idx):
        orb_previous = np.sum(self.orbnumbers[:idx])
        return int(orb_previous)
    
    def write_seedname_momentum(self):
        filename = 'seedname_tb_momentum.dat'
        lattice_dict_p = self._calculate_lattice_dict(operator='p')
        lattice_dict_S = self._calculate_lattice_dict(operator='S')
        lattice_dict_H = self._calculate_lattice_dict(operator='H')
        assert len(lattice_dict_H.keys()) == len(lattice_dict_p.keys())
        assert len(lattice_dict_S.keys()) == len(lattice_dict_p.keys())
        with open(filename, 'w') as f:
            f.write(str(np.datetime64('now'))+'\n')
            np.savetxt(f, self.abc)
            f.write(str(self.total_orbs)+'\n')
            f.write(str(self.n_lattice)+'\n')
            for i in range(self.n_lattice):
                f.write("1 ")
                if (i+1) % 15 == 0:
                    f.write("\n")
            f.write('\n')
            for point in lattice_dict_S.keys():
                f.write('\n')
                f.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
                S_array = lattice_dict_S[point]
                H_array = hartree_to_eV(lattice_dict_H[point])
                A = np.real(H_array)
                B = np.imag(H_array)
                C = np.real(S_array)
                D = np.imag(S_array)
                for i in range(np.shape(S_array)[0]):
                    for j in range(np.shape(S_array)[1]):
                        print(f"{i+1} {j+1}\t{A[i,j]:.18e}\t{B[i,j]:.18e}\t{C[i,j]:.18e}\t{D[i,j]:.18e}", file=f)
            conversion_angstrom_bohr_inv = physical_constants['atomic unit of length'][0] /angstrom 
            for point in lattice_dict_p.keys():
                f.write('\n')
                f.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
                p_array = lattice_dict_p[point] * conversion_angstrom_bohr_inv
                xre = np.real(p_array[0])
                xim = np.imag(p_array[0])
                yre = np.real(p_array[1])
                yim = np.imag(p_array[1])
                zre = np.real(p_array[2])
                zim = np.imag(p_array[2])
                for m in range(np.shape(p_array)[1]):
                    for n in range(np.shape(p_array)[2]):
                        i = m+1
                        j = n+1
                        print(f"{i} {j}\t{xre[m,n]:.18e}\t{xim[m,n]:.18e}\t{yre[m,n]:.18e}\t{yim[m,n]:.18e}\t{zre[m,n]:.18e}\t{zim[m,n]:.18e}", file=f)

    def write_seedname(self):
        """write to file in style of seedname_tb.dat form w90 program
        hamiltonian elements in eV, lengths in angstrom"""
        filename = 'seedname_tb.dat'
        lattice_dict_S = self._calculate_lattice_dict(operator='S')
        lattice_dict_H = self._calculate_lattice_dict(operator='H')
        lattice_dict_r = self._calculate_lattice_dict(operator='r')
        assert len(lattice_dict_H.keys()) == len(lattice_dict_r.keys())
        assert len(lattice_dict_S.keys()) == len(lattice_dict_r.keys())
        with open(filename, 'w') as f:
            f.write(str(np.datetime64('now'))+'\n')
            np.savetxt(f, self.abc)
            f.write(str(self.total_orbs)+'\n')
            f.write(str(self.n_lattice)+'\n')
            for i in range(self.n_lattice):
                f.write("1 ")
                if ((i+1) % 15 == 0) and (i+1 != self.n_lattice):
                    f.write("\n")
            f.write("\n")
            for point in lattice_dict_S.keys():
                f.write('\n')
                f.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
                S_array = lattice_dict_S[point]
                H_array = hartree_to_eV(lattice_dict_H[point])
                A = np.real(H_array)
                B = np.imag(H_array)
                C = np.real(S_array)
                D = np.imag(S_array)
                for i in range(np.shape(S_array)[0]):
                    for j in range(np.shape(S_array)[1]):
                        print(f"{i+1} {j+1}\t{A[i,j]:.18e}\t{B[i,j]:.18e}\t{C[i,j]:.18e}\t{D[i,j]:.18e}", file=f)
            for point in lattice_dict_r.keys():
                f.write('\n')
                f.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
                r_array = lattice_dict_r[point]
                xre = np.real(r_array[0])
                xim = np.imag(r_array[0])
                yre = np.real(r_array[1])
                yim = np.imag(r_array[1])
                zre = np.real(r_array[2])
                zim = np.imag(r_array[2])
                for m in range(np.shape(r_array)[1]):
                    for n in range(np.shape(r_array)[2]):
                        i = m+1
                        j = n+1
                        print(f"{i} {j}\t{xre[m,n]:.18e}\t{xim[m,n]:.18e}\t{yre[m,n]:.18e}\t{yim[m,n]:.18e}\t{zre[m,n]:.18e}\t{zim[m,n]:.18e}", file=f)
                
    def write_hamoversqr(self):
        """TODO: Check if units match with conventions of DFTB+"""
        matrix = self._calculate_lattice_dict(operator='S')[(0,0,0)]
        matrixH = self._calculate_lattice_dict(operator='H')[(0,0,0)]
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
    def __init__(self, table_type, table, deltaR, n_points, same_atom=None):
        assert table_type in OPERATOR_TYPES
        self.type = table_type
        self.table = table #first dimension distance, second dimension integral index, contains only unique values
        self.deltaR = deltaR
        self.n_points = n_points
        self.same_atom_vals = None #list for S and H, dict for r
        self._spline = None
        self._spline_full = None
        self.same_atom_vals = same_atom
        if table_type in ['S', 'H']:
            self.table_full = unique_to_full(table)
            assert np.shape(table)[1] == len(UNIQUE_INTEGRALS)
        if table_type == 'r':
            self.table_full = unique_to_full_posop(table)
            assert np.shape(table)[1] == len(UNIQUE_INTEGRALS_POSOP)

    @property
    def spline(self):
        if self._spline is None:
            R_grid = self.deltaR + self.deltaR * np.arange(self.n_points)
            self._spline = CubicSpline(R_grid, self.table)
        return self._spline

    @property
    def spline_full(self):
        if self._spline_full is None:
            R_grid = self.deltaR + self.deltaR * np.arange(self.n_points)
            self._spline_full = CubicSpline(R_grid, self.table_full)
        return self._spline_full