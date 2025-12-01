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
    def __init__(self, unit_cell, skpath, skpath_dipole, maxl_dict):
        self.ucell = unit_cell
        self.abc = unit_cell.get_cell()
        self.atomtypes = unit_cell.get_chemical_symbols()
        self.skpath = skpath
        self.skpath_dipole = skpath_dipole
        no_repeats_types = list(set(self.atomtypes))
        self.elem_pairs_unordered = list(itertools.combinations_with_replacement(no_repeats_types, 2))
        self.elem_pairs = list(itertools.product(no_repeats_types, repeat=2))
        self._get_interaction_cutoffs()
        self.maxl_dict = maxl_dict
        self.orbnumbers = [get_norbs(maxl_dict[key]) for key in self.atomtypes]
        self.total_orbs = int(np.sum(self.orbnumbers))

        if os.path.exists("identifier_nonzeros_overlap.pkl"):
            with open("identifier_nonzeros_overlap.pkl", 'rb') as f:
                quant_num_list = pickle.load(f)
                nonzeros = pickle.load(f)
        else:
            quant_num_list, nonzeros = get_index_list_overlap()
        self.quant_nums = quant_num_list
        self.sk_int_idx = nonzeros

        if os.path.exists("identifier_nonzeros_dipole.pkl"):
            with open("identifier_nonzeros_dipole.pkl", 'rb') as f:
                quant_num_list = pickle.load(f)
                nonzeros_dipole = pickle.load(f)
        else:
            quant_num_list, nonzeros_dipole = get_index_list_dipole()
        self.quant_nums_dipole = quant_num_list
        self.sk_int_idx_dipole = nonzeros_dipole

        if os.path.exists("symbolic_D_matrix.pkl"):
            pass
            print('Symbolic D matrix exists')
        else: 
            print('Calculate symbolic D-Matrix')            
            Wigner_D_real(euler_phi=PHI, euler_theta=THETA, euler_gamma=GAMMA)
        with open("symbolic_D_matrix.pkl", "rb") as f:
            M = pickle.load(f)
        self.D_symb = sym.lambdify((THETA, PHI, GAMMA), M, 'numpy') 

        self._create_SH_file_dict()
        self._create_dipole_file_dict()

    def _get_interaction_cutoffs(self):
        #TODO: Write assert to make sure that cutoff is direction invariant
        #TODO: Write assert to make sure that cutoff for dipole is the same
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
        if dipole:
            path = self.skpath_dipole + f"/{elem_pair[0]}-{elem_pair[1]}.skf" #write conditional for dipole
        else:
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
                same_atom = line2.split()
                if not dipole:
                    same_atom = np.flip(same_atom[:3])
            if extended == 1:
                parts = [p.strip() for p in line2.split()]
            delta_R, n_points = float(parts[0]), int(parts[1])
        if not homonuclear:
            extended -= 1
        data = np.loadtxt(path, skiprows=3+extended)
        if dipole:
            sk_table_r = data
            sorted_labels = sorted(INTEGRALS_DIPOLE.keys(), key=lambda x: x[0])
            sorted_labels = [l[1:] for l in sorted_labels]
            assert len(sorted_labels) == len(same_atom)
            atom_transitions = dict(zip(sorted_labels, same_atom))
        else:
            sk_table_S = data[:, len(self.sk_int_idx):] 
            sk_table_H = data[:, :len(self.sk_int_idx)]
        assert np.shape(data)[0] == n_points
        if dipole:
            return delta_R, n_points, sk_table_r, atom_transitions 
        else:
            return delta_R, n_points, sk_table_S, sk_table_H, same_atom
        
    def _create_SH_file_dict(self):
        """create a dictionary where for every element combiantion there is a custom object,
        that contains all the information from the .skf file
        """
        S_sk_dict = {}
        H_sk_dict = {}
        for element_comb in self.elem_pairs:
            delta_R, n_points, S, H, eigvals  = self._read_sk_file(elem_pair=element_comb, dipole=False)
            S_sk_dict[element_comb] = SKTable(table=S, deltaR=delta_R, n_points=n_points, same_atom=[1,1,1]) #assume the atomic functions to be orthonormal
            H_sk_dict[element_comb] = SKTable(table=H, deltaR=delta_R, n_points=n_points, same_atom=eigvals)
        self.S_sk_tables = S_sk_dict
        self.H_sk_tables = H_sk_dict
    
    def _create_dipole_file_dict(self):
        """create a dictionary where for every element combination there is a custom object,
        that contains all the information from the .skf file
        """
        r_sk_dict = {}
        for element_comb in self.elem_pairs:
            delta_R, n_points, r, atom_transitions = self._read_sk_file(elem_pair=element_comb, dipole=True)
            r_sk_dict[element_comb] = SKTable(table=r, deltaR=delta_R, n_points=n_points, same_atom=atom_transitions)
        self.r_sk_tables = r_sk_dict

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

    def _create_integral_dict(self, sk_table, posA, posB, operator, sk_table_dipole=None):
        """for one set of atom positions and a certain operator (S,H, r) create a dictionary for with quantum numbers as keys"""
        if operator == 'r':
            assert sk_table_dipole != None
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
        if operator == 'r':
            deltaR_dipole = sk_table_dipole.deltaR
            table_dipole = sk_table_dipole.table
            n_points_dipole = sk_table_dipole.n_points

            idx_pstart = 1
            idx_pend = 3
            D_r = D[idx_pstart:idx_pend+1, idx_pstart:idx_pend+1]
            D_dipole = np.kron(D_single, np.kron(D_r, D_single))
            D_dipole = np.real(D_dipole)

            R_grid_dipole = deltaR_dipole + deltaR_dipole * np.arange(n_points_dipole) 
            cs_dipole = CubicSpline(R_grid_dipole, table_dipole) 
            integral_vec_dipole = np.zeros((len(self.quant_nums_dipole)))
            for i, key in enumerate(sorted(INTEGRALS_DIPOLE, key= lambda x: x[0])):
                integral_vec_dipole[key[0]] = cs_dipole(R)[i]
            dipole_elements = D_dipole @ integral_vec_dipole

            #consider origin shift
            overlap_blocks = integrals.reshape(16,16)
            shift_term = np.tile(overlap_blocks, (1,3)).reshape(-1)
            space_factor = np.tile(np.repeat(posA, 16), 16)
            shift_term = shift_term * space_factor
            shifted_dipole = dipole_elements + shift_term
            integral_dict = {}
            for label in self.quant_nums_dipole:
                integral_dict[(label[1], label[2], label[3], label[4], label[5], label[6])] = shifted_dipole[label[0]]
        else:
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
    
    def _select_dipole_matrix_elements(self, max_lA, max_lB, integral_dict):
        """For dipole store components in 3rd dimension"""
        pair_matrix = np.zeros((3, get_norbs(max_lA), get_norbs(max_lB)))
        components = [(1, 1), (1,-1), (1,0)]
        for i, tup in enumerate(components):
            row_start = 0
            col_start = 0 
            for l1 in range(max_lA + 1):
                size_row = 2 * l1 +1
                for l2 in range(max_lB + 1):
                    size_col = 2 * l2 +1
                    block = np.zeros((size_row, size_col))
                    for mi, m in enumerate(range(-l1, l1 + 1)):
                        for ni, n in enumerate(range(-l2, l2 + 1)):
                            quant_nums = (l1, m, tup[0], tup[1], l2, n)
                            if any(quant_nums == item[1:] for item in INTEGRALS_DIPOLE.keys()):
                                # match = [item for item in INTEGRALS_DIPOLE.keys() if item[1:] == quant_nums][0]
                                block[mi, ni] = integral_dict[quant_nums]
                            else:
                                block[mi, ni] = 0
                    pair_matrix[i, row_start:row_start+size_row, col_start:col_start+size_col] = block
                    col_start += size_col
                col_start = 0
                row_start += size_row
        return pair_matrix

    def _calculate_atom_block(self, types, posA, posB, max_lA, max_lB, operator):
        """for two atoms, calculate the block of all relevant orbitals
        for dipole the block is a 3D array with the first axis for the 3 components x,y,z"""
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
                eigenvalues = sk_table.same_atom_vals
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
            sk_table = self.S_sk_tables[types]
            sk_table_dipole = self.r_sk_tables[types]
            if same_atom:
                integral_dict = sk_table_dipole.same_atom_vals
                block = self._select_dipole_matrix_elements(max_lA=max_lA, max_lB=max_lB, integral_dict=integral_dict)
            else: 
                integral_dict = self._create_integral_dict(sk_table=sk_table, posA=posA, posB=posB, operator='r', sk_table_dipole=sk_table_dipole)
                block = self._select_dipole_matrix_elements(max_lA=max_lA, max_lB=max_lB, integral_dict=integral_dict)
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
        self.n_lattice = np.shape(np.unique(R, axis=0))[0]
        for i, pair in enumerate(pairA):
            R_triple = (int(R[i,0]), int(R[i,1]), int(R[i,2]))
            if operator == 'r':
                matrix = lattice_dict.setdefault(R_triple, np.zeros((3, self.total_orbs, self.total_orbs)))
            else:
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
            if operator == 'r':
                matrix[:,start_rows:start_rows+n_rows, start_cols:start_cols+n_cols] = block
            else:
                matrix[start_rows:start_rows+n_rows, start_cols:start_cols+n_cols] = block
            lattice_dict[R_triple] = matrix
        assert np.shape(np.unique(R, axis=0))[0] == len(lattice_dict)
        return lattice_dict
    
    def _find_block_pos(self, idx):
        orb_previous = np.sum(self.orbnumbers[:idx])
        return int(orb_previous)
    
    def write_seedname(self):
        filename = 'seedname_tb.dat'
        lattice_dict_S = self._calculate_lattice_dict(operator='S')
        lattice_dict_H = self._calculate_lattice_dict(operator='H')
        lattice_dict_r = self._calculate_lattice_dict(operator='r')
        with open(filename, 'w') as f:
            f.write("Date\n")
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
                H_array = lattice_dict_H[point]
                A = np.real(H_array)
                B = np.imag(H_array)
                C = np.real(S_array)
                D = np.imag(S_array)
                for i in range(np.shape(S_array)[0]):
                    for j in range(np.shape(S_array)[1]):
                        print(f"{i} {j}\t{A[i,j]:.18e}\t{B[i,j]:.18e}\t{C[i,j]:.18e}\t{D[i,j]:.18e}", file=f)
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
                for m in range(np.shape(r_array)[0]):
                    for n in range(np.shape(r_array)[1]):
                        i = m+1
                        j = n+1
                        print(f"{i} {j}\t{xre[m,n]:.18e}\t{xim[m,n]:.18e}\t{yre[m,n]:.18e}\t{yim[m,n]:.18e}\t{zre[m,n]:.18e}\t{zim[m,n]:.18e}", file=f)
                
                
    
    def write_hamoversqr(self):
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
    def __init__(self, table, deltaR, n_points, same_atom=None):
        self.table = table
        self.deltaR = deltaR
        self.n_points = n_points
        self.same_atom_vals = same_atom #list for S and H, dict for r


cutoffs = {('C','C'): 1.85, ('H', 'C'): 1, 
           ('C', 'H'): 5,
             ('H', 'H'): 1}
max_l = {'C':1, 'H':0}
graphene = graphene('CC', size=(1,1,1), vacuum=10)
benzene = molecule('C6H6')

posA = graphene.positions[0]
posB = graphene.positions[1]
lcao_graphene = Seedname_TB(graphene, skpath="skfiles/self_made", maxl_dict=max_l, skpath_dipole="skfiles/self_made_dipole")
# block = lcao_graphene._calculate_atom_block(types=('C','C'), posA=posA, posB=posB, max_lA=1, max_lB=1, operator='r')
# lcao_graphene.write_seedname()
# print(lcao_graphene._calculate_lattice_dict('r')[(0,0,0)])
# sk_table = lcao_graphene.S_sk_tables[('C','C')]
# sk_table_dipole = lcao_graphene.r_sk_tables[('C','C')]
# dipoles = lcao_graphene._create_integral_dict(sk_table=sk_table, posA=posA, posB=posB, operator='r', sk_table_dipole=sk_table_dipole)
# pair_matrix = lcao_graphene._select_dipole_matrix_elements(max_lA=1, max_lB=1, integral_dict=dipoles)
# lcao_graphene._calculate_lattice_dict(operator='S')
# lcao_graphene._calculate_lattice_dict(operator='H')
# lcao_graphene._calculate_lattice_dict(operator='r')


# lattice_dict_S = lcao_graphene._calculate_lattice_dict(operator='S')
lcao_graphene.write_seedname()
