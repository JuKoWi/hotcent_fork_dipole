import numpy as np
from ase import Atoms 
from ase.neighborlist import *
from hotcent.new_dipole.assemble_integrals import SK_Integral
from hotcent.new_dipole.utils import *


def write_atom_matrix(structure, maxl_dict, skdir):
    """calculate oversqr.dat and hamsqr1.dat for a geometry to compare intermediate results 
    with dftb+ """
    idx1, idx2 = neighbor_list(quantities='ij', a=structure, self_interaction=True, cutoff=10)
    atomtypes = structure.get_chemical_symbols()
    orbnumbers = [get_norbs(maxl_dict[key]) for key in atomtypes]
    total_orbs = np.sum(orbnumbers)
    table_overlap = np.zeros((total_orbs, total_orbs))
    table_hamiltonian = np.zeros((total_orbs, total_orbs))
    block_starts = []

    count = 0
    for type in structure.get_chemical_symbols():
        block_starts.append(count)
        count += get_norbs(maxl_dict[type])

    for i, firstatom in enumerate(idx1): # iteration over pairs, not first atom
        secondatom = idx2[i]
        typeA = structure.symbols[firstatom]
        typeB = structure.symbols[secondatom]
        posA = structure.positions[firstatom]
        posB = structure.positions[secondatom]
        max_lA = maxl_dict[typeA]
        max_lB = maxl_dict[typeB]
        pair = Atoms(f'{typeA}{typeB}', positions=[posA, posB])
        skpath = skdir + f'/{typeA}-{typeB}.skf'
        
        with open(skpath, 'r') as f:
            next(f)
            line2 = f.readline() 
        eigenvalues = line2.replace(',', ' ').split()
        eigenvalues = np.flip(eigenvalues[:3])

            
        pair_integral = SK_Integral()
        pair_integral.load_atom_pair(atoms=pair)
        pair_integral.load_sk_file(path=skpath, homonuclear=(typeA== typeB))
        pair_integral.calculate(hamilton=True)
        pair_integral.calculate(hamilton=False)

        start_row = block_starts[firstatom]
        stop_row = start_row + get_norbs(max_lA)
        start_col = block_starts[secondatom]
        stop_col = start_col + get_norbs(max_lB) 
        if secondatom == firstatom:
            assert typeA == typeB
            table_overlap[start_row:stop_row, start_col:stop_col] = np.eye(get_norbs(max_lA))
            diag = np.eye(get_norbs(maxl=max_lA))
            count = 0
            for i in range(max_lA+1):
                for j in range(2*i+1):
                    diag[count+j, count+j] = eigenvalues[i]
                count += 2*i +1
            table_hamiltonian[start_row:stop_row, start_col:stop_col] = diag
        else:
            overlap, hamiltonian = pair_integral.select_matrix_elements(max_lA=max_lA, max_lB=max_lB)
            table_overlap[start_row:stop_row, start_col:stop_col] = overlap
            table_hamiltonian[start_row:stop_row, start_col:stop_col] = hamiltonian
    
    header1 = f"#\tREAL\tNALLORB\tNKPOINT\n"
    header2 = f"\tT\t{total_orbs}\t1\n" 
    header3 = "#IKPOINT\n"
    header4 = "\t1\n"
    header5 = "#MATRIX"
    header = header1 + header2 + header3 + header4 + header5
    np.savetxt(fname='oversqr_hotcent.dat', delimiter='\t', fmt='%+.18e', X=table_overlap, header=header, comments='')
    np.savetxt(fname='hamsqr1_hotcent.dat', delimiter='\t', fmt='%+.18e', X=table_hamiltonian, header=header, comments='')





    
