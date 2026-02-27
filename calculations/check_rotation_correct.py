from hotcent.new_dipole.compare_integration_methods import compare_integrals
import numpy as np

zeta = [1,1,1,1]
# compare_integrals(zeta1=zeta, use_existing_skf=False, dipole=True)

def convert_human_readable(filename):
    """convert table with lables for orbital cominations into numpy parsable file"""
    with open(filename, "r") as f:
        line1 = f.readline()
        vec1 = line1.split('[')[1]
        vec1 = vec1.split(']')[0]
        vec1 = np.array([float(v) for v in vec1.split()])
        line2 = f.readline()
        vec2 = line2.split('[')[1]
        vec2 = vec2.split(']')[0]
        vec2 = np.array([float(v) for v in vec2.split()])
        f.next()
        table = []
        while True:
            line = f.readline()
            if not line:
                break
            elif line.startswith('Testing'):
                continue
            else:
                table.append([float(num) for num in line.split()])
        table = np.array(table)
        np.savetxt(fname='comparison.txt', X=table, )



    
convert_human_readable(filename='comparison_dipole_rand_orient.txt')