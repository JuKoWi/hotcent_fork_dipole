from hotcent.new_dipole.compare_integration_methods import compare_integrals
from hotcent.new_dipole.slako_dipole import INTEGRALS_DIPOLE
from hotcent.new_dipole.integrals import get_index_list_dipole
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


zeta = [0.5,0.5,0.5,0.5]
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
        f.readline()
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
        header = str(vec1) + '\n' + str(vec2)
        print(table.shape)
        np.savetxt(fname='comparison.txt', X=table, header=header)

def plot_heatmap_orb_accuracy(filename):
    data = np.loadtxt(filename) 
    sk_minus_analyt = data[:,0] - data[:,1]
    diff_array = np.zeros(shape=(3,16,16))
    count = 0
    for i in range(16):
        for j in range(3):
            for k in range(16):
                diff_array[j,i,k] = sk_minus_analyt[count]
                count += 1
    fig, ax = plt.subplots(figsize=(10,10))
    qnum_labels = np.repeat([0,1,2,3], repeats=[1,3,5,7])
    qnum_labels = qnum_labels.astype(str)
    remove_label = [2,3,5,6,7,8,10,11,12,13,14,15]
    for i in remove_label:
        qnum_labels[i] = ''
    x = np.arange(len(qnum_labels))
    X, Y = np.meshgrid(x, x)
    cmap =ax.pcolormesh(X, Y, diff_array[0])
    ax.set_xticks(x)
    ax.set_xticklabels(qnum_labels)
    ax.set_yticks(x)
    ax.set_yticklabels(qnum_labels)
    ax.set_xlabel(r'$l$')
    ax.set_ylabel(r'$l$')
    fig.colorbar(cmap, ax=ax)
    ax.set_aspect('equal')
    plt.savefig('heatmap_orb_acc.pdf')
    plt.show()
    
    
    



    
# convert_human_readable(filename='comparison_dipole_rand_orient.txt')
plot_heatmap_orb_accuracy(filename='comparison.txt')