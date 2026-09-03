from hotcent.pos_op.compare_integration_methods import compare_integrals
from hotcent.pos_op.slako_dipole import INTEGRALS_POSOP
from hotcent.pos_op.symbolic_integrals import get_index_list_dipole
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

plt.rcParams.update({"font.size": 20})


zeta = [0.5, 0.5, 0.5, 0.5]
compare_integrals(zeta1=zeta, use_existing_skf=True, dipole=True)


def convert_human_readable(filename):
    """convert table with lables for orbital cominations into numpy parsable file"""
    with open(filename, "r") as f:
        line1 = f.readline()
        vec1 = line1.split("[")[1]
        vec1 = vec1.split("]")[0]
        vec1 = np.array([float(v) for v in vec1.split()])
        line2 = f.readline()
        vec2 = line2.split("[")[1]
        vec2 = vec2.split("]")[0]
        vec2 = np.array([float(v) for v in vec2.split()])
        f.readline()
        table = []
        while True:
            line = f.readline()
            if not line:
                break
            elif line.startswith("Testing"):
                continue
            else:
                table.append([float(num) for num in line.split()])
        table = np.array(table)
        header = str(vec1) + "\n" + str(vec2)
        print(table.shape)
        np.savetxt(fname="comparison.txt", X=table, header=header)


# def plot_heatmap_orb_accuracy(filename):
#     data = np.loadtxt(filename)
#     sk_minus_analyt = data[:,0] - data[:,1]
#     diff_array = np.zeros(shape=(3,16,16))
#     count = 0
#     for i in range(16):
#         for j in range(3):
#             for k in range(16):
#                 diff_array[j,i,k] = sk_minus_analyt[count]
#                 count += 1
#     fig, ax = plt.subplots(figsize=(10,10))
#     l_labels = np.repeat([0,1,2,3], repeats=[1,3,5,7])
#     l_labels = l_labels.astype(str)
#     remove_label = [2,3,5,6,7,8,10,11,12,13,14,15]
#     for i in remove_label:
#         l_labels[i] = ''
#     m_labels = []
#     for i in range(4):
#         for j in range(-i, i+1):
#             m_labels.append(j)
#     x = np.arange(len(l_labels))
#     X, Y = np.meshgrid(x, x)
#     cmap =ax.pcolormesh(X, Y, diff_array[0])
#     ax.set_xticks(x)
#     ax.set_xticklabels(l_labels)
#     ax.set_yticks(x)
#     ax.set_yticklabels(l_labels)
#     ax.set_xlabel(r'$l$')
#     ax.set_ylabel(r'$l$')

#     fig.colorbar(cmap, ax=ax)
#     ax.set_aspect('equal')
#     plt.savefig('heatmap_orb_acc.pdf')
#     plt.show()


def plot_heatmap_orb_accuracy(filename):
    data = np.loadtxt(filename)
    sk_minus_analyt = data[:, 0] - data[:, 1]
    diff_array = np.zeros(shape=(3, 16, 16))
    count = 0
    for i in range(16):
        for j in range(3):
            for k in range(16):
                diff_array[j, i, k] = sk_minus_analyt[count]
                count += 1

    # fig, ax = plt.subplots(figsize=(10,10))
    fig, (ax, cax) = plt.subplots(
        1, 2, figsize=(11, 10), gridspec_kw={"width_ratios": [20, 1], "wspace": 0.05}
    )

    # --- Labels ---
    m_labels = []
    for i in range(4):
        for j in range(-i, i + 1):
            m_labels.append(str(j))

    # l group centers and edges (in tick-position units)
    # l=0: pos 0 | l=1: 1-3 | l=2: 4-8 | l=3: 9-15
    l_centers = [0, 2, 6, 12]
    l_edges = [0.5, 3.5, 8.5]
    l_values = ["0", "1", "2", "3"]

    x = np.arange(16)
    X, Y = np.meshgrid(x, x)
    cmap = ax.pcolormesh(X, Y, diff_array[0])
    # fig.colorbar(cmap, ax=ax)
    fig.colorbar(cmap, cax=cax)
    ax.set_box_aspect(1)

    # --- Inner row: m_labels ---
    ax.set_xticks(x)
    ax.set_xticklabels(m_labels)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel(r"$m$")

    ax.set_yticks(x)
    ax.set_yticklabels(m_labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel(r"$m$")

    # --- Outer row x-axis: l_labels ---
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 48))
    ax2.set_xticks(l_centers)
    ax2.set_xticklabels(l_values)
    ax2.tick_params(axis="x", length=0)
    ax2.set_xlabel(r"$l$")

    # --- Outer row y-axis: l_labels ---
    ax3 = ax.twinx()
    ax3.set_ylim(ax.get_ylim())
    ax3.yaxis.set_ticks_position("left")
    ax3.yaxis.set_label_position("left")
    ax3.spines["left"].set_position(("outward", 48))
    ax3.set_yticks(l_centers)
    ax3.set_yticklabels(l_values)
    ax3.tick_params(axis="y", length=0)
    ax3.set_ylabel(r"$l$")

    # --- Dividing lines between l groups ---
    for edge in l_edges:
        ax.axvline(edge, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(edge, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    plt.savefig("heatmap_orb_acc.pdf", bbox_inches="tight")
    plt.show()


# convert_human_readable(filename='comparison_dipole_rand_orient.txt')
plot_heatmap_orb_accuracy(filename="comparison.txt")
