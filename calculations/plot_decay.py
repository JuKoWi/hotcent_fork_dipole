from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.pos_op.offsite_twocenter_new import Offsite2cTable
from hotcent.pos_op.offsite_twocenter_posop import Offsite2cTablePosOp
from hotcent.pos_op.utils import bohr_to_angstrom
from hotcent.pos_op.slako_new import INTEGRALS, get_hotcent_style_index
from hotcent.pos_op.slako_dipole import INTEGRALS_POSOP, convert_sk_index
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.bbox'] = 'tight'

READABLE_LABELS = {
    "d5": "d_{x^2 - y^2}",
    "d4": "d_{xz}",
    "d3": "d_{z^2}",
    "d1": "d_{xy}",
    "d2": "d_{yz}",
    "px": "p_x",
    "py": "p_y",
    "pz": "p_z",
    "ss": "s",
                   }

"""Plot data from file"""
def slowest_decay_from_file(filename, num_dipole, threshold, atol=1e-7):
    table = np.loadtxt(fname=filename, skiprows=3)
    label_list = sorted(INTEGRALS_POSOP.keys(), key=lambda x: x[0])

    top_keys = []
    top_indices = []
    top_rows = []

    table = np.abs(table.T)
    mask = table < threshold
    idx = np.where(
        mask.any(axis=1),
        table.shape[1] - 1 - mask[:, ::-1].argmax(axis=1),
        -1
    ) #compute slowest decay index per row 
    order = np.argsort(idx)[::-1] # rank by slowest decay
    for row_idx in order:
        row = table[row_idx]
        if len(top_rows) == num_dipole:
            break
        if idx[row_idx] < 0:
            continue
        if np.allclose(row, np.zeros_like(row)):
            continue
        if top_rows:
            duplicate = np.any(
                np.all(np.isclose(np.abs(top_rows), np.abs(row), atol=atol), axis=1)
            )
            if duplicate:
                continue
        top_keys.append(label_list[row_idx])
        top_indices.append(row_idx)
        top_rows.append(row)
    return top_keys, top_indices, top_rows

def plot_decay_file(filename, num_dipole, threshold, startline):
    skf_file = Path(filename) 
    keys, idx, data = slowest_decay_from_file(filename=filename, num_dipole=num_dipole, threshold=threshold)
    data = np.array(data)
    with open(filename, "r") as f:
        line1 = f.readline()
        line1 = line1.replace(",", " ")
        parameters = line1.split()
        dr, nr = float(parameters[0]), int(parameters[1])
    r_au = np.linspace(start=dr, stop=nr*dr, num=nr, endpoint=True)
    r_angst = bohr_to_angstrom(r_au)
    symbols = filename.replace("-", "_")
    symbols = symbols.split("_")
    typeA = symbols[0]
    typeB = symbols[1]
    fig, ax = plt.subplots(figsize=(6,4.5))
    for i, key in enumerate(keys):
        int_label = convert_sk_index(key)
        orba, comp, orbb = int_label[:2], int_label[3], int_label[4:6]
        ax.semilogy(r_angst[startline:], data[i,startline:], label=rf"$\langle {READABLE_LABELS[orba]}|\hat{{r}}_{{{comp}}}|{READABLE_LABELS[orbb]}\rangle$")
    ax.legend()
    ax.set_xlim(left=bohr_to_angstrom(dr), right=7)
    ax.set_ylim(bottom=1e-15)
    ax.set_xlabel(r'$R$ $[\mathrm{\AA}]$')
    ax.set_ylabel(r'$d$ $[\mathrm{\AA}]$')
    plt.savefig(f"dipole_distance_decay{typeA}-{typeB}_top{num_dipole}.pdf")
    plt.show()

def find_slowest_decay(offsite_obj, num_dipole, threshold, atol=1e-7):
    label_list = sorted(INTEGRALS_POSOP.keys(), key=lambda x: x[0])

    top_keys = []
    top_indices = []
    top_rows = []

    for key, table in offsite_obj.tables.items():
        table = np.abs(table.T)
        mask = table < threshold
        idx = np.where(
            mask.any(axis=1),
            table.shape[1] - 1 - mask[:, ::-1].argmax(axis=1),
            -1
        ) #compute slowest decay index per row 
        order = np.argsort(idx)[::-1] # rank by slowest decay
        for row_idx in order:
            row = table[row_idx]
            if len(top_rows) == num_dipole:
                break
            if idx[row_idx] < 0:
                continue
            if np.allclose(row, np.zeros_like(row)):
                continue
            if top_rows:
                duplicate = np.any(
                    np.all(np.isclose(np.abs(top_rows), np.abs(row), atol=atol), axis=1)
                )
                if duplicate:
                    continue
            top_keys.append(label_list[row_idx])
            top_indices.append(row_idx)
            top_rows.append(row)
    return top_keys, top_indices, top_rows

def plot_dipole_decay(offsite_obj, num_dipole, threshold):
    keys, idx, data = find_slowest_decay(offsite_obj=offsite_obj, num_dipole=num_dipole, threshold=threshold)
    data = [np.abs(bohr_to_angstrom(d)) for d in data]
    r = offsite_obj.Rgrid
    r_angst = bohr_to_angstrom(r)
    fig, ax = plt.subplots(figsize=(20,9))
    typeA = offsite_obj.pairs[0][0].symbol
    typeB = offsite_obj.pairs[0][1].symbol
    for i, key in enumerate(keys):
        int_label = convert_sk_index(key)
        orba, comp, orbb = int_label[:2], int_label[3], int_label[4:6]
        ax.semilogy(r_angst, data[i], label=rf"$\langle {READABLE_LABELS[orba]}|\hat{{r}}_{{{comp}}}|{READABLE_LABELS[orbb]}\rangle$")
    ax.legend()
    ax.set_xlim(left=0, right=6)
    ax.set_xlabel(r'$R$ $[\mathrm{\AA}]$')
    ax.set_ylabel(r'$d$ $[\mathrm{\AA}]$')
    plt.savefig(f"dipole_distance_decay{typeA}-{typeB}_top{num_dipole}.pdf")
    plt.show()

def plot_dipole_decay_selected(sk_file, homonuclear, labels, readable_labels, eigvals, startline):
    with open(sk_file, 'r') as f:
        line1 = f.readline()
        line1 = line1.replace(',', ' ')
        parts = [p.strip() for p in line1.split()]
        dr, nr = float(parts[0]), int(parts[1])
    if homonuclear:
        skiprows = 3
    else:
        skiprows = skiprows = 2
    data = np.loadtxt(fname=sk_file, skiprows=skiprows)
    sorted_tuple = sorted(INTEGRALS_POSOP.keys(), key=lambda k: k[0])
    sorted_labelnum = [k[0] for k in sorted_tuple]
    fig, ax = plt.subplots(figsize=(6,4.5))
    x = np.linspace(start=dr, stop=nr*dr, num=nr, endpoint=True)
    x_angst = bohr_to_angstrom(x)
    for i,l in enumerate(labels):
        column = sorted_labelnum.index(l)
        dipole = data[:,column]
        ax.plot(x_angst[startline:], dipole[startline:], label=readable_labels[i])
    for i, eig in enumerate(list(set(eigvals))):
        if i == 0:
            ax.hlines(y=eig, color='black', linestyle='--', xmin=0, xmax=6, label='atomic transition')
        else:
            ax.hlines(y=eig, color='black', linestyle='--', xmin=0, xmax=6) 
    ax.set_xlim(left=0, right=6)
    ax.set_xlabel(r'$R$ [$\AA$]')
    ax.set_ylabel(r'$d$ [$a_0$]')
    ax.legend(markerscale=2)
    element,_ = sk_file.split('.')
    plotname = "plotskf" + element
    for l in labels:
        plotname = plotname + (f"-{l}")
    plotname = plotname + '.pdf'
    plt.savefig(plotname)
    plt.show()
    
def find_slowest_overlap_decay(offsite_obj, num_dipole, threshold, atol=1e-7):
    label_list = sorted(INTEGRALS.keys(), key=lambda x: x[0])

    top_keys = []
    top_indices = []
    top_rows = []

    for key, table in offsite_obj.tables.items():
        table = np.abs(table.T)
        mask = table < threshold
        idx = np.where(
            mask.any(axis=1),
            table.shape[1] - 1 - mask[:, ::-1].argmax(axis=1),
            -1
        ) #compute slowest decay index per row 
        order = np.argsort(idx)[::-1] # rank by slowest decay
        for row_idx in order:
            row = table[row_idx]
            if len(top_rows) == num_dipole:
                break
            if idx[row_idx] < 0:
                continue
            if np.allclose(row, np.zeros_like(row)):
                continue
            if top_rows:
                duplicate = np.any(
                    np.all(np.isclose(np.abs(top_rows), np.abs(row), atol=atol), axis=1)
                )
                if duplicate:
                    continue
            top_keys.append(label_list[row_idx])
            top_indices.append(row_idx)
            top_rows.append(row)
    return top_keys, top_indices, top_rows
    
def plot_overlap_decay(offsite_obj, num_overlap, threshold):
    keys, idx, data = find_slowest_overlap_decay(offsite_obj=offsite_obj, num_dipole=num_overlap, threshold=threshold)
    r = offsite_obj.Rgrid
    r_angst = bohr_to_angstrom(r)
    fig, ax = plt.subplots(figsize=(20,9))
    typeA = offsite_obj.pairs[0][0].symbol
    typeB = offsite_obj.pairs[0][1].symbol
    for i, key in enumerate(keys):
        int_label = get_hotcent_style_index(key)
        orba, comp, orbb = int_label[:2], int_label[3:]
        ax.semilogy(r_angst, data[i], label=rf"{typeA}-{typeB}: $\langle {orba}|{orbb}\rangle$")
    ax.legend()
    ax.set_xlim(left=0, right=6)
    ax.set_xlabel(r'r / $\mathrm{\AA}$')
    ax.set_ylabel(r'$\mathrm{|\langle \phi_\mu|\phi_\nu \rangle|}$ / $\mathrm{\AA}$')
    plt.savefig(f"overlap_distance_decay{typeA}-{typeB}_top{num_overlap}.pdf")
    plt.show()

xc='GGA_X_PBE+GGA_C_PBE'

# atomMo = AtomicDFT('Mo',
#                 xc = xc,
#                 perturbative_confinement=False,
#                 configuration='[Kr] 4d4 5s2 5p0',
#                 valence=['4d', '5s', '5p'],
#                 confinement=PowerConfinement(r0=40, s=4),
#                 scalarrel=True,
#                 maxiter=2500,
#                 timing=False,
#                 # nodegpts=150,
#                 mix=0.2,
#                 txt='-',
#                 rmax=100,
#                 )
# atomMo.run()

rcovS = 3.9
rcovMo = 4.3
confS = PowerConfinement(r0=50, s=4)
confMo = PowerConfinement(r0=50, s=4)

wf_confS = {'3s': PowerConfinement(r0=rcovS, s=4.6),
           '3p': PowerConfinement(r0=rcovS, s=4.6),
           '3d': PowerConfinement(r0=rcovS, s=4.6),
           }

wf_confMo = {'4d': PowerConfinement(r0=rcovMo, s=11.6),
           '5s': PowerConfinement(r0=rcovMo, s=11.6),
           '5p': PowerConfinement(r0=rcovMo, s=11.6),
           }

# atomMo.set_confinement(confMo)
# atomMo.set_wf_confinement(wf_confinement=wf_confMo)
# atomMo.run()

rmin, dr, N = 0.4, 0.02, 600

# off2c_dipoleMo = Offsite2cTableDipole(atomMo, atomMo, timing=False)
# off2c_dipoleMo.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-10)
# off2c_dipoleMo.write_dipole()

# off2c_Mo = Offsite2cTable(atomMo, atomMo, timing=False)
# off2c_Mo.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-10)
# off2c_dipoleMo.write_dipole()

# element = 'C'
# xc = 'GGA_X_PBE+GGA_C_PBE'
# r0 = 3.2 # Bohr
# conf = PowerConfinement(r0=50.0, s=4)
# wf_conf = {'2s': PowerConfinement(r0=r0, s=8.2),
#            '2p': PowerConfinement(r0=r0, s=8.2),
#            }

# atom = AtomicDFT(element,
#                 xc = xc,
#                  confinement=conf,
#                  perturbative_confinement=False,
#                  configuration='[He] 2s2 2p2',
#                  valence=['2s', '2p'],
#                  scalarrel=True,
#                  maxiter=2500,
#                  timing=False,
#                  nodegpts=150,
#                  mix=0.2,
#                  txt='-',
#                  )
# atom.run()
# eigenvalues=atom.enl

# atom.set_confinement(conf)
# atom.set_wf_confinement(wf_confinement=wf_conf)
# atom.run()

# rmin, dr, N = 0.4, 0.02, 900
# off2c = Offsite2cTable(atom, atom, timing=False)
# off2c.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-9)
# off2c.write()

# print(atomMo.configuration)
# plot_dipole_decay(offsite_obj=off2c, num_dipole=5, threshold=1e-5)
# plot_overlap_decay(offsite_obj=off2c_dipoleMo, num_overlap=5, threshold=1e-7)

labels = [1,5,16]
readable_labels = [
    r'$\langle s | \hat{r}_y | p_y \rangle$',
    r'$\langle s | \hat{r}_y | d_{yz} \rangle$',
    r'$\langle s | \hat{r}_z | s \rangle$' 
    ]
eigvals = [-1.528113,0,0]

# plot_dipole_decay_selected(sk_file='Mo-Mo_dipole.skf', homonuclear=True, labels=labels, readable_labels=readable_labels, eigvals=eigvals, startline=20)
plot_decay_file(filename="Mo-Mo_dipole.skf", num_dipole=3, threshold=1e-7, startline=19)
