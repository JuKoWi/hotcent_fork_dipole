from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.new_dipole.offsite_twocenter_new import Offsite2cTable
from hotcent.new_dipole.offsite_twocenter_dipole import Offsite2cTableDipole
from hotcent.new_dipole.utils import bohr_to_angstrom
from hotcent.new_dipole.slako_dipole import INTEGRALS_DIPOLE, convert_sk_index
import matplotlib.pyplot as plt
import numpy as np
import sys

plt.rcParams.update({'font.size': 19})
plt.rcParams['savefig.bbox'] = 'tight'

import numpy as np

def find_slowest_decay(offsite_obj, num_dipole, threshold, atol=1e-7):
    label_list = sorted(INTEGRALS_DIPOLE.keys(), key=lambda x: x[0])

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
    fig, ax = plt.subplots(figsize=(16,9))
    typeA = offsite_obj.pairs[0][0].symbol
    typeB = offsite_obj.pairs[0][1].symbol
    for i, key in enumerate(keys):
        int_label = convert_sk_index(key)
        orba, comp, orbb = int_label[:2], int_label[3], int_label[4:6]
        ax.semilogy(r_angst, data[i], label=rf"{typeA}-{typeB}: $\langle {orba}|\hat{{r}}_{{{comp}}}|{orbb}\rangle$")
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_xlabel(r'r / $\mathrm{\AA}$')
    ax.set_ylabel(r'$\mathrm{|\langle \phi_\mu|\hat{r}_i|\phi_\nu \rangle|}$ / $\mathrm{\AA}$')
    plt.savefig(f"dipole_distance_decay{typeA}-{typeB}_top{num_dipole}.pdf")
    plt.show()


xc='GGA_X_PBE+GGA_C_PBE'

atomMo = AtomicDFT('Mo',
                xc = xc,
                perturbative_confinement=False,
                configuration='[Kr] 4d4 5s2 5p0',
                valence=['4d', '5s', '5p'],
                confinement=PowerConfinement(r0=40, s=4),
                scalarrel=True,
                maxiter=2500,
                timing=False,
                # nodegpts=150,
                mix=0.2,
                txt='-',
                rmax=100,
                )
atomMo.run()

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

atomMo.set_confinement(confMo)
atomMo.set_wf_confinement(wf_confinement=wf_confMo)
atomMo.run()

rmin, dr, N = 0.4, 0.02, 900

off2c_dipoleMo = Offsite2cTableDipole(atomMo, atomMo, timing=False)
off2c_dipoleMo.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-9)
off2c_dipoleMo.write_dipole()

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
# off2c = Offsite2cTableDipole(atom, atom, timing=False)
# off2c.run(rmin, dr, N, nr=200, ntheta=400, wflimit=1e-9)
# off2c.write_dipole()

plot_dipole_decay(offsite_obj=off2c_dipoleMo, num_dipole=5, threshold=1e-5)

