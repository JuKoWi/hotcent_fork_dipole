import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
import numpy as np
import h5py
import re

def plot_grid_vs_dftb(filename):
    with h5py.File(filename, 'r') as f:
        grid = np.imag(np.array(f['grid']['p_grid']))
        dftb = np.imag(np.array(f['dftb']['p_dftb']))
        grid_s = np.real(np.array(f['grid']['S_grid']))
        dftb_s = np.real(np.array(f['dftb']['S_dftb']))
 
    labels = ['x', 'y', 'z']
    rows = [
        [(lab, grid[k]) for k, lab in enumerate(labels)] ,
        [(lab, dftb[k]) for k, lab in enumerate(labels)] ,
        [(f'diff {lab}', dftb[k] - grid[k]) for k, lab in enumerate(labels)] ,
    ]
 
    all_data = [data for row in rows for _, data in row]
    vmin = min(np.nanmin(d) for d in all_data)
    vmax = max(np.nanmax(d) for d in all_data)
    norm = Normalize(vmin=vmin, vmax=vmax)
 
    fig, ax = plt.subplots(nrows=3, ncols=3, constrained_layout=True)
    for i, row in enumerate(rows):
        for j, (title, data) in enumerate(row):
            im = ax[i, j].imshow(data, norm=norm)
            ax[i, j].set_title(title)
 
    fig.colorbar(im, ax=ax, location='right', shrink=0.8)
    plt.show()

L_SYMBOLS = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
 
 
def _read_strings(dset):
    """h5py 3 returns stored strings as bytes (b'2p'); decode them to str."""
    return [s.decode() if isinstance(s, bytes) else str(s) for s in dset[()]]
 
 
def plot_truncation_error(files: list, lmax: list, show_l, show_comp, use_grid=False, log=False):
    """Bar plot of the basis-truncation error of x_c * phi for every basis function with l = show_l.
 
    files     : HDF5 files written by check_basis_completeness, one per lmax
    show_comp : Cartesian component 0, 1, 2 (x, y, z)
    use_grid  : plot 'diff_grid' (residual integrated on the grid) instead of 'diff'
    """
    l = L_SYMBOLS[show_l]
    series = []  # (lmax, labels, values) per file
    for file, lm in zip(files, lmax):
        with h5py.File(file, 'r') as f:
            res = np.array(f['diff_grid' if use_grid else 'diff'])  # shape (3, n_basis)
            atoms = _read_strings(f['atoms'])
            nls = _read_strings(f['nl'])
        idx = [j for j, nl in enumerate(nls) if nl[-1] == l]
        if not idx:
            print(f"{file}: no {l} functions (lmax = {lm}), skipped")
            continue
        # number the 2l+1 functions of each (atom, nl) so the bars are distinguishable
        seen, labels = {}, []
        for j in idx:
            key = (atoms[j], nls[j])
            labels.append(f"{nls[j]}({atoms[j]}) {seen.get(key, 0)}")
            seen[key] = seen.get(key, 0) + 1
        series.append((lm, labels, res[show_comp, idx]))
 
    if not series:
        raise ValueError(f"none of the files contain {l} functions")
    labels = series[0][1]
    if any(s[1] != labels for s in series):
        raise ValueError("files have different sets of basis functions for this l")
 
    n = len(series)
    width = 0.8 / n                       # the n bars of one group fill 80% of the slot
    x = np.arange(len(labels))
    fig, ax = plt.subplots(layout="constrained")
    for i, (lm, _, values) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * width, values, width=width, label=r"$l_\text{max}$ = "+f"{lm}")
    ax.set_xticks(x, labels=labels)       # ticks at the centre of each group
    ax.set_ylabel(rf"$F_{'xyz'[show_comp]}$" + (" (grid)" if use_grid else ""))
    if log:
        ax.set_yscale("log")
    ax.legend()
    plt.savefig('r_fidelity.pdf')
    plt.show()
    return fig, ax

if __name__ == "__main__":
    plot_truncation_error(files=['bs_completeness_Mo_l1.hdf5','bs_completeness_Mo_l2.hdf5'], lmax=[1,2], show_l=1, show_comp=2, use_grid=False)
    # plot_grid_vs_dftb(filename='graphene_rpS.hdf5')