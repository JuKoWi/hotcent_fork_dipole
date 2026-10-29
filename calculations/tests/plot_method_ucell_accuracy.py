import numpy as np
from ase.build import graphene, mx2, bulk
import matplotlib.pyplot as plt
from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator
import sys
import warnings
from hotcent.pos_op.utils import angstrom_to_bohr

_PAIRS = [(0, 1, "x", "y"), (1, 2, "y", "z"), (2, 0, "z", "x")]
_POSOP_LABELS = {
    "sk": "Full Slater-Koster",
    "onsite_expansion": "Onsite Expansion",
    "mulliken": "Mulliken",
}


def _commutator(A, B, S_inv=None):
    """[A, B] as a matrix. In a non-orthogonal basis the matrix of a product
    of two operators is A S^-1 B, not A B."""
    if S_inv is None:
        return A @ B - B @ A
    return A @ S_inv @ B - B @ S_inv @ A


def _plottable_part(C, rtol=1e-6):
    """Return whichever of Re(C) / Im(C) actually carries the commutator,
    and warn if the discarded part is not negligible."""
    re, im = np.real(C), np.imag(C)
    re_max, im_max = np.abs(re).max(), np.abs(im).max()
    big, small = max(re_max, im_max), min(re_max, im_max)
    if big > 0 and small > rtol * big:
        warnings.warn(
            f"commutator has both real (max {re_max:.2e}) and imaginary "
            f"(max {im_max:.2e}) parts; only the larger one is plotted"
        )
    return (re, "Re") if re_max >= im_max else (im, "Im")


def plot_component_commutation(sk_integrator, operator="r"):
    """Plot [O_x, O_y], [O_y, O_z], [O_z, O_x] for O = r (one row per
    position-operator approximation) or O = p (one row).

    """
    S_inv = np.linalg.inv(sk_integrator._calculate_lattice_dict(operator='S')[(0,0,0)])

    if operator == "r":
        approx = list(_POSOP_LABELS)
        original = sk_integrator.posop_approx
        try:
            ops = []
            for aprx in approx:
                sk_integrator.posop_approx = aprx
                ops.append(angstrom_to_bohr(sk_integrator._calculate_lattice_dict(operator="r")[(0, 0, 0)]))
        finally:
            # don't leave the integrator stuck in the last approximation
            sk_integrator.posop_approx = original
        row_labels = [_POSOP_LABELS[a] for a in approx]
    elif operator == "p":
        ops = [sk_integrator._calculate_lattice_dict(operator="p")[(0, 0, 0)]]
        row_labels = ["Momentum"]
    else:
        raise ValueError(f"operator must be 'r' or 'p', got {operator!r}")

    # grid[row][col] = (2-D array, "Re" or "Im")
    grid = [
        [_plottable_part(_commutator(op[i], op[j], S_inv)) for i, j, _, _ in _PAIRS]
        for op in ops
    ]

    vmax = max(np.abs(data).max() for row in grid for data, _ in row)
    if vmax == 0:
        print("All commutators are exactly zero.")
        vmax = 1.0  # avoid a degenerate colour scale

    fig, axs = plt.subplots(
        nrows=len(ops), ncols=3, squeeze=False, layout="constrained",
        figsize=(10, 3.2 * len(ops)),
    )
    bra = r"$\langle \phi_m |$"  
    ket = r"$| \phi_n \rangle$"
    for r, row in enumerate(grid):
        for c, (data, part) in enumerate(row):
            _, _, a, b = _PAIRS[c]
            im = axs[r, c].imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axs[r, c].set_title(f"{part} $[{operator}_{a}, {operator}_{b}]$")
        axs[r, 0].set_ylabel(f"{row_labels[r]}\n{bra}")
    for ax in axs[-1, :]:
        ax.set_xlabel(ket)
    fig.colorbar(im, ax=axs, shrink=0.9, label='[atomic units]')  # one shared colorbar: all panels use the same scale
    plt.savefig('component_commutator.pdf')
    plt.show()
    return fig, axs


if __name__ == '__main__':
    max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2, "Zn": 2, "O":1}
    graphene = graphene("CC", vacuum=10)
    zno = bulk('ZnO', crystalstructure='zincblende', a=4.62)
    MoS2 = mx2(vacuum=20)
    MoS2.pbc = (True, True, False)

    sk_intgrls_graphene = SlaterKosterIntegrator(
        graphene,
        skpath="sk_hdf5",
        maxl_dict=max_l,
        format="hdf5",
    )


    sk_integrals_mos2 = SlaterKosterIntegrator(
        MoS2,
        skpath="sk_hdf5",
        maxl_dict=max_l,
        format="hdf5",
    )
    plot_component_commutation(sk_integrator=sk_integrals_mos2, operator='r')
    sys.exit()

    approx = 'sk'
    sk_intgrls_graphene.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)
    approx = 'onsite_expansion'
    sk_intgrls_graphene.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)
    approx = 'mulliken'
    sk_intgrls_graphene.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)
    approx = 'onsite_only'
    sk_intgrls_graphene.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)
    approx = 'shift_only'
    sk_intgrls_graphene.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)

    lattice_vec = (0,0,0)
    sk_intgrls_graphene.posop_approx = 'sk'
    r_sk = np.real(sk_intgrls_graphene._calculate_lattice_dict(operator='r')[lattice_vec])
    sk_intgrls_graphene.posop_approx = "onsite_expansion"
    r_onsite_expand = np.real(sk_intgrls_graphene._calculate_lattice_dict(operator='r')[lattice_vec])
    sk_intgrls_graphene.posop_approx = 'mulliken'
    r_mulliken = np.real(sk_intgrls_graphene._calculate_lattice_dict(operator='r')[lattice_vec])
    sk_intgrls_graphene.posop_approx = 'onsite_only'
    r_onsite = np.real(sk_intgrls_graphene._calculate_lattice_dict(operator='r')[lattice_vec])
    sk_intgrls_graphene.posop_approx = "shift_only" 
    r_shift = np.real(sk_intgrls_graphene._calculate_lattice_dict(operator='r')[lattice_vec])


    print(r_onsite_expand[2,6,1])
    print(r_mulliken[2,6,1])

    rows = [
        (np.real(r_sk), r"$r^{\mathrm{exact}}_{\mu\nu}$ (Å)"),
        (np.real(r_onsite_expand - r_sk), r"$r^{\mathrm{onsite\,exp.}}_{\mu\nu} - r^{\mathrm{exact}}_{\mu\nu}$ (Å)"),
        (np.real(r_mulliken - r_sk), r"$r^{\mathrm{Mulliken}}_{\mu\nu} - r^{\mathrm{exact}}_{\mu\nu}$ (Å)"),
        (np.real(r_onsite - r_sk), r"$r^{\mathrm{onsite\,only}}_{\mu\nu} - r^{\mathrm{exact}}_{\mu\nu}$ (Å)"),
        (np.real(r_shift - r_sk), r"$r^{\mathrm{shift\,only}}_{\mu\nu} - r^{\mathrm{exact}}_{\mu\nu} (Å)$"),
    ]

    fig, axs = plt.subplots(nrows=5, ncols=3, layout="constrained")
    for i, (data, label) in enumerate(rows):
        vmax = np.abs(data).max()
        for c, comp in enumerate("xyz"):
            im = axs[i, c].imshow(data[c], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            if i == 0:
                axs[i, c].set_title(f"${comp}$")
        cbar = fig.colorbar(im, ax=axs[i, :], location="right", shrink=0.9)
        cbar.set_label(label)
    plt.show()
