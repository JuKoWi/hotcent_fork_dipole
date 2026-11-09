import numpy as np
from ase.build import graphene, mx2, bulk
import matplotlib.pyplot as plt
from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator
import sys
import warnings
from hotcent.pos_op.utils import angstrom_to_bohr
import matplotlib.colors as mcolors

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

def plot_posop_approximations(sk_integrator):
    """Plot dipole matrix elements generated with different approximations"""
    lattice_vec = (0,0,0)
    sk_integrator.posop_approx = 'sk'
    r_sk = np.real(sk_integrator._calculate_lattice_dict(operator='r')[lattice_vec])
    sk_integrator.posop_approx = "onsite_expansion"
    r_onsite_expand = np.real(sk_integrator._calculate_lattice_dict(operator='r')[lattice_vec])
    sk_integrator.posop_approx = 'mulliken'
    r_mulliken = np.real(sk_integrator._calculate_lattice_dict(operator='r')[lattice_vec])
    sk_integrator.posop_approx = 'onsite_only'
    r_onsite = np.real(sk_integrator._calculate_lattice_dict(operator='r')[lattice_vec])
    sk_integrator.posop_approx = "shift_only" 
    r_shift = np.real(sk_integrator._calculate_lattice_dict(operator='r')[lattice_vec])


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

def plot_rH_commutator(sk_integrator, cart_component, onsite=True):
    if onsite:
        atypes = list(set(sk_integrator.atomtypes))
        for el in atypes:
            p = sk_integrator._assemble_atom_block(types=(el,el), posA=np.zeros(3), posB=np.zeros(3), max_lA=sk_integrator.maxl_dict[el], max_lB=sk_integrator.maxl_dict[el], operator="p")
            r = sk_integrator._assemble_atom_block(types=(el,el), posA=np.zeros(3), posB=np.zeros(3), max_lA=sk_integrator.maxl_dict[el], max_lB=sk_integrator.maxl_dict[el], operator="r")
            r = angstrom_to_bohr(r)
            H = sk_integrator._assemble_atom_block(types=(el,el), posA=np.zeros(3), posB=np.zeros(3), max_lA=sk_integrator.maxl_dict[el], max_lB=sk_integrator.maxl_dict[el], operator="H")
            diff = p - 1j * np.einsum('ab, xbc -> xac',H,r) + 1j* np.einsum('xab, bc -> xac', r, H)
            break
        atom_type = atypes[0]
        print(atom_type)

        labels = ["s", "py", "pz", "px", "d", "d", "d", "d", "d"]
        comp = {0: "x", 1: "y", 2: "z"}

        im_p = np.imag(p[cart_component, ...])
        im_diff = np.imag(diff[cart_component, ...])

        # common color scale for both images
        norm = mcolors.Normalize(vmin=min(im_p.min(), im_diff.min()),
                                 vmax=max(im_p.max(), im_diff.max()))

        fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
        for ax, data, title in zip(axs, [im_p, im_diff], ["Im(p)", "Im(p-v)"]):
            im = ax.imshow(data, cmap="RdBu_r", norm=norm)
            ax.set_title(f"{comp[cart_component]}-component of {title}")
            ax.set_xticks(np.arange(np.shape(data)[0]))
            ax.set_xticklabels(labels[:np.shape(data)[0]])
            ax.set_yticks(np.arange(np.shape(data)[1]))
            ax.set_yticklabels(labels[:np.shape(data)[1]])

        fig.colorbar(im, ax=axs, shrink=0.6, label="a.u.")  # one colorbar spanning both axes
        plt.savefig(f"{atom_type}_rH_commutator.pdf")
        plt.show()
    else:
        r = sk_integrator._calculate_lattice_dict("r")[(0,0,0)]
        r = angstrom_to_bohr(r)
        p = sk_integrator._calculate_lattice_dict("p")[(0,0,0)]
        H = sk_integrator._calculate_lattice_dict("H")[(0,0,0)]
        S = sk_integrator._calculate_lattice_dict("S")[(0,0,0)]
        S_inv = np.linalg.inv(S)
        v = 1j * np.einsum('ab, bc, xcd-> xad', H, S_inv, r) - 1j * np.einsum('xab, bc, cd -> xad', r, S_inv, H)
        print(f"overlap hermitian:{np.allclose(S, S.T.conj())}")
        print(f"Hamiltonian hermitian:{np.allclose(H, H.T.conj())}")
        print(f"inverse overlap hermitian:{np.allclose(S_inv, S_inv.T.conj())}")
        xcomp = np.allclose(r[0,...], r[0,...].T.conj())
        ycomp = np.allclose(r[1,...], r[1,...].T.conj())
        zcomp = np.allclose(r[2,...], r[2,...].T.conj())
        print(f"position hermitian: {xcomp and ycomp and zcomp}")
        diff = p - v
        onsite = np.zeros(p.shape[1:], dtype=bool)
        for idx, n in enumerate(sk_integrator.orbnumbers):
            s = sk_integrator._find_block_pos(idx)
            onsite[s:s + n, s:s + n] = True

        comp = {0: "x", 1: "y", 2: "z"}

        im_p = np.imag(v[cart_component, ...])
        im_diff = np.imag(diff[cart_component, ...])

        # common color scale for both images
        norm = mcolors.Normalize(vmin=min(im_p.min(), im_diff.min()),
                                 vmax=max(im_p.max(), im_diff.max()))

        fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
        for ax, data, title in zip(axs, [im_p, im_diff], ["Im(v)", "Im(p-v)"]):
            im = ax.imshow(data, cmap="RdBu_r", norm=norm)
            ax.set_title(f"{comp[cart_component]}-component of {title}")
            # ax.set_xticks(np.arange(np.shape(data)[0]))
            # ax.set_yticks(np.arange(np.shape(data)[1]))

        fig.colorbar(im, ax=axs, shrink=0.6, label="a.u.")  # one colorbar spanning both axes
        plt.savefig("ucell_rH_commutator.pdf")
        plt.show()


if __name__ == '__main__':
    max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2, "Zn": 2, "O":1}
    graphene = graphene("CC", vacuum=10)
    zno = bulk('ZnO', crystalstructure='zincblende', a=4.62)
    MoS2 = mx2(vacuum=20)
    MoS2.pbc = (True, True, False)

    sk_integrator = SlaterKosterIntegrator(
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
    plot_rH_commutator(sk_integrator=sk_integrals_mos2, cart_component=0, onsite=True)
    plot_rH_commutator(sk_integrator=sk_integrals_mos2, cart_component=0, onsite=False)

