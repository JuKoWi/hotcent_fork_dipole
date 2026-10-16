import numpy as np
from hotcent.pos_op.utils import dim_atom_basis
from ase import Atoms
from ase.io import write
from ase.build import graphene
from ase.visualize import view
from ase.build import molecule
from ase.neighborlist import neighbor_list
import matplotlib.pyplot as plt
from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator


max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2}

# Two C atoms in a big cubic box. Nearest distances: L (periodic image) and sqrt(3)/2*L (the other atom).
L = 40.0  # Å
isolated = Atoms(
    "C2",
    positions=[[0.0, 0.0, 0.0], [L / 2, L / 2, L / 2]],
    cell=[L, L, L],
    pbc=True,
)

graphene = graphene("CC", vacuum=10)
# graphene.set_positions([[0,0,0], [1,1,1]])

sk_integrals = SlaterKosterIntegrator(
    graphene,
    skpath="skfiles_consistent/sk_unique",
    maxl_dict=max_l,
    skpath_posop="skfiles_consistent/sk_posop_unique",
    format="unique",
    path_p_onsite="skfiles_consistent/onsite_momentum/",
)

sk_integrals.posop_approx = "onsite_expansion"
r_onsite_expand = np.real(sk_integrals._calculate_lattice_dict(operator='r')[(0,0,0)])
sk_integrals.posop_approx = None
r_exact = np.real(sk_integrals._calculate_lattice_dict(operator='r')[(0,0,0)])
sk_integrals.posop_approx = "onsite_only" 
r_onsite_only = np.real(sk_integrals._calculate_lattice_dict(operator='r')[(0,0,0)])


rows = [
    (np.real(r_exact), r"$r^{\mathrm{exact}}_{\mu\nu}$ (Å)"),
    (np.real(r_onsite_expand - r_exact), r"$r^{\mathrm{onsite\,exp.}}_{\mu\nu} - r^{\mathrm{exact}}_{\mu\nu}$ (Å)"),
    (np.real(r_onsite_only - r_exact), r"$r^{\mathrm{onsite\,only}}_{\mu\nu}$ - r^{\mathrm{exact}}_{\mu\nu} (Å)"),
]

fig, axs = plt.subplots(nrows=3, ncols=3, layout="constrained")
for i, (data, label) in enumerate(rows):
    vmax = np.abs(data).max()
    for c, comp in enumerate("xyz"):
        im = axs[i, c].imshow(data[c], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        if i == 0:
            axs[i, c].set_title(f"${comp}$")
    cbar = fig.colorbar(im, ax=axs[i, :], location="right", shrink=0.9)
    cbar.set_label(label)
plt.show()