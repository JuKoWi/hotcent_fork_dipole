import numpy as np
from hotcent.pos_op.utils import dim_atom_basis
from ase import Atoms
from ase.io import write
from ase.build import graphene
from ase.visualize import view
from ase.build import molecule
from ase.neighborlist import neighbor_list
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

sk_integrals = SlaterKosterIntegrator(
    isolated,
    skpath="skfiles_consistent/sk_unique",
    maxl_dict=max_l,
    skpath_posop="skfiles_consistent/sk_posop_unique",
    format="unique",
    path_p_onsite="skfiles_consistent/onsite_momentum/",
)

sk_integrals.write_seedname()
sk_integrals.write_seedname_momentum()
sk_integrals.check_p_v_offsite()
