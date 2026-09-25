from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator
from ase.visualize import view
from ase.build import mx2
from ase import Atoms
from ase.io import read, write
import sys
import numpy as np
from hotcent.pos_op.utils import angstrom_to_bohr


MoS2 = mx2(vacuum=20)
MoS2.pbc = (True, True, False)


max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2}
seedname_mos2 = SlaterKosterIntegrator(
    MoS2,
    skpath="sk_experimental/sk_unique",
    maxl_dict=max_l,
    skpath_posop="sk_experimental/sk_posop_unique",
    format="unique",
    path_p_onsite="skfiles/onsite_momentum/"
)

seedname_mos2.write_seedname_momentum()