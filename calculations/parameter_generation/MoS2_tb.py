from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator
from ase.visualize import view
from ase.build import mx2
from ase import Atoms
from ase.io import read, write
import sys
import numpy as np
from hotcent.pos_op.utils import angstrom_to_bohr

max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2}
MoS2 = mx2(vacuum=20)
MoS2.pbc = (True, True, False)


seedname_mos2 = SlaterKosterIntegrator(
    MoS2,
    skpath="skfiles_consistent/sk_unique",
    maxl_dict=max_l,
    skpath_posop="skfiles_consistent/sk_posop_unique",
    format="unique",
    path_p_onsite="skfiles_consistent/onsite_momentum/"
)
seedname_mos2.write_seedname()
seedname_mos2.write_seedname_momentum()
seedname_mos2.check_p_v_onsite()
