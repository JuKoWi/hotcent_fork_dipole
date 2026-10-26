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


sk_integrals = SlaterKosterIntegrator(
    MoS2,
    skpath="sk_hdf5",
    maxl_dict=max_l,
    format="hdf5",
)

approx = 'sk'
sk_integrals.write_tb_hdf5(filename=f"mos2_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'onsite_expansion'
sk_integrals.write_tb_hdf5(filename=f"mos2_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'mulliken'
sk_integrals.write_tb_hdf5(filename=f"mos2_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'onsite_only'
sk_integrals.write_tb_hdf5(filename=f"mos2_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'shift_only'
sk_integrals.write_tb_hdf5(filename=f"mos2_{approx}.hdf5", momentum=True, posop_approx=approx)