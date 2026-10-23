from ase import Atoms
from ase.io import write
from ase.build import graphene
from ase.visualize import view
from ase.build import molecule
from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator
import sys


max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2}
graphene = graphene("CC", size=(1, 1, 1), vacuum=10)
# graphene.set_positions([[0,0,0], [0,2.4,0]])

sk_integrals = SlaterKosterIntegrator(
    graphene,
    skpath="sk_hdf5",
    maxl_dict=max_l,
    format="hdf5",
)

sk_integrals.write_seedname()
sk_integrals.write_seedname_momentum()

approx = 'sk'
sk_integrals.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'onsite_expansion'
sk_integrals.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'mulliken'
sk_integrals.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'onsite_only'
sk_integrals.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'shift_only'
sk_integrals.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)
