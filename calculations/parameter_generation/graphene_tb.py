from ase import Atoms
from ase.io import write
from ase.build import graphene
from ase.visualize import view
from ase.build import molecule
from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator


max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2}
graphene = graphene("CC", size=(1, 1, 1), vacuum=10)

sk_integrals = SlaterKosterIntegrator(
    graphene,
    skpath="skfiles_consistent/sk_unique",
    maxl_dict=max_l,
    skpath_posop="skfiles_consistent/sk_posop_unique",
    format="unique",
    path_p_onsite="skfiles_consistent/onsite_momentum/"
)
print(sk_integrals.H_sk_tables[("C","C")].same_atom_vals)
sk_integrals.write_seedname()
sk_integrals.write_seedname_momentum()
sk_integrals.check_p_v_onsite()
sk_integrals.check_p_v_offsite()
