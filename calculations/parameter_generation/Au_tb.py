from ase import Atoms
from ase.io import write
from ase.build import bulk
from ase.visualize import view
from ase.build import molecule
from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator


max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2, "Al": 2, "Au": 2}
gold = bulk("Au", "fcc", a=4.0786)
gold_tb = SlaterKosterIntegrator(
    gold,
    skpath="skfiles/sk_conventional",
    maxl_dict=max_l,
    skpath_posop="skfiles/sk_posop",
    conventional_skf=True,
)
gold_tb.write_seedname()
