from ase import Atoms
from ase.io import write
from ase.build import bulk
from ase.visualize import view
from ase.build import molecule
from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator


max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2, "Al": 2, "Au": 2}
aluminium = bulk("Al", "fcc", a=4.049)
aluminium_tb = SlaterKosterIntegrator(
    aluminium,
    skpath="skfiles/sk_conventional",
    maxl_dict=max_l,
    skpath_posop="skfiles/sk_posop",
    conventional_skf=True,
)
aluminium_tb.write_seedname()
