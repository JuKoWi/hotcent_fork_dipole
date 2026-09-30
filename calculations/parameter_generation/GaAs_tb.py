from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator
from ase.build.bulk import bulk
from ase.visualize import view

ml_dict = {"Ga": 2, "As": 2}
gaas = bulk("GaAs", "zincblende", a=5.653)
gaas_tb = SlaterKosterIntegrator(
    atoms_unit_cell=gaas,
    skpath="sk_experimental/sk_full",
    skpath_posop="sk_experimental/sk_posop",
    maxl_dict=ml_dict,
    format="full",
)
count = 0
while count < 20:
    gaas_tb.write_seedname()
    count += 1
