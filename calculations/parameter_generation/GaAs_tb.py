from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator
from ase.build.bulk import bulk
from ase.visualize import view

ml_dict = {"Ga": 2, "As": 2}
gaas = bulk("GaAs", "zincblende", a=5.653)
gaas_tb = SlaterKosterIntegrator(
    atoms_unit_cell=gaas,
    skpath='sk_hdf5',
    maxl_dict=ml_dict,
    format="hdf5",
)

approx = 'sk'
gaas_tb.write_tb_hdf5(filename=f"GaAs_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'onsite_expansion'
gaas_tb.write_tb_hdf5(filename=f"GaAs_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'mulliken'
gaas_tb.write_tb_hdf5(filename=f"GaAs_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'onsite_only'
gaas_tb.write_tb_hdf5(filename=f"GaAs_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'shift_only'
gaas_tb.write_tb_hdf5(filename=f"GaAs_{approx}.hdf5", momentum=True, posop_approx=approx)