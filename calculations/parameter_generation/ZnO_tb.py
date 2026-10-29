from ase.build import bulk
from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator

zno = bulk('ZnO', crystalstructure='zincblende', a=4.62)

ml_dict = {"Zn": 2, "O": 1}

zno_tb = SlaterKosterIntegrator(
    atoms_unit_cell=zno,
    skpath='sk_hdf5',
    maxl_dict=ml_dict,
    format="hdf5",
)

approx = 'sk'
zno_tb.write_tb_hdf5(filename=f"ZnO_zincblende_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'onsite_expansion'
zno_tb.write_tb_hdf5(filename=f"ZnO_zincblende_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'mulliken'
zno_tb.write_tb_hdf5(filename=f"ZnO_zincblende_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'onsite_only'
zno_tb.write_tb_hdf5(filename=f"ZnO_zincblende_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'shift_only'
zno_tb.write_tb_hdf5(filename=f"ZnO_zincblende_{approx}.hdf5", momentum=True, posop_approx=approx)