from ase.build import bulk
from ase.visualize import view
from hotcent.pos_op.tight_binding import SlaterKosterIntegrator

diamond = bulk('C', 'diamond', a=3.567)
maxl = {'C': 1}
tb_generator = SlaterKosterIntegrator(atoms_unit_cell=diamond, maxl_dict=maxl, skpath='skfiles/sk_conventional', skpath_posop='skfiles/sk_posop', conventional_skf=True)
tb_generator.write_seedname()

