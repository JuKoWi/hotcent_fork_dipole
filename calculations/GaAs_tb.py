from hotcent.pos_op.tight_binding import SlaterKosterIntegrator 
from ase.build.bulk import bulk
from ase.visualize import view

ml_dict = {'Ga':2, 'As':2}
gaas = bulk('GaAs', 'zincblende', a=5.653)
gaas_tb = SlaterKosterIntegrator(atoms_unit_cell=gaas,
                                 skpath='skfiles/sk_conventional',
                                 skpath_posop='skfiles/sk_posop',
                                 maxl_dict=ml_dict,
                                 conventional_skf=True)
count = 0
while count < 20:
    gaas_tb.write_seedname()
    count += 1

