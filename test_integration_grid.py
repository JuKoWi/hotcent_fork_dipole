from hotcent.new_dipole.compare_integration_methods import scan_grid_error
import numpy as np

pos = np.array([[0,0,0], [0,0,1]])
index_set = np.arange(16)
scan_grid_error(pos=pos, index_set=index_set, dipole=True)



