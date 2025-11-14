import numpy as np
from hotcent.new_dipole.compare_integration_methods import scan_distance

direction = np.random.rand(3)
direction = direction/np.linalg.norm(direction)
scan_distance(direction=direction, index=0, dipole=False, from_file=True, plot=True, n_dist=40, min_dist_angst=0.05, max_dist_angst=0.2) 
