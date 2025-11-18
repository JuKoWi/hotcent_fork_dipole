import numpy as np
from hotcent.new_dipole.utils import *
from hotcent.new_dipole.compare_integration_methods import scan_distance

direction = np.random.rand(3)
direction = direction/np.linalg.norm(direction)
min_dist_angst = bohr_to_angstrom(0.1)
max_dist_angst = bohr_to_angstrom(0.1 + 249*0.05)
direction = np.array([0,0,1])
scan_distance(direction=direction, index=0, dipole=True, from_file=False, plot=True, n_dist=25, min_dist_angst=min_dist_angst, max_dist_angst=max_dist_angst) 
