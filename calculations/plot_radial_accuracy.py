import numpy as np
from hotcent.new_dipole.utils import *
from hotcent.new_dipole.compare_integration_methods import scan_distance

direction = np.array([0,0,1])
direction = direction/np.linalg.norm(direction)
min_dist_angst = 0.2
dR_angst = 0.2
n_dist = 30
scan_distance(direction=direction, index=1, dipole=True, from_file=True, plot=True, n_dist=n_dist, min_dist_angst=min_dist_angst, d_dist_angst=dR_angst) 