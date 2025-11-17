import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from hotcent.new_dipole.compare_integration_methods import scan_distance

direction = np.random.rand(3)
direction = direction/np.linalg.norm(direction)
scan_distance(direction=direction, index=767, dipole=True, from_file=False, plot=True, n_dist=40, min_dist_angst=0.05, max_dist_angst=0.2) 
