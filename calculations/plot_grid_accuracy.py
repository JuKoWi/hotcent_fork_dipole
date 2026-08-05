from hotcent.pos_op.compare_integration_methods import scan_grid_error 
import numpy as np
from hotcent.pos_op.utils import *


pos = np.array([[0,0,0], [0,0,bohr_to_angstrom(2.8)]]) # roundabout 0.7 Angstrom, point on sk distance list -> no spline interpolation error
scan_grid_error(pos=pos, index=523, dipole=True, plot=True, from_file=False)