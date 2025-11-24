import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[2]))
from hotcent.new_dipole.compare_integration_methods import scan_grid_error 
import numpy as np
from hotcent.new_dipole.utils import *


pos = np.array([[0,0,0], [0,0,bohr_to_angstrom(1.3)]]) # roundabout 0.7 Angstrom, point on sk distance list -> no spline interpolation error
scan_grid_error(pos=pos, index=1, dipole=True, plot=True, from_file=True)




