import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from hotcent.new_dipole.compare_integration_methods import scan_grid_error 
import numpy as np

pos = np.array([[0,0,0], [0.3728,-0.8129,0.4476]])
scan_grid_error(pos=pos, index=767, dipole=True, plot=True, from_file=False)




