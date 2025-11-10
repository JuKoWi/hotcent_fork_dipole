import numpy as np
from ase import Atoms
from ase.io import write
from hotcent.new_dipole.assemble_integrals import SK_With_Shift
from hotcent.new_dipole.compare_integrals_shifted import compare_dipole_shifted
from hotcent.new_dipole.utils import *

compare_dipole_shifted([0.27, 0.27, 0.27, 0.27])

