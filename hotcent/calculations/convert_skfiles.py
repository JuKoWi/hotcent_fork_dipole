from hotcent.new_dipole.slako_new import convert_sk_table
import numpy as np

# convert_sk_table(path='skfiles/pbc_dftb_format/C-C.skf', homonuclear=True)
# convert_sk_table(path='skfiles/pbc_dftb_format/H-H.skf', homonuclear=True)
convert_sk_table(path='skfiles/pbc_dftb_format/H-C.skf', homonuclear=False)

