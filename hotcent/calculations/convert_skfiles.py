from hotcent.new_dipole.slako_new import convert_sk_table

"""convert .skf files used by dftb+ to the long format used in custom code 
to compare dftb+ results with own results based on exact same parameters"""

# convert_sk_table(path='skfiles/pbc_dftb_format/C-C.skf', homonuclear=True)
# convert_sk_table(path='skfiles/pbc_dftb_format/H-H.skf', homonuclear=True)
# convert_sk_table(path='skfiles/pbc_dftb_format/H-C.skf', homonuclear=False)
convert_sk_table(path="skfiles/sk_alex/C-C.skf", homonuclear=True)

