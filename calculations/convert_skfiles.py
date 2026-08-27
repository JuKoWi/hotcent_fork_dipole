from hotcent.pos_op.slako_new import dftbplus_to_full

"""convert .skf files used by dftb+ to the long format used in custom code 
to compare dftb+ results with own results based on exact same parameters"""

# convert_sk_table(path='skfiles/pbc_dftb_format/C-C.skf', homonuclear=True)
# convert_sk_table(path='skfiles/pbc_dftb_format/H-H.skf', homonuclear=True)
# convert_sk_table(path='skfiles/pbc_dftb_format/H-C.skf', homonuclear=False)
dftbplus_to_full(path="skfiles/sk_alex/C-C.skf", homonuclear=True)

