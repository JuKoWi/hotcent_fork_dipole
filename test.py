from ase.build import mx2

mos2 = mx2(size=(2,2,1))
print(mos2.get_distances(0, [1,2]))
