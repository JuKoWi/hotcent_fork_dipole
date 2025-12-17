import scipy as sc

def bohr_to_angstrom(bohr):
    meter = bohr*sc.constants.physical_constants['atomic unit of length'][0]
    angstrom = meter / sc.constants.angstrom
    return angstrom

def angstrom_to_bohr(angstrom):
    meter = angstrom * sc.constants.angstrom
    bohr = meter / sc.constants.physical_constants['atomic unit of length'][0]
    return bohr

def get_norbs(maxl):
    count = 0
    for l in range(maxl+1):
        count += 2*l +1
    return count

def hartree_to_eV(hartree):
    return hartree * sc.constants.physical_constants['hartree-electron volt relationship'][0]

def eV_to_hartree(eV):
    if eV == 0:
        return 0
    return eV/sc.constants.physical_constants['hartree-electron volt relationship'][0]