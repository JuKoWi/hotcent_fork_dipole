from scipy import constants as sc

def bohr_to_angstrom(bohr):
    meter = bohr*sc.physical_constants['atomic unit of length'][0]
    angstrom = meter / sc.angstrom
    return angstrom

def angstrom_to_bohr(angstrom):
    meter = angstrom * sc.angstrom
    bohr = meter / sc.physical_constants['atomic unit of length'][0]
    return bohr