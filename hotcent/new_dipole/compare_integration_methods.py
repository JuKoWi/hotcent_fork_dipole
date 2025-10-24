import numpy as np
from ase import Atoms
from ase.io import write
import pickle
import time
from scipy.integrate import nquad
import sympy as sp
from hotcent.new_dipole.dipole import SK_Integral

from hotcent.new_dipole.offsite_twocenter_dipole import Offsite2cTableDipole
from hotcent.new_dipole.onsite_twocenter_dipole import Onsite2cTable
from optparse import OptionParser
from ase.units import Bohr
from ase.data import covalent_radii, atomic_numbers
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

x, y, z = sp.symbols("x, y, z")
r = sp.sqrt(x**2 + y**2 + z**2)


s_1 = 1 / (2 * sp.sqrt(sp.pi))

px_1 = sp.sqrt(3 / (4 * sp.pi)) * x/r
py_1 = sp.sqrt(3 / (4 * sp.pi)) * y/r
pz_1 = sp.sqrt(3 / (4 * sp.pi)) * z/r

dxy_1 = sp.sqrt(15/(4 * sp.pi)) *x*y/r**2
dyz_1 = sp.sqrt(15 / (4*sp.pi)) * y*z/r**2
dxz_1 = sp.sqrt(15/(4* sp.pi)) * x*z/r**2
dx2y2_1 = sp.sqrt(15/(16*sp.pi)) * (x**2-y**2)/r**2
dz2_1 = sp.sqrt(5 / (16 * sp.pi)) * (2*z**2 -x**2 -y**2)/r**2

x0, y0, z0 = sp.symbols("x0, y0, z0")
r_0 = sp.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)

s_2 = s_1

px_2 = sp.sqrt(3 / (4 * sp.pi)) * (x-x0)/r_0
py_2 = sp.sqrt(3 / (4 * sp.pi)) * (y-y0)/r_0
pz_2 = sp.sqrt(3 / (4 * sp.pi)) * (z-z0)/r_0 

dxy_2 = sp.sqrt(15 / (4 * sp.pi)) * (x-x0)*(y-y0)/r_0**2
dyz_2 = sp.sqrt(15 / (4 * sp.pi)) * (y-y0)*(z-z0)/r_0**2
dxz_2 = sp.sqrt(15 / (4 * sp.pi)) * (x-x0)*(z-z0)/r_0**2
dx2y2_2 = sp.sqrt(15 / (16*sp.pi)) * ((x-x0)**2-(y-y0)**2)/r_0**2
dz2_2 = sp.sqrt(5 / (16 * sp.pi)) * (2*(z-z0)**2 -(x-x0)**2 -(y-y0)**2)/r_0**2

rx_1 = x
ry_1 = y 
rz_1 = z 

a, b, c = sp.symbols("a, b, c")
radial_1 = sp.exp(-b* (x**2 + y**2 + z**2))
radial_2 = sp.exp(-b* ((x-x0)**2 + (y-y0)**2 + (z-z0)**2))


first_center = {
    "ss": (s_1, 0,0),
    "py": (py_1, 1,-1),
    "pz": (pz_1, 1,0),
    "px": (px_1,1,1),
    "d1": (dxy_1,2,-2),
    "d2": (dyz_1,2,-1),
    "d3": (dz2_1,2,0),
    "d4": (dxz_1,2,1),
    "d5": (dx2y2_1,2,2),
}

second_center = {
    "ss": (s_2,0,0),
    "py": (py_2,1,-1),
    "pz": (pz_2,1,0),
    "px": (px_2,1,1),
    "d1": (dxy_2,2,-2),
    "d2": (dyz_2,2,-1),
    "d3": (dz2_2,2,0),
    "d4": (dxz_2,2,1),
    "d5": (dx2y2_2,2,2),
}

operator = {
    "ry": (ry_1,1,-1),
    "rz": (rz_1,1,0),
    "rx": (rx_1,1,1),
}



def get_2c_integrals(pos_at1, zeta1, zeta2, comparison):
    """Compute two-center integrals numerically with singularity-safe handling."""
    file = open("comparison.txt", 'w')
    print(f'coordinate: {pos_at1}', file=file)
    print("sk-value \t numerical", file=file)
    t_start = time.time()
    count = 0
    results = np.zeros((len(operator) * len(first_center) * len(second_center)))
    
    for name_i, i in first_center.items():
        for name_j, j in operator.items():
            for name_k, k in second_center.items():
                zeta1_val = zeta1[i[1]]
                zeta2_val = zeta2[k[1]]
                R1 = radial_1.subs({b: zeta1_val})
                R2 = radial_2.subs({b: zeta2_val})
                integrand = i[0] * R1 * j[0] * k[0] * R2 * r**i[1] * r_0**k[1] * 0.5# eliminate poles by multiplying with r as if it was part of the radial part
                print(integrand)
                integrand = integrand.subs({x0: pos_at1[0], y0: pos_at1[1], z0: pos_at1[2]})
                
                # Precompile symbolic -> numeric function once
                f_num = sp.lambdify((x, y, z), integrand, "numpy")

                print(f"Testing integral {name_i}-{name_j}-{name_k}", file=file)
                print(f"Testing integral {name_i}-{name_j}-{name_k}")
                num, err = nquad(f_num, [[-20, 20], [-20, 20], [-20, 20]])
                results[count] = num
                print(f"sk value:{comparison[count]}, numerical: {num} ")
                print(f'{comparison[count]} \t{num}',file=file)
                # print(f"Result = {num:.6e}, Estimated error = {err:.2e}")

                count += 1
    t_end = time.time()
    print(f"integration took {t_end-t_start}")
    with open("numerical_integrals_list.pkl", "wb") as f:
        pickle.dump(results, f)
    file.close()
    return results

def compare_matrix_elements(zeta1):
    USE_EXISTING_SKF = True

    if not USE_EXISTING_SKF:
        #set up atomic system with skf files
        p = OptionParser(usage='%prog')
        p.add_option('-f', '--functional', default='LDA',
                     help='Which density functional to apply? '
                          'E.g. LDA (default), GGA_X_PBE+GGA_C_PBE, ...')
        p.add_option('-s', '--superposition',
                     default='potential',
                     help='Which superposition scheme? '
                          'Choose "potential" (default) or "density"')
        p.add_option('-t', '--stride', default=1, type=int,
                     help='Which SK-table stride length? Default = 1. '
                          'See hotcent.slako.run for more information.')
        opt, args = p.parse_args()

        element = 'Ge'
        r0 = 1.85 * covalent_radii[atomic_numbers[element]] / Bohr
        atom = AtomicDFT(element,
                         xc=opt.functional,
                         confinement=PowerConfinement(r0=r0, s=2),
                         perturbative_confinement=False,
                         configuration='[Ar] 4s2 3d10 4p2',
                         valence=['4s', '4p', '3d'],
                         timing=True,
                         )
        atom.run()

        # Compute Slater-Koster integrals:
        zeta_dict = {'4s': (zeta1[0], 0), '4p': (zeta1[1],1), '3d': (zeta1[2], 2)}
        rmin, dr, N = 0.1, 0.05, 250
        off2c = Offsite2cTableDipole(atom, atom, timing=True)
        off2c.run(rmin, dr, N, superposition=opt.superposition,
                  xc=opt.functional, stride=opt.stride, zeta_dict=zeta_dict)
        off2c.write()
        off2c.plot_minimal()

    atoms = Atoms('Ge2', positions=[
        [0.0, 0.0, 0.0],
        [-1.0, -1.0, 1.0]
    ])

    #assemble actual matrix elements
    write('Ge2.xyz', atoms)
    method1 = SK_Integral('Ge', 'Ge')
    method1.load_atom_pair('Ge2.xyz')
    method1.choose_relevant_matrix()
    method1.load_SK_dipole_file('Ge-Ge_offsite2c-dipole.skf')
    res1 = method1.calculate_dipole()
    method1.check_rotation_implementation()

    #calculate directly brute force
    res2 = get_2c_integrals(pos_at1=method1.R_vec, zeta1=zeta1, zeta2=zeta1, comparison=res1)

    

