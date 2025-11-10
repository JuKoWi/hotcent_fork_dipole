import pickle
import sys
import time
from pathlib import Path
from optparse import OptionParser
from scipy.integrate import nquad
import sympy as sp
import numpy as np
import os
from ase import Atoms
from ase.io import write
from ase.units import Bohr
from ase.data import covalent_radii, atomic_numbers
from hotcent.new_dipole.assemble_integrals import SK_Integral_Dipole, SK_Integral_Overlap, SK_With_Shift
from hotcent.new_dipole.rotation_transform import to_spherical
from hotcent.new_dipole.offsite_twocenter_new import Offsite2cTable
from hotcent.new_dipole.offsite_twocenter_dipole import Offsite2cTableDipole
from hotcent.new_dipole.utils import angstrom_to_bohr, bohr_to_angstrom
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT

sys.path.append(str(Path(__file__).resolve().parents[1]))

x, y, z = sp.symbols("x, y, z")
x1, y1, z1 = sp.symbols("x1, y1, z1")
r_1 = sp.sqrt((x-x1)**2 + (y-y1)**2 + (z-z1)**2)


s_1 = 1 / (2 * sp.sqrt(sp.pi))

px_1 = sp.sqrt(3 / (4 * sp.pi)) * (x-x1)/r_1
py_1 = sp.sqrt(3 / (4 * sp.pi)) * (y-y1)/r_1
pz_1 = sp.sqrt(3 / (4 * sp.pi)) * (z-z1)/r_1

dxy_1 = sp.sqrt(15/(4 * sp.pi)) *(x-x1)*(y-y1)/r_1**2
dyz_1 = sp.sqrt(15 / (4*sp.pi)) * (y-y1)*(z-z1)/r_1**2
dxz_1 = sp.sqrt(15/(4* sp.pi)) * (x-x1)*(z-z1)/r_1**2
dx2y2_1 = sp.sqrt(15/(16*sp.pi)) * ((x-x1)**2-(y-y1)**2)/r_1**2
dz2_1 = sp.sqrt(5 / (16 * sp.pi)) * (3*(z-z1)**2 -r_1**2)/r_1**2

f1_1 = 1/4 * sp.sqrt(35/(2*sp.pi)) * (y-y1)*(3*(x-x1)**2 - (y-y1)**2)/r_1**3
f2_1 = 1/2 * sp.sqrt(105/sp.pi) * (x-x1)*(y-y1)*(z-z1)/r_1**3
f3_1 = 1/4 * sp.sqrt(21/(2*sp.pi)) * (y-y1) * (5*(z-z1)**2 - r_1**2)/r_1**3
f4_1 = 1/4 * sp.sqrt(7/sp.pi) * (5*(z-z1)**3 - 3* (z-z1) * r_1**2)/r_1**3
f5_1 = 1/4 * sp.sqrt(21/(2*sp.pi)) * (x-x1) * (5*(z-z1)**2 - r_1**2)/r_1**3
f6_1 = 1/4 * sp.sqrt(105/sp.pi) * ((x-x1)**2 - (y-y1)**2) * (z-z1) /r_1**3
f7_1 = 1/4 * sp.sqrt(35/(2*sp.pi)) * (x-x1) * ((x-x1)**2 - 3* (y-y1)**2) /r_1**3

x2, y2, z2 = sp.symbols("x2, y2, z2")
r_2 = sp.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2)

s_2 = s_1

px_2 = sp.sqrt(3 / (4 * sp.pi)) * (x-x2)/r_2
py_2 = sp.sqrt(3 / (4 * sp.pi)) * (y-y2)/r_2
pz_2 = sp.sqrt(3 / (4 * sp.pi)) * (z-z2)/r_2 

dxy_2 = sp.sqrt(15 / (4 * sp.pi)) * (x-x2)*(y-y2)/r_2**2
dyz_2 = sp.sqrt(15 / (4 * sp.pi)) * (y-y2)*(z-z2)/r_2**2
dxz_2 = sp.sqrt(15 / (4 * sp.pi)) * (x-x2)*(z-z2)/r_2**2
dx2y2_2 = sp.sqrt(15 / (16*sp.pi)) * ((x-x2)**2-(y-y2)**2)/r_2**2
dz2_2 = sp.sqrt(5 / (16 * sp.pi)) * (3 * (z-z2)**2 - r_2**2)/r_2**2

f1_2 = 1/4 * sp.sqrt(35/(2*sp.pi)) * (y-y2)*(3*(x-x2)**2 - (y-y2)**2)/r_2**3
f2_2 = 1/2 * sp.sqrt(105/sp.pi) * (x-x2)*(y-y2)*(z-z2)/r_2**3
f3_2 = 1/4 * sp.sqrt(21/(2*sp.pi)) * (y-y2) * (5*(z-z2)**2 - r_2**2)/r_2**3
f4_2 = 1/4 * sp.sqrt(7/sp.pi) * (5*(z-z2)**3 - 3* (z-z2) * r_2**2)/r_2**3
f5_2 = 1/4 * sp.sqrt(21/(2*sp.pi)) * (x-x2) * (5*(z-z2)**2 - r_2**2)/r_2**3
f6_2 = 1/4 * sp.sqrt(105/sp.pi) * ((x-x2)**2 - (y-y2)**2) * (z-z2) /r_2**3
f7_2 = 1/4 * sp.sqrt(35/(2*sp.pi)) * (x-x2) * ((x-x2)**2 - 3* (y-y2)**2) /r_2**3

rx_1 = x
ry_1 = y 
rz_1 = z 

a, b, c = sp.symbols("a, b, c")
radial_1 = (2 * b/sp.pi)**(3/4) * sp.exp(-b* ((x-x1)**2 + (y-y1)**2 + (z-z1)**2))
radial_2 = (2* b/sp.pi)**(3/4) * sp.exp(-b* ((x-x2)**2 + (y-y2)**2 + (z-z2)**2))


first_center_real = {
    "ss": (s_1, 0,0),
    "py": (py_1, 1,-1),
    "pz": (pz_1, 1,0),
    "px": (px_1,1,1),
    "d1": (dxy_1,2,-2),
    "d2": (dyz_1,2,-1),
    "d3": (dz2_1,2,0),
    "d4": (dxz_1,2,1),
    "d5": (dx2y2_1,2,2),
    "f1": (f1_1, 3, -3),
    "f2": (f2_1, 3, -2),
    "f3": (f3_1, 3, -1),
    "f4": (f4_1, 3, 0),
    "f5": (f5_1, 3, 1),
    "f6": (f6_1, 3, 2),
    "f7": (f7_1, 3, 3)
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
    "f1": (f1_2, 3, -3),
    "f2": (f2_2, 3, -2),
    "f3": (f3_2, 3, -1),
    "f4": (f4_2, 3, 0),
    "f5": (f5_2, 3, 1),
    "f6": (f6_2, 3, 2),
    "f7": (f7_2, 3, 3)
}

operator = {
    "ry": (ry_1,1,-1),
    "rz": (rz_1,1,0),
    "rx": (rx_1,1,1),
}


def get_analytic_2c_dipole(pos_at1, pos_at2, zeta1, zeta2, comparison):
    file = open("comparison_dipole.txt", 'w')
    print(f'coordinate1: {pos_at1}', file=file)
    print(f'coordinate2: {pos_at2}', file=file)
    print("sk-value \t analytic", file=file)
    t_start = time.time()
    count = 0
    results = np.zeros((len(operator) * len(first_center_real) * len(second_center)))
    print(f"first atom:{pos_at1}")
    print(f"second atom:{pos_at2}")
    
    for name_i, i in first_center_real.items():
        for name_j, j in operator.items():
            for name_k, k in second_center.items():
                # if (i[1]<2 and k[1]<2):
                zeta1_val = zeta1[i[1]]
                zeta2_val = zeta2[k[1]]
                R1 = radial_1.subs({b: zeta1_val})
                R2 = radial_2.subs({b: zeta2_val})
                integrand = i[0] * R1 * j[0] * k[0] * R2 * r_1**i[1] * r_2**k[1] # eliminate poles by multiplying with r as if it was part of the radial part
                # print(integrand)

                integrand = integrand.subs({x1: pos_at1[0], y1: pos_at1[1], z1: pos_at1[2]})
                integrand = integrand.subs({x2: pos_at2[0], y2: pos_at2[1], z2: pos_at2[2]})
                
                analyt_int = sp.integrate(sp.integrate(sp.integrate(integrand, (x, -sp.oo, sp.oo)), (y, -sp.oo, sp.oo)), (z, -sp.oo, sp.oo))
                analyt_int_value = analyt_int.evalf()
                
                print(f"Testing integral {name_i}-{name_j}-{name_k}", file=file)
                results[count] = analyt_int_value 
                print(f"Testing integral {name_i}-{name_j}-{name_k}")
                print(f"sk value:{comparison[count]}")
                print(f"analytical: {analyt_int_value}")
                print(f'{comparison[count]} \t{analyt_int_value}',file=file)

                count += 1
    t_end = time.time()
    print(f"integration took {t_end-t_start}")
    with open("analytical_dipole_list.pkl", "wb") as f:
        pickle.dump(results, f)
    file.close()
    return results


# def get_analytic_2c_overlap(pos_at1, zeta1, zeta2, comparison):
#     file = open("comparison_overlap.txt", 'w')
#     print(f'coordinate: {pos_at1}', file=file)
#     print("sk-value \t analytic", file=file)
#     t_start = time.time()
#     count = 0
#     results = np.zeros((len(first_center_real) * len(second_center)))

    
#     for name_i, i in first_center_real.items():
#         for name_k, k in second_center.items():
#             # if (i[1]<2 and k[1]<2):
#             zeta1_val = zeta1[i[1]]
#             zeta2_val = zeta2[k[1]]
#             R1 = radial_1.subs({b: zeta1_val})
#             R2 = radial_2.subs({b: zeta2_val})
#             integrand = i[0] * R1 * k[0] * R2 * r_1**i[1] * r_2**k[1] # eliminate poles by multiplying with r as if it was part of the radial part
#             integrand = integrand.subs({x2: pos_at1[0], y2: pos_at1[1], z2: pos_at1[2]})
#             analyt_int = sp.integrate(sp.integrate(sp.integrate(integrand, (x, -sp.oo, sp.oo)), (y, -sp.oo, sp.oo)), (z, -sp.oo, sp.oo))
#             analyt_int_value = analyt_int.evalf()
            
#             print(f"Testing integral {name_i}-{name_k}", file=file)
#             results[count] = analyt_int_value 
#             print(f"Testing integral {name_i}-{name_k}")
#             print(f"sk value:{comparison[count]}")
#             print(f"analytical: {analyt_int_value}")
#             print(f'{comparison[count]} \t{analyt_int_value}',file=file)

#             count += 1
#     t_end = time.time()
#     print(f"integration took {t_end-t_start}")
#     with open("analytical_overlap_list.pkl", "wb") as f:
#         pickle.dump(results, f)
#     file.close()
#     return results


def compare_dipole_shifted(zeta1):
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

        element = 'Eu'
        r0 = 1.85 * covalent_radii[atomic_numbers[element]] / Bohr
        atom = AtomicDFT(element,
                         xc=opt.functional,
                         confinement=PowerConfinement(r0=r0, s=2),
                         perturbative_confinement=False,
                         configuration='[Xe] 4f7 6s2 6p0 5d0',
                         valence=['5d', '6s', '6p', '4f'],
                         timing=True,
                         )
        atom.run()

        # Compute Slater-Koster integrals:
        zeta_dict = {'4f': (zeta1[0], 3), '5d': (zeta1[1],2), '6s': (zeta1[2], 0), '6p': (zeta1[3], 1)}
        rmin, dr, N = 0.4, 0.02, 250
        off2c = Offsite2cTable(atom, atom, timing=True)
        off2c.run(rmin, dr, N, superposition=opt.superposition,
                  xc=opt.functional, stride=opt.stride, zeta_dict=zeta_dict, 
                #   nr=200, ntheta=500
                  )
        off2c.write()

    # vec = np.random.normal(size=3)
    # vec = vec/np.linalg.norm(vec)
    shift_vec = bohr_to_angstrom(np.array([0, 1, 0]))
    inter_vec = bohr_to_angstrom(np.array([1, 2, 1.5]))

    atoms = Atoms('Eu2', positions=[
        shift_vec,
        inter_vec + shift_vec
    ])

    #assemble actual matrix elements
    write('Eu2.xyz', atoms)
    method1 = SK_With_Shift()
    method1.load_atom_pair('Eu2.xyz')
    method1.set_euler_angles()
    method1.choose_relevant_matrix()
    method1.load_sk_files(path='Eu-Eu_offsite2c.skf', path_dp='Eu-Eu_offsite2c-dipole.skf')
    res1 = method1.calculate_dipole_elements()
    
    #calculate directly brute force
    res2 = get_analytic_2c_dipole(pos_at1=angstrom_to_bohr(shift_vec), pos_at2=angstrom_to_bohr(inter_vec+shift_vec), zeta1=zeta1, zeta2=zeta1, comparison=res1)
