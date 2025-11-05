import numpy as np
import os
from ase import Atoms
from ase.io import write
import pickle
import time
from scipy.integrate import nquad
import sympy as sp
from hotcent.new_dipole.dipole import SK_Integral
from hotcent.new_dipole.compare_integration_methods import first_center_real, second_center, operator

from hotcent.new_dipole.offsite_twocenter_dipole import Offsite2cTableDipole
from hotcent.new_dipole.onsite_twocenter_dipole import Onsite2cTable
from optparse import OptionParser
from ase.units import Bohr
from ase.data import covalent_radii, atomic_numbers
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT

def get_analytic_2c_integrals(pos_at1, zeta1, zeta2, comparison):
    file = open("comparison.txt", 'w')
    print(f'coordinate: {pos_at1}', file=file)
    print("sk-value \t analytic", file=file)
    t_start = time.time()
    count = 0
    results = np.zeros((len(operator) * len(first_center_real) * len(second_center)))
    
    for name_i, i in first_center_real.items():
        for name_j, j in operator.items():
            for name_k, k in second_center.items():
                # if (i[1]<2 and k[1]<2):
                zeta1_val = zeta1[i[1]]
                zeta2_val = zeta2[k[1]]
                R1 = radial_1.subs({b: zeta1_val})
                R2 = radial_2.subs({b: zeta2_val})
                integrand = i[0] * R1 * j[0] * k[0] * R2 * r**i[1] * r_2**k[1] # eliminate poles by multiplying with r as if it was part of the radial part
                # print(integrand)

                integrand = integrand.subs({x0: pos_at1[0], y0: pos_at1[1], z0: pos_at1[2]})
                analyt_int = sp.integrate(sp.integrate(sp.integrate(integrand, (x, -sp.oo, sp.oo)), (y, -sp.oo, sp.oo)), (z, -sp.oo, sp.oo))
                # analyt_int = analyt_int.subs({x0: pos_at1[0], y0: pos_at1[1], z0: pos_at1[2]})
                analyt_int_value = analyt_int.evalf()
                
                # Precompile symbolic -> numeric function once
                # f_num = sp.lambdify((x, y, z), integrand, "numpy")

                print(f"Testing integral {name_i}-{name_j}-{name_k}", file=file)
                # num, err = nquad(f_num, [[-20, 20], [-20, 20], [-20, 20]])
                results[count] = analyt_int_value 
                abs_err = comparison[count] - analyt_int
                print(f"Testing integral {name_i}-{name_j}-{name_k}")
                print(f"sk value:{comparison[count]}")
                # print(f" numerical: {num}")
                print(f"analytical: {analyt_int_value}")
                print(f'{comparison[count]} \t{analyt_int_value}',file=file)
                # print(analyt_int_value/comparison[count])
                # print(f"Result = {num:.6e}, Estimated error = {err:.2e}")

                count += 1
    t_end = time.time()
    print(f"integration took {t_end-t_start}")
    with open("analytical_integrals_list.pkl", "wb") as f:
        pickle.dump(results, f)
    file.close()
    return results

def create_grid_error_chart():
    zeta1 = [1,1,1]
    VEC = np.random.normal(size=3)
    VEC = VEC/np.linalg.norm(VEC)

    ntheta_list = np.arange(50, 500, 50)
    nr_list = np.arange(10, 150, 20)

    mae_array = np.zeros((np.shape(ntheta_list)[0], np.shape(nr_list)[0]))
    mse_array = np.zeros((np.shape(ntheta_list)[0], np.shape(nr_list)[0]))


    for ntheta in ntheta_list:
        for nr in nr_list:
            element = 'Ge'
            r0 = 1.85 * covalent_radii[atomic_numbers[element]] / Bohr
            atom = AtomicDFT(element,
                             confinement=PowerConfinement(r0=r0, s=2),
                             perturbative_confinement=False,
                             configuration='[Ar] 4s2 3d10 4p2',
                             valence=['4s', '4p', '3d'],
                             timing=True,
                             )
            atom.run()

            # Compute Slater-Koster integrals:
            zeta_dict = {'4s': (zeta1[0], 0), '4p': (zeta1[1],1), '3d': (zeta1[2], 2)}
            rmin, dr, N = 0.0, 0.05, 250
            off2c = Offsite2cTableDipole(atom, atom, timing=True)
            off2c.run(rmin, dr, N, 
                      zeta_dict=zeta_dict, 
                      nr=nr, ntheta=ntheta
                      )
            off2c.write()
