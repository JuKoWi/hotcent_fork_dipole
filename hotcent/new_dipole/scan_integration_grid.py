import numpy as np
import os
from ase import Atoms
from ase.io import write
import pickle
import time
from scipy.integrate import nquad
import sympy as sp
from hotcent.new_dipole.assemble_integrals import SK_Integral_Dipole
from hotcent.new_dipole.compare_integration_methods import first_center_real, second_center, operator, radial_1, radial_2, r, r_2, b, x, y, z, x0, y0, z0

from hotcent.new_dipole.offsite_twocenter_dipole import Offsite2cTableDipole
from hotcent.new_dipole.onsite_twocenter_dipole import Onsite2cTable
from optparse import OptionParser
from ase.units import Bohr
from ase.data import covalent_radii, atomic_numbers
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT


def get_analytic_2c_integrals(pos_at1, zeta1, zeta2):
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
                N1 = (2 * zeta1_val/np.pi)**(3/4)
                N2 = (2 * zeta2_val/np.pi)**(3/4)
                R1 = N1*radial_1.subs({b: zeta1_val})
                R2 = N2*radial_2.subs({b: zeta2_val})
                integrand = i[0] * R1 * j[0] * k[0] * R2 * r**i[1] * r_2**k[1] # eliminate poles by multiplying with r as if it was part of the radial part
                # print(integrand)
                integrand = integrand.subs({x0: pos_at1[0], y0: pos_at1[1], z0: pos_at1[2]})
                analyt_int = sp.integrate(sp.integrate(sp.integrate(integrand, (x, -sp.oo, sp.oo)), (y, -sp.oo, sp.oo)), (z, -sp.oo, sp.oo))
                analyt_int_value = analyt_int.evalf()
                print(f"Testing integral {name_i}-{name_j}-{name_k}", file=file)
                print(f"analytical value{count}: {analyt_int_value}")
                results[count] = analyt_int_value 
                if count % 10 == 0:
                    np.save("analytical-integrals.npy", results)
                count += 1
    np.save("analytical-integrals.npy", results)
    t_end = time.time()
    print(f"analytical integration took {t_end-t_start}")
    file.close()
    return results

def create_grid_error_chart():
    t_total_1 = time.time()
    # exponents for exponentials
    zeta1 = [0.27,0.27,0.27,0.27]
    #set up random atom position
    # VEC = np.random.normal(size=3)
    # VEC = VEC/np.linalg.norm(VEC)
    VEC = [0.3, -0.7, 0.54]
    atoms = Atoms('Eu2', positions=[
        [0.0, 0.0, 0.0],
        [VEC[0], VEC[1], VEC[2]]
    ])
    write('Eu2.xyz', atoms)

    #dtheta and dr values to scan
    ntheta_list = np.arange(50, 500, 50)
    nr_list = np.arange(10, 150, 20)

    #initialize arrays
    mae_array = np.zeros((np.shape(ntheta_list)[0], np.shape(nr_list)[0]))
    mse_array = np.zeros((np.shape(ntheta_list)[0], np.shape(nr_list)[0]))

    #calculate directly brute force
    print('start analytical integrals')
    res2 = get_analytic_2c_integrals(pos_at1=[VEC[0], VEC[1], VEC[2]], zeta1=zeta1, zeta2=zeta1)
    print('finished analytical integrals')

    for i, ntheta in enumerate(ntheta_list):
        for j, nr in enumerate(nr_list):
            print('start calculation sk-tables with nr = {nr}, ntheta = {ntheta}')
            time1 = time.time()
            element = 'Eu'
            r0 = 1.85 * covalent_radii[atomic_numbers[element]] / Bohr
            atom = AtomicDFT(element,
                             confinement=PowerConfinement(r0=r0, s=2),
                             perturbative_confinement=False,
                             configuration='[Xe] 4f7 6s2 6p0 5d0',
                             valence=['5d', '6s', '6p', '4f'],
                             timing=True,
                             )
            atom.run()

            # Compute Slater-Koster integrals:
            zeta_dict = {'4f': (zeta1[0], 3), '5d': (zeta1[1],2), '6s': (zeta1[2], 0), '6p': (zeta1[3], 1)}
            rmin, dr, N = 0.0, 0.05, 250
            off2c = Offsite2cTableDipole(atom, atom, timing=True)
            off2c.run(rmin, dr, N, 
                      zeta_dict=zeta_dict, 
                      nr=nr, ntheta=ntheta
                      )
            off2c.write()
            time2 = time.time()
            print(f'finished after {time2-time1}')

            #assemble actual matrix elements
            print("start sk-transformation")
            time1 = time.time()
            method1 = SK_Integral_Dipole()
            method1.load_atom_pair('Eu2.xyz')
            method1.set_euler_angles()
            method1.choose_relevant_matrix()
            method1.load_SK_dipole_file('Eu-Eu_offsite2c-dipole.skf')
            res1 = method1.calculate_dipole()
            time2 = time.time()
            print(f'finished transformation after {time2-time1}')
            
            #calculate errors
            mae_array[i,j] = np.sum(np.abs(res1-res2))
            mse_array[i,j] = np.sum((res1-res2)**2)
    np.save('mae_grid_scan', mae_array)
    np.save("mse_grid_scan", mse_array)
    t_total_2 = time.time()
    print(f"finished scan after total of {t_total_2 -t_total_2}")

            

            

    