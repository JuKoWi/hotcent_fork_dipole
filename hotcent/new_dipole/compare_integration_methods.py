import pickle
import time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import write
from ase.units import Bohr
from ase.data import covalent_radii, atomic_numbers
from hotcent.new_dipole.assemble_integrals import SK_Integral
from hotcent.new_dipole.offsite_twocenter_new import Offsite2cTable
from hotcent.new_dipole.offsite_twocenter_dipole import Offsite2cTableDipole
from hotcent.new_dipole.utils import angstrom_to_bohr, bohr_to_angstrom
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT

x, y, z = sp.symbols("x, y, z")
x1, y1, z1 = sp.symbols("x1, y1, z1")
r_1 = sp.sqrt((x-x1)**2 + (y-y1)**2 + (z-z1)**2)
x2, y2, z2 = sp.symbols("x2, y2, z2")
r_2 = sp.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2)


"""first atom"""
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


"""second atom"""
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


"""position operator"""
rx_1 = x
ry_1 = y 
rz_1 = z 

operator = {
    "ry": (ry_1,1,-1),
    "rz": (rz_1,1,0),
    "rx": (rx_1,1,1),
}


"""radial part"""
a, b, c = sp.symbols("a, b, c")
radial_1 = (2 * b/sp.pi)**(3/4) * sp.exp(-b* ((x-x1)**2 + (y-y1)**2 + (z-z1)**2))
radial_2 = (2* b/sp.pi)**(3/4) * sp.exp(-b* ((x-x2)**2 + (y-y2)**2 + (z-z2)**2))


def analytic_2c_dipole(pos_at1, pos_at2, zeta1, zeta2, comparison=None, idx_list=np.arange(len(first_center_real) * len(second_center) * len(operator))):
    """
    Calculate dipole elements analytically, print reuslt to terminal, and to file
    """
    file = open("comparison_dipole.txt", 'w')
    print(f'coordinate1: {pos_at1}', file=file)
    print(f'coordinate2: {pos_at2}', file=file)
    print("sk-value \t analytic", file=file)
    t_start = time.time()
    count = 0
    results = np.zeros((len(operator) * len(first_center_real) * len(second_center)))
    pos_at1 = angstrom_to_bohr(pos_at1)
    pos_at2 = angstrom_to_bohr(pos_at2)
    
    for name_i, i in first_center_real.items():
        for name_j, j in operator.items():
            for name_k, k in second_center.items():
                if count in idx_list:
                    print(count)
                    zeta1_val = zeta1[i[1]]
                    zeta2_val = zeta2[k[1]]
                    R1 = radial_1.subs({b: zeta1_val})
                    R2 = radial_2.subs({b: zeta2_val})
                    integrand = i[0] * R1 * j[0] * k[0] * R2 * r_1**i[1] * r_2**k[1] # eliminate poles by multiplying with r as if it was part of the radial part
                    integrand = integrand.subs({x1: pos_at1[0], y1: pos_at1[1], z1: pos_at1[2]})
                    integrand = integrand.subs({x2: pos_at2[0], y2: pos_at2[1], z2: pos_at2[2]})
                
                    analyt_int = sp.integrate(sp.integrate(sp.integrate(integrand, (x, -sp.oo, sp.oo)), (y, -sp.oo, sp.oo)), (z, -sp.oo, sp.oo))
                    analyt_int_value = analyt_int.evalf()
                    results[count] = analyt_int_value 
                
                    if comparison != None:
                        print(f"Testing integral {name_i}-{name_j}-{name_k}", file=file)
                        print(f"Testing integral {name_i}-{name_j}-{name_k}")
                        print(f"sk value:\t{comparison[count]}")
                        print(f"analytical:\t{analyt_int_value}")
                        print(f'{comparison[count]} \t{analyt_int_value}',file=file)

                count += 1
    t_end = time.time()
    print(f"integration took {t_end-t_start}")
    with open("analytical_dipole_list.pkl", "wb") as f:
        pickle.dump(results, f)
    file.close()
    return results


def analytic_2c(pos_at1, pos_at2, zeta1, zeta2, comparison=None, idx_list=np.arange(len(first_center_real) * len(second_center))):
    """calculate overlap integrals analytically, print to terminal and file"""
    file = open("comparison_overlap.txt", 'w')
    print(f'coordinate: {pos_at1}', file=file)
    print(f'coordinate2: {pos_at2}', file=file)
    print("sk-value \t analytic", file=file)
    t_start = time.time()
    count = 0
    results = np.zeros((len(first_center_real) * len(second_center)))
    pos_at1 = angstrom_to_bohr(pos_at1)
    pos_at2 = angstrom_to_bohr(pos_at2)
    for name_i, i in first_center_real.items():
        for name_k, k in second_center.items():
            if count in idx_list:
                print(count)
                zeta1_val = zeta1[i[1]]
                zeta2_val = zeta2[k[1]]
                R1 = radial_1.subs({b: zeta1_val})
                R2 = radial_2.subs({b: zeta2_val})
                integrand = i[0] * R1 * k[0] * R2 * r_1**i[1] * r_2**k[1] # eliminate poles by multiplying with r as if it was part of the radial part
                integrand = integrand.subs({x1: pos_at1[0], y1: pos_at1[1], z1: pos_at1[2]})
                integrand = integrand.subs({x2: pos_at2[0], y2: pos_at2[1], z2: pos_at2[2]})

                analyt_int = sp.integrate(sp.integrate(sp.integrate(integrand, (x, -sp.oo, sp.oo)), (y, -sp.oo, sp.oo)), (z, -sp.oo, sp.oo))
                analyt_int_value = analyt_int.evalf()
                results[count] = analyt_int_value 

                if comparison != None:
                    print(f"Testing integral {name_i}-{name_k}", file=file)
                    print(f"Testing integral {name_i}-{name_k}")
                    print(f"sk value:\t{comparison[count]}")
                    print(f"analytical:\t{analyt_int_value}")
                    print(f'{comparison[count]} \t{analyt_int_value}',file=file)

            count += 1
    t_end = time.time()
    print(f"integration took {t_end-t_start}")
    with open("analytical_overlap_list.pkl", "wb") as f:
        pickle.dump(results, f)
    file.close()
    return results


def compare_integrals(zeta1, use_existing_skf=False, dipole=True):
    if not use_existing_skf:
        #set up atomic system with skf files
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
        rmin, dr, N = 0.4, 0.02, 500
        if dipole:
            off2c = Offsite2cTableDipole(atom, atom, timing=True)
        else:
            off2c = Offsite2cTable(atom, atom, timing=True)
        off2c.run(rmin, dr, N, 
                  zeta=zeta_dict, 
                #   nr=200, ntheta=500
                  )
        off2c.write()

    # set atom positions
    # vec = np.random.normal(size=3)
    # vec = vec/np.linalg.norm(vec)
    shift_vec = bohr_to_angstrom(np.array([0.3, 0.5, -0.7]))
    inter_vec = bohr_to_angstrom(np.array([1, 2, 1.5]))
    atoms = Atoms('Eu2', positions=[
        shift_vec,
        inter_vec + shift_vec
    ])
    write('Eu2.xyz', atoms)

    #assemble actual matrix elements
    method1 = SK_Integral()
    method1.load_atom_pair('Eu2.xyz')
    if dipole:
        method1.get_list_dipole()
        method1.load_sk_file_dipole(path='Eu-Eu_offsite2c.skf', path_dipole='Eu-Eu_offsite2c-dipole.skf')
        res1 = method1.calculate_dipole()
        res2 = analytic_2c_dipole(pos_at1=shift_vec, pos_at2=inter_vec+shift_vec, zeta1=zeta1, zeta2=zeta1, comparison=res1)
    else:
        method1.load_sk_file(path='Eu-Eu_offsite2c.skf')
        res1 = method1.calculate()
        res2 = analytic_2c(pos_at1=shift_vec, pos_at2=inter_vec+shift_vec, zeta1=zeta1, zeta2=zeta1, comparison=res1)

def scan_grid_error(pos, index, dipole=False, plot=False, from_file=False):
    t_total_1 = time.time()

    # exponents for exponentials
    zeta1 = [0.27,0.27,0.27,0.27]

    atoms = Atoms('Eu2', positions=pos)
    write('Eu2.xyz', atoms)

    #dtheta and dr values to scan
    ntheta_list = np.arange(50, 200, 20)
    nr_list = np.arange(10, 100, 20)

    #initialize arrays
    error_array = np.zeros((np.shape(ntheta_list)[0], np.shape(nr_list)[0]))
    rel_error_array = np.zeros((np.shape(ntheta_list)[0], np.shape(nr_list)[0]))
    file_error = f"error_grid_scan-{index}.npy"
    file_rel_error = f"rel_error_grid_scan-{index}.npy"

    if not from_file:
        #calculate directly brute force
        print('start analytical integrals')
        if dipole:
            res2 = analytic_2c_dipole(pos_at1=pos[0], pos_at2=pos[1], zeta1=zeta1, zeta2=zeta1, idx_list=[index])[index]
        else:
            res2 = analytic_2c(pos_at1=pos[0], pos_at2=pos[1], zeta1=zeta1, zeta2=zeta1, idx_list=[index])[index]
        print('finished analytical integrals')

        #set up atoms
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

        for i, ntheta in enumerate(ntheta_list):
            for j, nr in enumerate(nr_list):
                print(f'start calculation sk-tables with nr = {nr}, ntheta = {ntheta}')
                time1 = time.time()

                # Compute Slater-Koster integrals:
                zeta_dict = {'4f': (zeta1[0], 3), '5d': (zeta1[1],2), '6s': (zeta1[2], 0), '6p': (zeta1[3], 1)}
                rmin, dr, N = 0.4, 0.05, 250
                if dipole:
                    off2c = Offsite2cTableDipole(atom, atom, timing=True)
                else:
                    off2c = Offsite2cTable(atom, atom, timing=True)
                off2c.run(rmin, dr, N, 
                          zeta=zeta_dict, 
                          nr=nr, ntheta=ntheta
                          )
                off2c.write()
                time2 = time.time()
                print(f'finished after {time2-time1}')

                #assemble actual matrix elements
                print("start sk-transformation")
                time1 = time.time()
                method1 = SK_Integral()
                method1.load_atom_pair('Eu2.xyz')
                if dipole:
                    method1.get_list_dipole()
                    method1.load_sk_file_dipole(path='Eu-Eu_offsite2c.skf', path_dipole='Eu-Eu_offsite2c-dipole.skf')
                    res1 = method1.calculate_dipole()
                else:
                    method1.load_sk_file(path='Eu-Eu_offsite2c.skf')
                    res1 = method1.calculate()

                time2 = time.time()
                print(f'finished transformation after {time2-time1}')

                #calculate errors
                res1 = res1[index]
                error_array[i,j] = res1 - res2
                if res2 != 0:
                    rel_error_array[i,j] = (res1 - res2) /res2
                else:
                    rel_error_array[i,j] = 0

        np.save(file_error, error_array)
        np.save(file_rel_error, rel_error_array)
    else:
        error_array = np.load(file_error) 
        rel_error_array = np.load(file_rel_error) 

    t_total_2 = time.time()
    print(f"finished scan after total of {t_total_2 -t_total_1}")
    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
        ny, nx = error_array.shape
        xvals = nr_list
        yvals = ntheta_list

        # Extent boundaries
        xmin, xmax = xvals.min(), xvals.max()
        ymin, ymax = yvals.min(), yvals.max()

        # Cell widths in continuous extent
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny

        # Centers of each cell in extent coordinates
        xcenters = xmin + (np.arange(nx) + 0.5) * dx
        ycenters = ymin + (np.arange(ny) + 0.5) * dy
        err = axs[0].imshow(np.abs(error_array), extent=[nr_list.min(), nr_list.max(), ntheta_list.min(), ntheta_list.max()], origin='lower', aspect='auto')
        rel_err = axs[1].imshow(np.abs(rel_error_array), extent=[nr_list.min(), nr_list.max(), ntheta_list.min(), ntheta_list.max()], origin='lower', aspect='auto')
        axs[0].set_xticks(xcenters)
        axs[0].set_xticklabels(nr_list)
        axs[0].set_xlabel(r"$n(r)$")
        axs[0].set_ylabel(r"$n(\theta)$")
        axs[0].set_yticks(ycenters)
        axs[0].set_yticklabels(ntheta_list)
        axs[0].set_title(f"Error for integral {index}")

        axs[1].set_xticks(xcenters)
        axs[1].set_xticklabels(nr_list)
        axs[0].set_yticks(ycenters)
        axs[0].set_yticklabels(ntheta_list)
        axs[1].set_title(f"Relative error for integral {index}")
        
        fig.colorbar(err, ax=axs[0])
        fig.colorbar(rel_err, ax=axs[1])
        fig.suptitle("Error for chosen integrals for different grid discretizations while creating .skf file")
        plt.savefig("error_grid-plot.pdf")
        plt.show()

def scan_distance(direction, index, dipole=False, n_dist=20, min_dist_angst=0.4, max_dist_angst=4, from_file=False, plot=False):
    t_total_1 = time.time()

    # exponents for exponentials
    zeta1 = [0.27,0.27,0.27,0.27]

    #initialize arrays
    direction = direction/np.linalg.norm(direction)
    distance_factors = np.linspace(min_dist_angst, max_dist_angst, n_dist)
    error_array = np.zeros((len(distance_factors)))
    rel_error_array = np.zeros((len(distance_factors)))
    file_error = f'error_distance_scan-{index}.npy'
    file_rel_error = f"rel_error_distance_scan-{index}.npy"

    if not from_file:
        #set up atoms
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
        time1 = time.time()
        zeta_dict = {'4f': (zeta1[0], 3), '5d': (zeta1[1],2), '6s': (zeta1[2], 0), '6p': (zeta1[3], 1)}
        rmin, dr, N = 0.1, 0.05, 250
        if dipole:
            off2c = Offsite2cTableDipole(atom, atom, timing=True)
        else:
            off2c = Offsite2cTable(atom, atom, timing=True)
        off2c.run(rmin, dr, N, 
                  zeta=zeta_dict, 
                #   nr=100, ntheta=200
                  )
        off2c.write()
        time2 = time.time()
        print(f'finished after {time2-time1}')

        for i, dist in enumerate(distance_factors):
            pos = np.array([[0,0,0], direction * dist])
            atoms = Atoms('Eu2', positions=pos)
            write('Eu2.xyz', atoms)

            print('start analytical integrals')
            if dipole:
                res2 = analytic_2c_dipole(pos_at1=pos[0], pos_at2=pos[1], zeta1=zeta1, zeta2=zeta1, idx_list=[index])[index]
            else:
                res2 = analytic_2c(pos_at1=pos[0], pos_at2=pos[1], zeta1=zeta1, zeta2=zeta1, idx_list=[index])[index]
            print('finished analytical integrals')
    
            #assemble actual matrix elements
            print("start sk-transformation")
            time1 = time.time()
            method1 = SK_Integral()
            method1.load_atom_pair('Eu2.xyz')
            if dipole:
                method1.get_list_dipole()
                method1.load_sk_file_dipole(path='Eu-Eu_offsite2c.skf', path_dipole='Eu-Eu_offsite2c-dipole.skf')
                res1 = method1.calculate_dipole()
            else:
                method1.load_sk_file(path='Eu-Eu_offsite2c.skf')
                res1 = method1.calculate()

            time2 = time.time()
            print(f'finished transformation after {time2-time1}')

            #calculate errors
            res1 = res1[index]
            error_array[i] = res1 - res2
            if res2 != 0:
                rel_error_array[i] = (res1 -res2) / res2
            else:
                rel_error_array[i] = 0
        np.save(file_error, error_array)
        np.save(file_rel_error, rel_error_array)
    else:
        error_array = np.load(file_error) 
        rel_error_array = np.load(file_rel_error)

    t_total_2 = time.time()
    print(f"finished scan after total of {t_total_2 -t_total_1}")
    if plot:
        fig, axs = plt.subplots(ncols=2)
        axs[0].scatter(distance_factors, error_array, label="error")
        axs[1].scatter(distance_factors, rel_error_array, label="relative error")
        axs[0].set_xlabel(r"R / $\AA$")
        axs[0].set_xlabel(r"R / $\AA$")
        axs[0].set_ylabel(r"integral error")
        axs[0].legend()
        axs[1].set_ylabel(r"relative integral error")
        axs[0].set_xlabel(r"R / $\AA$")
        axs[0].set_xlabel(r"R / $\AA$")
        axs[1].legend()
        plt.show()
        
