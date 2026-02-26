import pickle
import sys
import time
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from ase import Atoms
from ase.io import write
from ase.units import Bohr
from ase.data import covalent_radii, atomic_numbers
from hotcent.new_dipole.assemble_integrals import SK_Integral
from hotcent.new_dipole.integrals import first_center, second_center, theta2, theta1, phi
from hotcent.new_dipole.offsite_twocenter_new import Offsite2cTable
from hotcent.new_dipole.offsite_twocenter_dipole import Offsite2cTableDipole
from hotcent.new_dipole.utils import angstrom_to_bohr, bohr_to_angstrom
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
plt.rcParams['savefig.bbox'] = 'tight'          
plt.rcParams["axes.formatter.limits"] = (-2,5)
plt.rcParams.update({'font.size':35})

x, y, z = sym.symbols("x, y, z")
x1, y1, z1 = sym.symbols("x1, y1, z1")
r_1 = sym.sqrt((x-x1)**2 + (y-y1)**2 + (z-z1)**2)
x2, y2, z2 = sym.symbols("x2, y2, z2")
r_2 = sym.sqrt((x-x2)**2 + (y-y2)**2 + (z-z2)**2)


"""first atom"""
s_1 = 1 / (2 * sym.sqrt(sym.pi))

px_1 = sym.sqrt(3 / (4 * sym.pi)) * (x-x1)/r_1
py_1 = sym.sqrt(3 / (4 * sym.pi)) * (y-y1)/r_1
pz_1 = sym.sqrt(3 / (4 * sym.pi)) * (z-z1)/r_1

dxy_1 = sym.sqrt(15/(4 * sym.pi)) *(x-x1)*(y-y1)/r_1**2
dyz_1 = sym.sqrt(15 / (4*sym.pi)) * (y-y1)*(z-z1)/r_1**2
dxz_1 = sym.sqrt(15/(4* sym.pi)) * (x-x1)*(z-z1)/r_1**2
dx2y2_1 = sym.sqrt(15/(16*sym.pi)) * ((x-x1)**2-(y-y1)**2)/r_1**2
dz2_1 = sym.sqrt(5 / (16 * sym.pi)) * (3*(z-z1)**2 -r_1**2)/r_1**2

f1_1 = 1/4 * sym.sqrt(35/(2*sym.pi)) * (y-y1)*(3*(x-x1)**2 - (y-y1)**2)/r_1**3
f2_1 = 1/2 * sym.sqrt(105/sym.pi) * (x-x1)*(y-y1)*(z-z1)/r_1**3
f3_1 = 1/4 * sym.sqrt(21/(2*sym.pi)) * (y-y1) * (5*(z-z1)**2 - r_1**2)/r_1**3
f4_1 = 1/4 * sym.sqrt(7/sym.pi) * (5*(z-z1)**3 - 3* (z-z1) * r_1**2)/r_1**3
f5_1 = 1/4 * sym.sqrt(21/(2*sym.pi)) * (x-x1) * (5*(z-z1)**2 - r_1**2)/r_1**3
f6_1 = 1/4 * sym.sqrt(105/sym.pi) * ((x-x1)**2 - (y-y1)**2) * (z-z1) /r_1**3
f7_1 = 1/4 * sym.sqrt(35/(2*sym.pi)) * (x-x1) * ((x-x1)**2 - 3* (y-y1)**2) /r_1**3

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

px_2 = sym.sqrt(3 / (4 * sym.pi)) * (x-x2)/r_2
py_2 = sym.sqrt(3 / (4 * sym.pi)) * (y-y2)/r_2
pz_2 = sym.sqrt(3 / (4 * sym.pi)) * (z-z2)/r_2 

dxy_2 = sym.sqrt(15 / (4 * sym.pi)) * (x-x2)*(y-y2)/r_2**2
dyz_2 = sym.sqrt(15 / (4 * sym.pi)) * (y-y2)*(z-z2)/r_2**2
dxz_2 = sym.sqrt(15 / (4 * sym.pi)) * (x-x2)*(z-z2)/r_2**2
dx2y2_2 = sym.sqrt(15 / (16*sym.pi)) * ((x-x2)**2-(y-y2)**2)/r_2**2
dz2_2 = sym.sqrt(5 / (16 * sym.pi)) * (3 * (z-z2)**2 - r_2**2)/r_2**2

f1_2 = 1/4 * sym.sqrt(35/(2*sym.pi)) * (y-y2)*(3*(x-x2)**2 - (y-y2)**2)/r_2**3
f2_2 = 1/2 * sym.sqrt(105/sym.pi) * (x-x2)*(y-y2)*(z-z2)/r_2**3
f3_2 = 1/4 * sym.sqrt(21/(2*sym.pi)) * (y-y2) * (5*(z-z2)**2 - r_2**2)/r_2**3
f4_2 = 1/4 * sym.sqrt(7/sym.pi) * (5*(z-z2)**3 - 3* (z-z2) * r_2**2)/r_2**3
f5_2 = 1/4 * sym.sqrt(21/(2*sym.pi)) * (x-x2) * (5*(z-z2)**2 - r_2**2)/r_2**3
f6_2 = 1/4 * sym.sqrt(105/sym.pi) * ((x-x2)**2 - (y-y2)**2) * (z-z2) /r_2**3
f7_2 = 1/4 * sym.sqrt(35/(2*sym.pi)) * (x-x2) * ((x-x2)**2 - 3* (y-y2)**2) /r_2**3

second_center_real = {
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
a, b, c = sym.symbols("a, b, c")
radial_1 = (2 * b/sym.pi)**(3/4) * 5 *  sym.exp(-b* ((x-x1)**2 + (y-y1)**2 + (z-z1)**2))
radial_2 = (2* b/sym.pi)**(3/4) * 5 * sym.exp(-b* ((x-x2)**2 + (y-y2)**2 + (z-z2)**2))

def compare_sphericals():
    for i, integral in enumerate(second_center.values()):
        expr1 = integral[0]
        for j, integral2 in enumerate(second_center_real.values()):
            bool1 = (integral2[1] == integral[1])
            bool2 = (integral2[2] == integral[2])
            if bool1 and bool2:
                expr2 = integral2[0]
                expr2 = expr2.subs({
                    x2: 0,
                    y2: 0,
                    z2: 0,
                    x: sym.sin(theta2)*sym.cos(phi),
                    y:sym.sin(theta2)*sym.sin(phi),
                    z: sym.cos(theta2)
                })
                print(sym.simplify(expr1 -expr2))


def analytic_2c_dipole(pos_at1, pos_at2, zeta1, zeta2, comparison=None, idx_list=np.arange(len(first_center_real) * len(second_center_real) * len(operator))):
    """
    Calculate dipole elements analytically, print result to terminal, and to file.
    Positions in angstrom
    """
    file = open("comparison_dipole.txt", 'w')
    print(f'coordinate1: {pos_at1}', file=file)
    print(f'coordinate2: {pos_at2}', file=file)
    print("sk-value \t analytic", file=file)
    t_start = time.time()
    count = 0
    results = np.zeros((len(operator) * len(first_center_real) * len(second_center_real)))
    pos_at1 = angstrom_to_bohr(pos_at1)
    pos_at2 = angstrom_to_bohr(pos_at2)
    
    for name_i, i in first_center_real.items():
        for name_j, j in operator.items():
            for name_k, k in second_center_real.items():
                if count in idx_list:
                    print(count)
                    zeta1_val = zeta1[i[1]]
                    zeta2_val = zeta2[k[1]]
                    R1 = radial_1.subs({b: zeta1_val})
                    R2 = radial_2.subs({b: zeta2_val})
                    integrand = i[0] * R1 * j[0] * k[0] * R2 * r_1**i[1] * r_2**k[1] # eliminate poles by multiplying with r as if it was part of the radial part
                    integrand = integrand.subs({x1: pos_at1[0], y1: pos_at1[1], z1: pos_at1[2]})
                    integrand = integrand.subs({x2: pos_at2[0], y2: pos_at2[1], z2: pos_at2[2]})
                
                    analyt_int = sym.integrate(sym.integrate(sym.integrate(integrand, (x, -sym.oo, sym.oo)), (y, -sym.oo, sym.oo)), (z, -sym.oo, sym.oo))
                    analyt_int_value = analyt_int.evalf()
                    results[count] = analyt_int_value 
                
                    if not(comparison is None):
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


def analytic_2c(pos_at1, pos_at2, zeta1, zeta2, comparison=None, idx_list=np.arange(len(first_center_real) * len(second_center_real))):
    """calculate overlap integrals analytically, print to terminal and file. 
        positions in angstrom
    """
    file = open("comparison_overlap.txt", 'w')
    print(f'coordinate: {pos_at1}', file=file)
    print(f'coordinate2: {pos_at2}', file=file)
    print("sk-value \t analytic", file=file)
    t_start = time.time()
    count = 0
    results = np.zeros((len(first_center_real) * len(second_center_real)))
    pos_at1 = angstrom_to_bohr(pos_at1)
    pos_at2 = angstrom_to_bohr(pos_at2)
    for name_i, i in first_center_real.items():
        for name_k, k in second_center_real.items():
            if count in idx_list:
                print(count)
                zeta1_val = zeta1[i[1]]
                zeta2_val = zeta2[k[1]]
                R1 = radial_1.subs({b: zeta1_val})
                R2 = radial_2.subs({b: zeta2_val})
                integrand = i[0] * R1 * k[0] * R2 * r_1**i[1] * r_2**k[1] # eliminate poles by multiplying with r as if it was part of the radial part
                integrand = integrand.subs({x1: pos_at1[0], y1: pos_at1[1], z1: pos_at1[2]})
                integrand = integrand.subs({x2: pos_at2[0], y2: pos_at2[1], z2: pos_at2[2]})

                analyt_int = sym.integrate(sym.integrate(sym.integrate(integrand, (x, -sym.oo, sym.oo)), (y, -sym.oo, sym.oo)), (z, -sym.oo, sym.oo))
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
    shift_vec = bohr_to_angstrom(np.array([0, 0, 0]))
    inter_vec = bohr_to_angstrom(np.array([0, 0, 0.8]))
    atoms = Atoms('Eu2', positions=[
        shift_vec,
        inter_vec + shift_vec
    ])
    write('Eu2.xyz', atoms)

    #assemble actual matrix elements
    method1 = SK_Integral()
    method1.load_atom_file('Eu2.xyz')
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
    zeta1 = [1,1,1,1]

    atoms = Atoms('Eu2', positions=pos)
    write('Eu2.xyz', atoms)

    #dtheta and dr values to scan
    ntheta_list = np.arange(50, 110, 10)
    nr_list = np.arange(10, 60, 10)
    # ntheta_list = np.linspace(start=50, stop=100, num=2)
    # nr_list = np.linspace(start=10, stop=50, num=4)

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
                rmin, dr, N = 0.4, 0.1, 12
                if dipole:
                    off2c = Offsite2cTableDipole(atom, atom, timing=True)
                    off2c.run(rmin, dr, N, 
                              zeta=zeta_dict, 
                              nr=nr, ntheta=ntheta,
                              wflimit=1e-8
                              )
                    off2c.write_dipole()
                else:
                    off2c = Offsite2cTable(atom, atom, timing=True)
                    off2c.run(rmin, dr, N, 
                              zeta=zeta_dict, 
                              nr=nr, ntheta=ntheta,
                              wflimit=1e-8
                              )
                    off2c.write()
                time2 = time.time()
                print(f'finished after {time2-time1}')

                #assemble actual matrix elements
                print("start sk-transformation")
                time1 = time.time()
                method1 = SK_Integral()
                method1.load_atom_file('Eu2.xyz')
                if dipole:
                    method1.get_list_dipole()
                    method1.load_sk_file_dipole(path='Eu-Eu.skf', path_dipole='Eu-Eu_dipole.skf')
                    res1 = method1.calculate_dipole()
                else:
                    method1.load_sk_file(path='Eu-Eu.skf')
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
        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(25,9)) 
        ny, nx = error_array.shape
        xvals = nr_list
        yvals = ntheta_list

        xv, yv = np.meshgrid(xvals, yvals)
        Z1 = bohr_to_angstrom(error_array)
        Z2 = bohr_to_angstrom(np.abs(error_array))
        err = axs[0].pcolormesh(xv, yv, Z1, shading='nearest')
        rel_err = axs[1].pcolormesh(xv, yv, Z2, shading='nearest', norm=colors.LogNorm(vmin=Z2.min(), vmax=Z2.max()))
        # err = axs[0].imshow(np.abs(error_array), extent=[nr_list.min(), nr_list.max(), ntheta_list.min(), ntheta_list.max()], norm='log', origin='lower', aspect='auto')
        # rel_err = axs[1].imshow(np.abs(rel_error_array), extent=[nr_list.min(), nr_list.max(), ntheta_list.min(), ntheta_list.max()], norm='log', origin='lower', aspect='auto')
        # axs[0].set_xticks(xcenters)
        axs[0].set_xlabel(r"$n(r)$")
        axs[0].set_ylabel(r"$n(\theta)$")
        # axs[0].set_yticks(ycenters)
        axs[0].set_title(f"Numerical - analytical") 

        # axs[1].set_xticks(xcenters)
        axs[1].set_xlabel(r"$n(r)$")
        # axs[0].set_yticks(ycenters)
        axs[1].set_ylabel(r"$n(\theta)$")
        axs[1].set_title(f"Absolute error logarithmic") 
        
        fig.colorbar(err, ax=axs[0])

        cbar1 = fig.colorbar(rel_err, ax=axs[1])
        cbar1.ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
        cbar1.ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10, subs=(1,2,3,4,5,6,7,8,9))
        )
        cbar1.ax.yaxis.set_minor_formatter(ticker.LogFormatter())

        # fig.suptitle("Error for chosen integrals for different grid discretizations while creating .skf file")
        plt.savefig(f"error_grid-plot{index}.pdf")
        plt.show()

def scan_distance(direction, index, dipole=False, n_dist=20, min_dist_angst=0.4, d_dist_angst=4, from_file=False, plot=False):
    """scan the dependence of the error of selected matrix elements on the internuclear distance
        For a selected matrix element measure how much the numerically integrated value deviates 
        from the analytical value. Overwrite the radial part of dummy atom Eu with gaussian* r^nl
        with zeta as gaussian exponent. 
            index: index of the matrix element
            direction: unit vector to set internuclear axis 
    """
    t_total_1 = time.time()

    # exponents for exponentials
    zeta1 = [1,1,1,1]

    #initialize arrays
    direction = direction/np.linalg.norm(direction)
    distance_factors = min_dist_angst + np.arange(n_dist) * d_dist_angst 
    error_array = np.zeros((len(distance_factors)))
    rel_error_array = np.zeros((len(distance_factors)))
    file_error = f'error_distance_scan-{index}.npy'
    file_rel_error = f"rel_error_distance_scan-{index}.npy"
    file_analytical = f"analytical_vals-{index}.npy"

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
        rmin, dr, N = angstrom_to_bohr(min_dist_angst), angstrom_to_bohr(d_dist_angst), n_dist
        if dipole:
            off2c = Offsite2cTableDipole(atom, atom, timing=True)
            off2c.run(rmin, dr, N, 
                      zeta=zeta_dict, wflimit=1e-11 
                      )
            off2c.write_dipole()
        else:
            off2c = Offsite2cTable(atom, atom, timing=True)
            off2c.run(rmin, dr, N, 
                      zeta=zeta_dict, wflimit=1e-11 
                      )
            off2c.write()
        time2 = time.time()
        print(f'finished after {time2-time1}')
        list_res1 = []
        list_res2 = []
        for i, dist in enumerate(distance_factors):
            pos = np.array([[0,0,0], direction * dist])
            atoms = Atoms('Eu2', positions=pos)
            write('Eu2.xyz', atoms)

            print('start analytical integrals')
            if dipole:
                res2 = analytic_2c_dipole(pos_at1=pos[0], pos_at2=pos[1], zeta1=zeta1, zeta2=zeta1, idx_list=[index])[index]
            else:
                res2 = analytic_2c(pos_at1=pos[0], pos_at2=pos[1], zeta1=zeta1, zeta2=zeta1, idx_list=[index])[index]
            list_res2.append(res2)
            print('finished analytical integrals')
    
            #assemble actual matrix elements
            print("start sk-transformation")
            time1 = time.time()
            method1 = SK_Integral()
            method1.load_atom_file('Eu2.xyz')
            if dipole:
                method1.get_list_dipole()
                method1.load_sk_file_dipole(path='Eu-Eu.skf', path_dipole='Eu-Eu_dipole.skf')
                res1 = method1.calculate_dipole()
            else:
                method1.load_sk_file(path='Eu-Eu.skf')
                res1 = method1.calculate()

            time2 = time.time()
            print(f'finished transformation after {time2-time1}')

            #calculate errors
            res1 = res1[index]
            list_res1.append(res1)
            error_array[i] = res1 - res2
            if res2 != 0:
                rel_error_array[i] = (res1 -res2) / res2
            else:
                rel_error_array[i] = 0
        np.save(file_error, error_array)
        np.save(file_rel_error, rel_error_array)
        np.save(file_analytical, np.array(list_res2))
    else:
        error_array = np.load(file_error) 
        rel_error_array = np.load(file_rel_error)
        list_res2 = np.load(file_analytical)

    t_total_2 = time.time()
    print(f"finished scan after total of {t_total_2 -t_total_1}")
    if plot:
        list_res2 = np.array(list_res2)
        fig, axs = plt.subplots(ncols=3, figsize=(45,9))
        axs[0].scatter(distance_factors, bohr_to_angstrom(list_res2))
        axs[0].set_xlabel(r"R / $\AA$")
        axs[0].set_ylabel(r"$d_\text{analytical}$ / $\AA$")
        axs[0].set_title('(a)')
        axs[1].scatter(distance_factors, bohr_to_angstrom(error_array)) 
        axs[1].set_xlabel(r"R / $\AA$")
        axs[1].set_xlabel(r"R / $\AA$")
        axs[1].set_ylabel(r"$d_\text{numerical}- d_\text{analytical}$ / $\AA$ ")
        axs[1].set_xlabel(r"R / $\AA$")
        axs[1].set_title('(b)')
        axs[2].scatter(distance_factors, rel_error_array)
        axs[2].set_ylabel(r"$|d_\text{numerical}-d_\text{analytical}|/d_\text{analytical}$")
        axs[2].set_xlabel(r"R / $\AA$")
        axs[2].set_title('(c)')
        plt.savefig(f"distance_error_range{index}.pdf")
        plt.ticklabel_format(style='sci')
        plt.show()
    print(list_res1)
    print(angstrom_to_bohr(distance_factors))

        
