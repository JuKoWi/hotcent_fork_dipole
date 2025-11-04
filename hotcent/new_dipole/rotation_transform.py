import numpy as np
import sympy as sp
import pickle
from sympy.physics.quantum.dagger import Dagger
from hotcent.new_dipole.integrals import first_center, phi, theta1, first_center_complex

PHI = sp.symbols('phi')
THETA = sp.symbols('theta')
GAMMA = sp.symbols('gamma')

def to_spherical(R):
    r = np.sqrt(np.sum(R**2))
    theta = np.arccos(R[2] / r)
    phi = np.arctan2(R[1], R[0])
    return np.array([r, theta, phi])

def d_mat_elem(l, m, n, phi):
    expr = 0
    k_min = max(0, m-n)
    k_max = min(l+m, l-n)
    for k in range(k_min, k_max+1):
        prefac = (-1)**(k-m+n) * sp.sqrt(sp.factorial(l+m) * sp.factorial(l-m) * sp.factorial(l+n) * sp.factorial(l-n)) / (sp.factorial(l+m-k) * sp.factorial(k) * sp.factorial(l-k-n) * sp.factorial(k-m+n))
        angle_part = (sp.cos(phi/2)**(2*l-2*k+m-n) * sp.sin(phi / 2)**(2*k-m+n))
        expr += prefac * angle_part 
    return sp.trigsimp(expr)


def d_mat(theta):
    """small-d Wigner matrix for the angle beta (up to d orbitals). 
    Returns a block-diagonal 9×9 symbolic matrix with j=0,1,2 blocks."""
    d = sp.zeros(9, 9)
    row_start = 0

    for j in range(3):  # j = 0, 1, 2
        size = 2*j + 1
        block = sp.zeros(size, size)
        for mi, m in enumerate(range(-j, j+1)):
            for ni, n in enumerate(range(-j, j+1)):
                block[mi, ni] = d_mat_elem(j, m, n, theta)
        d[row_start:row_start+size, row_start:row_start+size] = block
        row_start += size
    return d

def z_rot_mat(phi):
    Dz = sp.zeros(9,9)
    count = 0
    for l in range(3): # l=0,1,2
        for m in range(-l, l+1):
            Dz[count, count] = sp.exp(-sp.I*m*phi)
            count += 1
    # print('z-rotation')
    # print(Dz)
    return Dz
        
def Wigner_D_complex(euler_phi, euler_theta, euler_gamma):
    Dz = z_rot_mat(phi=euler_phi)
    dy = d_mat(theta=euler_theta)
    Dz2 = z_rot_mat(phi=euler_gamma) #last rotation around z-axis
    # print('small d-matrix')
    # print(dy)
    total = Dz2 * dy * Dz 
    # print('total transform for complex')
    # print(total)
    return total
    
def Wigner_D_real(euler_phi, euler_theta, euler_gamma):
    transform_to_real = sp.zeros(9,9)

    #s
    transform_to_real[0,0] = 1

    #p
    transform_to_real[1,1] = sp.I / sp.sqrt(2)
    transform_to_real[1,3] = sp.I / sp.sqrt(2)
    transform_to_real[2,2] = 1
    transform_to_real[3,1] = 1/sp.sqrt(2)
    transform_to_real[3,3] = -1/sp.sqrt(2)


    #d
    transform_to_real[4,4] = sp.I / sp.sqrt(2)
    transform_to_real[4,8] = -sp.I / sp.sqrt(2)
    transform_to_real[5,5] = sp.I / sp.sqrt(2)
    transform_to_real[5,7] = sp.I / sp.sqrt(2)
    transform_to_real[6,6] = 1
    transform_to_real[7,5] = 1/sp.sqrt(2)
    transform_to_real[7,7] = -1/sp.sqrt(2)
    transform_to_real[8,4] = 1 / sp.sqrt(2)
    transform_to_real[8,8] = 1 / sp.sqrt(2)
    


    transform_to_comp = transform_to_real.H
    D_total = transform_to_real * Wigner_D_complex(euler_phi=euler_phi, euler_theta=euler_theta, euler_gamma=euler_gamma) * transform_to_comp
    # print('check for identity of transform to real harmonics')
    # print(transform_to_comp * transform_to_real)
    # print(transform_to_real * Dagger(transform_to_real))
    with open("symbolic_D_matrix.pkl", "wb") as f:
        pickle.dump(D_total, f)
    return D_total


"""some tests"""

def evaluate_spherical(key, unit_vec):
    theta_val, phi_val = to_spherical(R=unit_vec)[1:]
    sh_func = first_center[key][0]
    res = sh_func.subs({phi: phi_val, theta1: theta_val})
    res = res.evalf()
    return res

def evaluate_spherical_complex(key, unit_vec):
    theta_val, phi_val = to_spherical(R=unit_vec)[1:]
    sh_func = first_center_complex[key][0]
    res = sh_func.subs({phi: phi_val, theta1: theta_val})
    res = res.evalf()
    return res

def check_rotation_complex():
    unit_vec1 = np.random.rand(3)
    # unit_vec1 = np.array([0,0,1])
    random_phi = np.random.uniform(0, 2*np.pi, size=1)[0]
    # random_phi = 0
    random_theta = np.random.uniform(0, np.pi, size=1)[0]
    # random_theta = 0
    Ry = np.array([[np.cos(random_theta), 0, np.sin(random_theta)], [0,1,0], [-np.sin(random_theta), 0, np.cos(random_theta)]])
    Rz = np.array([[np.cos(random_phi), - np.sin(random_phi), 0],[np.sin(random_phi), np.cos(random_phi), 0],[0,0,1]])
    unit_vec2 = Ry @ Rz @ unit_vec1
    theta1_val, phi1_val = to_spherical(unit_vec1)[1:]
    func_vec = np.zeros((9,), 'complex')
    for i, (key, item) in enumerate(first_center_complex.items()):
        func = item[0]
        func_val = func.subs({phi:phi1_val, theta1: theta1_val})
        func_val = func_val.evalf()
        func_vec[i] = func_val
    D_func = sp.lambdify((PHI, THETA, GAMMA), Wigner_D_complex(euler_gamma=GAMMA, euler_phi=PHI, euler_theta=THETA), 'numpy')
    D = D_func(-random_phi, -random_theta, 0)
    print(D)
    result_vec = D @ func_vec
    val1_vec = np.zeros((9,), 'complex')
    for i, (key, item) in enumerate(first_center_complex.items()):
        val1 = evaluate_spherical_complex(key=key, unit_vec=unit_vec2)
        val1_vec[i] = val1
    print(val1_vec)
    print(result_vec)
    print(np.isclose(val1_vec, result_vec))


def check_rotation():
    unit_vec1 = np.random.rand(3)
    # unit_vec1 = np.array([0,0,1])
    random_phi = np.random.uniform(0, 2*np.pi, size=1)[0]
    random_theta = np.random.uniform(0, np.pi, size=1)[0]
    Ry = np.array([[np.cos(random_theta), 0, np.sin(random_theta)], [0,1,0], [-np.sin(random_theta), 0, np.cos(random_theta)]])
    Rz = np.array([[np.cos(random_phi), - np.sin(random_phi), 0],[np.sin(random_phi), np.cos(random_phi), 0],[0,0,1]])
    unit_vec2 = Ry @ Rz @ unit_vec1
    theta1_val, phi1_val = to_spherical(unit_vec1)[1:]
    func_vec = np.zeros((9,))
    for i, (key, item) in enumerate(first_center.items()):
        func = item[0]
        func_val = func.subs({phi:phi1_val, theta1: theta1_val})
        func_val = func_val.evalf()
        func_vec[i] = func_val
    D_sym = Wigner_D_real(euler_phi=PHI, euler_gamma=GAMMA, euler_theta=THETA)
    D = sp.lambdify((PHI, GAMMA, THETA), D_sym, 'numpy')
    D = np.real(D(-random_phi, 0, -random_theta))
    print(D)
    result_vec = D @ func_vec
    val1_vec = np.zeros((9,))
    for i, (key, item) in enumerate(first_center.items()):
        val1 = evaluate_spherical(key=key, unit_vec=unit_vec2)
        val1_vec[i] = val1
    print(val1_vec)
    print(result_vec)
    print(np.isclose(val1_vec, result_vec))


def check_rotation_prod():
    unit_vec1 = np.random.rand(3)
    # unit_vec1 = np.array([0,0,1])
    random_phi = np.random.uniform(0, 2*np.pi, size=1)[0]
    random_theta = np.random.uniform(0, np.pi, size=1)[0]
    Ry = np.array([[np.cos(random_theta), 0, np.sin(random_theta)], [0,1,0], [-np.sin(random_theta), 0, np.cos(random_theta)]])
    Rz = np.array([[np.cos(random_phi), - np.sin(random_phi), 0],[np.sin(random_phi), np.cos(random_phi), 0],[0,0,1]])
    unit_vec2 = Ry @ Rz @ unit_vec1
    theta1_val, phi1_val = to_spherical(unit_vec1)[1:]
    func_vec = np.zeros((81,))
    count = 0
    for i, (key, item) in enumerate(first_center.items()):
        for j, (key2, item2) in enumerate(first_center.items()):
            
            func = item[0]
            func2 = item2[0]
            func_val = func.subs({phi:phi1_val, theta1: theta1_val})
            func_val = func_val.evalf()
            func_val2 = func2.subs({phi:phi1_val, theta1: theta1_val})
            func_val2 = func_val2.evalf()
            func_vec[count] = func_val * func_val2
            count += 1
    D_sym = Wigner_D_real(euler_phi=PHI, euler_gamma=GAMMA, euler_theta=THETA)
    D = sp.lambdify((PHI, THETA, GAMMA), D_sym, 'numpy')
    D = np.real(D(-random_phi, -random_theta, 0))
    D_tot = np.kron(D, D)
    result_vec = D_tot @ func_vec
    val1_vec = np.zeros((81,))
    count=0
    for i, (key, item) in enumerate(first_center.items()):
        for j, (key2, item2) in enumerate(first_center.items()):
            val1 = evaluate_spherical(key=key, unit_vec=unit_vec2)
            val2 = evaluate_spherical(key=key2, unit_vec=unit_vec2)
            val1_vec[count] = val1 * val2
            count += 1
    print(np.isclose(val1_vec, result_vec))

        
