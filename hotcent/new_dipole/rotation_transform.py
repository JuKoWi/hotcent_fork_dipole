import numpy as np
import sympy as sp
import pickle
from hotcent.new_dipole.integrals import first_center, phi, theta1, first_center_complex, operator


PHI = sp.symbols('phi')
THETA = sp.symbols('theta')
GAMMA = sp.symbols('gamma')

def d_mat_elem(l, m, n, theta):
    """single element for small Wigner matrix d"""
    expr = 0
    k_min = max(0, m-n)
    k_max = min(l+m, l-n)
    for k in range(k_min, k_max+1):
        prefac = (-1)**(k-m+n) * sp.sqrt(sp.factorial(l+m) * sp.factorial(l-m) * sp.factorial(l+n) * sp.factorial(l-n)) / (sp.factorial(l+m-k) * sp.factorial(k) * sp.factorial(l-k-n) * sp.factorial(k-m+n))
        angle_part = (sp.cos(theta/2)**(2*l-2*k+m-n) * sp.sin(theta / 2)**(2*k-m+n))
        expr += prefac * angle_part 
    return expr


def d_mat(theta):
    """ 
    Assemble small-d Wigner matrix (rotation of spherical harmonics in Condon-Shortley convention around y-axis) up to f orbitals. 
    Returns a block-diagonal 16×16 symbolic matrix with j=0,1,2,3 blocks.
    """
    d = sp.zeros(16,16)
    row_start = 0

    for j in range(4):  # j = 0, 1, 2, 3
        size = 2*j + 1
        block = sp.zeros(size, size)
        for mi, m in enumerate(range(-j, j+1)):
            for ni, n in enumerate(range(-j, j+1)):
                block[mi, ni] = d_mat_elem(j, m, n, theta)
        d[row_start:row_start+size, row_start:row_start+size] = block
        row_start += size
    return d

def z_rot_mat(phi):
    """
    Diagonal matrix for rotation of Condon-Shortley spherical harmonics 
    around z-axis
    """
    Dz = sp.zeros(16,16)
    count = 0
    for l in range(4): # l=0,1,2,3
        for m in range(-l, l+1):
            Dz[count, count] = sp.exp(-sp.I*m*phi)
            count += 1
    # print('Dz-matrix')
    # print(Dz)
    return Dz
        
def Wigner_D_complex(euler_phi, euler_theta, euler_gamma):
    """
    rotation matrix for complex harmonics for rotation sequence
    Rz(phi)Ry(theta)Rz(gamma)
    """
    Dz = z_rot_mat(phi=euler_phi)
    dy = d_mat(theta=euler_theta)
    Dz2 = z_rot_mat(phi=euler_gamma) #last rotation around z-axis
    total = Dz* dy * Dz2
    return total
    
def Wigner_D_real(euler_phi, euler_theta, euler_gamma):
    """
    Rotation matrix for real spherical harmonics for rotation sequence
    Rz(phi)Ry(theta)Rz(gamma)
    """
    transform_to_real = sp.zeros(16, 16)

    #s
    transform_to_real[0,0] = 1

    #p
    transform_to_real[1,1] = sp.I / sp.sqrt(2)
    transform_to_real[1,3] = sp.I / sp.sqrt(2)
    transform_to_real[2,2] = 1
    transform_to_real[3,1] = 1/sp.sqrt(2)
    transform_to_real[3,3] = -1/sp.sqrt(2)


    #d
    transform_to_real[4,4] =  sp.I / sp.sqrt(2)
    transform_to_real[4,8] = -sp.I / sp.sqrt(2)
    transform_to_real[5,5] = sp.I / sp.sqrt(2)
    transform_to_real[5,7] = sp.I / sp.sqrt(2)
    transform_to_real[6,6] = 1
    transform_to_real[7,5] = 1/sp.sqrt(2)
    transform_to_real[7,7] = -1/sp.sqrt(2)
    transform_to_real[8,4] = 1 / sp.sqrt(2)
    transform_to_real[8,8] = 1 / sp.sqrt(2)

    #f
    transform_to_real[9,9] = sp.I / sp.sqrt(2)
    transform_to_real[9, 15] = sp.I/sp.sqrt(2)
    transform_to_real[10,10] = sp.I/sp.sqrt(2)
    transform_to_real[10, 14] = -sp.I/sp.sqrt(2)
    transform_to_real[11, 11] = sp.I/sp.sqrt(2)
    transform_to_real[11, 13] = sp.I/sp.sqrt(2)
    transform_to_real[12, 12] = 1
    transform_to_real[13, 11] = 1/sp.sqrt(2)
    transform_to_real[13, 13] = -1/sp.sqrt(2)
    transform_to_real[14, 10] = 1/ sp.sqrt(2)
    transform_to_real[14, 14] = 1/sp.sqrt(2)
    transform_to_real[15, 9] = 1/sp.sqrt(2)
    transform_to_real[15, 15] = -1/sp.sqrt(2)

    transform_to_comp = transform_to_real.H
    D_total = transform_to_real * Wigner_D_complex(euler_phi=euler_phi, euler_theta=euler_theta, euler_gamma=euler_gamma) * transform_to_comp
    with open("symbolic_D_matrix.pkl", "wb") as f:
        pickle.dump(D_total, f)
    return D_total


"""some tests"""
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
dz2_1 = sp.sqrt(5 / (16 * sp.pi)) * (3*z**2 -r**2)/r**2

f1 = 1/4 * sp.sqrt(35/(2*sp.pi)) * y*(3*x**2 - y**2)/r**3
f2 = 1/2 * sp.sqrt(105/sp.pi) * x*y*z/r**3
f3 = 1/4 * sp.sqrt(21/(2*sp.pi)) * y * (5*z**2 - r**2)/r**3
f4 = 1/4 * sp.sqrt(7/sp.pi) * (5*z**3 - 3* z * r**2)/r**3
f5 = 1/4 * sp.sqrt(21/(2*sp.pi)) * x * (5*z**2 - r**2)/r**3
f6 = 1/4 * sp.sqrt(105/sp.pi) * (x**2 - y**2) * z /r**3
f7 = 1/4 * sp.sqrt(35/(2*sp.pi)) * x * (x**2 - 3* y**2) /r**3

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
    "f1": (f1, 3, -3),
    "f2": (f2, 3, -2),
    "f3": (f3, 3, -1),
    "f4": (f4, 3, 0),
    "f5": (f5, 3, 1),
    "f6": (f6, 3, 2),
    "f7": (f7, 3, 3)
}

def to_spherical(R):
    """spherical coordinates of vector"""
    r = np.sqrt(np.sum(R**2))
    theta = np.arccos(R[2] / r)
    phi = np.arctan2(R[1], R[0])
    return np.array([r, theta, phi])

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

def evaluate_spherical_operator(key, unit_vec):
    theta_val, phi_val = to_spherical(R=unit_vec)[1:]
    sh_func = operator[key][0]
    res = sh_func.subs({phi: phi_val, theta1: theta_val})
    res = res.evalf()
    return res

def check_rotation_complex():
    unit_vec1 = np.random.rand(3)
    random_phi = np.random.uniform(0, 2*np.pi, size=1)[0]
    random_phi = 0
    random_theta = np.random.uniform(0, np.pi, size=1)[0]
    random_theta = 0
    Ry = np.array([[np.cos(random_theta), 0, np.sin(random_theta)], [0,1,0], [-np.sin(random_theta), 0, np.cos(random_theta)]])
    Rz = np.array([[np.cos(random_phi), - np.sin(random_phi), 0],[np.sin(random_phi), np.cos(random_phi), 0],[0,0,1]])
    unit_vec2 = Rz @ Ry @ unit_vec1
    theta1_val, phi1_val = to_spherical(unit_vec1)[1:]
    func_vec = np.zeros((16,), 'complex')
    for i, (key, item) in enumerate(first_center_complex.items()):
        func = item[0]
        func_val = func.subs({phi:phi1_val, theta1: theta1_val})
        func_val = func_val.evalf()
        func_vec[i] = func_val
    D_func = sp.lambdify((PHI, THETA, GAMMA), Wigner_D_complex(euler_gamma=GAMMA, euler_phi=PHI, euler_theta=THETA), 'numpy')
    D = D_func(-random_phi, -random_theta, 0)
    result_vec = D @ func_vec
    val1_vec = np.zeros((16,), 'complex')
    for i, (key, item) in enumerate(first_center_complex.items()):
        val1 = evaluate_spherical_complex(key=key, unit_vec=unit_vec2)
        val1_vec[i] = val1
    print(np.isclose(val1_vec, result_vec))


def check_rotation():
    unit_vec1 = np.random.rand(3)
    random_phi = np.random.uniform(0, 2*np.pi, size=1)[0]
    # random_phi = 0
    random_theta = np.random.uniform(0, np.pi, size=1)[0]
    # random_theta = 0
    Ry = np.array([[np.cos(random_theta), 0, np.sin(random_theta)], [0,1,0], [-np.sin(random_theta), 0, np.cos(random_theta)]])
    Rz = np.array([[np.cos(random_phi), - np.sin(random_phi), 0],[np.sin(random_phi), np.cos(random_phi), 0],[0,0,1]])
    unit_vec2 = Rz @ Ry @ unit_vec1
    theta1_val, phi1_val = to_spherical(unit_vec1)[1:]
    func_vec = np.zeros((16,))
    for i, (key, item) in enumerate(first_center.items()):
        func = item[0]
        # print(func)
        func_val = func.subs({phi:phi1_val, theta1: theta1_val})
        func_val = func_val.evalf()
        func_vec[i] = func_val
    D_sym = Wigner_D_real(euler_phi=PHI, euler_gamma=GAMMA, euler_theta=THETA)
    D = sp.lambdify((PHI, GAMMA, THETA), D_sym, 'numpy')
    D = np.real(D(-random_phi, 0, -random_theta))
    result_vec = D @ func_vec
    val1_vec = np.zeros((16,))
    for i, (key, item) in enumerate(first_center.items()):
        val1 = evaluate_spherical(key=key, unit_vec=unit_vec2)
        val1_vec[i] = val1
    print(np.isclose(val1_vec, result_vec))

def check_rotation_prod():
    unit_vec1 = np.random.rand(3)
    random_phi = np.random.uniform(0, 2*np.pi, size=1)[0]
    random_theta = np.random.uniform(0, np.pi, size=1)[0]
    Ry = np.array([[np.cos(random_theta), 0, np.sin(random_theta)], [0,1,0], [-np.sin(random_theta), 0, np.cos(random_theta)]])
    Rz = np.array([[np.cos(random_phi), - np.sin(random_phi), 0],[np.sin(random_phi), np.cos(random_phi), 0],[0,0,1]])
    unit_vec2 = Rz @ Ry @ unit_vec1
    theta1_val, phi1_val = to_spherical(unit_vec1)[1:]
    func_vec = np.zeros((256,))
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
    val1_vec = np.zeros((256,))
    count=0
    for i, (key, item) in enumerate(first_center.items()):
        for j, (key2, item2) in enumerate(first_center.items()):
            val1 = evaluate_spherical(key=key, unit_vec=unit_vec2)
            val2 = evaluate_spherical(key=key2, unit_vec=unit_vec2)
            val1_vec[count] = val1 * val2
            count += 1
    print(np.isclose(val1_vec, result_vec))

def check_rot_triple():
    unit_vec1 = np.random.rand(3)
    random_phi = np.random.uniform(0, 2*np.pi, size=1)[0]
    random_theta = np.random.uniform(0, np.pi, size=1)[0]
    Ry = np.array([[np.cos(random_theta), 0, np.sin(random_theta)], [0,1,0], [-np.sin(random_theta), 0, np.cos(random_theta)]])
    Rz = np.array([[np.cos(random_phi), - np.sin(random_phi), 0],[np.sin(random_phi), np.cos(random_phi), 0],[0,0,1]])
    unit_vec2 = Rz @ Ry @ unit_vec1
    theta1_val, phi1_val = to_spherical(unit_vec1)[1:]
    func_vec = np.zeros((768,))
    count = 0
    for i, (key, item) in enumerate(first_center.items()):
        for k, (key3, item3) in enumerate(operator.items()):
            for j, (key2, item2) in enumerate(first_center.items()):
                func = item[0]
                func2 = item2[0]
                op = item3[0]
                func_val = func.subs({phi:phi1_val, theta1: theta1_val})
                func_val = func_val.evalf()
                func_val2 = func2.subs({phi:phi1_val, theta1: theta1_val})
                func_val2 = func_val2.evalf()
                op_val = op.subs({phi:phi1_val, theta1: theta1_val})
                op_val = op_val.evalf()
                func_vec[count] = func_val * func_val2 * op_val
                count += 1
    D_sym = Wigner_D_real(euler_phi=PHI, euler_gamma=GAMMA, euler_theta=THETA)
    D = sp.lambdify((PHI, THETA, GAMMA), D_sym, 'numpy')
    D = np.real(D(-random_phi, -random_theta, 0))
    D_op = D[1:4, 1:4]
    D_tot = np.kron(D, np.kron(D_op, D))
    result_vec = D_tot @ func_vec
    val1_vec = np.zeros((768,))
    count=0
    for i, (key, item) in enumerate(first_center.items()):
        for k, (key3, item3) in enumerate(operator.items()):
            for j, (key2, item2) in enumerate(first_center.items()):
                val1 = evaluate_spherical(key=key, unit_vec=unit_vec2)
                val2 = evaluate_spherical(key=key2, unit_vec=unit_vec2)
                val3 = evaluate_spherical_operator(key=key3, unit_vec=unit_vec2) 
                val1_vec[count] = val1 * val2 * val3
                count += 1
    print(np.isclose(val1_vec, result_vec))
    
def check_harmonics_equal():
    unit_vec = np.random.normal(size=3)
    unit_vec = unit_vec/np.linalg.norm(unit_vec)
    theta1_val, phi1_val = to_spherical(unit_vec)[1:]
    complex_sh = np.zeros((16,))
    real_sh = np.zeros((16,))
    for i, (key, item) in enumerate(first_center.items()):
        func = item[0]
        func = func.subs({phi: phi1_val, theta1: theta1_val})
        func = func.evalf()
        complex_sh[i] = func
    for i, (key, item) in enumerate(first_center.items()):
        func = item[0]
        func = func.subs({phi: phi1_val, theta1: theta1_val})
        func = func.evalf()
        real_sh[i] = func
    print(np.allclose(complex_sh, real_sh))

def check_vec_rotation():
    unit_vec = np.random.normal(size=3)
    unit_vec = unit_vec/np.linalg.norm(unit_vec)
    theta_val, phi_val= to_spherical(unit_vec)[1:]
    theta1_val = -theta_val 
    phi1_val = -phi_val
    Ry = np.array([[np.cos(theta1_val), 0, np.sin(theta1_val)], [0,1,0], [-np.sin(theta1_val), 0, np.cos(theta1_val)]])
    Rz = np.array([[np.cos(phi1_val), - np.sin(phi1_val), 0],[np.sin(phi1_val), np.cos(phi1_val), 0],[0,0,1]])
    z_vector = Ry @ Rz @ unit_vec
    print(np.allclose(z_vector, [0,0,1]))

    
    

        
