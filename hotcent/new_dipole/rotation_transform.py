import numpy as np
import sympy as sp
import pickle

phi = sp.symbols('phi')
theta = sp.symbols('theta')
gamma = sp.symbols('gamma')

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


def d_mat(phi):
    """small-d Wigner matrix for the angle beta (up to d orbitals). 
    Returns a block-diagonal 9×9 symbolic matrix with j=0,1,2 blocks."""
    d = sp.zeros(9, 9)
    row_start = 0

    for j in range(3):  # j = 0, 1, 2
        size = 2*j + 1
        block = sp.zeros(size, size)
        for mi, m in enumerate(range(-j, j+1)):
            for ni, n in enumerate(range(-j, j+1)):
                block[mi, ni] = d_mat_elem(j, m, n, phi)
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
    return Dz
        
def Wigner_D_complex(euler_phi, euler_theta, euler_gamma):
    Dz = z_rot_mat(euler_phi)
    dy = d_mat(euler_theta)
    Dz2 = z_rot_mat(euler_gamma) #last rotation around z-axis
    return dy * Dz * Dz2
    
def Wigner_D_real(euler_phi, euler_theta, euler_gamma):
    transform_to_real = sp.zeros(9,9)

    #s
    transform_to_real[0,0] = 1

    #p
    transform_to_real[1,1] = sp.I / sp.sqrt(2)
    transform_to_real[1,3] = sp.I / sp.sqrt(2)
    transform_to_real[3,1] = 1/sp.sqrt(2)
    transform_to_real[3,3] = -1/sp.sqrt(2)
    transform_to_real[2,2] = 1

    #d
    transform_to_real[4,4] = sp.I / sp.sqrt(2)
    transform_to_real[4,8] = -sp.I / sp.sqrt(2)
    transform_to_real[8,4] = 1/sp.sqrt(2)
    transform_to_real[8,8] = 1/sp.sqrt(2)
    transform_to_real[5,5] = sp.I /sp.sqrt(2)
    transform_to_real[5,7] = sp.I / sp.sqrt(2)
    transform_to_real[7,5] = 1/sp.sqrt(2)
    transform_to_real[7,7] = -1/sp.sqrt(2)
    transform_to_real[6,6] = 1

    transform_to_comp = transform_to_real.inv()
    D_total = transform_to_real * Wigner_D_complex(euler_phi=euler_phi, euler_theta=euler_theta, euler_gamma=euler_gamma) * transform_to_comp

    with open("symbolic_D_matrix.pkl", "wb") as f:
        pickle.dump(D_total, f)





 
