import numpy as np
import sympy as sp

# check euler angles

# define vector
R = np.array([0,0,-1])

# calculate angles
r = np.sqrt(R[0]**2 + R[1]**2 + R[2]**2)
theta = np.arccos(R[2]/r)
phi = np.arctan2(R[1], R[0])

# define matrix
def matrix(theta, phi):
    mat1 = np.array([[np.cos(-phi), -np.sin(-phi), 0], [np.sin(-phi), np.cos(-phi), 0], [0,0,1]])
    mat2 = np.array([[np.cos(-theta), 0, np.sin(-theta)], [0,1,0], [-np.sin(-theta), 0, np.cos(-theta)]])
    mat = mat2 @ mat1
    return mat 

beta = sp.symbols('beta')
theta = sp.symbols('theta')

def to_spherical(R):
    r = np.sqrt(np.sum(R**2))
    theta = np.arccos(R[2] / r)
    phi = np.arctan2(R[1], R[0])
    return np.array([r, theta, phi])

def d_mat_elem(l, m, n, beta):
    expr = 0
    k_min = max(0, m-n)
    k_max = min(l+m, l-n)
    for k in range(k_min, k_max+1):
        prefac = (-1)**(k-m+n) * sp.sqrt(sp.factorial(l+m) * sp.factorial(l-m) * sp.factorial(l+n) * sp.factorial(l-n)) / (sp.factorial(l+m-k) * sp.factorial(k) * sp.factorial(l-k-n) * sp.factorial(k-m+n))
        angle_part = (sp.cos(beta/2)**(2*l-2*k+m-n) * sp.sin(beta / 2)**(2*k-m+n))
        expr += prefac * angle_part 
    return sp.trigsimp(expr)


def d_mat(beta):
    """small-d Wigner matrix for the angle beta (up to d orbitals). 
    Returns a block-diagonal 9×9 symbolic matrix with j=0,1,2 blocks."""
    d = sp.zeros(9, 9)
    row_start = 0

    for j in range(3):  # j = 0, 1, 2
        size = 2*j + 1
        block = sp.zeros(size, size)
        for mi, m in enumerate(range(-j, j+1)):
            for ni, n in enumerate(range(-j, j+1)):
                block[mi, ni] = d_mat_elem(j, m, n, beta)
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
        
def Wigner_D_complex(euler_phi, euler_theta):
    Dz = z_rot_mat(euler_phi)
    dy = d_mat(euler_theta)
    return dy * Dz
    
def Wigner_D_real(euler_phi, euler_theta):
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
    D_total = transform_to_real * Wigner_D_complex(euler_phi=euler_phi, euler_theta=euler_theta) * transform_to_comp
    return D_total

def phi3_transform(R):
   pass 



"""get euler angles from internuclear vector"""
"""get list of 0 integrals"""
"""write map from string names to quantum number"""
def read_sk_table(R):
    """for an internuclear distance choose best column from sk-table
    bring it into order consistent with rotation matrix
    """
    pass

def calculate_dipole_element_set(n_basis):
    """calculate dipole vector for all different combination of atomic orbitals for an atom pair"""
    dipoles = np.zeros((n_basis, n_basis, 3))
    vec_ints = 

    pass



# d_mat(0)

# print(sp.simplify(d_mat_elem(1,1,1, beta=beta)- 1/2 * (1 +sp.cos(beta))) == 0)
# print(sp.simplify(d_mat_elem(1,1,0, beta=beta) - 1/sp.sqrt(2) * sp.sin(beta)) == 0)
# print(sp.simplify(d_mat_elem(1,1,-1, beta=beta)-1/2 * (1 -sp.cos(beta))) == 0)
# print(sp.simplify(d_mat_elem(1,0,0, beta=beta)-sp.cos(beta)) == 0)

# print(d_mat_elem(2,2,2, beta=beta).equals( +1/4 * (1 + sp.cos(beta))**2)) 
# print(d_mat_elem(2,2,1, beta=beta).equals( +1/2 * sp.sin(beta) *(1+sp.cos(beta))))
# print(d_mat_elem(2,2,0, beta=beta).equals( -sp.sqrt(3/8)* sp.sin(beta)**2))
# print(d_mat_elem(2,2,-1,beta=beta).equals( - 1/2 * sp.sin(beta) * (1-sp.cos(beta))))
# print(d_mat_elem(2,2,-2,beta=beta).equals( - 1/4 * (1-sp.cos(beta))**2)) 
# print(d_mat_elem(2,1,1, beta=beta).equals( - 1/2 * (2*sp.cos(beta)**2 -sp.cos(beta) -1)))
# print(d_mat_elem(2,1,0, beta=beta).equals( - sp.sqrt(3/8) * sp.sin(2*beta)))
# print(d_mat_elem(2,1,-1,beta=beta).equals( - 1/2 * (-2* sp.cos(beta)**2 + sp.cos(beta) + 1)))
# print(d_mat_elem(2,0,0, beta=beta).equals( - 1/2 * (3*sp.cos(beta)**2 -1))) 
 
# print(d_mat_elem(2,2,2, beta=beta).equals( sp.trigsimp(1/4 * (1 + sp.cos(beta))**2)))
# print(d_mat_elem(2,2,1, beta=beta).equals( sp.trigsimp(1/2 * sp.sin(beta) *(1+sp.cos(beta)))))
# print(d_mat_elem(2,2,0, beta=beta).equals( sp.trigsimp(sp.sqrt(3/8)* sp.sin(beta)**2)))
# print(d_mat_elem(2,2,-1,beta=beta).equals( sp.trigsimp(1/2 * sp.sin(beta) * (1-sp.cos(beta)))))
# print(d_mat_elem(2,2,-2,beta=beta).equals( sp.trigsimp(1/4 * (1-sp.cos(beta))**2)) )
# print(d_mat_elem(2,1,1, beta=beta).equals( sp.trigsimp(1/2 * (2*sp.cos(beta)**2 -sp.cos(beta) -1))))
# print(d_mat_elem(2,1,0, beta=beta).equals( sp.trigsimp(sp.sqrt(3/8) * sp.sin(2*beta))))
# print(d_mat_elem(2,1,-1,beta=beta).equals( sp.trigsimp(1/2 * (-2* sp.cos(beta)**2 + sp.cos(beta) + 1))))
# print(d_mat_elem(2,0,0, beta=beta).equals( sp.trigsimp(1/2 * (3*sp.cos(beta)**2 -1))) )
 
