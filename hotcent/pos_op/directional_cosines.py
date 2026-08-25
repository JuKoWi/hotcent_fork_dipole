import numpy as np
import sympy as sp
import sympy.tensor.array.expressions
import pickle
from hotcent.pos_op.integrals import first_center, phi, theta1, first_center_complex, operator
from hotcent.pos_op.slako_new import DFTBPLUS_SIMPLE, DICT_IDENTICAL

q, r, s = sp.symbols('l, m, n', real=True)
ALPHA, BETA, GAMMA = sp.symbols('alpha, beta, gamma', real=True)
P, Q = sp.symbols('P, Q', positive=True)


def d_mat_elem(l, m, n):
    """single element for small Wigner matrix d"""
    expr = 0
    k_min = max(0, n-m)
    k_max = min(l+n, l-m)
    for k in range(k_min, k_max+1):
        prefac = (-1)**(k-n+m) * sp.sqrt(sp.factorial(l+n) * sp.factorial(l-n) * sp.factorial(l+m) * sp.factorial(l-m)) / (sp.factorial(l+n-k) * sp.factorial(k) * sp.factorial(l-k-m) * sp.factorial(k-n+m))
        angle_part = ((P/sp.sqrt(2))**(2*l-2*k+n-m) * (-Q/sp.sqrt(2))**(2*k-n+m))
        expr += prefac * angle_part 
    return expr

def d_mat():
    """ 
    Assemble small-d Wigner matrix (rotation of spherical harmonics in Condon-Shortley convention around y-axis) up to f orbitals. 
    Returns a block-diagonal 16×16 symbolic matrix with j=0,1,2,3 blocks.
    """
    d = sp.zeros(16,16)
    row_start = 0

    for l in range(4):  # j = 0, 1, 2, 3
        size = 2*l + 1
        block = sp.zeros(size, size)
        for mi, m in enumerate(range(-l, l+1)):
            for ni, n in enumerate(range(-l, l+1)):
                block[mi, ni] = d_mat_elem(l=l, m=m, n=n)
        d[row_start:row_start+size, row_start:row_start+size] = block
        row_start += size
    return d

def z_rot_mat():
    """
    Diagonal matrix for rotation of Condon-Shortley spherical harmonics 
    around z-axis
    """
    Dz = sp.zeros(16,16)
    count = 0
    for l in range(4): # l=0,1,2,3
        for m in range(-l, l+1):
            # Dz[count, count] = sp.exp(-sp.I*m*alpha)
            Dz[count, count] = ((q + sp.I * r)/(P*Q))**m
            # phase = (q - sp.I*r)/(P*Q) if m >= 0 else (q + sp.I*r)/(P*Q)
            # Dz[count, count] = phase**abs(m)
            count += 1
    return Dz
        
def Wigner_D_complex():
    """
    rotation matrix for complex harmonics for rotation sequence
    Rz(gamma)Ry(theta)Rz(phi)
    """
    # Dz = z_rot_mat(alpha=euler_alpha)
    dy = d_mat()
    Dz2 = z_rot_mat() 
    total = dy * Dz2 
    return total

    
def Wigner_D_real(save=True):
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
    D_total = transform_to_real * Wigner_D_complex().T * transform_to_comp
    D_total = D_total.as_mutable()
    if save:
        with open("symbolic_D_matrix.pkl", "wb") as f:
            pickle.dump(D_total, f)
    D_total = D_total.subs(P**2, 1 + s)
    D_total = D_total.subs(Q**2, 1 - s)
    D_total = sp.cancel(sp.together(sp.expand(D_total)))
    D_total = sp.simplify(D_total)
    return D_total

def print_sk_rules():
    """
    Generate Python code for the explicit Slater Koster rules for H/S.
    Resulting function takes a (16,16) array from the Slater Koster files as input
    """
    D = Wigner_D_real()
    X = sp.tensor.array.expressions.ArraySymbol('X', (16, 16))

    rep, kept = {}, set()
    for num in DFTBPLUS_SIMPLE:
        i, j = divmod(num, 16)
        kept.add(num)
        for idx in DICT_IDENTICAL[num]:
            a, b = divmod(idx, 16)
            assert rep.get(X[a, b], X[i, j]) == X[i, j], f"conflicting rule for {idx}"
            rep[X[a, b]] = X[i, j]
            kept.add(idx)

    for num in set(range(256)) - kept:          # everything not covered is zero
        a, b = divmod(num, 16)
        rep[X[a, b]] = sp.Integer(0)

    A = lambda a, b: rep.get(X[a, b], X[a, b])
    with open("skrules.py", 'w') as handle:
        print("import numpy as np", file=handle)
        print("def matrix_elements(X, cb, cg, sb, sg):", file=handle)
        print("\tC = np.zeros((16,16))", file=handle)
        dim = 4
        for i in range(dim):
            for j in range(dim):
                e = 0
                for a in range(dim):
                    for b in range(dim):
                        e += A(a,b) * D[i,a] * D[j,b]
                e = e.subs(P**2, s+1)
                e = e.subs(Q**2, 1-s)
                e = sp.simplify(e)
                if e != 0:
                    print(f"\tC[{i},{j}] = {e}", file=handle)
                    print(f"\tC[{i},{j}] = {e}")
        print("\treturn C", file=handle)

if __name__=="__main__":
    # print(Wigner_D_real())
    print_sk_rules()