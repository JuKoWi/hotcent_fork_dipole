from sympy import *
import time
import pickle
CODE = True #print python code to file
init_printing(use_unicode=True)

phi = symbols('phi')
theta1 = symbols('theta1')
theta2 = symbols('theta2')
s1, s2, c1, c2 = symbols('s1 s2 c1 c2')


"""complex, first atom"""
s_1_comp_1 = 1/(2 * sqrt(pi))

p1_comp_1 = 1/2 * sqrt(3/(2*pi)) * exp(- I * phi) * sin(theta1)
p2_comp_1 = 1/2 * sqrt(3/pi) * cos(theta1) 
p3_comp_1 = -1/2 * sqrt(3/(2*pi)) * exp(I * phi) * sin(theta1)

d1_comp_1 = 1/4 * sqrt(15/(2*pi)) * exp(-2* I * phi) * sin(theta1)**2
d2_comp_1 = 1/2 * sqrt(15/(2*pi)) * exp(-I * phi) * sin(theta1) * cos(theta1)
d3_comp_1 = 1/4 * sqrt(5/pi) * (3*cos(theta1)**2-1)
d4_comp_1 = -1/2 * sqrt(15/(2*pi)) * exp(I * phi) * sin(theta1) * cos(theta1)
d5_comp_1 = 1/4 * sqrt(15/(2*pi)) * exp(2* I * phi) * sin(theta1)**2

f1_comp_1 = 1/8 * sqrt(35/pi) * exp(-3*I*phi) * sin(theta1)**3
f2_comp_1 = 1/4 * sqrt(105/(2*pi)) * exp(-2 * I* phi) * sin(theta1)**2 * cos(theta1)
f3_comp_1 = 1/8 * sqrt(21/pi) * exp(-I * phi) * sin(theta1) * (5* cos(theta1)**2 -1)
f4_comp_1 = 1/4 * sqrt(7/pi) * (5*cos(theta1)**3 - 3* cos(theta1))
f5_comp_1 = -1/8 * sqrt(21/pi) * exp(I * phi) * sin(theta1) * (5*cos(theta1)**2 -1)
f6_comp_1 = 1/4 * sqrt(105/(2*pi)) * exp(2*I*phi) * sin(theta1)**2 * cos(theta1)
f7_comp_1 = -1/8 * sqrt(35/pi) * exp(3* I * phi) * sin(theta1)**3


"""complex, second atom"""
s_2_comp_2 = 1/(2 * sqrt(pi))

p1_comp_2 = 1/2 * sqrt(3/(2*pi)) * exp(- I * phi) * sin(theta2)
p2_comp_2 = 1/2 * sqrt(3/pi) * cos(theta2) 
p3_comp_2 = -1/2 * sqrt(3/(2*pi)) * exp(I * phi) * sin(theta2)

d1_comp_2 = 1/4 * sqrt(15/(2*pi)) * exp(-2* I * phi) * sin(theta2)**2
d2_comp_2 = 1/2 * sqrt(15/(2*pi)) * exp(-I * phi) * sin(theta2) * cos(theta2)
d3_comp_2 = 1/4 * sqrt(5/pi) * (3*cos(theta2)**2-1)
d4_comp_2 = -1/2 * sqrt(15/(2*pi)) * exp(I * phi) * sin(theta2) * cos(theta2)
d5_comp_2 = 1/4 * sqrt(15/(2*pi)) * exp(2* I * phi) * sin(theta2)**2

f1_comp_2 = 1/8 * sqrt(35/pi) * exp(-3*I*phi) * sin(theta2)**3
f2_comp_2 = 1/4 * sqrt(105/(2*pi)) * exp(-2 * I* phi) * sin(theta2)**2 * cos(theta2)
f3_comp_2 = 1/8 * sqrt(21/pi) * exp(-I * phi) * sin(theta2) * (5* cos(theta2)**2 -1)
f4_comp_2 = 1/4 * sqrt(7/pi) * (5*cos(theta2)**3 - 3* cos(theta2))
f5_comp_2 = -1/8 * sqrt(21/pi) * exp(I * phi) * sin(theta2) * (5*cos(theta2)**2 -1)
f6_comp_2 = 1/4 * sqrt(105/(2*pi)) * exp(2*I*phi) * sin(theta2)**2 * cos(theta2)
f7_comp_2 = -1/8 * sqrt(35/pi) * exp(3* I * phi) * sin(theta2)**3


"""real centered at first atom"""
ss_1 = s_1_comp_1

py_1 = 1/2 * sqrt(3/pi) * sin(theta1) * sin(phi)
pz_1 = 1/2 * sqrt(3/pi) * cos(theta1) 
px_1 = 1/2 * sqrt(3/pi) * cos(phi) * sin(theta1) 

dxy_1 = 1/4 * sqrt(15/pi) * sin(theta1)**2 * sin(2*phi)
dyz_1 = 1/2 * sqrt(15/pi) * cos(theta1) * sin(theta1) * sin(phi)
dz2_1 = 1/4 * sqrt(5/pi) * (3*cos(theta1)**2-1)
dzx_1 = 1/2 * sqrt(15/pi) * cos(theta1) * cos(phi) * sin(theta1)
dx2y2_1 = 1/4 * sqrt(15/pi) * cos(2 * phi) * sin(theta1)**2

f1_1 = simplify(I/sqrt(2) * (f1_comp_1 + f7_comp_1))
f2_1 = simplify(I/sqrt(2) * (f2_comp_1 - f6_comp_1))
f3_1 = simplify(I/sqrt(2) * (f3_comp_1 + f5_comp_1))
f4_1 = f4_comp_1
f5_1 = simplify(1/sqrt(2) * (f3_comp_1 - f5_comp_1))
f6_1 = simplify(1/sqrt(2) * (f2_comp_1 + f6_comp_1))
f7_1 = simplify(1/sqrt(2) * (f1_comp_1 - f7_comp_1))

"""real, centered at second atom"""
ss_2 = s_2_comp_2

py_2 = 1/2 * sqrt(3/pi) * sin(theta2) * sin(phi)
pz_2 = 1/2 * sqrt(3/pi) * cos(theta2) 
px_2 = 1/2 * sqrt(3/pi) * cos(phi) * sin(theta2) 

dxy_2 = 1/4 * sqrt(15/pi) * sin(theta2)**2 * sin(2*phi)
dyz_2 = 1/2 * sqrt(15/pi) * cos(theta2) * sin(theta2) * sin(phi)
dz2_2 = 1/4 * sqrt(5/pi) * (3*cos(theta2)**2-1)
dzx_2 = 1/2 * sqrt(15/pi) * cos(theta2) * cos(phi) * sin(theta2)
dx2y2_2 = 1/4 * sqrt(15/pi) * cos(2 * phi) * sin(theta2)**2

f1_2 = simplify(I/sqrt(2) * (f1_comp_2 + f7_comp_2))
f2_2 = simplify(I/sqrt(2) * (f2_comp_2 - f6_comp_2))
f3_2 = simplify(I/sqrt(2) * (f3_comp_2 + f5_comp_2))
f4_2 = f4_comp_2
f5_2 = simplify(1/sqrt(2) * (f3_comp_2 - f5_comp_2))
f6_2 = simplify(1/sqrt(2) * (f2_comp_2 + f6_comp_2))
f7_2 = simplify(1/sqrt(2) * (f1_comp_2 - f7_comp_2))

first_center = {
    "ss": (ss_1, 0,0),
    "py": (py_1, 1,-1),
    "pz": (pz_1, 1,0),
    "px": (px_1,1,1),
    "d1": (dxy_1,2,-2),
    "d2": (dyz_1,2,-1),
    "d3": (dz2_1,2,0),
    "d4": (dzx_1,2,1),
    "d5": (dx2y2_1,2,2),
    "f1": (f1_1, 3, -3),
    "f2": (f2_1, 3, -2),
    "f3": (f3_1, 3, -1),
    "f4": (f4_1, 3, 0),
    "f5": (f5_1, 3, 1),
    "f6": (f6_1, 3, 2),
    "f7": (f7_1, 3, 3)
}

first_center_complex = {
    "ss": (s_1_comp_1, 0,0),
    "py": (p1_comp_1, 1,-1),
    "pz": (p2_comp_1, 1,0),
    "px": (p3_comp_1,1,1),
    "d1": (d1_comp_1,2,-2),
    "d2": (d2_comp_1,2,-1),
    "d3": (d3_comp_1,2,0),
    "d4": (d4_comp_1,2,1),
    "d5": (d5_comp_1,2,2),
    "f1": (f1_comp_1, 3,-3),
    "f2": (f2_comp_1, 3,-2),
    "f3": (f3_comp_1, 3,-1),
    "f4": (f4_comp_1, 3,0),
    "f5": (f5_comp_1, 3,1),
    "f6": (f6_comp_1, 3,2),
    "f7": (f7_comp_1, 3,3)
}

second_center = {
    "ss": (ss_2,0,0),
    "py": (py_2,1,-1),
    "pz": (pz_2,1,0),
    "px": (px_2,1,1),
    "d1": (dxy_2,2,-2),
    "d2": (dyz_2,2,-1),
    "d3": (dz2_2,2,0),
    "d4": (dzx_2,2,1),
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
    "py": (py_1,1,-1),
    "pz": (pz_1,1,0),
    "px": (px_1,1,1),
}

def pick_quantum_number(dictionary, lm):
    """map from quantum numbers to respective (function, l,m) """
    for key, value in dictionary.items():
        if value[1] == lm[0] and value[2] == lm[1]:
            return value
    raise ValueError("Element missing: No spherical harmonic for this quantum number combination")

def get_index_list_dipole():
    """list of nonzero phi3 integrals"""
    count = 0
    identifier = []
    nonzeros = []
    for name_i, i in first_center.items():
        for name_j, j in operator.items():
            for name_k, k in second_center.items():
                integral = integrate(i[0] * j[0] * k[0], (phi, 0, 2 * pi))
                if integral != 0:
                    nonzeros.append(count)
                tuple = (count, i[1], i[2], j[1], j[2], k[1], k[2])
                identifier.append(tuple)
                count += 1
    with open("identifier_nonzeros_dipole.pkl", "wb") as f:
        pickle.dump(identifier, f)
        pickle.dump(nonzeros, f)
    return identifier, nonzeros 

def get_index_list_overlap():
    """list of nonzero phi2 integrals """
    count = 0
    identifier = []
    nonzeros = []
    for name_i, i in first_center.items():
        for name_k, k in second_center.items():
            integral = integrate(i[0] * k[0], (phi, 0, 2 * pi))
            if integral != 0:
                nonzeros.append(count)
            tuple = (count, i[1], i[2], k[1], k[2])
            identifier.append(tuple)
            count += 1
    with open("identifier_nonzeros_overlap.pkl", "wb") as f:
        pickle.dump(identifier, f)
        pickle.dump(nonzeros, f)
    return identifier, nonzeros 


def print_dipole_integrals():
    counter = 0
    f= open("phi3_expr.txt", 'w')
    print("INTEGRALS = {", file=f)
    time_start = time.time()
    for name_i, i in first_center.items():
        for name_j, j in operator.items():
            for name_k, k in second_center.items():
                tuple = (counter, i[1], i[2], j[1], j[2], k[1], k[2])
                integral = integrate(i[0] * j[0] * k[0], (phi, 0, 2 * pi))
                counter += 1
                if integral != 0:
                    if CODE:
                        txt = f"{integral}"
                        txt = txt.replace('sqrt', 'np.sqrt')
                        txt = txt.replace('sin(theta2)', 's2')
                        txt = txt.replace('cos(theta2)', 'c2')
                        txt = txt.replace('cos(theta1)', 'c1')
                        txt = txt.replace('sin(theta1)', 's1')
                        txt = txt.replace('pi', 'np.pi')
                        print(f'\t{tuple}: lambda c1, c2, s1, s2: '+ txt + ',', file=f)
    time_end = time.time()
    print('}', file=f)
    print(f"finished integrals in {time_end-time_start} s")

def print_overlap_integrals():
    """print python code for numerical phi2 integrals to file"""
    counter = 0
    f= open("phi2_expr.txt", 'w')
    print("INTEGRALS = {", file=f)
    time_start = time.time()
    for name_i, i in first_center.items():
            for name_k, k in second_center.items():
                tuple = (counter, i[1], i[2], k[1], k[2])
                integral = integrate(i[0] * k[0], (phi, 0, 2 * pi))
                counter += 1
                if integral != 0:
                    if CODE:
                        txt = f"{integral}"
                        txt = txt.replace('sqrt', 'np.sqrt')
                        txt = txt.replace('sin(theta2)', 's2')
                        txt = txt.replace('cos(theta2)', 'c2')
                        txt = txt.replace('cos(theta1)', 'c1')
                        txt = txt.replace('sin(theta1)', 's1')
                        txt = txt.replace('pi', 'np.pi')
                        print(f'\t{tuple}: lambda c1, c2, s1, s2: '+ txt + ',', file=f)
    time_end = time.time()
    print('}', file=f)
    print(f"finished integrals in {time_end-time_start} s")

def print_overlap_derivatives():
    """print python code for derivatives of phi2 integrals to file"""
    counter = 0
    f= open("deriv-phi2_expr.txt", 'w')
    print("INTEGRAL_DERIVATIVE = {", file=f)
    time_start = time.time()
    for name_i, i in first_center.items():
            for name_k, k in second_center.items():
                tuple = (counter, i[1], i[2], k[1], k[2])
                integral = integrate(i[0] * k[0], (phi, 0, 2 * pi))
                integral = integral.subs({sin(theta1): s1, sin(theta2): s2, cos(theta1): c1, cos(theta2): c2})
                ds1 = diff(integral, s1)
                ds2 = diff(integral, s2)
                dc1 = diff(integral, c1)
                dc2 = diff(integral, c2)
                counter += 1
                if integral != 0:
                    print(ds1)
                    print(ds2)
                    print(dc1)
                    print(dc2)
                    if CODE:
                        txt = f"[{dc1}, {dc2}, {ds1}, {ds2}],"
                        txt = txt.replace('sqrt', 'np.sqrt')
                        txt = txt.replace('pi', 'np.pi')
                        print(f'\t{tuple}: lambda c1, c2, s1, s2: '+ txt , file=f)
    time_end = time.time()
    print('}', file=f)
    print(f"finished integrals in {time_end-time_start} s")
    
    


if __name__ == "__main__":
    # print_dipole_integrals()
    # print_overlap_integrals()
    print_overlap_derivatives()

