from sympy import *
CODE =False 
init_printing(use_unicode=True)

phi, theta1 = symbols('phi theta1')

"""centered at first atom"""
s_1 = 1/(2 * sqrt(pi))

py_1 = 1/2 * sqrt(3/pi) * sin(theta1) * sin(phi)
px_1 = 1/2 * sqrt(3/pi) * cos(phi) * sin(theta1) 
pz_1 = 1/2 * sqrt(3/pi) * cos(theta1) 

dxy_1 = 1/4 * sqrt(15/pi) * sin(theta1)**2 * sin(2*phi)
dyz_1 = 1/2 * sqrt(15/pi) * cos(theta1) * sin(theta1) * sin(phi)
dz2_1 = 1/8 * sqrt(5/pi) * (1 + 3 * cos(2 * theta1))
dzx_1 = 1/2 * sqrt(15/pi) * cos(theta1) * cos(phi) * sin(theta1)
dx2y2_1 = 1/4 * sqrt(15/pi) * cos(2 * phi) * sin(theta1)**2


phi, theta2 = symbols('phi theta2')

"""centered at second atom"""
s_2 = 1/(2 * sqrt(pi))

py_2 = 1/2 * sqrt(3/pi) * sin(theta2) * sin(phi)
px_2 = 1/2 * sqrt(3/pi) * cos(phi) * sin(theta2) 
pz_2 = 1/2 * sqrt(3/pi) * cos(theta2) 

dxy_2 = 1/4 * sqrt(15/pi) * sin(theta2)**2 * sin(2*phi)
dyz_2 = 1/2 * sqrt(15/pi) * cos(theta2) * sin(theta2) * sin(phi)
dz2_2 = 1/8 * sqrt(5/pi) * (1 + 3 * cos(2 * theta2))
dzx_2 = 1/2 * sqrt(15/pi) * cos(theta2) * cos(phi) * sin(theta2)
dx2y2_2 = 1/4 * sqrt(15/pi) * cos(2 * phi) * sin(theta2)**2

first_center = {
    "ss": (s_1, 0,0),
    "py": (py_1, 1,-1),
    "pz": (pz_1, 1,0),
    "px": (px_1,1,1),
    "d1": (dxy_1,2,-2),
    "d2": (dyz_1,2,-1),
    "d3": (dz2_1,2,0),
    "d4": (dzx_1,2,1),
    "d5": (dx2y2_1,2,2),
}

second_center = {
    "ss": (s_2,0,0),
    "py": (py_2,1,-1),
    "pz": (pz_2,1,0),
    "px": (px_2,1,1),
    "d1": (dxy_2,2,-2),
    "d2": (dyz_2,2,-1),
    "d3": (dz2_2,2,0),
    "d4": (dzx_2,2,1),
    "d5": (dx2y2_2,2,2),
}

operator = {
    "py": (py_1,1,-1),
    "pz": (pz_1,1,0),
    "px": (px_1,1,1),
}

def get_index_list():
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
    return identifier, nonzeros 

    


if __name__ == "__main__":
    counter = 0
    unique_integrals = []
    zero_indices = []
    identifier = []
    for name_i, i in first_center.items():
        for name_j, j in operator.items():
            for name_k, k in second_center.items():
                tuple = (counter, i[1], i[2], j[1], j[2], k[1], k[2])
                integral = integrate(i[0] * j[0] * k[0], (phi, 0, 2 * pi))
                counter += 1
                if integral != 0:
                    if CODE:
                        txt = f"elif sk_label == {tuple}:\n\treturn {integral}"
                        txt = txt.replace('sqrt', 'np.sqrt')
                        txt = txt.replace('sin(theta2)', 's2')
                        txt = txt.replace('cos(theta2)', 'c2')
                        txt = txt.replace('cos(theta1)', 'c1')
                        txt = txt.replace('sin(theta1)', 's1')
                        txt = txt.replace('cos(2*theta2)', '(2*c2**2-1)')
                        txt = txt.replace('cos(2*theta1)', '(2*c1**2-1)')
                        txt = txt.replace('pi', 'np.pi')
                        print(txt) 
                    else:
                        print(f"{tuple},")

