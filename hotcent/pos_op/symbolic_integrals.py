import sympy as sym
import numpy as np
import time
from hotcent.pos_op.slako_new import DFTBPLUS_SIMPLE

CODE = True  # print python code to file
sym.init_printing(use_unicode=True)

phi = sym.symbols("phi")
theta1 = sym.symbols("theta1")
theta2 = sym.symbols("theta2")
s1, s2, c1, c2 = sym.symbols("s1 s2 c1 c2")


"""complex, first atom"""
s_1_comp_1 = 1 / (2 * sym.sqrt(sym.pi))

p1_comp_1 = 1 / 2 * sym.sqrt(3 / (2 * sym.pi)) * sym.exp(-sym.I * phi) * sym.sin(theta1)
p2_comp_1 = 1 / 2 * sym.sqrt(3 / sym.pi) * sym.cos(theta1)
p3_comp_1 = -1 / 2 * sym.sqrt(3 / (2 * sym.pi)) * sym.exp(sym.I * phi) * sym.sin(theta1)

d1_comp_1 = (
    1
    / 4
    * sym.sqrt(15 / (2 * sym.pi))
    * sym.exp(-2 * sym.I * phi)
    * sym.sin(theta1) ** 2
)
d2_comp_1 = (
    1
    / 2
    * sym.sqrt(15 / (2 * sym.pi))
    * sym.exp(-sym.I * phi)
    * sym.sin(theta1)
    * sym.cos(theta1)
)
d3_comp_1 = 1 / 4 * sym.sqrt(5 / sym.pi) * (3 * sym.cos(theta1) ** 2 - 1)
d4_comp_1 = (
    -1
    / 2
    * sym.sqrt(15 / (2 * sym.pi))
    * sym.exp(sym.I * phi)
    * sym.sin(theta1)
    * sym.cos(theta1)
)
d5_comp_1 = (
    1
    / 4
    * sym.sqrt(15 / (2 * sym.pi))
    * sym.exp(2 * sym.I * phi)
    * sym.sin(theta1) ** 2
)

f1_comp_1 = (
    1 / 8 * sym.sqrt(35 / sym.pi) * sym.exp(-3 * sym.I * phi) * sym.sin(theta1) ** 3
)
f2_comp_1 = (
    1
    / 4
    * sym.sqrt(105 / (2 * sym.pi))
    * sym.exp(-2 * sym.I * phi)
    * sym.sin(theta1) ** 2
    * sym.cos(theta1)
)
f3_comp_1 = (
    1
    / 8
    * sym.sqrt(21 / sym.pi)
    * sym.exp(-sym.I * phi)
    * sym.sin(theta1)
    * (5 * sym.cos(theta1) ** 2 - 1)
)
f4_comp_1 = (
    1 / 4 * sym.sqrt(7 / sym.pi) * (5 * sym.cos(theta1) ** 3 - 3 * sym.cos(theta1))
)
f5_comp_1 = (
    -1
    / 8
    * sym.sqrt(21 / sym.pi)
    * sym.exp(sym.I * phi)
    * sym.sin(theta1)
    * (5 * sym.cos(theta1) ** 2 - 1)
)
f6_comp_1 = (
    1
    / 4
    * sym.sqrt(105 / (2 * sym.pi))
    * sym.exp(2 * sym.I * phi)
    * sym.sin(theta1) ** 2
    * sym.cos(theta1)
)
f7_comp_1 = (
    -1 / 8 * sym.sqrt(35 / sym.pi) * sym.exp(3 * sym.I * phi) * sym.sin(theta1) ** 3
)


"""complex, second atom"""
s_2_comp_2 = 1 / (2 * sym.sqrt(sym.pi))

p1_comp_2 = 1 / 2 * sym.sqrt(3 / (2 * sym.pi)) * sym.exp(-sym.I * phi) * sym.sin(theta2)
p2_comp_2 = 1 / 2 * sym.sqrt(3 / sym.pi) * sym.cos(theta2)
p3_comp_2 = -1 / 2 * sym.sqrt(3 / (2 * sym.pi)) * sym.exp(sym.I * phi) * sym.sin(theta2)

d1_comp_2 = (
    1
    / 4
    * sym.sqrt(15 / (2 * sym.pi))
    * sym.exp(-2 * sym.I * phi)
    * sym.sin(theta2) ** 2
)
d2_comp_2 = (
    1
    / 2
    * sym.sqrt(15 / (2 * sym.pi))
    * sym.exp(-sym.I * phi)
    * sym.sin(theta2)
    * sym.cos(theta2)
)
d3_comp_2 = 1 / 4 * sym.sqrt(5 / sym.pi) * (3 * sym.cos(theta2) ** 2 - 1)
d4_comp_2 = (
    -1
    / 2
    * sym.sqrt(15 / (2 * sym.pi))
    * sym.exp(sym.I * phi)
    * sym.sin(theta2)
    * sym.cos(theta2)
)
d5_comp_2 = (
    1
    / 4
    * sym.sqrt(15 / (2 * sym.pi))
    * sym.exp(2 * sym.I * phi)
    * sym.sin(theta2) ** 2
)

f1_comp_2 = (
    1 / 8 * sym.sqrt(35 / sym.pi) * sym.exp(-3 * sym.I * phi) * sym.sin(theta2) ** 3
)
f2_comp_2 = (
    1
    / 4
    * sym.sqrt(105 / (2 * sym.pi))
    * sym.exp(-2 * sym.I * phi)
    * sym.sin(theta2) ** 2
    * sym.cos(theta2)
)
f3_comp_2 = (
    1
    / 8
    * sym.sqrt(21 / sym.pi)
    * sym.exp(-sym.I * phi)
    * sym.sin(theta2)
    * (5 * sym.cos(theta2) ** 2 - 1)
)
f4_comp_2 = (
    1 / 4 * sym.sqrt(7 / sym.pi) * (5 * sym.cos(theta2) ** 3 - 3 * sym.cos(theta2))
)
f5_comp_2 = (
    -1
    / 8
    * sym.sqrt(21 / sym.pi)
    * sym.exp(sym.I * phi)
    * sym.sin(theta2)
    * (5 * sym.cos(theta2) ** 2 - 1)
)
f6_comp_2 = (
    1
    / 4
    * sym.sqrt(105 / (2 * sym.pi))
    * sym.exp(2 * sym.I * phi)
    * sym.sin(theta2) ** 2
    * sym.cos(theta2)
)
f7_comp_2 = (
    -1 / 8 * sym.sqrt(35 / sym.pi) * sym.exp(3 * sym.I * phi) * sym.sin(theta2) ** 3
)


"""real centered at first atom"""
ss_1 = s_1_comp_1

py_1 = 1 / 2 * sym.sqrt(3 / sym.pi) * sym.sin(theta1) * sym.sin(phi)
pz_1 = 1 / 2 * sym.sqrt(3 / sym.pi) * sym.cos(theta1)
px_1 = 1 / 2 * sym.sqrt(3 / sym.pi) * sym.cos(phi) * sym.sin(theta1)

dxy_1 = 1 / 4 * sym.sqrt(15 / sym.pi) * sym.sin(theta1) ** 2 * sym.sin(2 * phi)
dyz_1 = 1 / 2 * sym.sqrt(15 / sym.pi) * sym.cos(theta1) * sym.sin(theta1) * sym.sin(phi)
dz2_1 = 1 / 4 * sym.sqrt(5 / sym.pi) * (3 * sym.cos(theta1) ** 2 - 1)
dzx_1 = 1 / 2 * sym.sqrt(15 / sym.pi) * sym.cos(theta1) * sym.cos(phi) * sym.sin(theta1)
dx2y2_1 = 1 / 4 * sym.sqrt(15 / sym.pi) * sym.cos(2 * phi) * sym.sin(theta1) ** 2

f1_1 = sym.simplify(sym.I / sym.sqrt(2) * (f1_comp_1 + f7_comp_1))
f2_1 = sym.simplify(sym.I / sym.sqrt(2) * (f2_comp_1 - f6_comp_1))
f3_1 = sym.simplify(sym.I / sym.sqrt(2) * (f3_comp_1 + f5_comp_1))
f4_1 = f4_comp_1
f5_1 = sym.simplify(1 / sym.sqrt(2) * (f3_comp_1 - f5_comp_1))
f6_1 = sym.simplify(1 / sym.sqrt(2) * (f2_comp_1 + f6_comp_1))
f7_1 = sym.simplify(1 / sym.sqrt(2) * (f1_comp_1 - f7_comp_1))

"""real, centered at second atom"""
ss_2 = s_2_comp_2

py_2 = 1 / 2 * sym.sqrt(3 / sym.pi) * sym.sin(theta2) * sym.sin(phi)
pz_2 = 1 / 2 * sym.sqrt(3 / sym.pi) * sym.cos(theta2)
px_2 = 1 / 2 * sym.sqrt(3 / sym.pi) * sym.cos(phi) * sym.sin(theta2)

dxy_2 = 1 / 4 * sym.sqrt(15 / sym.pi) * sym.sin(theta2) ** 2 * sym.sin(2 * phi)
dyz_2 = 1 / 2 * sym.sqrt(15 / sym.pi) * sym.cos(theta2) * sym.sin(theta2) * sym.sin(phi)
dz2_2 = 1 / 4 * sym.sqrt(5 / sym.pi) * (3 * sym.cos(theta2) ** 2 - 1)
dzx_2 = 1 / 2 * sym.sqrt(15 / sym.pi) * sym.cos(theta2) * sym.cos(phi) * sym.sin(theta2)
dx2y2_2 = 1 / 4 * sym.sqrt(15 / sym.pi) * sym.cos(2 * phi) * sym.sin(theta2) ** 2

f1_2 = sym.simplify(sym.I / sym.sqrt(2) * (f1_comp_2 + f7_comp_2))
f2_2 = sym.simplify(sym.I / sym.sqrt(2) * (f2_comp_2 - f6_comp_2))
f3_2 = sym.simplify(sym.I / sym.sqrt(2) * (f3_comp_2 + f5_comp_2))
f4_2 = f4_comp_2
f5_2 = sym.simplify(1 / sym.sqrt(2) * (f3_comp_2 - f5_comp_2))
f6_2 = sym.simplify(1 / sym.sqrt(2) * (f2_comp_2 + f6_comp_2))
f7_2 = sym.simplify(1 / sym.sqrt(2) * (f1_comp_2 - f7_comp_2))

first_center = {
    "ss": (ss_1, 0, 0),
    "py": (py_1, 1, -1),
    "pz": (pz_1, 1, 0),
    "px": (px_1, 1, 1),
    "d1": (dxy_1, 2, -2),
    "d2": (dyz_1, 2, -1),
    "d3": (dz2_1, 2, 0),
    "d4": (dzx_1, 2, 1),
    "d5": (dx2y2_1, 2, 2),
    "f1": (f1_1, 3, -3),
    "f2": (f2_1, 3, -2),
    "f3": (f3_1, 3, -1),
    "f4": (f4_1, 3, 0),
    "f5": (f5_1, 3, 1),
    "f6": (f6_1, 3, 2),
    "f7": (f7_1, 3, 3),
}

first_center_complex = {
    "ss": (s_1_comp_1, 0, 0),
    "py": (p1_comp_1, 1, -1),
    "pz": (p2_comp_1, 1, 0),
    "px": (p3_comp_1, 1, 1),
    "d1": (d1_comp_1, 2, -2),
    "d2": (d2_comp_1, 2, -1),
    "d3": (d3_comp_1, 2, 0),
    "d4": (d4_comp_1, 2, 1),
    "d5": (d5_comp_1, 2, 2),
    "f1": (f1_comp_1, 3, -3),
    "f2": (f2_comp_1, 3, -2),
    "f3": (f3_comp_1, 3, -1),
    "f4": (f4_comp_1, 3, 0),
    "f5": (f5_comp_1, 3, 1),
    "f6": (f6_comp_1, 3, 2),
    "f7": (f7_comp_1, 3, 3),
}

second_center = {
    "ss": (ss_2, 0, 0),
    "py": (py_2, 1, -1),
    "pz": (pz_2, 1, 0),
    "px": (px_2, 1, 1),
    "d1": (dxy_2, 2, -2),
    "d2": (dyz_2, 2, -1),
    "d3": (dz2_2, 2, 0),
    "d4": (dzx_2, 2, 1),
    "d5": (dx2y2_2, 2, 2),
    "f1": (f1_2, 3, -3),
    "f2": (f2_2, 3, -2),
    "f3": (f3_2, 3, -1),
    "f4": (f4_2, 3, 0),
    "f5": (f5_2, 3, 1),
    "f6": (f6_2, 3, 2),
    "f7": (f7_2, 3, 3),
}

operator = {
    "py": (py_1, 1, -1),
    "pz": (pz_1, 1, 0),
    "px": (px_1, 1, 1),
}


def interchange_related_phi2():
    """Print dictionary with
    keys: integral index of the integrals appearing in the conventional .skf format
    values: list of integrals that are related to the
    same integral with centers interchanged by M_ij = (-1)**(l_i + l_j) * M_ji

    """
    tmp1, tmp2 = sym.symbols("tmp1 tmp2")
    integrals_DFTB = {}
    interchange_related_integrals = {}
    for i in DFTBPLUS_SIMPLE:
        interchange_related_integrals[i] = []

    # calculate integral expressions for those integrals appearing in the conventional .skf format
    count = 0
    for name_i, i in first_center.items():
        for name_k, k in second_center.items():
            if count in DFTBPLUS_SIMPLE:
                integral = sym.integrate(i[0] * k[0], (phi, 0, 2 * sym.pi))
                integrals_DFTB[count] = integral
            count += 1

    # check, which of the nonzero phi2 integrals are identical to the integrals appearing in the conventional .skf format
    count = 0
    for name_i, i in first_center.items():
        for name_k, k in second_center.items():
            integral = sym.integrate(i[0] * k[0], (phi, 0, 2 * sym.pi))
            for dftb in DFTBPLUS_SIMPLE:
                identical = sym.simplify(integral - integrals_DFTB[dftb])
                integral_swapped = integral.subs({theta1: tmp1, theta2: tmp2})
                integral_swapped = integral_swapped.subs({tmp1: theta2, tmp2: theta1})
                centers_exchanged = sym.simplify(
                    integral_swapped - integrals_DFTB[dftb]
                )
                parity_factor = (-1) ** (i[1] + k[1])
                if (centers_exchanged == 0) and (identical != 0):
                    interchange_related_integrals[dftb].append(parity_factor * count)
            count += 1
    print(interchange_related_integrals)


def minimal_phi2():
    tmp1, tmp2 = sym.symbols("tmp1 tmp2")
    integrals_DFTB = {}
    identical_integrals = {}
    for i in DFTBPLUS_SIMPLE:
        identical_integrals[i] = []

    # calculate integral expressions for those integrals appearing in the conventional .skf format
    count = 0
    for name_i, i in first_center.items():
        for name_k, k in second_center.items():
            if count in DFTBPLUS_SIMPLE:
                integral = sym.integrate(i[0] * k[0], (phi, 0, 2 * sym.pi))
                integrals_DFTB[count] = integral
            count += 1

    # check, which of the nonzero phi2 integrals are identical to the integrals appearing in the conventional .skf format
    count = 0
    for name_i, i in first_center.items():
        for name_k, k in second_center.items():
            integral = sym.integrate(i[0] * k[0], (phi, 0, 2 * sym.pi))
            for dftb in DFTBPLUS_SIMPLE:
                identical = sym.simplify(integral - integrals_DFTB[dftb])
                if identical == 0:
                    identical_integrals[dftb].append(count)
            count += 1
    print(identical_integrals)


def identical_phi2():
    unique_integrals = []
    unique_indices = []
    equivalence_classes = {}

    count = 0
    for name_i, i in first_center.items():
        for name_k, k in second_center.items():
            integral = sym.integrate(i[0] * k[0], (phi, 0, 2 * sym.pi))
            if integral != 0:
                if integral not in unique_integrals:
                    unique_integrals.append(integral)
                    unique_indices.append(count)
                    equivalence_classes[count] = []
                equivalence_classes[
                    unique_indices[unique_integrals.index(integral)]
                ].append(count)
            count += 1

    print(equivalence_classes)
    print(len(equivalence_classes.keys()))


def identical_phi3():
    unique_integrals = []
    unique_indices = []
    equivalence_classes = {}
    count = 0
    for name_i, i in first_center.items():
        for name_j, j in operator.items():
            for name_k, k in second_center.items():
                integral = sym.integrate(i[0] * j[0] * k[0], (phi, 0, 2 * sym.pi))
                if integral != 0:
                    if integral in unique_integrals:
                        pos, sign = unique_integrals.index(integral), 1
                    elif -integral in unique_integrals:
                        pos, sign = unique_integrals.index(-integral), -1
                    else:
                        unique_integrals.append(integral)
                        unique_indices.append(count)
                        equivalence_classes[count] = []
                        pos, sign = len(unique_integrals) - 1, 1
                    equivalence_classes[unique_indices[pos]].append(sign * count)
                count += 1
    print(equivalence_classes)
    print(len(equivalence_classes.keys()))


def identical_atomic_transitions():
    unique_integrals = []
    unique_indices = []
    equivalence_classes = {}
    count = 0
    for name_i, i in first_center.items():
        for name_j, j in operator.items():
            for name_k, k in second_center.items():
                integral = sym.integrate(
                    sym.integrate(
                        i[0] * j[0] * k[0].subs(theta2, theta1) * sym.sin(theta1),
                        (phi, 0, 2 * sym.pi),
                    ),
                    (theta1, 0, sym.pi),
                )
                if integral != 0:
                    if integral in unique_integrals:
                        pos, sign = unique_integrals.index(integral), 1
                    elif -integral in unique_integrals:
                        pos, sign = unique_integrals.index(-integral), -1
                    else:
                        unique_integrals.append(integral)
                        unique_indices.append(count)
                        equivalence_classes[count] = []
                        pos, sign = len(unique_integrals) - 1, 1
                    equivalence_classes[unique_indices[pos]].append(sign * count)
                count += 1
    print(equivalence_classes)
    print(len(equivalence_classes.keys()))


def pick_quantum_number(dictionary, lm):
    """map from quantum numbers to respective (function, l,m)"""
    for key, value in dictionary.items():
        if value[1] == lm[0] and value[2] == lm[1]:
            return value
    raise ValueError(
        "Element missing: No spherical harmonic for this quantum number combination"
    )


def get_index_list_dipole():
    """list of nonzero phi3 integrals"""
    count = 0
    identifier = []
    nonzeros = []
    for name_i, i in first_center.items():
        for name_j, j in operator.items():
            for name_k, k in second_center.items():
                integral = sym.integrate(i[0] * j[0] * k[0], (phi, 0, 2 * sym.pi))
                if integral != 0:
                    nonzeros.append(count)
                tuple = (count, i[1], i[2], j[1], j[2], k[1], k[2])
                identifier.append(tuple)
                count += 1
    np.savez("identifier_nonzeros_posop.npz", np.array(identifier), np.array(nonzeros))
    return identifier, nonzeros


def get_index_list_overlap():
    """list of nonzero phi2 integrals"""
    count = 0
    identifier = []
    nonzeros = []
    for name_i, i in first_center.items():
        for name_k, k in second_center.items():
            integral = sym.integrate(i[0] * k[0], (phi, 0, 2 * sym.pi))
            if integral != 0:
                nonzeros.append(count)
            tuple = (count, i[1], i[2], k[1], k[2])
            identifier.append(tuple)
            count += 1
    np.savez(
        "identifier_nonzeros_overlap.npz", np.array(identifier), np.array(nonzeros)
    )
    return identifier, nonzeros


def print_dipole_integrals():
    counter = 0
    f = open("phi3_expr.txt", "w")
    print("INTEGRALS = {", file=f)
    time_start = time.time()
    for name_i, i in first_center.items():
        for name_j, j in operator.items():
            for name_k, k in second_center.items():
                tuple = (counter, i[1], i[2], j[1], j[2], k[1], k[2])
                integral = sym.integrate(i[0] * j[0] * k[0], (phi, 0, 2 * sym.pi))
                counter += 1
                if integral != 0:
                    if CODE:
                        txt = f"{integral}"
                        txt = txt.replace("sqrt", "np.sqrt")
                        txt = txt.replace("sin(theta2)", "s2")
                        txt = txt.replace("cos(theta2)", "c2")
                        txt = txt.replace("cos(theta1)", "c1")
                        txt = txt.replace("sin(theta1)", "s1")
                        txt = txt.replace("pi", "np.pi")
                        print(f"\t{tuple}: lambda c1, c2, s1, s2: " + txt + ",", file=f)
    time_end = time.time()
    print("}", file=f)
    print(f"finished integrals in {time_end - time_start} s")


def print_overlap_integrals():
    """print python code for numerical phi2 integrals to file"""
    counter = 0
    f = open("phi2_expr.txt", "w")
    print("INTEGRALS = {", file=f)
    time_start = time.time()
    for name_i, i in first_center.items():
        for name_k, k in second_center.items():
            tuple = (counter, i[1], i[2], k[1], k[2])
            integral = sym.integrate(i[0] * k[0], (phi, 0, 2 * sym.pi))
            counter += 1
            if integral != 0:
                if CODE:
                    txt = f"{integral}"
                    txt = txt.replace("sqrt", "np.sqrt")
                    txt = txt.replace("sin(theta2)", "s2")
                    txt = txt.replace("cos(theta2)", "c2")
                    txt = txt.replace("cos(theta1)", "c1")
                    txt = txt.replace("sin(theta1)", "s1")
                    txt = txt.replace("pi", "np.pi")
                    print(f"\t{tuple}: lambda c1, c2, s1, s2: " + txt + ",", file=f)
    time_end = time.time()
    print("}", file=f)
    print(f"finished integrals in {time_end - time_start} s")


def print_overlap_derivatives():
    """print python code for derivatives of phi2 integrals to file"""
    counter = 0
    f = open("deriv-phi2_expr.txt", "w")
    print("INTEGRAL_DERIVATIVE = {", file=f)
    time_start = time.time()
    for name_i, i in first_center.items():
        for name_k, k in second_center.items():
            tuple = (counter, i[1], i[2], k[1], k[2])
            integral = sym.integrate(i[0] * k[0], (phi, 0, 2 * sym.pi))
            integral = integral.subs(
                {
                    sym.sin(theta1): s1,
                    sym.sin(theta2): s2,
                    sym.cos(theta1): c1,
                    sym.cos(theta2): c2,
                }
            )
            ds1 = sym.diff(integral, s1)
            ds2 = sym.diff(integral, s2)
            dc1 = sym.diff(integral, c1)
            dc2 = sym.diff(integral, c2)
            counter += 1
            if integral != 0:
                print(ds1)
                print(ds2)
                print(dc1)
                print(dc2)
                if CODE:
                    txt = f"[{dc1}, {dc2}, {ds1}, {ds2}],"
                    txt = txt.replace("sqrt", "np.sqrt")
                    txt = txt.replace("pi", "np.pi")
                    print(f"\t{tuple}: lambda c1, c2, s1, s2: " + txt, file=f)
    time_end = time.time()
    print("}", file=f)
    print(f"finished integrals in {time_end - time_start} s")


def print_f_orbitals():
    for orb in first_center.items():
        print(orb)


if __name__ == "__main__":
    # print_dipole_integrals()
    # print_overlap_integrals()
    # print_overlap_derivatives()
    # interchange_related_phi2()
    # identical_phi2()
    # identical_phi3()
    identical_atomic_transitions()
