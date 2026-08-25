"""
    Symbolic expression of Wigner D matrix with first euler angle equal to 0.
    generated with rotation_transform.py
    """
import sympy as sym
import math
import sympy.tensor.array.expressions
from hotcent.pos_op.slako_new import EQUIVALENT_INTEGRALS, UNIQUE_INTEGRALS
from hotcent.pos_op.slako_dipole import EQUIVALENT_INTEGRALS_POSOP, UNIQUE_INTEGRALS_POSOP

q, r, s = sym.symbols('l, m, n', real=True)
ALPHA, BETA, GAMMA = sym.symbols('alpha, beta, gamma', real=True)
P, Q = sym.symbols('P, Q', positive=True)


D_SYMB = sym.Matrix([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, sym.cos(GAMMA), sym.cos(BETA - GAMMA)/2 - sym.cos(BETA + GAMMA)/2, -sym.sin(GAMMA)*sym.cos(BETA), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, sym.cos(BETA), sym.sin(BETA), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, sym.sin(GAMMA), -sym.sin(BETA - GAMMA)/2 - sym.sin(BETA + GAMMA)/2, sym.cos(BETA - GAMMA)/2 + sym.cos(BETA + GAMMA)/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 2*(sym.cos(GAMMA)**2 - 1/2)*sym.cos(BETA), (1 - 2*sym.cos(GAMMA)**2)*sym.sin(BETA), -2*sym.sqrt(3)*sym.sin(BETA/2)**2*sym.sin(2*GAMMA)*sym.cos(BETA/2)**2, sym.cos(2*(BETA - GAMMA))/4 - sym.cos(2*(BETA + GAMMA))/4, 2*(-2*sym.sin(BETA/2)**4 + 2*sym.sin(BETA/2)**2 - 1)*sym.sin(GAMMA)*sym.cos(GAMMA), 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, sym.sin(BETA)*sym.cos(GAMMA), sym.cos(BETA)*sym.cos(GAMMA), sym.sqrt(3)*(sym.cos(2*BETA - GAMMA) - sym.cos(2*BETA + GAMMA))/4, (-2*sym.sin(BETA/2)**4 + 2*sym.sin(BETA/2)**2 + 3*sym.sin(BETA)**2/2 - 1)*sym.sin(GAMMA), -sym.cos(2*BETA - GAMMA)/4 + sym.cos(2*BETA + GAMMA)/4, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, (1 - sym.cos(BETA))**2/2 + sym.cos(BETA) + sym.cos(2*BETA)/2 - 1/2, sym.sqrt(3)*sym.sin(2*BETA)/2, sym.sqrt(3)*(1 - sym.cos(2*BETA))/4, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, sym.sin(BETA)*sym.sin(GAMMA), sym.sin(GAMMA)*sym.cos(BETA), -sym.sqrt(3)*(sym.sin(2*BETA - GAMMA) + sym.sin(2*BETA + GAMMA))/4, (2*sym.sin(BETA/2)**4 - 2*sym.sin(BETA/2)**2 - 3*sym.sin(BETA)**2/2 + 1)*sym.cos(GAMMA), sym.sin(2*BETA - GAMMA)/4 + sym.sin(2*BETA + GAMMA)/4, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, -sym.sin(BETA - 2*GAMMA)/2 + sym.sin(BETA + 2*GAMMA)/2, -sym.cos(BETA - 2*GAMMA)/2 + sym.cos(BETA + 2*GAMMA)/2, sym.sqrt(3)*(1 - sym.cos(2*BETA))*sym.cos(2*GAMMA)/4, -sym.sin(2*(BETA - GAMMA))/4 - sym.sin(2*(BETA + GAMMA))/4, sym.sqrt(2)*(sym.sqrt(2)*sym.sin(BETA/2)**4*sym.cos(2*GAMMA)/2 + sym.sqrt(2)*sym.cos(BETA/2)**4*sym.cos(2*GAMMA)/2), 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, (3*sym.sin(BETA/2)**4 - 3*sym.sin(BETA/2)**2 + 1)*sym.cos(3*GAMMA), -sym.sqrt(6)*(sym.sin(2*BETA - 3*GAMMA) + sym.sin(2*BETA + 3*GAMMA))/8, sym.sqrt(15)*sym.sin(BETA)**2*sym.cos(3*GAMMA)/4, 2*sym.sqrt(10)*sym.sin(BETA/2)**3*sym.sin(3*GAMMA)*sym.cos(BETA/2)**3, sym.sqrt(15)*(sym.sin(BETA/2)**2 - sym.cos(BETA/2)**2)*sym.sin(BETA/2)**2*sym.sin(3*GAMMA)*sym.cos(BETA/2)**2, sym.sqrt(6)*(2*sym.sin(BETA/2)**4 - 2*sym.sin(BETA/2)**2 + 1)*sym.sin(BETA/2)*sym.sin(3*GAMMA)*sym.cos(BETA/2), 15*sym.sin(BETA - 3*GAMMA)/32 + sym.sin(3*(BETA - GAMMA))/32 - sym.sin(3*(BETA + GAMMA))/32 - 15*sym.sin(BETA + 3*GAMMA)/32], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, sym.sqrt(6)*(sym.sin(2*(BETA - GAMMA)) + sym.sin(2*(BETA + GAMMA)))/8, sym.sqrt(2)*(-sym.sqrt(2)*(5 - 6*sym.sin(BETA/2)**2)*sym.sin(BETA/2)**4*sym.cos(2*GAMMA)/2 + sym.sqrt(2)*(sym.cos(BETA) + 1)**2*(3*sym.cos(BETA) - 2)*sym.cos(2*GAMMA)/8)/2 - sym.sqrt(2)*(sym.sqrt(2)*(5 - 6*sym.sin(BETA/2)**2)*sym.sin(BETA/2)**4*sym.cos(2*GAMMA)/2 - sym.sqrt(2)*(sym.cos(BETA) + 1)**2*(3*sym.cos(BETA) - 2)*sym.cos(2*GAMMA)/8)/2, -sym.sqrt(10)*(sym.sin(2*(BETA - GAMMA)) + sym.sin(2*(BETA + GAMMA)))/8, -sym.sqrt(15)*(-sym.sin(BETA - 2*GAMMA) + sym.sin(BETA + 2*GAMMA) + sym.sin(3*BETA - 2*GAMMA) - sym.sin(3*BETA + 2*GAMMA))/16, sym.sqrt(10)*(-sym.sin(BETA) + 3*sym.sin(3*BETA))*sym.sin(GAMMA)*sym.cos(GAMMA)/8, sym.sqrt(2)*(-sym.sqrt(2)*(5 - 6*sym.sin(BETA/2)**2)*sym.sin(BETA/2)**4*sym.sin(2*GAMMA)/2 - sym.sqrt(2)*(sym.cos(BETA) + 1)**2*(3*sym.cos(BETA) - 2)*sym.sin(2*GAMMA)/8), sym.sqrt(6)*(-(1 - sym.cos(BETA))**2*sym.cos(BETA - 2*GAMMA) + (1 - sym.cos(BETA))**2*sym.cos(BETA + 2*GAMMA) - sym.cos(2*(BETA - GAMMA)) + sym.cos(2*(BETA + GAMMA)))/8], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, sym.sqrt(15)*sym.sin(BETA)**2*sym.cos(GAMMA)/4, sym.sqrt(10)*(sym.sin(2*BETA - GAMMA) + sym.sin(2*BETA + GAMMA))/8, 2*(15*sym.sin(BETA/2)**6/2 - 20*sym.sin(BETA/2)**4 + 20*sym.sin(BETA/2)**2 + 15*sym.cos(BETA/2)**6/2 - 7)*sym.cos(GAMMA), sym.sqrt(6)*(sym.sin(BETA)*sym.cos(BETA)**2 + sym.sin(3*BETA))*sym.sin(GAMMA)/4, 2*(-15*sym.sin(BETA/2)**4 + 15*sym.sin(BETA/2)**2 - 1)*sym.sin(GAMMA)*sym.sin((2*BETA + sym.pi)/4)*sym.cos((2*BETA + sym.pi)/4), sym.sqrt(10)*(4*sym.sin(BETA)**3 - sym.sin(BETA) - 5*sym.sin(3*BETA))*sym.sin(GAMMA)/32, sym.sqrt(15)*(sym.sin(BETA/2)**2 - sym.cos(BETA/2)**2)*sym.sin(BETA/2)**2*sym.sin(GAMMA)*sym.cos(BETA/2)**2], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5*(1 - sym.cos(BETA))**2*sym.cos(BETA)/2 - 4*sym.cos(BETA) + 5*sym.cos(2*BETA)/2 + 5/2, sym.sqrt(6)*(sym.sin(BETA) + 5*sym.sin(3*BETA))/16, 2*sym.sqrt(15)*(-sym.sin(BETA/2)**4*sym.cos(BETA/2)**2 + sym.sin(BETA/2)**2*sym.cos(BETA/2)**4), sym.sqrt(10)*(3*sym.sin(BETA) - sym.sin(3*BETA))/16], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, sym.sqrt(15)*sym.sin(BETA)**2*sym.sin(GAMMA)/4, sym.sqrt(10)*(sym.cos(2*BETA - GAMMA) - sym.cos(2*BETA + GAMMA))/8, 2*(15*sym.sin(BETA/2)**6/2 - 20*sym.sin(BETA/2)**4 + 20*sym.sin(BETA/2)**2 + 15*sym.cos(BETA/2)**6/2 - 7)*sym.sin(GAMMA), -sym.sqrt(6)*(sym.sin(BETA)*sym.cos(BETA)**2 + sym.sin(3*BETA))*sym.cos(GAMMA)/4, 2*(15*sym.sin(BETA/2)**4 - 15*sym.sin(BETA/2)**2 + 1)*sym.sin((2*BETA + sym.pi)/4)*sym.cos(GAMMA)*sym.cos((2*BETA + sym.pi)/4), sym.sqrt(10)*(-4*sym.sin(BETA)**3 + sym.sin(BETA) + 5*sym.sin(3*BETA))*sym.cos(GAMMA)/32, sym.sqrt(15)*(-sym.sin(BETA/2)**2 + sym.cos(BETA/2)**2)*sym.sin(BETA/2)**2*sym.cos(BETA/2)**2*sym.cos(GAMMA)], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, sym.sqrt(6)*(sym.cos(2*(BETA - GAMMA)) - sym.cos(2*(BETA + GAMMA)))/8, (-10*(1 - sym.cos(BETA/2)**2)**2 + 12*sym.sin(BETA/2)**6 + 3*sym.cos(BETA)**3/2 + 2*sym.cos(BETA)**2 - sym.cos(BETA)/2 - 1)*sym.sin(GAMMA)*sym.cos(GAMMA), sym.sqrt(10)*(-sym.cos(2*(BETA - GAMMA)) + sym.cos(2*(BETA + GAMMA)))/8, sym.sqrt(15)*(sym.cos(BETA - 2*GAMMA) + sym.cos(BETA + 2*GAMMA) - sym.cos(3*BETA - 2*GAMMA) - sym.cos(3*BETA + 2*GAMMA))/16, sym.sqrt(10)*(-8*sym.sin(BETA)**3*sym.sin(GAMMA)**2 + 4*sym.sin(BETA)**3 + 2*sym.sin(BETA)*sym.sin(GAMMA)**2 - sym.sin(BETA) + 10*sym.sin(3*BETA)*sym.sin(GAMMA)**2 - 5*sym.sin(3*BETA))/32, sym.sqrt(2)*(sym.sqrt(2)*(5 - 6*sym.sin(BETA/2)**2)*sym.sin(BETA/2)**4*sym.cos(2*GAMMA)/2 + sym.sqrt(2)*(sym.cos(BETA) + 1)**2*(3*sym.cos(BETA) - 2)*sym.cos(2*GAMMA)/8), sym.sqrt(6)*(-4*sym.sin(BETA/2)**4*sym.sin(GAMMA)**2 + 2*sym.sin(BETA/2)**4 + 4*sym.sin(BETA/2)**2*sym.sin(GAMMA)**2 - 2*sym.sin(BETA/2)**2 - 2*sym.sin(GAMMA)**2 + 1)*sym.sin(BETA/2)*sym.cos(BETA/2)], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, (3*sym.sin(BETA/2)**4 - 3*sym.sin(BETA/2)**2 + 1)*sym.sin(3*GAMMA), sym.sqrt(6)*(-sym.cos(2*BETA - 3*GAMMA) + sym.cos(2*BETA + 3*GAMMA))/8, sym.sqrt(15)*sym.sin(BETA)**2*sym.sin(3*GAMMA)/4, -2*sym.sqrt(10)*sym.sin(BETA/2)**3*sym.cos(BETA/2)**3*sym.cos(3*GAMMA), sym.sqrt(15)*(-sym.sin(BETA/2)**2 + sym.cos(BETA/2)**2)*sym.sin(BETA/2)**2*sym.cos(BETA/2)**2*sym.cos(3*GAMMA), sym.sqrt(6)*(-2*sym.sin(BETA/2)**4 + 2*sym.sin(BETA/2)**2 - 1)*sym.sin(BETA/2)*sym.cos(BETA/2)*sym.cos(3*GAMMA), 15*sym.cos(BETA - 3*GAMMA)/32 + sym.cos(3*(BETA - GAMMA))/32 + sym.cos(3*(BETA + GAMMA))/32 + 15*sym.cos(BETA + 3*GAMMA)/32]
    ]
    )

def print_sk_rules(minimal=True, cse=True):
    """
    Generate Python code for the explicit Slater Koster rules for H/S.
        minimal: if input of unique SK elements is enough or if 16x16 array of all quantum 
                number combinations is required
        cse: preevaluate common subexpressions. Makes evaluation faster but impairs readability
    """
    D = sym.expand_trig(D_SYMB)
    D = D.subs(sym.sin(BETA), -P*Q)
    D = D.subs(sym.cos(BETA), s)
    D = D.subs(sym.cos(GAMMA), q/(P*Q))
    D = D.subs(sym.sin(GAMMA), -r/(P*Q))
    D = D.subs(sym.sin(BETA/2), -Q/sym.sqrt(2))
    D = D.subs(sym.cos(BETA/2), P/sym.sqrt(2))
    D= sym.simplify(D)
    if minimal:
        X = sym.IndexedBase('X', (len(UNIQUE_INTEGRALS)))
    else:
        X = sym.IndexedBase('X', (16, 16))
    Y = sym.tensor.array.expressions.ArraySymbol('Y', (16, 16))

    rep, kept = {}, set()
    for count, num in enumerate(UNIQUE_INTEGRALS):
        i, j = divmod(num, 16)
        kept.add(num)
        for idx in EQUIVALENT_INTEGRALS[num]:
            a, b = divmod(idx, 16)
            assert rep.get(Y[a, b], Y[i, j]) == Y[i, j], f"conflicting rule for {idx}"
            if minimal:
                rep[Y[a,b]] = X[count]
            else:
                rep[Y[a, b]] = X[i, j]
            kept.add(idx)

    for num in set(range(256)) - kept:          # everything not covered is zero
        a, b = divmod(num, 16)
        rep[Y[a, b]] = sym.Integer(0)

    A = lambda a, b: rep.get(Y[a, b], Y[a, b])

    if minimal:
        str1 = '_mini'
    else: 
        str1 =''
    if cse:
        str2 = '_cse'
    else:
        str2 = ''
    outfile = 'skrules' + str1 + str2 + '.py'
    with open(outfile, 'w') as handle:
        print("import numpy as np", file=handle)
        print("import math", file=handle)
        print("def matrix_elements(X, l, m, n):", file=handle)
        print("\tC = np.zeros((16,16))", file=handle)
        expressions = []
        locations = []
        dim = 16
        for i in range(dim):
            for j in range(dim):
                e = 0
                for a in range(dim):
                    for b in range(dim):
                        e += A(a,b) * D[i,a] * D[j,b]
                e = e.subs(P**2, 1 + s)
                e = e.subs(Q**2, 1 - s)
                e = sym.simplify(e)
                num, den = sym.fraction(sym.cancel(sym.together(sym.expand(e))))
                num = sym.rem(sym.Poly(num, q), sym.Poly(q**2 - (1 - r**2 - s**2), q)).as_expr()
                e = sym.cancel(num / den)
                e = sym.simplify(e)
                if e != 0:
                    if cse:
                        expressions.append(e)
                        locations.append((i,j))
                    else:
                        print(f"\tC[{i},{j}] = {sym.pycode(e)}", file=handle)
        if cse:
            replacements, expressions = sym.cse(expressions, symbols=sym.numbered_symbols("x"))
            for symbol, expr in replacements:
                print(f"\t{symbol} = {sym.pycode(expr)}", file=handle)

            for (i, j), e in zip(locations, expressions):
                print(f"\tC[{i},{j}] = {sym.pycode(e)}", file=handle)
        print("\treturn C", file=handle)

def print_sk_rules_posop(minimal=True, cse=True):
    """
    Generate Python code for the explicit Slater Koster rules for position matrix elements.
        minimal: determines if input of unique SK elements is enough or if 16x3x16 array of all quantum 
                number combinations is required
        cse: preevaluate common subexpressions. Makes evaluation faster but impairs readability
    """
    D = sym.expand_trig(D_SYMB)
    D = D.subs(sym.sin(BETA), -P*Q)
    D = D.subs(sym.cos(BETA), s)
    D = D.subs(sym.cos(GAMMA), q/(P*Q))
    D = D.subs(sym.sin(GAMMA), -r/(P*Q))
    D = D.subs(sym.sin(BETA/2), -Q/sym.sqrt(2))
    D = D.subs(sym.cos(BETA/2), P/sym.sqrt(2))
    D= sym.simplify(D)

    if minimal:
        X = sym.IndexedBase('X', (len(UNIQUE_INTEGRALS_POSOP)))
    else:
        X = sym.IndexedBase('X', (16, 3, 16))
    Y = sym.tensor.array.expressions.ArraySymbol('Y', (16, 3, 16))

    rep, kept = {}, set()
    for count, num in enumerate(UNIQUE_INTEGRALS_POSOP):
        x, rem = divmod(num, 16*3)
        y, z = divmod(rem, 16)

        kept.add(num)
        for idx in EQUIVALENT_INTEGRALS_POSOP[num]:
            idx_abs = abs(idx)
            a, p = divmod(idx_abs, 16 * 3)
            b, c = divmod(p, 16)
            assert rep.get(Y[a,b,c], Y[x,y,z]) == Y[x,y,z], f"conflicting rule for {idx}"
            sign = -1 if idx < 0 else 1
            if minimal:
                rep[Y[a,b,c]] = sign * X[count]
            else:
                rep[Y[a,b,c]] = sign * X[x,y,z]
            kept.add(abs(idx))

    for num in set(range(16*16*3)) - kept:          # everything not covered is zero
        a, rem = divmod(num, 16 * 3)
        b, c = divmod(rem, 16)
        rep[Y[a,b,c]] = sym.Integer(0)

    A = lambda a, b, c: rep.get(Y[a, b, c], Y[a, b, c])

    if minimal:
        str1 = '_mini'
    else: 
        str1 =''
    if cse:
        str2 = '_cse'
    else:
        str2 = ''
    outfile = 'skrules_posop' + str1 + str2 + '.py'
    with open(outfile, 'w') as handle:
        print("import numpy as np", file=handle)
        print("import math", file=handle)
        print("def matrix_elements_posop(X, l, m, n):", file=handle)
        print("\tC = np.zeros((16,3,16))", file=handle)
        dim_basis = 16
        dim_realspace = 3
        expressions = []
        locations = []
        for x in range(dim_basis):
            for y in range(dim_realspace):
                for z in range(dim_basis):
                    e = 0
                    for a in range(dim_basis):
                        for b in range(dim_realspace):
                            for c in range(dim_basis):
                                e += A(a,b,c) * D[x,a] * D[y+1,b+1] * D[z,c]
                    e = e.subs(P**2, 1 + s)
                    e = e.subs(Q**2, 1 - s)
                    num, den = sym.fraction(sym.cancel(sym.together(sym.expand(e))))
                    num = sym.rem(sym.Poly(num, q), sym.Poly(q**2 - (1 - r**2 - s**2), q)).as_expr()
                    e = sym.cancel(num / den)
                    if e != 0:
                        if cse:
                            expressions.append(e)
                            locations.append((x,y,z))
                        else:
                            print(f"\tC[{x},{y},{z}] = {sym.pycode(e)}", file=handle)
        if cse:
            replacements, expressions = sym.cse(expressions, symbols=sym.numbered_symbols("x"))
            for symbol, expr in replacements:
                print(f"\t{symbol} = {sym.pycode(expr)}", file=handle)

            for (i, j, k), e in zip(locations, expressions):
                print(f"\tC[{i},{j},{k}] = {sym.pycode(e)}", file=handle)
        print("\treturn C", file=handle)


if __name__ == "__main__":
    print_sk_rules(minimal=True, cse=False)
    print_sk_rules_posop(minimal=True, cse=False)

                


