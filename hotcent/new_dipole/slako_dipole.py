import numpy as np

"""nonvanishing SlaKo integrals for dipole elements named in the form
Y1*Y1*Y2
unified format for the label: 1., 3. and 5. letter give nl, others just for distinguishing
(x,y,z) for p, (1,2,3,4,5) for d
"""
INTEGRALS = [
        (1, 0, 0, 1, -1, 1, -1),
        (5, 0, 0, 1, -1, 2, -1),
        (9, 0, 0, 1, 0, 0, 0),
        (11, 0, 0, 1, 0, 1, 0),
        (15, 0, 0, 1, 0, 2, 0),
        (21, 0, 0, 1, 1, 1, 1),
        (25, 0, 0, 1, 1, 2, 1),
        (27, 1, -1, 1, -1, 0, 0),
        (29, 1, -1, 1, -1, 1, 0),
        (33, 1, -1, 1, -1, 2, 0),
        (35, 1, -1, 1, -1, 2, 2),
        (37, 1, -1, 1, 0, 1, -1),
        (41, 1, -1, 1, 0, 2, -1),
        (49, 1, -1, 1, 1, 2, -2),
        (55, 1, 0, 1, -1, 1, -1),
        (59, 1, 0, 1, -1, 2, -1),
        (63, 1, 0, 1, 0, 0, 0),
        (65, 1, 0, 1, 0, 1, 0),
        (69, 1, 0, 1, 0, 2, 0),
        (75, 1, 0, 1, 1, 1, 1),
        (79, 1, 0, 1, 1, 2, 1),
        (85, 1, 1, 1, -1, 2, -2),
        (93, 1, 1, 1, 0, 1, 1),
        (97, 1, 1, 1, 0, 2, 1),
        (99, 1, 1, 1, 1, 0, 0),
        (101, 1, 1, 1, 1, 1, 0),
        (105, 1, 1, 1, 1, 2, 0),
        (107, 1, 1, 1, 1, 2, 2),
        (111, 2, -2, 1, -1, 1, 1),
        (115, 2, -2, 1, -1, 2, 1),
        (121, 2, -2, 1, 0, 2, -2),
        (127, 2, -2, 1, 1, 1, -1),
        (131, 2, -2, 1, 1, 2, -1),
        (135, 2, -1, 1, -1, 0, 0),
        (137, 2, -1, 1, -1, 1, 0),
        (141, 2, -1, 1, -1, 2, 0),
        (143, 2, -1, 1, -1, 2, 2),
        (145, 2, -1, 1, 0, 1, -1),
        (149, 2, -1, 1, 0, 2, -1),
        (157, 2, -1, 1, 1, 2, -2),
        (163, 2, 0, 1, -1, 1, -1),
        (167, 2, 0, 1, -1, 2, -1),
        (171, 2, 0, 1, 0, 0, 0),
        (173, 2, 0, 1, 0, 1, 0),
        (177, 2, 0, 1, 0, 2, 0),
        (183, 2, 0, 1, 1, 1, 1),
        (187, 2, 0, 1, 1, 2, 1),
        (193, 2, 1, 1, -1, 2, -2),
        (201, 2, 1, 1, 0, 1, 1),
        (205, 2, 1, 1, 0, 2, 1),
        (207, 2, 1, 1, 1, 0, 0),
        (209, 2, 1, 1, 1, 1, 0),
        (213, 2, 1, 1, 1, 2, 0),
        (215, 2, 1, 1, 1, 2, 2),
        (217, 2, 2, 1, -1, 1, -1),
        (221, 2, 2, 1, -1, 2, -1),
        (233, 2, 2, 1, 0, 2, 2),
        (237, 2, 2, 1, 1, 1, 1),
        (241, 2, 2, 1, 1, 2, 1),
] 

NUMSK = len(INTEGRALS)

def convert_quant_num(l):
        if l == 0:
                return 's'
        elif l == 1:
                return 'p'
        elif l == 2:
                return 'd'
        else:
              raise ValueError("invalid quantum number for angular momentum")
        
def convert_sk_index(lm_tuple):
        """
        Convert (l1, m1, l2, m2, l3, m3) into a string like 'sspxd1'.
        """
        if (len(lm_tuple)-1) % 2 != 0:
            raise ValueError("Tuple must have pairs of (l, m) quantum numbers.")
    
        # mapping from l to orbital letter
        l_map = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
        # mapping for p orbitals
        p_map = {-1: 'y', 0: 'z', 1: 'x'}
        # mapping for d orbitals (numbered)
        d_map = {-2: '1', -1: '2', 0: '3', 1: '4', 2: '5'}
    
        out = []
        for i in range(0, len(lm_tuple)-1, 2):
            l, m = lm_tuple[i+1], lm_tuple[i+2]
            if l not in l_map:
                raise ValueError(f"Unsupported l={l}")
            l_char = l_map[l]
        
            if l == 0:
                part = 's' + 's'  # always "ss"
            elif l == 1:
                part = 'p' + p_map.get(m, '?')
            elif l == 2:
                part = 'd' + d_map.get(m, '?')
            else:
                part = l_char  # fallback
            out.append(part)
    
        return ''.join(out)


        

def phi3(c1, c2, s1, s2, sk_label): 
    """ Returns the angle-dependent part of the given two-center dipole-integral,
    with c1 and s1 (c2 and s2) the sine and cosine of theta_1 (theta_2)
    for the atom at origin (atom at z=Rz). These expressions are obtained
    by integrating analytically over phi.
    """
    if sk_label == (1, 0, 0, 1, -1, 1, -1):
        return 0.375*s1*s2/np.sqrt(np.pi)
    elif sk_label == (5, 0, 0, 1, -1, 2, -1):
            return 0.375*np.sqrt(5)*s1*s2*c2/np.sqrt(np.pi)
    elif sk_label == (9, 0, 0, 1, 0, 0, 0):
            return 0.25*np.sqrt(3)*c1/np.sqrt(np.pi)
    elif sk_label == (11, 0, 0, 1, 0, 1, 0):
            return 0.75*c1*c2/np.sqrt(np.pi)
    elif sk_label == (15, 0, 0, 1, 0, 2, 0):
            return 0.0625*np.sqrt(15)*(3*(2*c2**2-1) + 1)*c1/np.sqrt(np.pi)
    elif sk_label == (21, 0, 0, 1, 1, 1, 1):
            return 0.375*s1*s2/np.sqrt(np.pi)
    elif sk_label == (25, 0, 0, 1, 1, 2, 1):
            return 0.375*np.sqrt(5)*s1*s2*c2/np.sqrt(np.pi)
    elif sk_label == (27, 1, -1, 1, -1, 0, 0):
            return 0.375*s1**2/np.sqrt(np.pi)
    elif sk_label == (29, 1, -1, 1, -1, 1, 0):
            return 0.375*np.sqrt(3)*s1**2*c2/np.sqrt(np.pi)
    elif sk_label == (33, 1, -1, 1, -1, 2, 0):
            return 0.09375*np.sqrt(5)*(3*(2*c2**2-1) + 1)*s1**2/np.sqrt(np.pi)
    elif sk_label == (35, 1, -1, 1, -1, 2, 2):
            return -0.09375*np.sqrt(15)*s1**2*s2**2/np.sqrt(np.pi)
    elif sk_label == (37, 1, -1, 1, 0, 1, -1):
            return 0.375*np.sqrt(3)*s1*s2*c1/np.sqrt(np.pi)
    elif sk_label == (41, 1, -1, 1, 0, 2, -1):
            return 0.375*np.sqrt(15)*s1*s2*c1*c2/np.sqrt(np.pi)
    elif sk_label == (49, 1, -1, 1, 1, 2, -2):
            return 0.09375*np.sqrt(15)*s1**2*s2**2/np.sqrt(np.pi)
    elif sk_label == (55, 1, 0, 1, -1, 1, -1):
            return 0.375*np.sqrt(3)*s1*s2*c1/np.sqrt(np.pi)
    elif sk_label == (59, 1, 0, 1, -1, 2, -1):
            return 0.375*np.sqrt(15)*s1*s2*c1*c2/np.sqrt(np.pi)
    elif sk_label == (63, 1, 0, 1, 0, 0, 0):
            return 0.75*c1**2/np.sqrt(np.pi)
    elif sk_label == (65, 1, 0, 1, 0, 1, 0):
            return 0.75*np.sqrt(3)*c1**2*c2/np.sqrt(np.pi)
    elif sk_label == (69, 1, 0, 1, 0, 2, 0):
            return 0.1875*np.sqrt(5)*(3*(2*c2**2-1) + 1)*c1**2/np.sqrt(np.pi)
    elif sk_label == (75, 1, 0, 1, 1, 1, 1):
            return 0.375*np.sqrt(3)*s1*s2*c1/np.sqrt(np.pi)
    elif sk_label == (79, 1, 0, 1, 1, 2, 1):
            return 0.375*np.sqrt(15)*s1*s2*c1*c2/np.sqrt(np.pi)
    elif sk_label == (85, 1, 1, 1, -1, 2, -2):
            return 0.09375*np.sqrt(15)*s1**2*s2**2/np.sqrt(np.pi)
    elif sk_label == (93, 1, 1, 1, 0, 1, 1):
            return 0.375*np.sqrt(3)*s1*s2*c1/np.sqrt(np.pi)
    elif sk_label == (97, 1, 1, 1, 0, 2, 1):
            return 0.375*np.sqrt(15)*s1*s2*c1*c2/np.sqrt(np.pi)
    elif sk_label == (99, 1, 1, 1, 1, 0, 0):
            return 0.375*s1**2/np.sqrt(np.pi)
    elif sk_label == (101, 1, 1, 1, 1, 1, 0):
            return 0.375*np.sqrt(3)*s1**2*c2/np.sqrt(np.pi)
    elif sk_label == (105, 1, 1, 1, 1, 2, 0):
            return 0.09375*np.sqrt(5)*(3*(2*c2**2-1) + 1)*s1**2/np.sqrt(np.pi)
    elif sk_label == (107, 1, 1, 1, 1, 2, 2):
            return 0.09375*np.sqrt(15)*s1**2*s2**2/np.sqrt(np.pi)
    elif sk_label == (111, 2, -2, 1, -1, 1, 1):
            return 0.09375*np.sqrt(15)*s1**3*s2/np.sqrt(np.pi)
    elif sk_label == (115, 2, -2, 1, -1, 2, 1):
            return 0.46875*np.sqrt(3)*s1**3*s2*c2/np.sqrt(np.pi)
    elif sk_label == (121, 2, -2, 1, 0, 2, -2):
            return 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi)
    elif sk_label == (127, 2, -2, 1, 1, 1, -1):
            return 0.09375*np.sqrt(15)*s1**3*s2/np.sqrt(np.pi)
    elif sk_label == (131, 2, -2, 1, 1, 2, -1):
            return 0.46875*np.sqrt(3)*s1**3*s2*c2/np.sqrt(np.pi)
    elif sk_label == (135, 2, -1, 1, -1, 0, 0):
            return 0.375*np.sqrt(5)*s1**2*c1/np.sqrt(np.pi)
    elif sk_label == (137, 2, -1, 1, -1, 1, 0):
            return 0.375*np.sqrt(15)*s1**2*c1*c2/np.sqrt(np.pi)
    elif sk_label == (141, 2, -1, 1, -1, 2, 0):
            return 0.46875*(3*(2*c2**2-1) + 1)*s1**2*c1/np.sqrt(np.pi)
    elif sk_label == (143, 2, -1, 1, -1, 2, 2):
            return -0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi)
    elif sk_label == (145, 2, -1, 1, 0, 1, -1):
            return 0.375*np.sqrt(15)*s1*s2*c1**2/np.sqrt(np.pi)
    elif sk_label == (149, 2, -1, 1, 0, 2, -1):
            return 1.875*np.sqrt(3)*s1*s2*c1**2*c2/np.sqrt(np.pi)
    elif sk_label == (157, 2, -1, 1, 1, 2, -2):
            return 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi)
    elif sk_label == (163, 2, 0, 1, -1, 1, -1):
            return 0.09375*np.sqrt(5)*(3*(2*c1**2-1) + 1)*s1*s2/np.sqrt(np.pi)
    elif sk_label == (167, 2, 0, 1, -1, 2, -1):
            return 0.46875*(3*(2*c1**2-1) + 1)*s1*s2*c2/np.sqrt(np.pi)
    elif sk_label == (171, 2, 0, 1, 0, 0, 0):
            return 0.0625*np.sqrt(15)*(3*(2*c1**2-1) + 1)*c1/np.sqrt(np.pi)
    elif sk_label == (173, 2, 0, 1, 0, 1, 0):
            return 0.1875*np.sqrt(5)*(3*(2*c1**2-1) + 1)*c1*c2/np.sqrt(np.pi)
    elif sk_label == (177, 2, 0, 1, 0, 2, 0):
            return 0.078125*np.sqrt(3)*(3*(2*c1**2-1) + 1)*(3*(2*c2**2-1) + 1)*c1/np.sqrt(np.pi)
    elif sk_label == (183, 2, 0, 1, 1, 1, 1):
            return 0.09375*np.sqrt(5)*(3*(2*c1**2-1) + 1)*s1*s2/np.sqrt(np.pi)
    elif sk_label == (187, 2, 0, 1, 1, 2, 1):
            return 0.46875*(3*(2*c1**2-1) + 1)*s1*s2*c2/np.sqrt(np.pi)
    elif sk_label == (193, 2, 1, 1, -1, 2, -2):
            return 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi)
    elif sk_label == (201, 2, 1, 1, 0, 1, 1):
            return 0.375*np.sqrt(15)*s1*s2*c1**2/np.sqrt(np.pi)
    elif sk_label == (205, 2, 1, 1, 0, 2, 1):
            return 1.875*np.sqrt(3)*s1*s2*c1**2*c2/np.sqrt(np.pi)
    elif sk_label == (207, 2, 1, 1, 1, 0, 0):
            return 0.375*np.sqrt(5)*s1**2*c1/np.sqrt(np.pi)
    elif sk_label == (209, 2, 1, 1, 1, 1, 0):
            return 0.375*np.sqrt(15)*s1**2*c1*c2/np.sqrt(np.pi)
    elif sk_label == (213, 2, 1, 1, 1, 2, 0):
            return 0.46875*(3*(2*c2**2-1) + 1)*s1**2*c1/np.sqrt(np.pi)
    elif sk_label == (215, 2, 1, 1, 1, 2, 2):
            return 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi)
    elif sk_label == (217, 2, 2, 1, -1, 1, -1):
            return -0.09375*np.sqrt(15)*s1**3*s2/np.sqrt(np.pi)
    elif sk_label == (221, 2, 2, 1, -1, 2, -1):
            return -0.46875*np.sqrt(3)*s1**3*s2*c2/np.sqrt(np.pi)
    elif sk_label == (233, 2, 2, 1, 0, 2, 2):
            return 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi)
    elif sk_label == (237, 2, 2, 1, 1, 1, 1):
            return 0.09375*np.sqrt(15)*s1**3*s2/np.sqrt(np.pi)
    elif sk_label == (241, 2, 2, 1, 1, 2, 1):
            return 0.46875*np.sqrt(3)*s1**3*s2*c2/np.sqrt(np.pi)

def select_integrals(e1, e2):
    """ Return list of integrals (integral, nl1, nl2)
    to be done for element pair e1, e2. """
    selected = []
    for ival1, valence1 in enumerate(e1.basis_sets):
        for ival2, valence2 in enumerate(e2.basis_sets):
            for sk_label in INTEGRALS:
                nl1, nl2 = select_subshells(valence1, valence2, sk_label)
                if nl1 is not None and nl2 is not None:
                    selected.append((sk_label, nl1, nl2))
    return selected

def select_subshells(val1, val2, sk_label):
    """
    Select subshells from given valence sets to calculate given
    Slater-Koster integral.

    Parameters
    ----------
    val1, val2 : list of str
        Valence subshell sets (e.g. ['2s', '2p'], ['4s', '3d']).
    integral : str
        Slater-Koster integral label (e.g. 'pzpzpz').

    Returns
    -------
    nl1, nl2 : str
        Matching subshell pair (e.g. ('2s', '3d') in this example).
    """
    nl1 = None
    for nl in val1:
        if nl[1] == convert_quant_num(sk_label[1]):
            nl1 = nl

    nl2 = None
    for nl in val2:
        if nl[1] == convert_quant_num(sk_label[5]):
            nl2 = nl

    return nl1, nl2
    
    
def print_integral_overview(e1, e2, selected, file):
    """ Prints an overview of the selected Slater-Koster integrals. """
    for bas1 in range(len(e1.basis_sets)):
        for bas2 in range(len(e2.basis_sets)):
            sym1 = e1.get_symbol() + '+'*bas1
            sym2 = e2.get_symbol() + '+'*bas2
            print('Integrals for %s-%s pair:' % (sym1, sym2), end=' ',
                  file=file)
            for integral, nl1, nl2 in selected:
                if e1.get_basis_set_index(nl1) == bas1 and \
                   e2.get_basis_set_index(nl2) == bas2:
                    print('_'.join([nl1, nl2, convert_sk_index(integral)]), end=' ', file=file)
            print(file=file, flush=True)
    return

def tail_smoothening(x, y_in, eps_inner=1e-8, eps_outer=1e-16, window_size=5):
    """ Smoothens the tail for the given function y(x).

    Parameters
    ----------
    x : np.array
        Array with grid points (strictly increasing).
    y_in : np.array
        Array with function values.
    eps_inner : float, optional
        Inner threshold. Tail values with magnitudes between this value and
        the outer threshold are subjected to moving window averaging to
        reduce noise.
    eps_outer : float, optional
        Outer threshold. Tail values with magnitudes below this value
        are set to zero.
    window_size : int, optional
        Moving average window size (odd integers only).

    Returns
    -------
    y_out : np.array
        Array with function values with a smoothed tail.
    """
    assert window_size % 2 == 1, 'Window size needs to be odd.'

    y_out = np.copy(y_in)
    N = len(y_out)

    if np.all(abs(y_in) < eps_outer):
        return y_out

    Nzero = 0
    izero = -1
    for izero in range(N-1, 1, -1):
        if abs(y_out[izero]) < eps_outer:
            Nzero += 1
        else:
            break

    y_out[izero+1:] = 0.

    Nsmall = 0
    for ismall in range(izero, 1, -1):
        if abs(y_out[ismall]) < eps_inner:
            Nsmall += 1
        else:
            break
    else:
        ismall -= 1

    if Nsmall > 0:
        tail = np.empty(Nsmall-1)
        half = (window_size - 1) // 2
        for j, i in enumerate(range(ismall+1, izero)):
            tail[j] = np.mean(y_out[i-half:i+half+1])

        y_out[ismall+1:izero] = tail

    return y_out
    

def write_skf(handle, Rgrid, table, has_diagonal_data, is_extended, eigval,
              hubval, occup, spe, mass, has_offdiagonal_data, offdiag_H,
              offdiag_S):
    """
    Writes a parameter file in '.skf' format.

    Parameters
    ----------
    handle : file handle
        Handle of an open file.
    Rgrid : list or array
        Lists with interatomic distances.
    table : nd.ndarray
        Two-dimensional array with the Slater-Koster table.

    Other parameters
    ----------------
    See Offsite2cTable.write()
    """
    # TODO find out what all the other quantities are, that do not come from table
#     print(table)
    assert not (has_diagonal_data and has_offdiagonal_data)

    if is_extended:
        print('@', file=handle)

    grid_dist = Rgrid[1] - Rgrid[0]
    grid_npts, numint = np.shape(table)
    assert (numint % NUMSK) == 0
    nzeros = int(np.round(Rgrid[0] / grid_dist)) - 1
    assert nzeros >= 0
    print("%.12f, %d" % (grid_dist, grid_npts + nzeros), file=handle)

    if has_diagonal_data or has_offdiagonal_data:
        if has_diagonal_data:
            prefixes, dicts = ['E', 'U', 'f'], [eigval, hubval, occup]
            fields = ['E_f', 'E_d', 'E_p', 'E_s', 'SPE', 'U_f', 'U_d',
                      'U_p', 'U_s', 'f_f', 'f_d', 'f_p', 'f_s']
            labels = {'SPE': spe}
        elif has_offdiagonal_data:
            prefixes, dicts = ['H', 'S'], [offdiag_H, offdiag_S]
            fields = ['H_f', 'H_d', 'H_p', 'H_s',
                      'S_f', 'S_d', 'S_p', 'S_s']
            labels = {}

        if not is_extended:
            fields = [field for field in fields if field[-1] != 'f']

        for prefix, d in zip(prefixes, dicts):
            for l in ['s', 'p', 'd', 'f']:
                if l in d:
                    key = '%s_%s' % (prefix, l)
                    labels[key] = d[l]

        line = ' '.join(fields)
        for field in fields:
            val = labels[field] if field in labels else 0
            s = '%d' % val if isinstance(val, int) else '%.6f' % val
            line = line.replace(field, s)

        print(line, file=handle)

    print("%.3f, 19*0.0" % mass, file=handle) # TODO change number of columns

    # Table containing the Slater-Koster integrals
    numtab = numint // NUMSK
    assert numtab > 0

    if is_extended:
        indices = list(range(numtab*NUMSK))
    else:
        selected = [INTEGRALS.index(name) for name in INTEGRALS
                    if 'f' not in name[:2]]
        indices = []
        for itab in range(numtab):
            indices.extend([itab*NUMSK+j for j in selected])

#     for i in range(nzeros):
#       print('%d*0.0,' % len(indices), file=handle)

    for i in range(grid_npts):
        line = ''
        for j in indices:
                line += '{0: 1.12e}  '.format(table[i, j])
        print(line, file=handle)
    
    