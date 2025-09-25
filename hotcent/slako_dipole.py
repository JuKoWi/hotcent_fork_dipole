import numpy as np

"""nonvanishing SlaKo integrals for dipole elements named in the form
Y1*Y1*Y2
"""
INTEGRALS = ['pzpzpz', 'pzpzss', 'sspzss', 'sspzpz', 'sspxpx', 'pxpxss', 'pxpxpz', 'pxpzpx' ] 

NUMSK = len(INTEGRALS)

def phi3(c1, c2, s1, s2, sk_label): #TODO write integrals including d
    """ Returns the angle-dependent part of the given two-center dipole-integral,
    with c1 and s1 (c2 and s2) the sine and cosine of theta_1 (theta_2)
    for the atom at origin (atom at z=Rz). These expressions are obtained
    by integrating analytically over phi.
    """
    if sk_label == 'pzpzpz':
        return 3/4 * np.sqrt(3/np.pi) * c1**2 * c2
    elif sk_label == 'pzpzss':
        return 3/( 4 * np.sqrt(np.pi)) * c1**2
    elif sk_label == 'sspzss':
        return 1/4 * np.sqrt(3/np.pi) * c1
    elif sk_label == 'sspzpz':
        return 3/(4 * np.sqrt(np.pi)) * c1 * c2
    elif sk_label == 'sspxpx':
        return 3/(8 * np.sqrt(np.pi)) * s1 * s2
    elif sk_label == 'pxpxss':
        return 3/(8 * np.sqrt(np.pi)) * s1**2 
    elif sk_label == 'pxpxpz':
        return 3/8 * np.sqrt(3/np.pi) * s1**2 * c2
    elif sk_label == 'pxpzpx': 
        return 3/8 * np.sqrt(3/np.pi) * s1 * c1 * s2 

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
        if nl[1] == sk_label[0]:
            nl1 = nl

    nl2 = None
    for nl in val2:
        if nl[1] == sk_label[4]:
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
                    print('_'.join([nl1, nl2, integral]), end=' ', file=file)
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

    print("%.3f, 19*0.0" % mass, file=handle)

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

    for i in range(nzeros):
        print('%d*0.0,' % len(indices), file=handle)

    for i in range(grid_npts):
        line = ''
        num_zero = 0
        zero_str = ''

        for j in indices:
            if table[i, j] == 0:
                num_zero += 1
                zero_str = str(num_zero) + '*0.0 ' # WTF, which machine can read this without getting confused?
            else:
                num_zero = 0
                line += zero_str
                zero_str = ''
                line += '{0: 1.12e}  '.format(table[i, j])

        if zero_str != '':
            line += zero_str

        print(line, file=handle)
    
    
