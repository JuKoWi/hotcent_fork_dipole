import numpy as np

"""nonvanishing SlaKo integrals for dipole elements named in the form
Y1*Y1*Y2
unified format for the label: 1., 3. and 5. letter give nl, others just for distinguishing
(x,y,z) for p, (1,2,3,4,5) for d
"""
INTEGRALS = [
    'sspxpx',
    'sspxd2',
    'sspypy',
    'sspyd4',
    'sspzss',
    'sspzpz',
    'sspzd3',
    'pxpxss',
    'pxpxpz',
    'pxpxd3',
    'pxpxd5',
    'pxpyd1',
    'pxpzpx',
    'pxpzd2',
    'pypxd1',
    'pypyss',
    'pypypz',
    'pypyd3',
    'pypyd5',
    'pypzpy',
    'pypzd4',
    'pzpxpx',
    'pzpxd2',
    'pzpypy',
    'pzpyd4',
    'pzpzss',
    'pzpzpz',
    'pzpzd3',
    'd1pxpy',
    'd1pxd4',
    'd1pypx',
    'd1pyd2',
    'd1pzd1',
    'd2pxss',
    'd2pxpz',
    'd2pxd3',
    'd2pxd5',
    'd2pyd1',
    'd2pzpx',
    'd2pzd2',
    'd3pxpx',
    'd3pxd2',
    'd3pypy',
    'd3pyd4',
    'd3pzss',
    'd3pzpz',
    'd3pzd3',
    'd4pxd1',
    'd4pyss',
    'd4pypz',
    'd4pyd3',
    'd4pyd5',
    'd4pzpy',
    'd4pzd4',
    'd5pxpx',
    'd5pxd2',
    'd5pypy',
    'd5pyd4',
    'd5pzd5'
] 

NUMSK = len(INTEGRALS)

def phi3(c1, c2, s1, s2, sk_label): 
    """ Returns the angle-dependent part of the given two-center dipole-integral,
    with c1 and s1 (c2 and s2) the sine and cosine of theta_1 (theta_2)
    for the atom at origin (atom at z=Rz). These expressions are obtained
    by integrating analytically over phi.
    """
    if sk_label == 'sspxpx':
            return 0.375*s1*s2/np.sqrt(np.pi)
    elif sk_label == 'sspxd2':
            return 0.375*np.sqrt(5)*s1*s2*c2/np.sqrt(np.pi)
    elif sk_label == 'sspypy':
            return 0.375*s1*s2/np.sqrt(np.pi)
    elif sk_label == 'sspyd4':
            return 0.375*np.sqrt(5)*s1*s2*c2/np.sqrt(np.pi)
    elif sk_label == 'sspzss':
            return 0.25*np.sqrt(3)*c1/np.sqrt(np.pi)
    elif sk_label == 'sspzpz':
            return 0.75*c1*c2/np.sqrt(np.pi)
    elif sk_label == 'sspzd3':
            return 0.0625*np.sqrt(15)*(3*2*c2**2-1 + 1)*c1/np.sqrt(np.pi)
    elif sk_label == 'pxpxss':
            return 0.375*s1**2/np.sqrt(np.pi)
    elif sk_label == 'pxpxpz':
            return 0.375*np.sqrt(3)*s1**2*c2/np.sqrt(np.pi)
    elif sk_label == 'pxpxd3':
            return 0.09375*np.sqrt(5)*(3*2*c2**2-1 + 1)*s1**2/np.sqrt(np.pi)
    elif sk_label == 'pxpxd5':
            return -0.09375*np.sqrt(15)*s1**2*s2**2/np.sqrt(np.pi)
    elif sk_label == 'pxpyd1':
            return 0.09375*np.sqrt(15)*s1**2*s2**2/np.sqrt(np.pi)
    elif sk_label == 'pxpzpx':
            return 0.375*np.sqrt(3)*s1*s2*c1/np.sqrt(np.pi)
    elif sk_label == 'pxpzd2':
            return 0.375*np.sqrt(15)*s1*s2*c1*c2/np.sqrt(np.pi)
    elif sk_label == 'pypxd1':
            return 0.09375*np.sqrt(15)*s1**2*s2**2/np.sqrt(np.pi)
    elif sk_label == 'pypyss':
            return 0.375*s1**2/np.sqrt(np.pi)
    elif sk_label == 'pypypz':
            return 0.375*np.sqrt(3)*s1**2*c2/np.sqrt(np.pi)
    elif sk_label == 'pypyd3':
            return 0.09375*np.sqrt(5)*(3*2*c2**2-1 + 1)*s1**2/np.sqrt(np.pi)
    elif sk_label == 'pypyd5':
            return 0.09375*np.sqrt(15)*s1**2*s2**2/np.sqrt(np.pi)
    elif sk_label == 'pypzpy':
            return 0.375*np.sqrt(3)*s1*s2*c1/np.sqrt(np.pi)
    elif sk_label == 'pypzd4':
            return 0.375*np.sqrt(15)*s1*s2*c1*c2/np.sqrt(np.pi)
    elif sk_label == 'pzpxpx':
            return 0.375*np.sqrt(3)*s1*s2*c1/np.sqrt(np.pi)
    elif sk_label == 'pzpxd2':
            return 0.375*np.sqrt(15)*s1*s2*c1*c2/np.sqrt(np.pi)
    elif sk_label == 'pzpypy':
            return 0.375*np.sqrt(3)*s1*s2*c1/np.sqrt(np.pi)
    elif sk_label == 'pzpyd4':
            return 0.375*np.sqrt(15)*s1*s2*c1*c2/np.sqrt(np.pi)
    elif sk_label == 'pzpzss':
            return 0.75*c1**2/np.sqrt(np.pi)
    elif sk_label == 'pzpzpz':
            return 0.75*np.sqrt(3)*c1**2*c2/np.sqrt(np.pi)
    elif sk_label == 'pzpzd3':
            return 0.1875*np.sqrt(5)*(3*2*c2**2-1 + 1)*c1**2/np.sqrt(np.pi)
    elif sk_label == 'd1pxpy':
            return 0.09375*np.sqrt(15)*s1**3*s2/np.sqrt(np.pi)
    elif sk_label == 'd1pxd4':
            return 0.46875*np.sqrt(3)*s1**3*s2*c2/np.sqrt(np.pi)
    elif sk_label == 'd1pypx':
            return 0.09375*np.sqrt(15)*s1**3*s2/np.sqrt(np.pi)
    elif sk_label == 'd1pyd2':
            return 0.46875*np.sqrt(3)*s1**3*s2*c2/np.sqrt(np.pi)
    elif sk_label == 'd1pzd1':
            return 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi)
    elif sk_label == 'd2pxss':
            return 0.375*np.sqrt(5)*s1**2*c1/np.sqrt(np.pi)
    elif sk_label == 'd2pxpz':
            return 0.375*np.sqrt(15)*s1**2*c1*c2/np.sqrt(np.pi)
    elif sk_label == 'd2pxd3':
            return 0.46875*(3*2*c2**2-1 + 1)*s1**2*c1/np.sqrt(np.pi)
    elif sk_label == 'd2pxd5':
            return -0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi)
    elif sk_label == 'd2pyd1':
            return 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi)
    elif sk_label == 'd2pzpx':
            return 0.375*np.sqrt(15)*s1*s2*c1**2/np.sqrt(np.pi)
    elif sk_label == 'd2pzd2':
            return 1.875*np.sqrt(3)*s1*s2*c1**2*c2/np.sqrt(np.pi)
    elif sk_label == 'd3pxpx':
            return 0.09375*np.sqrt(5)*(3*2*c1**2-1 + 1)*s1*s2/np.sqrt(np.pi)
    elif sk_label == 'd3pxd2':
            return 0.46875*(3*2*c1**2-1 + 1)*s1*s2*c2/np.sqrt(np.pi)
    elif sk_label == 'd3pypy':
            return 0.09375*np.sqrt(5)*(3*2*c1**2-1 + 1)*s1*s2/np.sqrt(np.pi)
    elif sk_label == 'd3pyd4':
            return 0.46875*(3*2*c1**2-1 + 1)*s1*s2*c2/np.sqrt(np.pi)
    elif sk_label == 'd3pzss':
            return 0.0625*np.sqrt(15)*(3*2*c1**2-1 + 1)*c1/np.sqrt(np.pi)
    elif sk_label == 'd3pzpz':
            return 0.1875*np.sqrt(5)*(3*2*c1**2-1 + 1)*c1*c2/np.sqrt(np.pi)
    elif sk_label == 'd3pzd3':
            return 0.078125*np.sqrt(3)*(3*2*c1**2-1 + 1)*(3*2*c2**2-1 + 1)*c1/np.sqrt(np.pi)
    elif sk_label == 'd4pxd1':
            return 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi)
    elif sk_label == 'd4pyss':
            return 0.375*np.sqrt(5)*s1**2*c1/np.sqrt(np.pi)
    elif sk_label == 'd4pypz':
            return 0.375*np.sqrt(15)*s1**2*c1*c2/np.sqrt(np.pi)
    elif sk_label == 'd4pyd3':
            return 0.46875*(3*2*c2**2-1 + 1)*s1**2*c1/np.sqrt(np.pi)
    elif sk_label == 'd4pyd5':
            return 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi)
    elif sk_label == 'd4pzpy':
            return 0.375*np.sqrt(15)*s1*s2*c1**2/np.sqrt(np.pi)
    elif sk_label == 'd4pzd4':
            return 1.875*np.sqrt(3)*s1*s2*c1**2*c2/np.sqrt(np.pi)
    elif sk_label == 'd5pxpx':
            return -0.09375*np.sqrt(15)*s1**3*s2/np.sqrt(np.pi)
    elif sk_label == 'd5pxd2':
            return -0.46875*np.sqrt(3)*s1**3*s2*c2/np.sqrt(np.pi)
    elif sk_label == 'd5pypy':
            return 0.09375*np.sqrt(15)*s1**3*s2/np.sqrt(np.pi)
    elif sk_label == 'd5pyd4':
            return 0.46875*np.sqrt(3)*s1**3*s2*c2/np.sqrt(np.pi)
    elif sk_label == 'd5pzd5':
            return 0.46875*np.sqrt(3)*s1**2*s2**2*c1/np.sqrt(np.pi)

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
    
    
