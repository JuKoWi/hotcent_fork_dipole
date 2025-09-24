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

    

    
    
