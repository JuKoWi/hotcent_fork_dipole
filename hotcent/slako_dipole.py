import numpy as np

"""nonvanishing SlaKo integrals for dipole elements named in the form
Y-p_n-Y-symmetry
"""
INTEGRALS = ['ppps', 'ppss', 'pppp', 'spss', 'spps'] 

def phi3(c1, c2, s1, s2, integral):
    """ Returns the angle-dependent part of the given two-center integral,
    with c1 and s1 (c2 and s2) the sine and cosine of theta_1 (theta_2)
    for the atom at origin (atom at z=Rz). These expressions are obtained
    by integrating analytically over phi.
    """
    if integral == 'ppps':
        return 3/4 * np.sqrt(3/np.pi) * c1**2 * c2
    elif integral == 'ppss':
        return 3/( 4 * np.sqrt(np.pi)) * c1**2
    
    
