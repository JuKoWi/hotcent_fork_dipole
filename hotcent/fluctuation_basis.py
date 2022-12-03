#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
from hotcent.interpolation import build_interpolator


class AuxiliaryBasis:
    """
    Class for handling sets of auxiliary basis functions.
    """
    def __init__(self):
        self.Anlg = {}
        self.Anl_fct = {}

    def build_basis_functions(self, el, lmax=4):
        """
        Builds a set of radial auxiliary basis functions associated
        with the radial functions of the 'regular' basis set.

        Parameters
        ----------
        el : AtomicBase instance
            Provides the 'regular' basis set and an integration grid.
        lmax : int, optional
            Maximum angular momentum of the auxiliary basis functions
            (default: 4).
        """
        for valence in el.basis_sets:
            for nl in valence:
                rc = el.rcutnl[nl] if nl in el.rcutnl else None

                for l in range(lmax+1):
                    A = el.Rnlg[nl]**2
                    self.Anlg[(nl, l)] = A
                    self.Anl_fct[(nl, l)] = build_interpolator(el.rgrid, A, rc)
        return

    def __call__(self, r, nl, l, der=0):
        """
        Evaluates the selected auxiliary basis function or its derivatives.

        Parameters
        ----------
        r : float or np.ndarray
            Distance from the basis function origin.
        nl : str
            Subshell label defining the radial function.
        l : int
            Angular momentum (also defining the radial function).
        der : int, optional
            Order of the derivative (default: 0).

        Returns
        -------
        A : float or np.ndarray
            Function value(s) or derivative(s).
        """
        A = self.Anl_fct[(nl, l)](r, der=der)
        return A