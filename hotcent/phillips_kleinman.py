#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
from ase.data import atomic_numbers
from hotcent.separable_pseudopotential import SeparablePP


class PhillipsKleinmanPP(SeparablePP):
    """
    Class representing a Phillips-Kleinman(-like) pseudopotential
    (doi:10.1103/PhysRev.116.287).

    Parameters
    ----------
    symbol : str
        The chemical symbol.
    verbose : bool, optional
        Verbosity flag (default: False).
    """
    def __init__(self, symbol, verbose=False):
        SeparablePP.__init__(self, verbose=verbose)
        self.symbol = symbol
        self.Z = atomic_numbers[self.symbol]

    def all_zero_onsite_overlaps(self):
        """ Returns whether all on-site overlaps of the (valence)
        states with the PP projectors are zero.
        """
        return True

    def local_potential(self, r):
        """ Returns the local part of the pseudopotential at r. """
        return -self.Z / r

    def build_projectors(self, e3):
        """ Build the pseudopotential projectors and the
        associated energies.
        """
        assert e3.get_symbol() == self.symbol
        self.projectors = {}
        self.energies = {}

        for n3, l3, nl3 in e3.list_states():
            if nl3 not in e3.valence:
                self.projectors[nl3] = e3.construct_wfn_interpolator(e3.rgrid,
                                                            e3.Rnlg[nl3], nl3)
                self.energies[nl3] = -e3.get_eigenvalue(nl3)
                self.overlap_onsite[nl3] = 0.
