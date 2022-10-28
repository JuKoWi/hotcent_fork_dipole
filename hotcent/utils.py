#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
from ase.data import chemical_symbols


def verify_chemical_symbols(*symbols):
    """
    Checks whether the given chemical symbols correspond to known
    elements.

    Parameters
    ----------
    symbols : str
        Chemical symbol(s).
    """
    for symbol in symbols:
        assert symbol in chemical_symbols and symbol != 'X', \
               'Unknown element: {0}'.format(symbol)
    return
