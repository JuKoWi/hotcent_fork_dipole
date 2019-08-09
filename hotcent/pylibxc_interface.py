""" GPAW-style GGA XC class but with the kernel
being evaluated via the pylibxc module shipped
with LibXC (at least in recent versions).

This allows to perform atomic DFT calculations with
GPAW using LibXC-implemented GGA functionals, but
without needing to link GPAW to LibXC (mostly for
when GPAW cannot be compiled with the development
version of LibXC).

Currently only LDA- and GGA-type functionals can
be used, i.e. no meta-GGAs.

Note: one tiny change in gpaw/atom/all_electron.py
is needed for this to work. Around line 191,
replace this:
         self.xc = XC(self.xcname)
by this:
         if type(self.xcname) == str:
             self.xc = XC(self.xcname)
         else:
             self.xc = self.xcname


Example usage for the BLYP functional:

>>> from hotcent.pylibxc_interface import PyLibXCGGA
>>> from hotcent.atom_gpaw import GPAWAE

>>> xc = PyLibXCGGA(['gga_x_b88', 'gga_c_lyp'])
>>> ae = GPAWAE(..., xc=xc)
>>> ae.run()

In case your GPAW-LibXC combination works fine,
you can get away with the regular LibXC syntax
in GPAW:

>>> xc =  'XC_GGA_X_B88+XC_GGA_C_LYP'
"""

from __future__ import print_function
import numpy as np
from gpaw.xc.functional import XCFunctional
from gpaw.xc.gga import GGA
try:
    from pylibxc import LibXCFunctional
except ImportError:
    print('Could not find pylibxc -- make sure LibXC has been compiled'
          'and the $PYTHONPATH has been set accordingly.')
    raise


class PyLibXCKernel(XCFunctional):
    def __init__(self, names):
        self.names = names
        self.functionals = [LibXCFunctional(name, 'unpolarized')
                            for name in self.names]
        self.types = []
        for name in self.names:
            if 'mgga' in name.lower():
                raise ValueError('Meta-GGA functionals not allowed:', name)
            if 'lda' in name.lower():
                self.types.append('LDA')
            elif 'gga' in name.lower():
                self.types.append('GGA')
            else:
                raise ValueError('XC func is not LDA or GGA:', name)

        XCFunctional.__init__(self, '+'.join(self.names),
                              '+'.join(self.types))

    def calculate(self, e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg):
        inp = {'rho': n_sg, 'sigma': sigma_xg}
        for i, func in enumerate(self.functionals):
            out = func.compute(inp)
            e_g += out['zk'][0]
            dedn_sg += out['vrho']
            if self.types[i] == 'GGA':
                dedsigma_xg += out['vsigma']


class PyLibXCGGA(GGA):
    def __init__(self, func_names, stencil=2):
        self.kernel = PyLibXCKernel(func_names)
        self.stencil_range = stencil
        self.type = self.kernel.type
        self.name = self.kernel.name

    def __repr__(self):
        return self.kernel.tostring()
