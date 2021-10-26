#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import numpy as np
from scipy.interpolate import CubicSpline
try:
    from pylibxc import LibXCFunctional
    from pylibxc.version import __version__ as pylibxc_version
    has_pylibxc = True
    assert int(pylibxc_version[0]) >= 5, \
           'PyLibXC >= v5 is needed (found {0})'.format(pylibxc_version)
except ImportError:
    print('Warning -- could not load LibXC')
    has_pylibxc = False


class LibXC:
    def __init__(self, xcname):
        """ Interface to PyLibXC.

        Parameters
        ----------
        xcname : str
            Combination of LibXC functional names,
            e.g. 'GGA_X_PBE+GGA_C_PBE'.
        """
        assert has_pylibxc, 'Using XC other than LDA requires PyLibXC!'

        self.xcname = xcname
        self.names = self.xcname.split('+')
        self.functionals = []
        self.types = []
        self.add_gradient_corrections = False

        for name in self.names:
            try:
                self.functionals.append(LibXCFunctional(name, 'unpolarized'))
            except KeyError as err:
                print('KeyError:', err)
                print('>>> Bad XC name. For valid LibXC functional names, see')
                print('>>> https://www.tddft.org/programs/libxc/functionals/')
                raise

            if 'mgga' in name.lower():
                raise ValueError('Meta-GGA functionals not allowed:', name)
            if 'lda' in name.lower():
                self.types.append('LDA')
            elif 'gga' in name.lower():
                self.types.append('GGA')
                self.add_gradient_corrections = True
            else:
                raise ValueError('XC func %s is not LDA or GGA' % name)

    def compute_exc(self, rho, sigma=None):
        """ Returns the exchange-correlation energy density.

        Parameters
        ----------
        rho : np.ndarray
            Electron density.
        sigma : np.ndarray, optional
            Norm of the electron density gradient
            (only needed if GGA functionals are involved).

        Returns
        -------
        exc: np.ndarray
            Exchange-correlation energy density.
        """
        inp = {'rho': rho, 'sigma': sigma}
        exc = np.zeros_like(rho)

        for i, func in enumerate(self.functionals):
            out = func.compute(inp, do_exc=True, do_vxc=False)
            exc += out['zk'][:, 0]

        return exc

    def compute_vxc(self, rho, sigma=None, fxc=False):
        """ Returns a dictionary with the arrays needed to compute
        the XC potential.

        Parameters
        ----------
        rho : np.ndarray
            Electron density.
        sigma : np.ndarray, optional
            Norm of the electron density gradient
            (only needed if GGA functionals are involved).
        fxc : bool, optional
            Whether to also compute selected second derivatives
            with respect to the electron density.

        Returns
        -------
        results: dict of np.ndarray
            Dictionary with 'vrho' and 'vsigma' and (if fxc=True)
            'v2rhosigma' and 'v2sigma2'.
        """
        inp = {'rho': rho, 'sigma': sigma}
        results = {'vrho': np.zeros_like(rho)}
        if self.add_gradient_corrections:
            grad_keys = ['vsigma']
            if fxc:
                grad_keys.extend(['v2rhosigma', 'v2sigma2'])

            for key in grad_keys:
                results[key] = np.copy(results['vrho'])

        for i, func in enumerate(self.functionals):
            is_gga = self.types[i] == 'GGA'
            do_fxc = fxc and is_gga
            out = func.compute(inp, do_exc=False, do_vxc=True, do_fxc=do_fxc)

            results['vrho'] += out['vrho'][:, 0]
            if self.add_gradient_corrections and is_gga:
                for key in grad_keys:
                    results[key] += out[key][:, 0]

        return results

    def compute_all(self, rho, sigma=None):
        """ Returns the results of self.compute_vxc() together
        with a 'zk' entry with the result from self.compute_exc().
        """
        results = self.compute_vxc(rho, sigma=sigma)
        results['zk'] = self.compute_exc(rho, sigma=sigma)
        return results

    def evaluate(self, rho, gd):
        """ Returns the XC energy density and the XC potential.

        Parameters
        ----------
        rho : np.ndarray
            Electron density.
        gd : Grid
            An object that can carry out gradient and divergence
            operations on a grid-based array.

        Returns
        -------
        exc : np.ndarray
            XC energy density.
        vxc : np.ndarray
            XC potential.
        """
        grad = gd.gradient(rho)
        sigma = grad ** 2
        out = self.compute_all(rho, sigma)
        exc = out['zk']
        vxc = out['vrho']
        if self.add_gradient_corrections:
            assert out['vsigma'] is not None
            vxc += -2 * gd.divergence(out['vsigma'] * grad)
        return exc, vxc


class XC_PW92:
    def __init__(self):
        """ The Perdew-Wang 1992 LDA exchange-correlation functional. """
        self.small = 1e-90
        self.a1 = 0.21370
        self.c0 = 0.031091
        self.c1 = 0.046644
        self.b1 = 1.0 / 2.0 / self.c0 * np.exp(-self.c1 / 2.0 / self.c0)
        self.b2 = 2 * self.c0 * self.b1 ** 2
        self.b3 = 1.6382
        self.b4 = 0.49294
        self.add_gradient_corrections = False

    def exc(self, n, der=0):
        """ Exchange-correlation with electron density n. """
        is_scalar = type(n) == np.float64
        n = np.array([n]) if is_scalar else n
        indices = n < self.small
        n[indices] = self.small
        e = self.e_x(n, der=der) + self.e_corr(n, der=der)
        e[indices] = 0.
        if is_scalar:
            e = e[0]
        return e

    def e_x(self, n, der=0):
        """ Exchange. """
        if der == 0:
            return -3. / 4 * (3 * n / np.pi) ** (1. / 3)
        elif der == 1:
            return -3. / (4 * np.pi) * (3 * n / np.pi) ** (-2. / 3)

    def e_corr(self, n, der=0):
        """ Correlation energy. """
        rs = (3. / (4 * np.pi * n)) ** (1. / 3)
        aux = 2 * self.c0
        aux *= self.b1 * np.sqrt(rs) + self.b2 * rs + self.b3 * rs ** (3. / 2) + self.b4 * rs ** 2
        if der == 0:
            return -2 * self.c0 * (1 + self.a1 * rs) * np.log(1 + aux ** -1)
        elif der == 1:
            return (-2 * self.c0 * self.a1 * np.log(1 + aux ** -1) \
                    -2 * self.c0 * (1 + self.a1 * rs) * (1 + aux ** -1) ** -1 * (-aux ** -2) \
                   * 2 * self.c0 * (self.b1 / (2 * np.sqrt(rs)) + self.b2 + 3 * self.b3 * np.sqrt(rs) / 2 \
                   + 2 * self.b4 * rs)) * (-(4 * np.pi * n ** 2 * rs ** 2) ** -1)

    def vxc(self, n):
        """ Exchange-correlation potential (functional derivative of exc). """
        indices = n < self.small
        n[indices] = self.small
        v = self.exc(n) + n * self.exc(n, der=1)
        v[indices] = 0.
        return v

    def evaluate(self, n, *args, **kwargs):
        """ Return the XC energy and potential

            n: array-like, the electron density
        """
        return self.exc(n), self.vxc(n)


class LDA_Spline(CubicSpline):
    """
    Class for representing LDA functionals using a (cubic) spline.

    Parameters
    ----------
    func : function
        Function for evaluating the functional when constructing
        the spline.
    rho_min, rho_max : float, optional
        Lower and upper bounds for the electron density. Zero will
        be returned for electron densities outside these bounds.
    num : int, optional
        The number of grid points to use in the spline.
    """
    def __init__(self, func, rho_min=1e-12, rho_max=1e12, num=1000):
        self.rho_min = rho_min
        self.rho_max = rho_max
        x = np.exp(np.linspace(np.log(rho_min), np.log(rho_max), num=num,
                               endpoint=True))
        y = func(x)
        CubicSpline.__init__(self, x, y, bc_type=('clamped', 'natural'),
                             extrapolate=False)
        self.add_gradient_corrections = False

    def __call__(self, rho, **kwargs):
        """
        Evaluates the exchange-correlation functional.

        Parameters
        ----------
        rho : float or np.ndarray
            The electron density.

        Returns
        -------
        y : float or np.ndarray
            The interpolated function value.
        """
        y = CubicSpline.__call__(self, rho, **kwargs)
        np.nan_to_num(y, copy=False)
        return y


class EXC_PW92_Spline(LDA_Spline):
    """
    The Perdew-Wang 1992 LDA exchange-correlation energy density
    represented by a cubic spline.
    """
    def __init__(self, **kwargs):
        LDA_Spline.__init__(self, XC_PW92().exc, **kwargs)


class VXC_PW92_Spline(LDA_Spline):
    """
    The Perdew-Wang 1992 LDA exchange-correlation potential
    represented by a cubic spline.
    """
    def __init__(self, **kwargs):
        LDA_Spline.__init__(self, XC_PW92().vxc, **kwargs)
