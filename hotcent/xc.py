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
    def __init__(self, xcname, spin_polarized=False):
        """ Interface to PyLibXC.

        Parameters
        ----------
        xcname : str
            Combination of LibXC functional names,
            e.g. 'GGA_X_PBE+GGA_C_PBE'.
        spin_polarized : bool, optional
            Whether to select the spin-polarized version
            of the functionals (default: False).
        """
        assert has_pylibxc, 'Using XC other than LDA requires PyLibXC!'

        self.xcname = xcname
        self.names = self.xcname.split('+')
        self.functionals = []
        self.types = []
        self.add_gradient_corrections = False

        for name in self.names:
            try:
                spin = 'polarized' if spin_polarized else 'unpolarized'
                func = LibXCFunctional(name, spin)
                self.functionals.append(func)
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

    def compute_vxc_polarized(self, rho_up, rho_down, sigma_up=None,
                              sigma_updown=None, sigma_down=None):
        """
        Returns a dictionary with the arrays needed to compute
        the spin-resolved XC potentials.

        Parameters
        ----------
        rho_up : np.ndarray
            Electron density for the 'up' spin channel.
        rho_down : np.ndarray
            Electron density for the 'down' spin channel.
        sigma_up : np.ndarray, optional
            Norm of the electron density gradient for the 'up' spin
            channel (only needed if GGA functionals are involved).
        sigma_updown : np.ndarray, optional
            Dot product of the density gradients for the 'up'
            and 'down' spin channels (only needed if GGA functionals
            are involved).
        sigma_down : np.ndarray, optional
            Norm of the electron density gradient for the 'down' spin
            channel (only needed if GGA functionals are involved).

        Returns
        -------
        results: dict of np.ndarray
            Dictionary with 'vrho_up' and 'vrho_down' and also
            (if GGA functionals are involved), 'vsigma_up',
            'vsigma_updown' and 'vsigma_down'.
        """
        rho = np.array([rho_up, rho_down])
        sigma = np.array([sigma_up, sigma_updown, sigma_down])
        inp = {
            'rho': np.ascontiguousarray(rho.T),
            'sigma': np.ascontiguousarray(sigma.T),
        }

        N = len(rho_up)
        results = {}

        for suffix in ['up', 'down']:
            results['vrho_' + suffix] = np.zeros(N)

        if self.add_gradient_corrections:
            grad_keys = ['vsigma']
            for key in grad_keys:
                for suffix in ['up', 'updown', 'down']:
                    results[key + '_' + suffix] = np.zeros(N)

        for i, func in enumerate(self.functionals):
            is_gga = self.types[i] == 'GGA'
            out = func.compute(inp, do_exc=False, do_vxc=True, do_fxc=False)

            for j, suffix in enumerate(['up', 'down']):
                results['vrho_' + suffix] += out['vrho'][:, j]

            if self.add_gradient_corrections and is_gga:
                for key in grad_keys:
                    for j, suffix in enumerate(['up', 'updown', 'down']):
                        results[key + '_' + suffix] += out[key][:, j]

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

    def evaluate_polarized(self, rho_up, rho_down, gd):
        """ Returns the XC potential for the 'up' spin channel.

        Parameters
        ----------
        rho_up : np.ndarray
            Electron density for the 'up' spin channel.
        rho_down : np.ndarray
            Electron density for the 'down' spin channel.
        gd : Grid
            An object that can carry out gradient and divergence
            operations on a grid-based array.

        Returns
        -------
        vxc_up : np.ndarray
            XC potential for spin 'up' electrons.
        """
        grad_up = gd.gradient(rho_up)
        grad_down = gd.gradient(rho_down)

        sigma_up = grad_up**2
        sigma_down = grad_down**2
        sigma_updown = grad_up * grad_down

        out = self.compute_vxc_polarized(rho_up, rho_down, sigma_up,
                                         sigma_updown, sigma_down)
        vxc_up = out['vrho_up']
        if self.add_gradient_corrections:
            vxc_up -= 2. * gd.divergence(out['vsigma_up'] * grad_up)
            vxc_up -= gd.divergence(out['vsigma_updown'] * grad_down)
        return vxc_up


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
