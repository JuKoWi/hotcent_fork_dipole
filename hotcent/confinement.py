""" Definition of confinement potentials """
import numpy as np


class ZeroConfinement():
    def __call__(self, r):
        return np.zeros_like(r)

    def __str__(self):
        return 'ZeroConfinement'


class PowerConfinement():
    def __init__(self, r0=1., s=2):
        self.r0 = r0
        self.s = s

    def __call__(self, r):
        return (r / self.r0) ** self.s

    def __str__(self):
        return 'PowerConfinement(r0=%.6f, s=%.6f)' % (self.r0, self.s)


class WoodsSaxonConfinement():
    def __init__(self, w=1., r0=1., a=1.):
        self.w = w
        self.r0 = r0
        self.a = a

    def __call__(self, r):
        return self.w / (1 + np.exp(self.a * (self.r0 - r)))

    def __str__(self):
        return 'WoodsSaxonConfinement(w=%.6f, r0=%.6f, a=%.6f)' % \
               (self.w, self.r0, self.a)
