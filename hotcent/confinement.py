""" Definition of confinement potentials """
import numpy as np

class PowerConfinement():
    def __init__(self, r0=1., s=2):
        self.r0 = r0
        self.s = s

    def __call__(self, r):
        return (r / self.r0) ** self.s


class WoodsSaxonConfinement():
    def __init__(self, v0=1., r0=1., a=1.): 
        self.v0 = v0
        self.r0 = r0
        self.a = a

    def __call__(self, r):
        return -1 * self.v0 / (1 + np.exp((r - self.r0) / self.a))
