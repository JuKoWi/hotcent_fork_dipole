import numpy as np
from hotcent.multiatom_integrator import MultiAtomIntegrator

class Offsite2cTableDipole(MultiAtomIntegrator):
    def __init__(self, *args, **kwargs):
        MultiAtomIntegrator.__init__(self, *args, grid_type='bipolar', **kwargs)

    def run(self, rmin=0.4, dr=0.02, N=None, superposition='density', xc='LDA', stride=1):

        self.print_header()

        assert N is not None, 'Need to set number of grid points N!'
        assert rmin >= 1e-3, 'For stability, please set rmin >= 1e-3'
        assert np.isclose(rmin / dr, np.round(rmin / dr)), \
               'rmin must be a multiple of dr'
        assert superposition in ['density', 'potential']
        print('finished') 
