""" Defintion of the GPAWAllElectron class for 
calculations with the GPAW atomic DFT code.
"""
from __future__ import print_function
import pickle
from hotcent.atom import AllElectron
from gpaw.atom.all_electron import AllElectron as GPAWAllElectron
try:
    import pylab as pl
except:
    pl = None


class GPAWAE(AllElectron, GPAWAllElectron):
    def __init__(self, symbol, **kwargs):
        """
        Run Kohn-Sham all-electron calculation for a given atom 
        using the atomic DFT calculator in GPAW.
        """
        AllElectron.__init__(self, symbol, **kwargs)

        config = kwargs['configuration']
        config = config.replace('[', '').replace(']', '')
        config = ','.join(config.split())
 
        GPAWAllElectron.__init__(self, xcname=kwargs['xcname'], 
                                 configuration=config,
                                 scalarrel=kwargs['scalarrel'], 
                                 gpernode=kwargs['nodegpts']) 

        self.timer.stop('init')

    def run(self):
        self.timer.start('run')

        GPAWAllElectron.run(self)

        val = self.get_valence_orbitals()
        enl = {}
        Rnlg = {}
        unlg = {}

        if self.confinement is not None:
            # run with confinement potential for the density

        confinement = self.confinement
        for nl, wf_confinement in self.wf_confinement.items():
            assert nl in val, "Confinement: %s not in %s" % (nl, str(val))
            self.confinement = wf_confinement
            GPAWAllElectron.solve_confined(j, rcut, vconf=vconf)

            Rnlg[nl] = self.Rnlg[nl].copy()
            unlg[nl] = self.unlg[nl].copy()
            enl[nl] = self.enl[nl]

        self.Rnlg.update(Rnlg)
        self.unlg.update(unlg)
        self.enl.update(enl)
        for nl in val:
            self.Rnl_fct[nl] = Function('spline', self.rgrid, self.Rnlg[nl])
            self.unl_fct[nl] = Function('spline', self.rgrid, self.unlg[nl])

        self.veff = self.calculate_veff()

        if self.write != None:
            with open(self.write, 'w') as f:
                pickle.dump(self.rgrid, f)
                pickle.dump(self.veff, f)
                pickle.dump(self.dens, f)

        self.solved = True
        self.timer.stop('run')

        self.timer.summary()
        self.txt.flush()
