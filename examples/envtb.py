""" Test/example for the 'environmental' tight-binding extensions
(beyond 1-center and 2-center expansions for the on- and off-site
matrix elements, respectively).
"""
import numpy as np
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import SoftConfinement
from hotcent.offsite_threecenter import Offsite3cTable
from hotcent.onsite_twocenter import Onsite2cTable
from hotcent.onsite_threecenter import Onsite3cTable
from hotcent.repulsion_twocenter import Repulsion2cTable
from hotcent.repulsion_threecenter import Repulsion3cTable

element = 'C'
xc = 'LDA'
configuration, valence, occupation = '[He] 2s2 2p2', ['2s', '2p'], [2., 2., 0.]
wf_conf = {'2s': SoftConfinement(amp=12., x_ri=0.6, rc=5.58),
           '2p': SoftConfinement(amp=12., x_ri=0.6, rc=6.87)}

atom = AtomicDFT(element,
                 nodegpts=1000,
                 xc=xc,
                 wf_confinement=wf_conf,
                 perturbative_confinement=True,
                 configuration=configuration,
                 valence=valence,
                 scalarrel=True,
                 txt='-',
                 )
atom.run()

eigenvalues = {nl: atom.get_eigenvalue(nl) for nl in valence}
on1c = {nl: atom.get_onecenter_integral(nl) for nl in valence}

atom.pp.build_projectors(atom)
atom.pp.build_overlaps(atom, atom, rmin=0.05, rmax=8.)

rep2c = Repulsion2cTable(atom, atom)
rep2c.run(rmin=1.2, dr=0.1, N=46, xc='LDA')
rep2c.write()

rmin, dr, N = 0.4, 0.04, 300
on2c = Onsite2cTable(atom, atom)
on2c.run(atom, rmin, dr, N, superposition='density', xc=xc)

min_rAB, max_rAB, num_rAB = 2.6456165761716997, 2.6456165761716997, 1
min_rCM, max_rCM, num_rCM = 2.2911711636379, 2.2911711636379, 1
num_theta = 3
Rgrid = np.exp(np.linspace(np.log(min_rAB), np.log(max_rAB), num=num_rAB,
               endpoint=True))
Sgrid = np.exp(np.linspace(np.log(min_rCM), np.log(max_rCM), num=num_rCM,
               endpoint=True))
Tgrid = np.linspace(0., np.pi, num=num_theta)

on3c = Onsite3cTable(atom, atom)
on3c.run(atom, atom, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc)

off3c = Offsite3cTable(atom, atom)
off3c.run(atom, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc)

rep3c = Repulsion3cTable(atom, atom)
rep3c.run(atom, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc)
