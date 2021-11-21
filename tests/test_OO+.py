""" Tests for a double-numerical basis set for oxygen.

Reference values come from the BeckeHarris tool, unless mentioned otherwise.
"""
import pytest
import numpy as np
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT
from hotcent.slako import INTEGRALS


R1 = 2.4

LDA = 'LDA'


@pytest.fixture(scope='module')
def atom(request):
    xcname = request.param

    configuration = '[He] 2s2 2p4'

    wf_confinements = {
        '2s': SoftConfinement(rc=4.01),
        '2p': SoftConfinement(rc=4.80),
    }

    valence = list(wf_confinements.keys())

    element = 'O'
    pp = KleinmanBylanderPP('./pseudos/{0}.psf'.format(element), valence,
                            verbose=True)

    atom = PseudoAtomicDFT(element, pp,
                           xc=xcname,
                           nodegpts=1000,
                           valence=valence,
                           configuration=configuration,
                           wf_confinement=wf_confinements,
                           perturbative_confinement=True,
                           scalarrel=False,
                           timing=False,
                           txt=None,
                           )
    atom.run()
    atom.generate_nonminimal_basis(size='dz', tail_norm=0.15)
    atom.pp.build_projectors(atom)
    atom.pp.build_overlaps(atom, atom, rmin=1., rmax=4.)
    return atom


@pytest.mark.parametrize('atom', [LDA], indirect=True)
def test_on1c(atom):
    xc = atom.xcname

    HS_ref = {
        LDA: {
            ('2s', '2s'): (-0.79874591, 0.99999468),
            ('2p', '2p'): (-0.25260263, 0.99999468),
            ('2s', '2s+'): (-0.73938783, 0.94299966),
            ('2p', '2p+'): (-0.22966216, 0.94982839),
            ('2s+', '2s'): (-0.74060276, 0.94299966),
            ('2p+', '2p'): (-0.22835834, 0.94982839),
            ('2s+', '2s+'): (-0.57275898, 0.99999468),
            ('2p+', '2p+'): (-0.13232958, 0.99999469),
        },
    }

    stol = 1e-5
    msg = 'Too large error for {0}_{1}-{2} (value={3})'
    for (nl1, nl2), ref in HS_ref[xc].items():
        H, S = atom.get_onecenter_integrals(nl1, nl2)
        htol = 1e-4
        if '+' in nl2:
            # Larger deviations due to use of KB transformation
            # in BeckeHarris calculations.
            htol = 5e-3

        H_diff = abs(H - ref[0])
        assert H_diff < htol, msg.format('H', nl1, nl2, H)

        S_diff = abs(S - ref[1])
        assert S_diff < stol, msg.format('S', nl1, nl2, S)


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atom', [LDA], indirect=True)
def test_off2c(R, atom):
    from hotcent.offsite_twocenter import Offsite2cTable

    xc = atom.xcname

    rmin, dr, N = R, R, 2
    off2c = Offsite2cTable(atom, atom)
    off2c.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
              smoothen_tails=False, ntheta=300, nr=100)
    HS = off2c.tables

    HS_ref = {
        (R1, LDA): {
            (0, 0, 0): {
                'sss': (-0.37502703, 0.24459516),
                'sps': (0.43535478, -0.34981181),
                'pps': (0.30997467, -0.28858338),
                'ppp': (-0.15800151, 0.18783366),
            },
            (0, 0, 1): {
                'sss': (-0.28889358, 0.14105552),
                'sps': (0.31934193, -0.19191517),
                'pps': (0.32966136, -0.25526033),
                'ppp': (-0.13312948, 0.10124055),
            },
            (0, 1, 0): {
                'sss': (-0.28930004, 0.14105552),
                'sps': (0.42329127, -0.26913362),
                'pps': (0.32957689, -0.25526033),
                'ppp': (-0.13286351, 0.10124055),
            },
            (0, 1, 1): {
                'sss': (-0.20116785, 0.05818130),
                'sps': (0.29805990, -0.13395224),
                'pps': (0.33298175, -0.23139294),
                'ppp': (-0.09435624, 0.04956880),
            },
        },
    }

    stol = 5e-5
    msg = 'Too large error for {0}_{1} [{2}] (value={3})'

    for key, integrals_refs in HS_ref[(R, xc)].items():
        for integral, ref in integrals_refs.items():
            index = INTEGRALS.index(integral)
            htol = 5e-5
            if (key[1] > 0 or key[2] > 0):
                # Larger deviations related to kinetic energy
                htol = 5e-4

            val = HS[key][0, index]
            H_diff = abs(val - ref[0])
            assert H_diff < htol, msg.format('H', integral, key, val)

            val = HS[key][0, index+20]
            S_diff = abs(val - ref[1])
            assert S_diff < stol, msg.format('S', integral, key, val)


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atom', [LDA], indirect=True)
def test_on2c(R, atom):
    from hotcent.onsite_twocenter import Onsite2cTable

    xc = atom.xcname

    rmin, dr, N = R, R, 2
    on2c = Onsite2cTable(atom, atom)
    on2c.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
             smoothen_tails=False, ntheta=300, nr=100)
    H = on2c.tables

    H_ref = {
        (R1, LDA): {
            (0, 0): {
                'sss': -0.06792868,
                'sps': -0.09856299,
                'pps': -0.16365362,
                'ppp': -0.03296093,
            },
            (0, 1): {
                'sss': -0.04648836,
                'sps': -0.07643658,
                'pps': -0.11687865,
                'ppp': -0.02694682,
            },
            (1, 0): {
                'sss': -0.04648836,
                'sps': -0.05877341,
                'pps': -0.11687865,
                'ppp': -0.02694682,
            },
            (1, 1): {
                'sss': -0.04216209,
                'sps': -0.05714434,
                'pps': -0.10282754,
                'ppp': -0.02613588,
            },
        },
    }

    msg = 'Too large error for H_{0} [{1}] (value={2})'

    for key, integrals_refs in H_ref[(R, xc)].items():
        for integral, ref in integrals_refs.items():
            index = INTEGRALS.index(integral)
            val = H[key][0, index]
            diff = abs(val - ref)
            assert diff < 2e-5, msg.format(integral, key, val)


@pytest.fixture(scope='module')
def grids(request):
    all_grids = {
        R1: (R1, np.array([R1]), np.array([1.]), np.array([0.6*np.pi])),
    }
    return all_grids[request.param]


@pytest.mark.parametrize('nphi', ['adaptive', 13])
@pytest.mark.parametrize('grids', [R1], indirect=True)
@pytest.mark.parametrize('atom', [LDA], indirect=True)
def test_off3c(nphi, grids, atom):
    from hotcent.offsite_threecenter import Offsite3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    xc = atom.xcname

    off3c = Offsite3cTable(atom, atom)
    H = off3c.run(atom, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                  ntheta=300, nr=100, nphi=nphi, write=False)

    H_ref = {
        (R1, LDA): {
            ('O', 0, 0, 0): {
                's_s': -0.14513380,
                's_px': -0.07153752,
                's_pz': 0.23273429,
                'px_s': -0.07592540,
                'px_px': -0.08057081,
                'px_pz': 0.12971555,
                'py_py': -0.06718708,
                'pz_s': -0.19588074,
                'pz_px': -0.09415060,
                'pz_pz': 0.27737727,
            },
            ('O', 0, 0, 1): {
                's_s': -0.08055232,
                's_px': -0.04634292,
                's_pz': 0.15418093,
                'px_s': -0.02192888,
                'px_px': -0.03902317,
                'px_pz': 0.05415374,
                'py_py': -0.04277687,
                'pz_s': -0.13809558,
                'pz_px': -0.08002882,
                'pz_pz': 0.23227257,
            },
            ('O', 0, 1, 0): {
                's_s': -0.10714404,
                's_px': -0.03104567,
                's_pz': 0.19586654,
                'px_s': -0.05923435,
                'px_px': -0.05058672,
                'px_pz': 0.11652169,
                'py_py': -0.05595607,
                'pz_s': -0.16338977,
                'pz_px': -0.05698134,
                'pz_pz': 0.25538340,
            },
            ('O', 0, 1, 1): {
                's_s': -0.04914880,
                's_px': -0.01519286,
                's_pz': 0.11962572,
                'px_s': -0.01128544,
                'px_px': -0.01869574,
                'px_pz': 0.04390940,
                'py_py': -0.03450254,
                'pz_s': -0.10822087,
                'pz_px': -0.04856542,
                'pz_pz': 0.20884367,
            },
        },
    }

    msg = 'Too large error for H_{0} [{1}] (value={2})'

    for key, integrals_refs in H_ref[(R, xc)].items():
        for integral, ref in integrals_refs.items():
            val = H[key][integral][0][1]
            diff = abs(val - ref)
            assert diff < 1e-4, msg.format(integral, key, val)


@pytest.mark.parametrize('nphi', ['adaptive', 13])
@pytest.mark.parametrize('grids', [R1], indirect=True)
@pytest.mark.parametrize('atom', [LDA], indirect=True)
def test_on3c(nphi, grids, atom):
    from hotcent.onsite_threecenter import Onsite3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    xc = atom.xcname

    on3c = Onsite3cTable(atom, atom)
    H = on3c.run(atom, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                 ntheta=300, nr=100, nphi=nphi, write=False)

    H_ref = {
        (R1, LDA): {
            (0, 0, 0): {
                's_s': 0.00708275,
                's_px': 0.00279041,
                's_pz': 0.01040898,
                'px_s': 0.00279041,
                'px_px': 0.00447262,
                'px_pz': 0.00365672,
                'py_py': 0.00389250,
                'pz_s': 0.01040898,
                'pz_px': 0.00365672,
                'pz_pz': 0.01691855,
            },
            (0, 0, 1): {
                's_s': 0.00416779,
                's_px': 0.00197603,
                's_pz': 0.00751428,
                'px_s': 0.00145023,
                'px_px': 0.00294593,
                'px_pz': 0.00228509,
                'py_py': 0.00255736,
                'pz_s': 0.00551233,
                'pz_px': 0.00228509,
                'pz_pz': 0.01121707,
            },
            (0, 1, 0): {
                's_s': 0.00416779,
                's_px': 0.00145023,
                's_pz': 0.00551233,
                'px_s': 0.00197603,
                'px_px': 0.00294593,
                'px_pz': 0.00228509,
                'py_py': 0.00255736,
                'pz_s': 0.00751428,
                'pz_px': 0.00228509,
                'pz_pz': 0.01121707,
            },
            (0, 1, 1): {
                's_s': 0.00340006,
                's_px': 0.00136317,
                's_pz': 0.00512853,
                'px_s': 0.00136317,
                'px_px': 0.00250592,
                'px_pz': 0.00187392,
                'py_py': 0.00217054,
                'pz_s': 0.00512853,
                'pz_px': 0.00187392,
                'pz_pz': 0.00935529,
            },
        },
    }

    msg = 'Too large error for H_{0} [{1}] (value={2})'

    for key, integrals_refs in H_ref[(R, xc)].items():
        for integral, ref in integrals_refs.items():
            val = H[key][integral][0][1]
            diff = abs(val - ref)
            assert diff < 1e-6, msg.format(integral, key, val)
