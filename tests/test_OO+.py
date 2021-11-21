""" Tests for a double-numerical basis set for oxygen.

Reference values come from the BeckeHarris tool, unless mentioned otherwise.
"""
import pytest
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
