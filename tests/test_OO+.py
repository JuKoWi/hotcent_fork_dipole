""" Tests for a double-numerical basis set for oxygen.

Reference values come from the BeckeHarris tool, unless mentioned otherwise.
"""
import pytest
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT


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
            # Larger deviations in this case due to use
            # of KB transformation in BeckeHarris.
            htol = 5e-3

        H_diff = abs(H - ref[0])
        assert H_diff < htol, msg.format('H', nl1, nl2, H)

        S_diff = abs(S - ref[1])
        assert S_diff < stol, msg.format('S', nl1, nl2, S)
