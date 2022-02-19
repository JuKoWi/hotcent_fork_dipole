""" Tests with double-zeta and single-zeta-plus-polarization basis sets for O.

Reference values come from the BeckeHarris tool, unless mentioned otherwise.
"""
import pytest
import numpy as np
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT
from hotcent.slako import INTEGRALS


R1 = 2.4

SZP = 'szp'
DZ = 'dz'
DZP = 'dzp'

LDA = 'LDA'
PBE = 'GGA_X_PBE+GGA_C_PBE'

# The following will be used to avoid unnecessary fixture rebuilds
SZP_LDA = SZP + '-' + LDA
SZP_PBE = SZP + '-' + PBE
DZ_LDA = DZ + '-' + LDA
DZ_PBE = DZ + '-' + PBE
DZP_LDA = DZP + '-' + LDA
DZP_PBE = DZP + '-' + PBE


@pytest.fixture(scope='module')
def atom(request):
    size, xc = request.param.split('-')

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
                           xc=xc,
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
    atom.generate_nonminimal_basis(size=size, tail_norm=0.15, r_pol=1.125)
    atom.pp.build_projectors(atom)
    atom.pp.build_overlaps(atom, atom, rmin=1., rmax=4., N=200)
    return atom


@pytest.mark.parametrize('atom', [SZP_LDA, DZ_LDA], indirect=True)
def test_on1c(atom):
    size = atom.basis_size

    HS_ref = {
        SZP: {
            ('2s', '2s'): (-0.79874591, 0.99999468),
            ('2p', '2p'): (-0.25260263, 0.99999468),
            ('0d', '0d'): (1.68192139, 0.99999468),
        },
        DZ: {
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
    for (nl1, nl2), ref in HS_ref[size].items():
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


@pytest.mark.parametrize('atom', [SZP_LDA, DZ_LDA], indirect=True)
def test_hubbard(atom):
    def mix(U_a, U_b):
        U = 8. / 5. * ((U_a * U_b) / (U_a + U_b) + \
                       (U_a * U_b)**2 / (U_a + U_b)**3)
        return U

    tol = 2e-2
    msg = 'Too large difference between d(eps_{0})/d(occ_{1}) ' + \
          'and the value obtained from the mixing rule: {2}'

    for valence1 in atom.basis_sets:
        for nl1 in valence1:
            for valence2 in atom.basis_sets:
                for nl2 in valence2:
                    U = atom.get_hubbard_value(nl1, nl2, scheme=None,
                                               maxstep=0.5)
                    if nl1 == nl2:
                        assert U > 0, 'U = {0} < 0 for {1}'.format(U, nl1)
                    else:
                        U_a = atom.get_hubbard_value(nl1, nl1, scheme=None,
                                                     maxstep=0.5)
                        U_b = atom.get_hubbard_value(nl2, nl2, scheme=None,
                                                     maxstep=0.5)
                        U_mix = mix(U_a, U_b)
                        U_diff = abs(U - U_mix)
                        assert U_diff < tol, msg.format(nl1, nl2, U_diff)


@pytest.mark.parametrize('atom', [SZP_LDA, DZ_LDA, SZP_PBE, DZ_PBE],
                         indirect=True)
def test_hubbard_analytical(atom):
    tol = 2e-4
    msg = 'Too large diff. for U_{0}-{1} (analytical: {2}, numerical: {3})'

    for valence1 in atom.basis_sets:
        for nl1 in valence1:
            for valence2 in atom.basis_sets:
                for nl2 in valence2:
                    U_num = atom.get_hubbard_value(nl1, nl2, scheme=None,
                                                   maxstep=0.25)
                    U_ana = atom.get_analytical_hubbard_value(nl1, nl2)
                    U_diff = abs(U_num - U_ana)
                    assert U_diff < tol, msg.format(nl1, nl2, U_ana, U_num)


@pytest.mark.parametrize('atom', [SZP_LDA, DZ_LDA], indirect=True)
def test_spin(atom):
    size = atom.basis_size

    # Regression test; values compare well with those
    # from e.g. the pbc-0-3 DFTB parameter set
    W_ref = {
        SZP: {
            ('2s', '2s'): -0.031736,
            ('2s', '2p'): -0.029346,
            ('2s', '0d'): -0.022399,
            ('2p', '2s'): -0.029120,
            ('2p', '2p'): -0.028053,
            ('2p', '0d'): -0.021191,
            ('0d', '2s'): -0.022427,
            ('0d', '2p'): -0.021382,
            ('0d', '0d'): -0.031324,
        },
        DZ: {
            ('2s', '2s'): -0.031736,
            ('2s', '2p'): -0.029346,
            ('2s', '2s+'): -0.037900,
            ('2s', '2p+'): -0.035232,
            ('2p', '2s'): -0.029120,
            ('2p', '2p'): -0.028053,
            ('2p', '2s+'): -0.034169,
            ('2p', '2p+'): -0.033071,
            ('2s+', '2s'): -0.037919,
            ('2s+', '2p'): -0.034455,
            ('2s+', '2s+'): -0.049712,
            ('2s+', '2p+'): -0.044614,
            ('2p+', '2s'): -0.035226,
            ('2p+', '2p'): -0.033329,
            ('2p+', '2s+'): -0.044585,
            ('2p+', '2p+'): -0.042220,
        },
    }

    msg = 'Too large error for W_{0}-{1} (value={2})'
    tol = 1e-4

    for valence1 in atom.basis_sets:
        for nl1 in valence1:
            for valence2 in atom.basis_sets:
                for nl2 in valence2:
                    W = atom.get_spin_constant(nl1, nl2, scheme=None,
                                               maxstep=0.5)
                    W_diff = abs(W - W_ref[size][(nl1, nl2)])
                    assert W_diff < tol, msg.format(nl1, nl2, W)


@pytest.mark.parametrize('atom', [SZP_LDA, DZ_LDA, SZP_PBE, DZ_PBE],
                         indirect=True)
def test_spin_analytical(atom):
    tol = 2e-4
    msg = 'Too large diff. for W_{0}-{1} (analytical: {2}, numerical: {3})'

    for valence1 in atom.basis_sets:
        for nl1 in valence1:
            for valence2 in atom.basis_sets:
                for nl2 in valence2:
                    W_num = atom.get_spin_constant(nl1, nl2, scheme=None,
                                                   maxstep=0.25)
                    W_ana = atom.get_analytical_spin_constant(nl1, nl2)
                    W_diff = abs(W_num - W_ana)
                    assert W_diff < tol, msg.format(nl1, nl2, W_ana, W_num)


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atom', [SZP_LDA, DZ_LDA], indirect=True)
def test_off2c(R, atom):
    from hotcent.offsite_twocenter import Offsite2cTable

    xc = atom.xcname
    size = atom.basis_size

    rmin, dr, N = R, R, 2
    off2c = Offsite2cTable(atom, atom)
    off2c.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
              smoothen_tails=False, ntheta=300, nr=100)
    HS = off2c.tables

    HS_ref = {
        (R1, SZP): {
            (0, 0, 0): {
                'sss': (-0.37502703, 0.24459516),
                'sps': (0.43535478, -0.34981181),
                'sds': (-0.24666439, 0.24752688),
                'pps': (0.30997467, -0.28858338),
                'ppp': (-0.15800151, 0.18783366),
                'pds': (-0.12496644, 0.22400675),
                'pdp': (0.11533162, -0.21547920),
                'dds': (0.39997486, 0.18863603),
                'ddp': (-0.28419754, -0.36484078),
                'ddd': (0.00558790, 0.10273658),
            },
        },
        (R1, DZ): {
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

    for key, integrals_refs in HS_ref[(R, size)].items():
        for integral, ref in integrals_refs.items():
            index = INTEGRALS.index(integral)
            htol = 5e-5
            if (size == DZ and (key[1] > 0 or key[2] > 0)) or \
               (size == SZP and integral[1] == 'd'):
                # Larger deviations related to kinetic energy
                htol = 5e-4

            val = HS[key][0, index]
            H_diff = abs(val - ref[0])
            assert H_diff < htol, msg.format('H', integral, key, val)

            val = HS[key][0, index+20]
            S_diff = abs(val - ref[1])
            assert S_diff < stol, msg.format('S', integral, key, val)


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atom', [SZP_LDA, DZ_LDA], indirect=True)
def test_on2c(R, atom):
    from hotcent.onsite_twocenter import Onsite2cTable

    xc = atom.xcname
    size = atom.basis_size

    rmin, dr, N = R, R, 2
    on2c = Onsite2cTable(atom, atom)
    on2c.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
             smoothen_tails=False, shift=False, ntheta=300, nr=100)
    H = on2c.tables

    H_ref = {
        (R1, SZP): {
            (0, 0): {
                'sps': -0.09856299,
                'sss': -0.06792868,
                'sds': -0.10830885,
                'pps': -0.16365362,
                'ppp': -0.03296093,
                'pds': -0.18812456,
                'pdp': -0.04953338,
                'dds': -0.27832485,
                'ddp': -0.10611034,
                'ddd': -0.02447832,
            },
        },
        (R1, DZ): {
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
    htol = 5e-5 if size == SZP else 2e-5

    for key, integrals_refs in H_ref[(R, size)].items():
        for integral, ref in integrals_refs.items():
            index = INTEGRALS.index(integral)
            val = H[key][0, index]
            diff = abs(val - ref)
            assert diff < htol, msg.format(integral, key, val)


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atom', [DZP_LDA, DZP_PBE], indirect=True)
def test_chg2c(R, atom):
    # Regression test
    from hotcent.offsite_chargetransfer import Offsite2cGammaTable
    from hotcent.onsite_chargetransfer import Onsite2cGammaTable

    xc = atom.xcname
    size = atom.basis_size
    rmin, dr, N = R, R, 2

    tol = 1e-9
    msg = 'Too large error for {0}_{1}-{2} [{3}] (value={4})'

    chg2c = Offsite2cGammaTable(atom, atom)
    chg2c.run(rmin=rmin, dr=dr, N=N, xc=xc, smoothen_tails=False,
              shift=False, ntheta=300, nr=100)
    G = chg2c.tables

    G_ref = {
        (R1, DZP, LDA): {
            (0, 0, 0):
                [-9.35969499e-03, -1.34282890e-02, -1.46553291e-02,  0.,
                 -1.34277776e-02, -1.75928447e-02, -1.94190229e-02,  0.,
                 -1.46544664e-02, -1.94185700e-02, -2.36020571e-02,  0.,
                  0.,  0.,  0.,  0.],
            (0, 0, 1):
                [-4.40057174e-03, -5.77791810e-03,  0.,  0.,
                 -8.04183907e-03, -9.57411647e-03,  0.,  0.,
                 -7.32470849e-03, -9.52989235e-03,  0.,  0., 0.,  0.,  0.,  0.],
            (0, 1, 0):
                [-4.40133432e-03, -8.04316075e-03, -7.32625544e-03,  0.,
                 -5.77845065e-03, -9.57518213e-03, -9.53122151e-03,  0.,
                  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
            (0, 1, 1):
                [-9.63066950e-04, -1.80356765e-03,  0.,  0.,
                 -1.80335255e-03, -2.83041102e-03,  0.,  0.,
                  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
        },
        (R1, DZP, PBE): {
            (0, 0, 0):
                [-9.00995470e-03, -1.28995855e-02, -1.44252673e-02,  0.,
                 -1.28990967e-02, -1.68684518e-02, -1.90054744e-02,  0.,
                 -1.44243934e-02, -1.90049914e-02, -2.38822099e-02,  0.,
                  0.,  0.,  0.,  0.],
            (0, 0, 1):
                [-4.09013354e-03, -5.40906473e-03,  0.,  0.,
                 -7.55906629e-03, -9.02019415e-03,  0.,  0.,
                 -6.94431673e-03, -9.13163401e-03,  0.,  0., 0.,  0.,  0.,  0.],
            (0, 1, 0):
                [-4.09088455e-03, -7.56035247e-03, -6.94586665e-03,  0.,
                 -5.40959575e-03, -9.02123598e-03, -9.13297449e-03,  0.,
                  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
            (0, 1, 1):
                [-7.84423332e-04, -1.55638676e-03,  0.,  0.,
                 -1.55618112e-03, -2.51279306e-03,  0.,  0.,
                  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
        },
    }

    for key, val in G.items():
        val_ref = G_ref[R, size, xc][key]
        for i, (item, item_ref) in enumerate(zip(val[0, :], val_ref)):
            diff = abs(item - item_ref)
            assert diff < tol, msg.format('Goff2c', key, val[0, i], i, item)

    chg2c = Onsite2cGammaTable(atom, atom)
    chg2c.run(rmin=rmin, dr=dr, N=N, xc=xc, smoothen_tails=False,
              ntheta=300, nr=100)
    G = chg2c.tables

    G_ref = {
        (R1, DZP, LDA): {
            (0, 0):
                [1.42296952e-03, 1.42910128e-03, 2.11772456e-03, 0.0,
                 1.42910128e-03, 1.56374110e-03, 2.19621864e-03, 0.0,
                 2.11772456e-03, 2.19621864e-03, 4.00294625e-03, 0.0,
                 0.,  0.,  0.,  0.],
            (0, 1):
                [1.05548720e-03, 1.18507853e-03,  0.,  0.,
                 9.46946512e-04, 1.12067380e-03,  0.,  0.,
                 1.11553459e-03, 1.52803407e-03,  0.,  0., 0.,  0.,  0.,  0.],
            (1, 0):
                [1.05548720e-03, 9.46946512e-04, 1.11553459e-03,  0.,
                 1.18507853e-03, 1.12067380e-03, 1.52803407e-03,  0.,
                 0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
            (1, 1):
                [1.08711046e-03, 1.05680334e-03,  0.,  0.,
                 1.05680334e-03, 1.11845459e-03,  0.,  0.,
                 0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
        },
        (R1, DZP, PBE): {
            (0, 0):
                [1.17013670e-03, 1.14447229e-03, 1.91674530e-03, 0.0,
                 1.14447229e-03, 1.24051561e-03, 2.04458829e-03, 0.0,
                 1.91674530e-03, 2.04458829e-03, 3.99886074e-03, 0.0,
                 0.,  0.,  0.,  0.],
            (0, 1):
                [7.77752684e-04, 8.54187106e-04,  0.,  0.,
                 6.28692947e-04, 8.21059843e-04,  0.,  0.,
                 7.89245073e-04, 1.22506566e-03,  0.,  0., 0.,  0.,  0.,  0.],
            (1, 0):
                [7.77752684e-04, 6.28692947e-04, 7.89245073e-04,  0.,
                 8.54187106e-04, 8.21059843e-04, 1.22506566e-03,  0.,
                 0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
            (1, 1):
                [8.13443727e-04, 6.81652471e-04,  0.,  0.,
                 6.81652471e-04, 7.41159208e-04,  0.,  0.,
                 0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
        }
    }

    for key, val in G.items():
        val_ref = G_ref[R, size, xc][key]
        for i, (item, item_ref) in enumerate(zip(val[0, :], val_ref)):
            diff = abs(item - item_ref)
            assert diff < tol, msg.format('Gon2c', key, val[0, i], i, item)


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atom', [DZP_LDA, DZP_PBE], indirect=True)
def test_mag2c(R, atom):
    # Regression test
    from hotcent.offsite_magnetization import Offsite2cWTable
    from hotcent.onsite_magnetization import Onsite2cWTable

    xc = atom.xcname
    size = atom.basis_size
    rmin, dr, N = R, R, 2

    tol = 1e-9
    msg = 'Too large error for {0}_{1}-{2} [{3}] (value={4})'

    mag2c = Offsite2cWTable(atom, atom)
    mag2c.run(rmin=rmin, dr=dr, N=N, xc=xc, smoothen_tails=False,
              ntheta=300, nr=100)
    W = mag2c.tables

    W_ref = {
        (R1, DZP, LDA): {
            (0, 0, 0):
                [-1.79165855e-03, -2.00311480e-03, -2.89578338e-03,  0.,
                 -2.00311480e-03, -2.22735265e-03, -3.08172118e-03,  0.,
                 -2.89578338e-03, -3.08172118e-03, -4.35845865e-03,  0.,
                  0.,  0.,  0.,  0.],
            (0, 0, 1):
                [-1.08039042e-03, -1.32933282e-03,  0.,  0.,
                 -1.29743684e-03, -1.54043621e-03,  0.,  0.,
                 -2.02415879e-03, -2.33307716e-03,  0.,  0., 0.,  0.,  0.,  0.],
            (0, 1, 0):
                [-1.08039042e-03, -1.29743684e-03, -2.02415879e-03,  0.,
                 -1.32933282e-03, -1.54043621e-03, -2.33307716e-03,  0.,
                  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
            (0, 1, 1):
                [-4.44143146e-04, -6.67809564e-04,  0.,  0.,
                 -6.67809564e-04, -9.01765450e-04,  0.,  0.,
                  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
        },
        (R1, DZP, PBE): {
            (0, 0, 0):
                [-1.44371837e-03, -1.70547842e-03, -2.62816975e-03,  0.,
                 -1.70547842e-03, -1.96020498e-03, -2.90657791e-03,  0.,
                 -2.62816975e-03, -2.90657791e-03, -4.45325807e-03,  0.,
                  0.,  0.,  0.,  0.],
            (0, 0, 1):
                [-6.62576966e-04, -9.23649368e-04,  0.,  0.,
                 -9.15637638e-04, -1.16763742e-03,  0.,  0.,
                 -1.51396278e-03, -1.87807817e-03,  0.,  0., 0.,  0.,  0.,  0.],
            (0, 1, 0):
                [-6.62576966e-04, -9.15637638e-04, -1.51396278e-03,  0.,
                 -9.23649368e-04, -1.16763742e-03, -1.87807817e-03,  0.,
                  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
            (0, 1, 1):
                [-5.69857983e-05, -2.72208140e-04,  0.,  0.,
                 -2.72208140e-04, -4.99898470e-04,  0.,  0.,
                  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
        },
    }

    for key, val in W.items():
        val_ref = W_ref[R, size, xc][key]
        for i, (item, item_ref) in enumerate(zip(val[0, :], val_ref)):
            diff = abs(item - item_ref)
            assert diff < tol, msg.format('Woff2c', key, val[0, i], i, item)

    mag2c = Onsite2cWTable(atom, atom)
    mag2c.run(rmin=rmin, dr=dr, N=N, xc=xc, smoothen_tails=False,
              ntheta=300, nr=100)
    W = mag2c.tables

    W_ref = {
        (R1, DZP, LDA): {
            (0, 0):
                [9.61652033e-04, 9.46355500e-04, 1.38206143e-03, 0.0,
                 9.46355500e-04, 9.96262206e-04, 1.41085940e-03, 0.0,
                 1.38206143e-03, 1.41085940e-03, 2.57319291e-03, 0.0,
                 0.,  0.,  0.,  0.],
            (0, 1):
                [7.51922824e-04, 8.25861445e-04,  0.,  0.,
                 6.73925789e-04, 7.76604103e-04,  0.,  0.,
                 7.64947714e-04, 1.02052139e-03,  0.,  0., 0.,  0.,  0.,  0.],
            (1, 0):
                [7.51922824e-04, 6.73925789e-04, 7.64947714e-04,  0.,
                 8.25861445e-04, 7.76604103e-04, 1.02052139e-03,  0.,
                 0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
            (1, 1):
                [7.89504958e-04, 7.61853632e-04,  0.,  0.,
                 7.61853632e-04, 7.95211355e-04,  0.,  0.,
                 0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
        },
        (R1, DZP, PBE): {
            (0, 0):
                [7.58271913e-04, 7.81682545e-04, 1.27827733e-03, 0.0,
                 7.81682545e-04, 8.52514313e-04, 1.42311351e-03, 0.0,
                 1.27827733e-03, 1.42311351e-03, 2.10749741e-03, 0.0,
                 0.,  0.,  0.,  0.],
            (0, 1):
                [4.61449939e-04, 5.27949937e-04,  0.,  0.,
                 4.17255114e-04, 5.49669291e-04,  0.,  0.,
                 7.21302211e-04, 8.77317244e-04,  0.,  0., 0.,  0.,  0.,  0.],
            (1, 0):
                [4.61449939e-04, 4.17255114e-04, 7.21302211e-04,  0.,
                 5.27949937e-04, 5.49669291e-04, 8.77317244e-04,  0.,
                 0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
            (1, 1):
                [3.34637923e-04,  3.24033619e-04,  0.,  0.,
                 3.24033619e-04,  3.86201150e-04,  0.,  0.,
                 0.,  0.,  0.,  0., 0.,  0.,  0.,  0.],
        },
    }

    for key, val in W.items():
        val_ref = W_ref[R, size, xc][key]
        for i, (item, item_ref) in enumerate(zip(val[0, :], val_ref)):
            diff = abs(item - item_ref)
            assert diff < tol, msg.format('Won2c', key, val[0, i], i, item)


@pytest.fixture(scope='module')
def grids(request):
    all_grids = {
        R1: (R1, np.array([R1]), np.array([1.]), np.array([0.6*np.pi])),
    }
    return all_grids[request.param]


@pytest.mark.parametrize('nphi', ['adaptive', 13])
@pytest.mark.parametrize('grids', [R1], indirect=True)
@pytest.mark.parametrize('atom', [SZP_LDA, DZ_LDA], indirect=True)
def test_off3c(nphi, grids, atom):
    from hotcent.offsite_threecenter import Offsite3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    xc = atom.xcname
    size = atom.basis_size

    off3c = Offsite3cTable(atom, atom)
    H = off3c.run(atom, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                  ntheta=300, nr=100, nphi=nphi, write=False)

    H_ref = {
        (R1, SZP): {
            ('O', 0, 0, 0): {
                's_s': -0.14513380,
                's_px': -0.07153752,
                's_pz': 0.23273429,
                's_dxz': 0.12409047,
                's_dx2-y2': -0.00710350,
                's_dz2': -0.25429537,
                'px_s': -0.07592540,
                'px_px': -0.08057081,
                'px_pz': 0.12971555,
                'px_dxz': 0.11846876,
                'px_dx2-y2': -0.01898184,
                'px_dz2': -0.10411828,
                'py_py': -0.06718708,
                'py_dxy': -0.05147703,
                'py_dyz': 0.13823137,
                'pz_s': -0.19588074,
                'pz_px': -0.09415060,
                'pz_pz': 0.27737727,
                'pz_dxz': 0.17435741,
                'pz_dx2-y2': -0.02020218,
                'pz_dz2': -0.32433343,
                'dxy_py': -0.06224529,
                'dxy_dxy': -0.08506329,
                'dxy_dyz': 0.13050945,
                'dyz_py': -0.11445817,
                'dyz_dxy': -0.09873114,
                'dyz_dyz': 0.25661722,
                'dxz_s': -0.13824009,
                'dxz_px': -0.14477121,
                'dxz_pz': 0.18976408,
                'dxz_dxz': 0.27665309,
                'dxz_dx2-y2': -0.06128564,
                'dxz_dz2': -0.16439247,
                'dx2-y2_s': -0.00895321,
                'dx2-y2_px': -0.03690125,
                'dx2-y2_pz': 0.02266394,
                'dx2-y2_dxz': 0.02808808,
                'dx2-y2_dx2-y2': -0.03556367,
                'dx2-y2_dz2': 0.01959273,
                'dz2_s': -0.19514948,
                'dz2_px': -0.08112433,
                'dz2_pz': 0.24717968,
                'dz2_dxz': 0.19678484,
                'dz2_dx2-y2': -0.03410039,
                'dz2_dz2': -0.30908415,
            },
        },
        (R1, DZ): {
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
    htol = 4e-4 if size == SZP else 1e-4

    for key, integrals_refs in H_ref[(R, size)].items():
        for integral, ref in integrals_refs.items():
            val = H[key][integral][0][1]
            diff = abs(val - ref)
            assert diff < htol, msg.format(integral, key, val)


@pytest.mark.parametrize('nphi', ['adaptive', 13])
@pytest.mark.parametrize('grids', [R1], indirect=True)
@pytest.mark.parametrize('atom', [SZP_LDA, DZ_LDA], indirect=True)
def test_on3c(nphi, grids, atom):
    from hotcent.onsite_threecenter import Onsite3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    xc = atom.xcname
    size = atom.basis_size

    on3c = Onsite3cTable(atom, atom)
    H = on3c.run(atom, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                 ntheta=300, nr=100, nphi=nphi, write=False)

    H_ref = {
        (R1, SZP): {
            (0, 0): {
                's_s': 0.00708275,
                's_px': 0.00279041,
                's_pz': 0.01040898,
                's_dxz': 0.00497859,
                's_dx2-y2': 0.00038799,
                's_dz2': 0.01062247,
                'px_s': 0.00279041,
                'px_px': 0.00447262,
                'px_pz': 0.00365672,
                'px_dxz': 0.00725993,
                'px_dx2-y2': 0.00161921,
                'px_dz2': 0.00211912,
                'py_py': 0.00389250,
                'py_dxy': 0.00173547,
                'py_dyz': 0.00650814,
                'pz_s': 0.01040898,
                'pz_px': 0.00365672,
                'pz_pz': 0.01691855,
                'pz_dxz': 0.00702511,
                'pz_dx2-y2': 0.00037590,
                'pz_dz2': 0.01826269,
                'dxy_py': 0.00173547,
                'dxy_dxy': 0.00388589,
                'dxy_dyz': 0.00277126,
                'dyz_py': 0.00650814,
                'dyz_dxy': 0.00277126,
                'dyz_dyz': 0.01384782,
                'dxz_s': 0.00497859,
                'dxz_px': 0.00725993,
                'dxz_pz': 0.00702511,
                'dxz_dxz': 0.01526283,
                'dxz_dx2-y2': 0.00251854,
                'dxz_dz2': 0.00608610,
                'dx2-y2_s': 0.00038799,
                'dx2-y2_px': 0.00161921,
                'dx2-y2_pz': 0.00037590,
                'dx2-y2_dxz': 0.00251854,
                'dx2-y2_dx2-y2': 0.00376423,
                'dx2-y2_dz2': 0.00003398,
                'dz2_s': 0.01062247,
                'dz2_px': 0.00211912,
                'dz2_pz': 0.01826269,
                'dz2_dxz': 0.00608610,
                'dz2_dx2-y2': 0.00003398,
                'dz2_dz2': 0.02587294,
            },
        },
        (R1, DZ): {
            (0, 0): {
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
            (0, 1): {
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
            (1, 0): {
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
            (1, 1): {
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
    htol = 4e-6 if size == SZP else 1e-6

    for key, integrals_refs in H_ref[(R, size)].items():
        for integral, ref in integrals_refs.items():
            val = H[key][integral][0][1]
            diff = abs(val - ref)
            assert diff < htol, msg.format(integral, key, val)
