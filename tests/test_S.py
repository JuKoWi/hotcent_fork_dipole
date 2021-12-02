""" Tests with one element (S), a KB pseudopotential, and LDA
and GGA functionals.

Reference values come from the BeckeHarris tool, unless mentioned otherwise.
"""
import pytest
import numpy as np
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT
from hotcent.slako import INTEGRALS


R1 = 2.4
R2 = 6.0

LDA = 'LDA'
LDA_LibXC = 'LDA_X+LDA_C_PW'
PBE_LibXC = 'GGA_X_PBE+GGA_C_PBE'


@pytest.fixture(scope='module')
def atom(request):
    wf_confinement = {
        '3s': SoftConfinement(rc=5.550736),
        '3p': SoftConfinement(rc=7.046078),
    }
    valence = list(wf_confinement.keys())
    pp = KleinmanBylanderPP('./pseudos/S.psf', valence)
    xcname = request.param
    atom = PseudoAtomicDFT('S', pp,
                           xc=xcname,
                           nodegpts=1000,
                           valence=valence,
                           configuration='[Ne] 3s2 3p4',
                           wf_confinement=wf_confinement,
                           perturbative_confinement=True,
                           scalarrel=False,
                           timing=False,
                           txt=None,
                           )
    atom.run()
    atom.pp.build_projectors(atom)
    atom.pp.build_overlaps(atom, atom, rmin=1., rmax=7., N=300)
    return atom


@pytest.mark.parametrize('atom', [PBE_LibXC], indirect=True)
def test_energies(atom):
    # Reference values obtained with Siesta v4.1.5
    from ase.units import Ha

    # Total energy
    e = atom.get_energy()
    e_ref = -273.865791 / Ha
    e_tol = 5e-4
    e_diff = abs(e - e_ref)
    assert e_diff < e_tol

    # Eigenvalues
    H_ref = {
        '3s': -0.163276829e+02 / Ha,
        '3p': -0.620202160e+01 / Ha,
    }
    H_tol = 1e-4
    for nl in atom.valence:
        H = atom.get_onecenter_integrals(nl, nl)[0]
        H_diff = abs(H - H_ref[nl])
        assert H_diff < H_tol, (nl, H)

    # Hubbard parameter for 3p
    U = atom.get_hubbard_value('3p', scheme='central', maxstep=1.)
    U_ref = (2*273.865791 - 261.846966 - 274.229224) / Ha
    U_tol = 1e-3
    U_diff = abs(U - U_ref)
    assert U_diff < U_tol


@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_on1c(atom):
    xc = atom.xcname

    H_ref = {
        PBE_LibXC: {
            '3s': -0.60003038,
            '3p': -0.22792123,
        },
        LDA_LibXC: {
            '3s': -0.60020793,
            '3p': -0.22964775,
        },
    }
    H_ref[LDA] = H_ref[LDA_LibXC]

    htol = 5e-6
    msg = 'Too large error for H_{0} (value={1})'
    for nl, ref in H_ref[xc].items():
        H = atom.get_onecenter_integrals(nl, nl)[0]
        H_diff = abs(H - ref)
        assert H_diff < htol, msg.format(nl, H)


@pytest.mark.parametrize('R', [R1, R2])
@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_off2c(R, atom):
    from hotcent.offsite_twocenter import Offsite2cTable

    xc = atom.xcname

    rmin, dr, N = R, R, 2
    off2c = Offsite2cTable(atom, atom)
    off2c.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
              smoothen_tails=False, ntheta=300, nr=100)
    H, S = off2c.tables[(0, 0, 0)][0, :20], off2c.tables[(0, 0, 0)][0, 20:41]

    HS_ref = {
        (R1, PBE_LibXC): {
            'sss': (-0.56647099, 0.49948243),
            'sps': (0.54222362, -0.56984662),
            'pps': (0.23151521, -0.18330690),
            'ppp': (-0.29839103, 0.46900355),
        },
        (R1, LDA_LibXC): {
            'sss': (-0.57068710, 0.50148036),
            'sps': (0.54653009, -0.57206506),
            'pps': (0.23100346, -0.17645749),
            'ppp': (-0.30160758, 0.47370045),
        },
        (R2, PBE_LibXC): {
            'sss': (-0.01444882, 0.01200375),
            'sps': (0.04311914, -0.05187151),
            'pps': (0.06293949, -0.12927919),
            'ppp': (-0.01213259, 0.02230937),
        },
        (R2, LDA_LibXC): {
            'sss': (-0.01488318, 0.01230404),
            'sps': (0.04448793, -0.05320122),
            'pps': (0.06553254, -0.13289531),
            'ppp': (-0.01271585, 0.02303343),
        },
    }
    HS_ref[(R1, LDA)] = HS_ref[(R1, LDA_LibXC)]
    HS_ref[(R2, LDA)] = HS_ref[(R2, LDA_LibXC)]

    htol = 2e-4
    stol = 1e-4
    msg = 'Too large error for {0}_{1} (value={2})'

    for integral, ref in HS_ref[(R, xc)].items():
        index = INTEGRALS.index(integral)

        H_diff = abs(H[index] - ref[0])
        assert H_diff < htol, msg.format('H', integral, H[index])

        S_diff = abs(S[index] - ref[1])
        assert S_diff < stol, msg.format('S', integral, S[index])


@pytest.mark.parametrize('R', [R1, R2])
@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_on2c(R, atom):
    from hotcent.onsite_twocenter import Onsite2cTable

    xc = atom.xcname

    rmin, dr, N = R, R, 2
    on2c = Onsite2cTable(atom, atom)
    on2c.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc, shift=False,
             smoothen_tails=False, ntheta=300, nr=100)
    H = on2c.tables[(0, 0)]

    H_ref = {
        (R1, PBE_LibXC): {
            'sss': -0.20680849,
            'sps': -0.23136608,
            'pps': -0.32695284,
            'ppp': -0.12726185,
        },
        (R1, LDA_LibXC): {
            'sss': -0.21141976,
            'sps': -0.23403501,
            'pps': -0.33032949,
            'ppp': -0.13063652,
        },
        (R2, PBE_LibXC): {
            'sss': -0.00043645,
            'sps': -0.00122827,
            'pps': -0.00466045,
            'ppp': -0.00035467,
        },
        (R2, LDA_LibXC): {
            'sss': -0.00064242,
            'sps': -0.00168316,
            'pps': -0.00587111,
            'ppp': -0.00058351,
        }
    }
    H_ref[(R1, LDA)] = H_ref[(R1, LDA_LibXC)]
    H_ref[(R2, LDA)] = H_ref[(R2, LDA_LibXC)]

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        index = INTEGRALS.index(integral)
        val = H[0, index]
        diff = abs(val - ref)
        assert diff < 1e-4, msg.format(integral, val)


@pytest.mark.parametrize('R', [R1, R2])
@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_rep2c(R, atom):
    from hotcent.repulsion_twocenter import Repulsion2cTable

    xc = atom.xcname

    rmin, dr, N = R, R, 3
    rep2c = Repulsion2cTable(atom, atom)
    rep2c.run(rmin=rmin, dr=dr, N=N, xc=xc, smoothen_tails=False, shift=False,
              ntheta=600, nr=200)
    E = rep2c.erep[0]

    E_ref = {
        (R1, PBE_LibXC): 2.40963604,
        (R1, LDA_LibXC): 2.44188670,
        (R2, PBE_LibXC): 0.00626943,
        (R2, LDA_LibXC): 0.00685118,
    }
    E_ref[(R1, LDA)] = E_ref[(R1, LDA_LibXC)]
    E_ref[(R2, LDA)] = E_ref[(R2, LDA_LibXC)]

    etol = 2e-4
    E_diff = abs(E - E_ref[(R, xc)])
    msg = 'Too large error for E_rep (value={0})'
    assert E_diff < etol, msg.format(E)


@pytest.fixture(scope='module')
def grids(request):
    all_grids = {
        R1: (R1, np.array([R1]), np.array([1.2]), np.array([np.pi/2])),
        R2: (R2, np.array([R2]), np.array([0.]), np.array([0.])),
    }
    return all_grids[request.param]


@pytest.mark.parametrize('nphi', ['adaptive', 13])
@pytest.mark.parametrize('grids', [R1, R2], indirect=True)
@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_off3c(nphi, grids, atom):
    from hotcent.offsite_threecenter import Offsite3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    xc = atom.xcname

    off3c = Offsite3cTable(atom, atom)
    H = off3c.run(atom, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                  ntheta=300, nr=100, nphi=nphi, write=False)

    H_ref = {
        (R1, PBE_LibXC): {
            's_s': -0.20053834,
            's_px': -0.06779214,
            's_pz': 0.23955453,
            'px_s': -0.06779214,
            'px_px': -0.12579050,
            'px_pz': 0.08002474,
            'py_py': -0.15488876,
            'pz_s': -0.23955453,
            'pz_px': -0.08002474,
            'pz_pz': 0.19302300,
        },
        (R1, LDA_LibXC): {
            's_s': -0.20386716,
            's_px': -0.06891464,
            's_pz': 0.24284659,
            'px_s': -0.06891464,
            'px_px': -0.12838910,
            'px_pz': 0.08132229,
            'py_py': -0.15680786,
            'pz_s': -0.24284659,
            'pz_px': -0.08132229,
            'pz_pz': 0.19281573,
        },
        (R2, PBE_LibXC): {
            's_s': -0.00766321,
            's_px': -0.00000000,
            's_pz': 0.02727241,
            'px_s': -0.00000000,
            'px_px': -0.00799655,
            'px_pz': 0.00000000,
            'py_py': -0.00799655,
            'pz_s': -0.02727241,
            'pz_px': -0.00000000,
            'pz_pz': 0.07184050,
        },
        (R2, LDA_LibXC): {
            's_s': -0.00774572,
            's_px': -0.00000000,
            's_pz': 0.02782506,
            'px_s': -0.00000000,
            'px_px': -0.00815735,
            'px_pz': 0.00000000,
            'py_py': -0.00815735,
            'pz_s': -0.02782506,
            'pz_px': -0.00000000,
            'pz_pz': 0.07331421,
        },
    }
    H_ref[(R1, LDA)] = H_ref[(R1, LDA_LibXC)]
    H_ref[(R2, LDA)] = H_ref[(R2, LDA_LibXC)]

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        key = ('S', 0, 0, 0)
        val = H[key][integral][0][1]
        diff = abs(val - ref)
        assert diff < 5e-4, msg.format(integral, val)


@pytest.mark.parametrize('nphi', ['adaptive', 13])
@pytest.mark.parametrize('grids', [R1, R2], indirect=True)
@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_on3c(nphi, grids, atom):
    from hotcent.onsite_threecenter import Onsite3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    xc = atom.xcname

    on3c = Onsite3cTable(atom, atom)
    H = on3c.run(atom, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                 ntheta=300, nr=100, nphi=nphi, write=False)

    H_ref = {
        (R1, PBE_LibXC): {
            's_s': 0.01211223,
            's_px': 0.00435629,
            's_pz': 0.01536477,
            'px_s': 0.00435629,
            'px_px': 0.00899547,
            'px_pz': 0.00508989,
            'py_py': 0.00899430,
            'pz_s': 0.01536477,
            'pz_px': 0.00508989,
            'pz_pz': 0.02464314,
        },
        (R1, LDA_LibXC): {
            's_s': 0.01312959,
            's_px': 0.00466225,
            's_pz': 0.01602632,
            'px_s': 0.00466225,
            'px_px': 0.01015447,
            'px_pz': 0.00505370,
            'py_py': 0.00997600,
            'pz_s': 0.01602632,
            'pz_px': 0.00505370,
            'pz_pz': 0.02542099,
        },
        (R2, PBE_LibXC): {
            's_s': 0.00015298,
            's_px': 0.00000000,
            's_pz': 0.00052767,
            'px_s': 0.00000000,
            'px_px': 0.00016053,
            'px_pz': 0.00000000,
            'py_py': 0.00016053,
            'pz_s': 0.00052767,
            'pz_px': 0.00000000,
            'pz_pz': 0.00216021,
        },
        (R2, LDA_LibXC): {
            's_s': 0.00033167,
            's_px': 0.00000000,
            's_pz': 0.00092070,
            'px_s': 0.00000000,
            'px_px': 0.00035231,
            'px_pz': 0.00000000,
            'py_py': 0.00035231,
            'pz_s': 0.00092070,
            'pz_px': 0.00000000,
            'pz_pz': 0.00315674,
        },
    }
    H_ref[(R1, LDA)] = H_ref[(R1, LDA_LibXC)]
    H_ref[(R2, LDA)] = H_ref[(R2, LDA_LibXC)]

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        key = (0, 0)
        val = H[key][integral][0][1]
        diff = abs(val - ref)
        assert diff < 1e-6, msg.format(integral, val)


@pytest.mark.parametrize('nphi', ['adaptive', 13])
@pytest.mark.parametrize('grids', [R1, R2], indirect=True)
@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_rep3c(nphi, grids, atom):
    from hotcent.repulsion_threecenter import Repulsion3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    xc = atom.xcname

    rep3c = Repulsion3cTable(atom, atom)
    E = rep3c.run(atom, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, nphi=nphi, xc=xc,
                  write=False)

    E_ref = {
        (R1, PBE_LibXC): -0.06149992,
        (R1, LDA_LibXC): -0.05907973,
        (R2, PBE_LibXC): -0.00253134,
        (R2, LDA_LibXC): -0.00293521,
    }
    E_ref[(R1, LDA)] = E_ref[(R1, LDA_LibXC)]
    E_ref[(R2, LDA)] = E_ref[(R2, LDA_LibXC)]

    val = E[('S', 'S')]['s_s'][0][1]
    diff = abs(val - E_ref[(R, xc)])
    msg = 'Too large error for E_rep (value={0})'
    assert diff < 5e-5, msg.format(val)
