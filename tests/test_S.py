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
    atom.pp.build_overlaps(atom, atom, rmin=1., rmax=7.)
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
    e_ref = {
        '3s': -0.163276829e+02 / Ha,
        '3p': -0.620202160e+01 / Ha,
    }
    e_tol = 1e-4
    for nl in atom.valence:
        e = atom.get_onecenter_integral(nl)
        e_diff = abs(e - e_ref[nl])
        assert e_diff < e_tol

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
            '3s': -0.60003476,
            '3p': -0.22792440,
        },
        LDA_LibXC: {
            '3s': -0.60021238,
            '3p': -0.22965093,
        },
    }
    H_ref[LDA] = H_ref[LDA_LibXC]

    htol = 1e-5
    msg = 'Too large error for H_{0} (value={1})'
    for nl, ref in H_ref[xc].items():
        H = atom.get_onecenter_integral(nl)
        H_diff = abs(H - ref)
        assert H_diff < htol, msg.format(nl, H)


@pytest.mark.parametrize('R', [R1, R2])
@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_off2c(R, atom):
    from hotcent.slako import SlaterKosterTable

    xc = atom.xcname

    rmin, dr, N = R, R, 2
    off2c = SlaterKosterTable(atom, atom)
    off2c.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
              smoothen_tails=False, ntheta=300, nr=100)
    H, S = off2c.tables[0][0, :20], off2c.tables[0][0, 20:41]

    HS_ref = {
        (R1, PBE_LibXC): {
            'sss': (-0.56677916,  0.49949084),
            'sps': ( 0.54239200, -0.56984428),
            'pps': ( 0.23153307, -0.18330357),
            'ppp': (-0.29839091,  0.46900432),
        },
        (R1, LDA_LibXC): {
            'sss': (-0.57099349,  0.50148895),
            'sps': ( 0.54669416, -0.57206251),
            'pps': ( 0.23101831, -0.17645402),
            'ppp': (-0.30160739,  0.47370130),
        },
        (R2, PBE_LibXC): {
            'sss': (-0.01439856, 0.01201897),
            'sps': (0.04313330, -0.05188745),
            'pps': (0.06294089, -0.12927059),
            'ppp': (-0.01214645, 0.02231471),
        },
        (R2, LDA_LibXC): {
            'sss': (-0.01483195, 0.01231958),
            'sps': (0.04450229, -0.05321746),
            'pps': (0.06553380, -0.13288662),
            'ppp': (-0.01273031, 0.02303890),
        },
    }
    HS_ref[(R1, LDA)] = HS_ref[(R1, LDA_LibXC)]
    HS_ref[(R2, LDA)] = HS_ref[(R2, LDA_LibXC)]

    htol = 5e-4
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
    H = on2c.run(atom, rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
                 smoothen_tails=False, ntheta=300, nr=100, write=False)

    H_ref = {
        (R1, PBE_LibXC): {
            'sss': -0.20686456,
            'sps': -0.23146656,
            'pps': -0.32711570,
            'ppp': -0.12725461,
        },
        (R1, LDA_LibXC): {
            'sss': -0.21148071,
            'sps': -0.23413930,
            'pps': -0.33049489,
            'ppp': -0.13063231,
        },
        (R2, PBE_LibXC): {
            'sss': -0.00043656,
            'sps': -0.00122881,
            'pps': -0.00466089,
            'ppp': -0.00035469,
        },
        (R2, LDA_LibXC): {
            'sss': -0.00064243,
            'sps': -0.00168346,
            'pps': -0.00587112,
            'ppp': -0.00058349,
        }
    }
    H_ref[(R1, LDA)] = H_ref[(R1, LDA_LibXC)]
    H_ref[(R2, LDA)] = H_ref[(R2, LDA_LibXC)]

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        pair = ('S', 'S')
        val = H[pair][integral][0]
        diff = abs(val - ref)
        assert diff < 5e-4, msg.format(integral, val)


@pytest.mark.parametrize('R', [R1, R2])
@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_rep2c(R, atom):
    from hotcent.slako import SlaterKosterTable

    xc = atom.xcname

    rmin, dr, N = R, R, 3
    off2c = SlaterKosterTable(atom, atom)
    off2c.run_repulsion(rmin=rmin, dr=dr, N=N, xc=xc, smoothen_tails=False,
                        ntheta=600, nr=200)
    E = off2c.erep[0]

    E_ref = {
        (R1, PBE_LibXC): 2.40963669,
        (R1, LDA_LibXC): 2.44188747,
        (R2, PBE_LibXC): 0.00626949,
        (R2, LDA_LibXC): 0.00685124,
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
    }
    return all_grids[request.param]


@pytest.mark.parametrize('grids', [R1], indirect=True)
@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_off3c(grids, atom):
    from hotcent.offsite_threecenter import Offsite3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    xc = atom.xcname

    off3c = Offsite3cTable(atom, atom)
    H = off3c.run(atom, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                  ntheta=300, nr=100, write=False)

    H_ref = {
        (R1, PBE_LibXC): {
            's_s': -0.20053906,
            's_px': -0.06779262,
            's_pz': 0.23955519,
            'px_s': -0.06779262,
            'px_px': -0.12579070,
            'px_pz': 0.08002509,
            'py_py': -0.15488882,
            'pz_s': -0.23955519,
            'pz_px': -0.08002509,
            'pz_pz': 0.19302398,
        },
        (R1, LDA_LibXC): {
            's_s': -0.20386829,
            's_px': -0.06891501,
            's_pz': 0.24284765,
            'px_s': -0.06891501,
            'px_px': -0.12838952,
            'px_pz': 0.08132246,
            'py_py': -0.15680802,
            'pz_s': -0.24284765,
            'pz_px': -0.08132246,
            'pz_pz': 0.19281687,
        },
    }
    H_ref[(R1, LDA)] = H_ref[(R1, LDA_LibXC)]

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        pair = ('S', 'S')
        val = H[pair][integral][0][1]
        diff = abs(val - ref)
        assert diff < 5e-4, msg.format(integral, val)


@pytest.mark.parametrize('grids', [R1], indirect=True)
@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_on3c(grids, atom):
    from hotcent.onsite_threecenter import Onsite3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    xc = atom.xcname

    on3c = Onsite3cTable(atom, atom)
    H = on3c.run(atom, atom, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                 ntheta=300, nr=100, write=False)

    H_ref = {
        (R1, PBE_LibXC): {
            's_s': 0.01211108,
            's_px': 0.00435683,
            's_pz': 0.01536382,
            'px_s': 0.00435683,
            'px_px': 0.00899474,
            'px_pz': 0.00509048,
            'py_py': 0.00899400,
            'pz_s': 0.01536382,
            'pz_px': 0.00509048,
            'pz_pz': 0.02464206,
        },
        (R1, LDA_LibXC): {
            's_s': 0.01312961,
            's_px': 0.00466226,
            's_pz': 0.01602638,
            'px_s': 0.00466226,
            'px_px': 0.01015447,
            'px_pz': 0.00505369,
            'py_py': 0.00997600,
            'pz_s': 0.01602638,
            'pz_px': 0.00505369,
            'pz_pz': 0.02542100,
        },
    }
    H_ref[(R1, LDA)] = H_ref[(R1, LDA_LibXC)]

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        pair = ('S', 'S')
        val = H[pair][integral][0][1]
        diff = abs(val - ref)
        assert diff < 1e-5, msg.format(integral, H[index])


@pytest.mark.parametrize('grids', [R1], indirect=True)
@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_rep3c(grids, atom):
    from hotcent.offsite_threecenter import Offsite3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    xc = atom.xcname

    off3c = Offsite3cTable(atom, atom)
    E = off3c.run_repulsion(atom, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                            write=False)

    E_ref = {
        (R1, PBE_LibXC): -0.06149970,
        (R1, LDA_LibXC): -0.05907971,
    }
    E_ref[(R1, LDA)] = E_ref[(R1, LDA_LibXC)]

    val = E[('S', 'S')]['s_s'][0][1]
    diff = abs(val - E_ref[(R, xc)])
    msg = 'Too large error for E_rep (value={0})'
    assert diff < 5e-5, msg.format(val)
