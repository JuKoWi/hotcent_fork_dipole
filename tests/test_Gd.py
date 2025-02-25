""" Tests with an f-element (Gd), all-electron, and LDA and GGA functionals.

Reference values come from the BeckeHarris tool, unless mentioned otherwise.
"""
import pytest
from hotcent.confinement import SoftConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.slako import INTEGRALS


R1 = 4.0

LDA = 'LDA'
PBE_LibXC = 'GGA_X_PBE+GGA_C_PBE'


@pytest.fixture(scope='module')
def atom(request):
    wf_confinement = {
        '4f': SoftConfinement(rc=5.),
        '5d': SoftConfinement(rc=6.),
        '6s': SoftConfinement(rc=7.),
        '6p': SoftConfinement(rc=9.),
    }
    valence = list(wf_confinement.keys())
    xcname = request.param
    atom = AtomicDFT('Gd',
                     xc=xcname,
                     nodegpts=1000,
                     valence=valence,
                     configuration='[Xe] 4f7 5d1 6s2 6p0',
                     wf_confinement=wf_confinement,
                     perturbative_confinement=True,
                     scalarrel=True,
                     timing=False,
                     txt=None,
                     )
    atom.run()
    atom.pp.build_projectors(atom)
    atom.pp.build_overlaps(atom, atom, rmin=3., rmax=5., N=100)
    return atom


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atom', [PBE_LibXC, LDA], indirect=True)
def test_off2c(R, atom):
    from hotcent.offsite_twocenter import Offsite2cTable

    xc = atom.xcname

    rmin, dr, N = R, R, 2
    off2c = Offsite2cTable(atom, atom)
    off2c.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
              smoothen_tails=False, ntheta=300, nr=100)
    off2c.write()
    H, S = off2c.tables[(0, 0, 0)][0, :20], off2c.tables[(0, 0, 0)][0, 20:41]

    HS_ref = {
        (R1, PBE_LibXC): {
            'sss': (-0.04546340, 0.35939491),
            'sps': (-0.00318568, 0.43841218),
            'sds': (-0.01624041, 0.00322715),
            'sfs': (0.01246280, -0.02042406),
            'pps': (-0.04404193, -0.05529406),
            'ppp': (-0.00990217, 0.49558360),
            'pds': (0.05625428, -0.16784240),
            'pdp': (0.03056944, -0.15276787),
            'pfs': (-0.01889029, 0.02624270),
            'pfp': (0.00032220, -0.00082892),
            'dds': (-0.08293694, 0.12843255),
            'ddp': (0.12149843, -0.33481693),
            'ddd': (-0.04328317, 0.10911230),
            'dfs': (0.04104442, -0.03126264),
            'dfp': (-0.04639077, 0.05769863),
            'dfd': (0.01967116, -0.02677881),
            'ffs': (0.02438273, -0.01697512),
            'ffp': (-0.02182108, 0.02011762),
            'ffd': (0.00953207, -0.00955611),
            'fff': (-0.00143984, 0.00136783),
        },
        (R1, LDA): {
            'sss': (-0.04795308, 0.35763797),
            'sps': (-0.00582536, 0.43818523),
            'sds': (-0.01588192, 0.00050996),
            'sfs': (0.01240072, -0.02009205),
            'pps': (-0.04423500, -0.05462753),
            'ppp': (-0.01194286, 0.49486010),
            'pds': (0.05802347, -0.16662596),
            'pdp': (0.03148717, -0.15315067),
            'pfs': (-0.01908609, 0.02613384),
            'pfp': (0.00024189, -0.00057715),
            'dds': (-0.08426760, 0.12940552),
            'ddp': (0.12459300, -0.33603126),
            'ddd': (-0.04418465, 0.10927541),
            'dfs': (0.04150963, -0.03136080),
            'dfp': (-0.04711406, 0.05798783),
            'dfd': (0.01995583, -0.02685326),
            'ffs': (0.02460971, -0.01697958),
            'ffp': (-0.02206260, 0.02014547),
            'ffd': (0.00963948, -0.00957236),
            'fff': (-0.00145425, 0.00137026),
        },
    }

    htol = 1e-4
    stol = 1e-4
    msg = 'Too large error for {0}_{1} (value={2})'

    for integral, ref in HS_ref[(R, xc)].items():
        index = INTEGRALS.index(integral)

        H_diff = abs(H[index] - ref[0])
        assert H_diff < htol, msg.format('H', integral, H[index])

        S_diff = abs(S[index] - ref[1])
        assert S_diff < stol, msg.format('S', integral, S[index])


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atom', [PBE_LibXC, LDA], indirect=True)
def test_on2c(R, atom):
    from hotcent.onsite_twocenter import Onsite2cTable

    xc = atom.xcname

    rmin, dr, N = R, R, 2
    on2c = Onsite2cTable(atom, atom)
    on2c.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc, shift=False,
             smoothen_tails=False, ntheta=900, nr=600, wflimit=1e-12)
    H = on2c.tables[(0, 0)]

    H_ref = {
        (R1, PBE_LibXC): {
            'sss': 0.00549407,
            'sps': -0.00934387,
            'sds': -0.02273461,
            'sfs': -0.00656944,
            'pps': -0.00045978,
            'ppp': -0.01252857,
            'pds': 0.03704286,
            'pdp': -0.00675542,
            'pfs': 0.01072312,
            'pfp': 0.00099897,
            'dds': -0.03548983,
            'ddp': -0.05825463,
            'ddd': -0.02040735,
            'dfs': -0.01783307,
            'dfp': -0.01919489,
            'dfd': -0.00852624,
            'ffs': -0.02722755,
            'ffp': -0.02517198,
            'ffd': -0.01859051,
            'fff': -0.01190265,
        },
        (R1, LDA): {
            'sss': 0.00551890,
            'sps': -0.00925365,
            'sds': -0.02260827,
            'sfs': -0.00650503,
            'pps': -0.00196092,
            'ppp': -0.01333250,
            'pds': 0.03694796,
            'pdp': -0.00665622,
            'pfs': 0.01075374,
            'pfp': 0.00101677,
            'dds': -0.03610375,
            'ddp': -0.05868062,
            'ddd': -0.02046285,
            'dfs': -0.01808605,
            'dfp': -0.01940579,
            'dfd': -0.00854609,
            'ffs': -0.02712045,
            'ffp': -0.02502416,
            'ffd': -0.01836976,
            'fff': -0.01167828,
        },
    }

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        index = INTEGRALS.index(integral)
        val = H[0, index]
        diff = abs(val - ref)
        assert diff < 1e-4, msg.format(integral, val)


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atom', [PBE_LibXC, LDA], indirect=True)
def test_rep2c(R, atom):
    from hotcent.repulsion_twocenter import Repulsion2cTable

    xc = atom.xcname

    rmin, dr, N = R, R, 3
    rep2c = Repulsion2cTable(atom, atom)
    rep2c.run(rmin=rmin, dr=dr, N=N, xc=xc, smoothen_tails=False, shift=False,
              ntheta=900, nr=300)
    E = rep2c.erep[0]

    E_ref = {
        (R1, PBE_LibXC): 0.49887749,
        (R1, LDA): 0.46719540,
    }

    etol = 5e-5
    E_diff = abs(E - E_ref[(R, xc)])
    msg = 'Too large error for E_rep (value={0})'
    assert E_diff < etol, msg.format(E)
