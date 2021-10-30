""" Tests with three different elements (Mo/Au/S), KB pseudopotentials
and LDA and GGA functionals.

Reference values come from the BeckeHarris tool, unless mentioned otherwise.
"""
import pytest
import numpy as np
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT
from hotcent.slako import INTEGRALS


R1 = 4.0
R2 = 8.0

LDA = 'LDA'
PBE_LibXC = 'GGA_X_PBE+GGA_C_PBE'


@pytest.fixture(scope='module')
def atoms(request):
    xcname = request.param

    configurations = {
        'Mo': '[Kr] 4d5 5s1 5p0',
        'Au': '[Xe] 4f14 5d10 6s1',
        'S': '[Ne] 3s2 3p4',
    }

    wf_confinements = {
        'Mo': {
            '4d': SoftConfinement(rc=7.183434),
            '5s': SoftConfinement(rc=9.907957),
            '5p': SoftConfinement(rc=13.664737),
        },
        'Au': {
            '5d': SoftConfinement(rc=5.837419),
            '6s': SoftConfinement(rc=8.340823),
        },
        'S': {
            '3s': SoftConfinement(rc=5.550736),
            '3p': SoftConfinement(rc=7.046078),
        },
    }

    atoms = []
    for element in ['Mo', 'Au', 'S']:
        valence = list(wf_confinements[element].keys())
        pp = KleinmanBylanderPP('./pseudos/{0}.psf'.format(element), valence,
                                verbose=True)
        atom = PseudoAtomicDFT(element, pp,
                               xc=xcname,
                               nodegpts=1000,
                               valence=valence,
                               configuration=configurations[element],
                               wf_confinement=wf_confinements[element],
                               perturbative_confinement=True,
                               scalarrel=False,
                               timing=False,
                               txt=None,
                               )
        atom.run()
        atom.pp.build_projectors(atom)
        atoms.append(atom)

    for atom1 in atoms:
        for atom2 in atoms:
            atom1.pp.build_overlaps(atom2, atom1, rmin=2., rmax=9.)
    return atoms


@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_on1c(atoms):
    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    H_ref = {
        PBE_LibXC: {
            '4d': -0.08227156,
            '5s': -0.12145480,
            '5p': -0.01686089,
        },
        LDA: {
            '4d': -0.08259557,
            '5s': -0.12780923,
            '5p': -0.02060831,
        },
    }

    htol = 1e-5
    msg = 'Too large error for H_{0} (value={1})'
    for nl, ref in H_ref[xc].items():
        H = atom_Mo.get_onecenter_integral(nl)
        H_diff = abs(H - ref)
        assert H_diff < htol, msg.format(nl, H)


@pytest.mark.parametrize('R', [R1, R2])
@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_off2c(R, atoms):
    from hotcent.slako import SlaterKosterTable

    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    rmin, dr, N = R, R, 2
    off2c = SlaterKosterTable(atom_Mo, atom_Au)
    off2c.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
              smoothen_tails=False, ntheta=300, nr=100)

    HS_ref = {
        (R1, PBE_LibXC, 0): {
            'sss': (-0.13625527, 0.53477109),
            'sds': (-0.02059730, 0.04140634),
            'pds': (0.01272648, -0.01597811),
            'pdp': (0.03715148, -0.08335847),
            'dds': (-0.11243781, 0.11452742),
            'ddp': (0.10998473, -0.17262141),
            'ddd': (-0.02992819, 0.05149114),
        },
        (R1, PBE_LibXC, 1): {
            'sss': (-0.13626730, 0.53477109),
            'sps': (0.12034448, -0.63723093),
            'sds': (-0.06540094, 0.15204890),
            'dds': (-0.11243812, 0.11452742),
            'ddp': (0.10995282, -0.17262141),
            'ddd': (-0.02986330, 0.05149114),
        },
        (R1, LDA, 0): {
            'sss': (-0.14218103, 0.53331329),
            'sds': (-0.02104170, 0.04132913),
            'pds': (0.01300529, -0.01714769),
            'pdp': (0.03903974, -0.08648998),
            'dds': (-0.11361213, 0.11424419),
            'ddp': (0.11213245, -0.17544371),
            'ddd': (-0.03080280, 0.05278001),
        },
        (R1, LDA, 1): {
            'sss': (-0.14219221, 0.53331329),
            'sps': (0.12760934, -0.63823215),
            'sds': (-0.06808161, 0.15440045),
            'dds': (-0.11361284, 0.11424419),
            'ddp': (0.11209929, -0.17544371),
            'ddd': (-0.03073484, 0.05278001),
        },
        (R2, PBE_LibXC, 0): {
            'sss': (-0.02623596, 0.08289264),
            'sds': (-0.01048895, 0.02164274),
            'pds': (-0.01321184, 0.02911574),
            'pdp': (0.00883598, -0.01877728),
            'dds': (-0.00592549, 0.00848859),
            'ddp': (0.00194103, -0.00200137),
            'ddd': (-0.00016917, 0.00013151),
        },
        (R2, PBE_LibXC, 1): {
            'sss': (-0.02624584, 0.08289264),
            'sps': (0.06448772, -0.28770736),
            'sds': (-0.01152764, 0.02873165),
            'dds': (-0.00606628, 0.00848859),
            'ddp': (0.00194864, -0.00200137),
            'ddd': (-0.00016495, 0.00013151),
        },
        (R2, LDA, 0): {
            'sss': (-0.02721740, 0.08196722),
            'sds': (-0.01088095, 0.02195029),
            'pds': (-0.01434407, 0.03106861),
            'pdp': (0.00899208, -0.01872935),
            'dds': (-0.00626375, 0.00885245),
            'ddp': (0.00205225, -0.00209347),
            'ddd': (-0.00017877, 0.00013782),
        },
        (R2, LDA, 1): {
            'sss': (-0.02722679, 0.08196722),
            'sps': (0.06642561, -0.27886886),
            'sds': (-0.01205394, 0.02893220),
            'dds': (-0.00640841, 0.00885245),
            'ddp': (0.00205986, -0.00209347),
            'ddd': (-0.00017436, 0.00013782),
        },
    }

    for i in range(2):
        H, S = off2c.tables[i][0, :20], off2c.tables[i][0, 20:41]

        htol = 5e-4
        stol = 1e-4
        msg = 'Too large error for {0}[{1}]_{2} (value={3})'

        for integral, ref in HS_ref[(R, xc, i)].items():
            index = INTEGRALS.index(integral)

            H_diff = abs(H[index] - ref[0])
            assert H_diff < htol, msg.format('H', i, integral, H[index])

            S_diff = abs(S[index] - ref[1])
            assert S_diff < stol, msg.format('S', i, integral, S[index])


@pytest.mark.parametrize('R', [R1, R2])
@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_on2c(R, atoms):
    from hotcent.onsite_twocenter import Onsite2cTable

    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    rmin, dr, N = R, R, 2
    on2c = Onsite2cTable(atom_Mo, atom_Mo)
    H = on2c.run(atom_Au, rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
                 smoothen_tails=False, ntheta=300, nr=100, write=False)

    H_ref = {
        (R1, PBE_LibXC): {
            'sss': -0.03066006,
            'sps': -0.03817858,
            'sds': -0.01804072,
            'pps': -0.06001689,
            'ppp': -0.01407190,
            'pds': -0.01947427,
            'pdp': -0.01322642,
            'dds': -0.03675210,
            'ddp': -0.03301960,
            'ddd': -0.00867401,
        },
        (R1, LDA): {
            'sss': -0.03267520,
            'sps': -0.04050045,
            'sds': -0.01905343,
            'pps': -0.06308134,
            'ppp': -0.01573625,
            'pds': -0.02076073,
            'pdp': -0.01455591,
            'dds': -0.03966799,
            'ddp': -0.03565135,
            'ddd': -0.00958096,
        },
        (R2, PBE_LibXC): {
            'sss': -0.00141768,
            'sps': -0.00430863,
            'sds': -0.00040599,
            'pps': -0.01723770,
            'ppp': -0.00122478,
            'pds': -0.00095157,
            'pdp': -0.00007427,
            'dds': -0.00018447,
            'ddp': -0.00002972,
            'ddd': -0.00000131,
        },
        (R2, LDA): {
            'sss': -0.00195895,
            'sps': -0.00536222,
            'sds': -0.00066912,
            'pps': -0.01896093,
            'ppp': -0.00150422,
            'pds': -0.00142890,
            'pdp': -0.00017212,
            'dds': -0.00036952,
            'ddp': -0.00007939,
            'ddd': -0.00000628,
        },
    }

    htol = 2e-4
    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        pair = ('Mo', 'Mo')
        val = H[pair][integral][0]
        diff = abs(val - ref)
        assert diff < 5e-4, msg.format(integral, val)


@pytest.mark.parametrize('R', [R1, R2])
@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_rep2c(R, atoms):
    from hotcent.slako import SlaterKosterTable

    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    for i, (a1, a2) in enumerate([(atom_Mo, atom_Au), (atom_Au, atom_Mo)]):
        rmin, dr, N = R, R, 3
        off2c = SlaterKosterTable(a1, a2)
        off2c.run_repulsion(rmin=rmin, dr=dr, N=N, xc=xc, smoothen_tails=False,
                            ntheta=600, nr=200)
        E = off2c.erep[0]

        E_ref = {
            (R1, PBE_LibXC): 0.51045989,
            (R1, LDA): 0.51516297,
            (R2, PBE_LibXC): 0.00124824,
            (R2, LDA): 0.00152799,
        }

        etol = 5e-5
        E_diff = abs(E - E_ref[(R, xc)])
        msg = 'Too large error for E_rep (value={0})'
        assert E_diff < etol, msg.format(E)


@pytest.fixture(scope='module')
def grids(request):
    all_grids = {
        R1: (R1, np.array([R1]), np.array([2.]), np.array([0.6*np.pi])),
    }
    return all_grids[request.param]


@pytest.mark.parametrize('grids', [R1], indirect=True)
@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_off3c(grids, atoms):
    from hotcent.offsite_threecenter import Offsite3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    off3c = Offsite3cTable(atom_Mo, atom_Au)
    H = off3c.run(atom_S, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                  write=False)
    H_ref = {
        (R1, PBE_LibXC): {
            's_s': -0.08474124,
            's_dxz': 0.05222017,
            's_dx2-y2': -0.01594812,
            's_dz2': -0.03377308,
            'px_s': -0.05533098,
            'px_dxz': 0.03389649,
            'px_dx2-y2': -0.01536485,
            'px_dz2': -0.00753834,
            'py_dxy': -0.01073777,
            'py_dyz': 0.01680702,
            'pz_s': -0.06610126,
            'pz_dxz': 0.04847177,
            'pz_dx2-y2': -0.01792681,
            'pz_dz2': -0.02419966,
            'dxy_dxy': -0.01195056,
            'dxy_dyz': 0.01995707,
            'dyz_dxy': -0.01576844,
            'dyz_dyz': 0.03873723,
            'dxz_s': -0.07971265,
            'dxz_dxz': 0.05792652,
            'dxz_dx2-y2': -0.01510574,
            'dxz_dz2': -0.04361923,
            'dx2-y2_s': -0.02447859,
            'dx2-y2_dxz': 0.01270348,
            'dx2-y2_dx2-y2': -0.00795919,
            'dx2-y2_dz2': -0.00534208,
            'dz2_s': -0.04457528,
            'dz2_dxz': 0.03950836,
            'dz2_dx2-y2': -0.00842968,
            'dz2_dz2': -0.05899668,
        },
        (R1, LDA): {
            's_s': -0.08505762,
            's_dxz': 0.05275620,
            's_dx2-y2': -0.01605709,
            's_dz2': -0.03426070,
            'px_s': -0.05601983,
            'px_dxz': 0.03479452,
            'px_dx2-y2': -0.01575772,
            'px_dz2': -0.00774840,
            'py_dxy': -0.01108928,
            'py_dyz': 0.01740890,
            'pz_s': -0.06763416,
            'pz_dxz': 0.04980483,
            'pz_dx2-y2': -0.01837093,
            'pz_dz2': -0.02490493,
            'dxy_dxy': -0.01221202,
            'dxy_dyz': 0.02033114,
            'dyz_dxy': -0.01608626,
            'dyz_dyz': 0.03961876,
            'dxz_s': -0.08023384,
            'dxz_dxz': 0.05914376,
            'dxz_dx2-y2': -0.01546334,
            'dxz_dz2': -0.04429537,
            'dx2-y2_s': -0.02438830,
            'dx2-y2_dxz': 0.01304442,
            'dx2-y2_dx2-y2': -0.00819303,
            'dx2-y2_dz2': -0.00548303,
            'dz2_s': -0.04517071,
            'dz2_dxz': 0.04018504,
            'dz2_dx2-y2': -0.00862280,
            'dz2_dz2': -0.05983252,
        },
    }

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        pair = ('Mo', 'Au')
        val = H[pair][integral][0][1]
        diff = abs(val - ref)
        assert diff < 5e-4, msg.format(integral, val)


@pytest.mark.parametrize('grids', [R1], indirect=True)
@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_on3c(grids, atoms):
    from hotcent.onsite_threecenter import Onsite3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    on3c = Onsite3cTable(atom_Mo, atom_Mo)
    H = on3c.run(atom_Au, atom_S, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                 write=False)

    H_ref = {
        (R1, PBE_LibXC): {
            's_s': 0.00829577,
            's_px': 0.00380357,
            's_pz': 0.01007909,
            's_dxz': 0.00404084,
            's_dx2-y2': 0.00039111,
            's_dz2': 0.00641591,
            'px_s': 0.00380357,
            'px_px': 0.00362646,
            'px_pz': 0.00491688,
            'px_dxz': 0.00290991,
            'px_dx2-y2': 0.00065705,
            'px_dz2': 0.00169039,
            'py_py': 0.00234985,
            'py_dxy': 0.00075148,
            'py_dyz': 0.00213596,
            'pz_s': 0.01007909,
            'pz_px': 0.00491688,
            'pz_pz': 0.01368827,
            'pz_dxz': 0.00433637,
            'pz_dx2-y2': 0.00038697,
            'pz_dz2': 0.00745141,
            'dxy_py': 0.00075148,
            'dxy_dxy': 0.00082934,
            'dxy_dyz': 0.00101106,
            'dyz_py': 0.00213596,
            'dyz_dxy': 0.00101106,
            'dyz_dyz': 0.00344459,
            'dxz_s': 0.00404084,
            'dxz_px': 0.00290991,
            'dxz_pz': 0.00433637,
            'dxz_dxz': 0.00432475,
            'dxz_dx2-y2': 0.00078149,
            'dxz_dz2': 0.00292772,
            'dx2-y2_s': 0.00039111,
            'dx2-y2_px': 0.00065705,
            'dx2-y2_pz': 0.00038697,
            'dx2-y2_dxz': 0.00078149,
            'dx2-y2_dx2-y2': 0.00068368,
            'dx2-y2_dz2': 0.00010435,
            'dz2_s': 0.00641591,
            'dz2_px': 0.00169039,
            'dz2_pz': 0.00745141,
            'dz2_dxz': 0.00292772,
            'dz2_dx2-y2': 0.00010435,
            'dz2_dz2': 0.00775556,
        },
        (R1, LDA): {
            's_s': 0.00898654,
            's_px': 0.00410273,
            's_pz': 0.01078539,
            's_dxz': 0.00421723,
            's_dx2-y2': 0.00039659,
            's_dz2': 0.00665361,
            'px_s': 0.00410273,
            'px_px': 0.00418072,
            'px_pz': 0.00515865,
            'px_dxz': 0.00337534,
            'px_dx2-y2': 0.00081077,
            'px_dz2': 0.00159929,
            'py_py': 0.00283963,
            'py_dxy': 0.00089537,
            'py_dyz': 0.00261155,
            'pz_s': 0.01078539,
            'pz_px': 0.00515865,
            'pz_pz': 0.01456810,
            'pz_dxz': 0.00447619,
            'pz_dx2-y2': 0.00038190,
            'pz_dz2': 0.00785794,
            'dxy_py': 0.00089537,
            'dxy_dxy': 0.00125161,
            'dxy_dyz': 0.00118733,
            'dyz_py': 0.00261155,
            'dyz_dxy': 0.00118733,
            'dyz_dyz': 0.00458365,
            'dxz_s': 0.00421723,
            'dxz_px': 0.00337534,
            'dxz_pz': 0.00447619,
            'dxz_dxz': 0.00545781,
            'dxz_dx2-y2': 0.00101080,
            'dxz_dz2': 0.00293144,
            'dx2-y2_s': 0.00039659,
            'dx2-y2_px': 0.00081077,
            'dx2-y2_pz': 0.00038190,
            'dx2-y2_dxz': 0.00101080,
            'dx2-y2_dx2-y2': 0.00114567,
            'dx2-y2_dz2': 0.00008670,
            'dz2_s': 0.00665361,
            'dz2_px': 0.00159929,
            'dz2_pz': 0.00785794,
            'dz2_dxz': 0.00293144,
            'dz2_dx2-y2': 0.00008670,
            'dz2_dz2': 0.00893035,
        },
    }

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        pair = ('Au', 'S')
        val = H[pair][integral][0][1]
        diff = abs(val - ref)
        assert diff < 1e-5, msg.format(integral, val)


@pytest.mark.parametrize('grids', [R1], indirect=True)
@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_rep3c(grids, atoms):
    from hotcent.offsite_threecenter import Offsite3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    off3c = Offsite3cTable(atom_Mo, atom_Au)
    E = off3c.run_repulsion(atom_S, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                            write=False)

    E_ref = {
        (R1, PBE_LibXC): -0.01991227,
        (R1, LDA): -0.01891973,
    }

    tol = 1e-5
    val = E[('Mo', 'Au')]['s_s'][0][1]
    diff = abs(val - E_ref[(R, xc)])
    msg = 'Too large error for E_rep (value={0})'
    assert diff < tol, msg.format(val)
