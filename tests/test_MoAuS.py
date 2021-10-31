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
            '4d': -0.08227130,
            '5s': -0.12145440,
            '5p': -0.01686066,
        },
        LDA: {
            '4d': -0.08259531,
            '5s': -0.12780887,
            '5p': -0.02060811,
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
            'sss': (-0.13629618, 0.53477087),
            'sds': (-0.02050255, 0.04140646),
            'pds': (0.01269322, -0.01597736),
            'pdp': (0.03684588, -0.08335874),
            'dds': (-0.11241881, 0.11452769),
            'ddp': (0.10996349, -0.17262107),
            'ddd': (-0.02987887, 0.05149103),
        },
        (R1, PBE_LibXC, 1): {
            'sss': (-0.13627977, 0.53477087),
            'sps': (0.12039709, -0.63723142),
            'sds': (-0.06545480, 0.15204946),
            'dds': (-0.11245464, 0.11452769),
            'ddp': (0.10992119, -0.17262107),
            'ddd': (-0.02985929, 0.05149103),
        },
        (R1, LDA, 0): {
            'sss': (-0.14222240, 0.53331310),
            'sds': (-0.02094463, 0.04132923),
            'pds': (0.01297138, -0.01714697),
            'pdp': (0.03872405, -0.08649027),
            'dds': (-0.11359304, 0.11424443),
            'ddp': (0.11211154, -0.17544340),
            'ddd': (-0.03075145, 0.05277989),
        },
        (R1, LDA, 1): {
            'sss': (-0.14220665, 0.53331310),
            'sps': (0.12766483, -0.63823267),
            'sds': (-0.06813746, 0.15440103),
            'dds': (-0.11362979, 0.11424443),
            'ddp': (0.11206783, -0.17544340),
            'ddd': (-0.03073072, 0.05277989),
        },
        (R2, PBE_LibXC, 0): {
            'sss': (-0.02625157, 0.08289152),
            'sds': (-0.01049890, 0.02164280),
            'pds': (-0.01319974, 0.02911567),
            'pdp': (0.00885725, -0.01877764),
            'dds': (-0.00602126, 0.00848859),
            'ddp': (0.00192793, -0.00200130),
            'ddd': (-0.00016546, 0.00013147),
        },
        (R2, PBE_LibXC, 1): {
            'sss': (-0.02623593, 0.08289152),
            'sps': (0.06448467, -0.28770566),
            'sds': (-0.01165471, 0.02873122),
            'dds': (-0.00604405, 0.00848859),
            'ddp': (0.00198205, -0.00200130),
            'ddd': (-0.00017971, 0.00013147),
        },
        (R2, LDA, 0): {
            'sss': (-0.02723221, 0.08196614),
            'sds': (-0.01089092, 0.02195035),
            'pds': (-0.01433199, 0.03106853),
            'pdp': (0.00901210, -0.01872974),
            'dds': (-0.00636185, 0.00885245),
            'ddp': (0.00203844, -0.00209340),
            'ddd': (-0.00017493, 0.00013777),
        },
        (R2, LDA, 1): {
            'sss': (-0.02721722, 0.08196614),
            'sps': (0.06642412, -0.27886726),
            'sds': (-0.01218490, 0.02893175),
            'dds': (-0.00638514, 0.00885245),
            'ddp': (0.00209490, -0.00209340),
            'ddd': (-0.00018984, 0.00013777),
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
            'sss': -0.03067390,
            'sps': -0.03820173,
            'sds': -0.01805640,
            'pps': -0.06005506,
            'ppp': -0.01406964,
            'pds': -0.01950308,
            'pdp': -0.01322453,
            'dds': -0.03677458,
            'ddp': -0.03301580,
            'ddd': -0.00867340,
        },
        (R1, LDA): {
            'sss': -0.03268958,
            'sps': -0.04052471,
            'sds': -0.01906977,
            'pps': -0.06312153,
            'ppp': -0.01573458,
            'pds': -0.02079088,
            'pdp': -0.01455406,
            'dds': -0.03969156,
            'ddp': -0.03564729,
            'ddd': -0.00958039,
        },
        (R2, PBE_LibXC): {
            'sss': -0.00141769,
            'sps': -0.00430800,
            'sds': -0.00040605,
            'pps': -0.01723373,
            'ppp': -0.00122336,
            'pds': -0.00095167,
            'pdp': -0.00007439,
            'dds': -0.00018444,
            'ddp': -0.00002978,
            'ddd': -0.00000132,
        },
        (R2, LDA): {
            'sss': -0.00195895,
            'sps': -0.00536216,
            'sds': -0.00066911,
            'pps': -0.01896033,
            'ppp': -0.00150399,
            'pds': -0.00142888,
            'pdp': -0.00017213,
            'dds': -0.00036951,
            'ddp': -0.00007939,
            'ddd': -0.00000628,
        },
    }

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        pair = ('Mo', 'Mo')
        val = H[pair][integral][0]
        diff = abs(val - ref)
        assert diff < 1e-4, msg.format(integral, val)


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
            's_s': -0.08468792,
            's_dxz': 0.05218732,
            's_dx2-y2': -0.01592988,
            's_dz2': -0.03375553,
            'px_s': -0.05529695,
            'px_dxz': 0.03386928,
            'px_dx2-y2': -0.01534619,
            'px_dz2': -0.00754192,
            'py_dxy': -0.01073421,
            'py_dyz': 0.01679929,
            'pz_s': -0.06606535,
            'pz_dxz': 0.04845203,
            'pz_dx2-y2': -0.01791401,
            'pz_dz2': -0.02418833,
            'dxy_dxy': -0.01195118,
            'dxy_dyz': 0.01994505,
            'dyz_dxy': -0.01576464,
            'dyz_dyz': 0.03870742,
            'dxz_s': -0.07968780,
            'dxz_dxz': 0.05792226,
            'dxz_dx2-y2': -0.01508556,
            'dxz_dz2': -0.04364708,
            'dx2-y2_s': -0.02444356,
            'dx2-y2_dxz': 0.01267145,
            'dx2-y2_dx2-y2': -0.00792973,
            'dx2-y2_dz2': -0.00536507,
            'dz2_s': -0.04455632,
            'dz2_dxz': 0.03951567,
            'dz2_dx2-y2': -0.00844304,
            'dz2_dz2': -0.05892671,
        },
        (R1, LDA): {
            's_s': -0.08500540,
            's_dxz': 0.05272382,
            's_dx2-y2': -0.01603899,
            's_dz2': -0.03424394,
            'px_s': -0.05598554,
            'px_dxz': 0.03476719,
            'px_dx2-y2': -0.01573873,
            'px_dz2': -0.00775236,
            'py_dxy': -0.01108564,
            'py_dyz': 0.01740105,
            'pz_s': -0.06759865,
            'pz_dxz': 0.04978519,
            'pz_dx2-y2': -0.01835807,
            'pz_dz2': -0.02489401,
            'dxy_dxy': -0.01221289,
            'dxy_dyz': 0.02031944,
            'dyz_dxy': -0.01608257,
            'dyz_dyz': 0.03958910,
            'dxz_s': -0.08021105,
            'dxz_dxz': 0.05914125,
            'dxz_dx2-y2': -0.01544340,
            'dxz_dz2': -0.04432435,
            'dx2-y2_s': -0.02435337,
            'dx2-y2_dxz': 0.01301241,
            'dx2-y2_dx2-y2': -0.00816329,
            'dx2-y2_dz2': -0.00550642,
            'dz2_s': -0.04515487,
            'dz2_dxz': 0.04019263,
            'dz2_dx2-y2': -0.00863665,
            'dz2_dz2': -0.05976538,
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
            's_s': 0.00829476,
            's_px': 0.00380302,
            's_pz': 0.01007826,
            's_dxz': 0.00403895,
            's_dx2-y2': 0.00039091,
            's_dz2': 0.00641582,
            'px_s': 0.00380302,
            'px_px': 0.00362620,
            'px_pz': 0.00491643,
            'px_dxz': 0.00290922,
            'px_dx2-y2': 0.00065699,
            'px_dz2': 0.00168968,
            'py_py': 0.00234864,
            'py_dxy': 0.00075127,
            'py_dyz': 0.00213557,
            'pz_s': 0.01007826,
            'pz_px': 0.00491643,
            'pz_pz': 0.01368770,
            'pz_dxz': 0.00433487,
            'pz_dx2-y2': 0.00038683,
            'pz_dz2': 0.00745151,
            'dxy_py': 0.00075127,
            'dxy_dxy': 0.00082895,
            'dxy_dyz': 0.00101041,
            'dyz_py': 0.00213557,
            'dyz_dxy': 0.00101041,
            'dyz_dyz': 0.00344466,
            'dxz_s': 0.00403895,
            'dxz_px': 0.00290922,
            'dxz_pz': 0.00433487,
            'dxz_dxz': 0.00432187,
            'dxz_dx2-y2': 0.00078109,
            'dxz_dz2': 0.00292518,
            'dx2-y2_s': 0.00039091,
            'dx2-y2_px': 0.00065699,
            'dx2-y2_pz': 0.00038683,
            'dx2-y2_dxz': 0.00078109,
            'dx2-y2_dx2-y2': 0.00068368,
            'dx2-y2_dz2': 0.00010353,
            'dz2_s': 0.00641582,
            'dz2_px': 0.00168968,
            'dz2_pz': 0.00745151,
            'dz2_dxz': 0.00292518,
            'dz2_dx2-y2': 0.00010353,
            'dz2_dz2': 0.00776057,
        },
        (R1, LDA): {
            's_s': 0.00898655,
            's_px': 0.00410272,
            's_pz': 0.01078541,
            's_dxz': 0.00421721,
            's_dx2-y2': 0.00039656,
            's_dz2': 0.00665372,
            'px_s': 0.00410272,
            'px_px': 0.00418070,
            'px_pz': 0.00515865,
            'px_dxz': 0.00337530,
            'px_dx2-y2': 0.00081075,
            'px_dz2': 0.00159929,
            'py_py': 0.00283959,
            'py_dxy': 0.00089537,
            'py_dyz': 0.00261155,
            'pz_s': 0.01078541,
            'pz_px': 0.00515865,
            'pz_pz': 0.01456815,
            'pz_dxz': 0.00447618,
            'pz_dx2-y2': 0.00038187,
            'pz_dz2': 0.00785805,
            'dxy_py': 0.00089537,
            'dxy_dxy': 0.00125160,
            'dxy_dyz': 0.00118732,
            'dyz_py': 0.00261155,
            'dyz_dxy': 0.00118732,
            'dyz_dyz': 0.00458366,
            'dxz_s': 0.00421721,
            'dxz_px': 0.00337530,
            'dxz_pz': 0.00447618,
            'dxz_dxz': 0.00545769,
            'dxz_dx2-y2': 0.00101075,
            'dxz_dz2': 0.00293157,
            'dx2-y2_s': 0.00039656,
            'dx2-y2_px': 0.00081075,
            'dx2-y2_pz': 0.00038187,
            'dx2-y2_dxz': 0.00101075,
            'dx2-y2_dx2-y2': 0.00114567,
            'dx2-y2_dz2': 0.00008667,
            'dz2_s': 0.00665372,
            'dz2_px': 0.00159929,
            'dz2_pz': 0.00785805,
            'dz2_dxz': 0.00293157,
            'dz2_dx2-y2': 0.00008667,
            'dz2_dz2': 0.00893071,
        },
    }

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        pair = ('Au', 'S')
        val = H[pair][integral][0][1]
        diff = abs(val - ref)
        assert diff < 5e-6, msg.format(integral, val)


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
        (R1, PBE_LibXC): -0.01991109,
        (R1, LDA): -0.01891928,
    }

    tol = 1e-5
    val = E[('Mo', 'Au')]['s_s'][0][1]
    diff = abs(val - E_ref[(R, xc)])
    msg = 'Too large error for E_rep (value={0})'
    assert diff < tol, msg.format(val)
