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
            '4d': -0.08227063,
            '5s': -0.12145420,
            '5p': -0.01686008,
        },
        LDA: {
            '4d': -0.08259461,
            '5s': -0.12780867,
            '5p': -0.02060761,
        },
    }

    htol = 5e-6
    msg = 'Too large error for H_{0} (value={1})'
    for nl, ref in H_ref[xc].items():
        H = atom_Mo.get_onecenter_integrals(nl, nl)[0]
        H_diff = abs(H - ref)
        assert H_diff < htol, msg.format(nl, H)


@pytest.mark.parametrize('R', [R1, R2])
@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_off2c(R, atoms):
    from hotcent.offsite_twocenter import Offsite2cTable

    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    rmin, dr, N = R, R, 2
    off2c = Offsite2cTable(atom_Mo, atom_Au)
    off2c.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
              smoothen_tails=False, ntheta=300, nr=100)

    HS_ref = {
        (R1, PBE_LibXC, 0): {
            'sss': (-0.13627975, 0.53476654),
            'sds': (-0.02055839, 0.04139772),
            'pds': (0.01278117, -0.01596903),
            'pdp': (0.03680884, -0.08334578),
            'dds': (-0.11244298, 0.11452413),
            'ddp': (0.10992556, -0.17262639),
            'ddd': (-0.02985695, 0.05148756),
        },
        (R1, PBE_LibXC, 1): {
            'sss': (-0.13627887, 0.53476654),
            'sps': (0.12039688, -0.63723445),
            'sds': (-0.06542687, 0.15204281),
            'dds': (-0.11244274, 0.11452413),
            'ddp': (0.10992695, -0.17262639),
            'ddd': (-0.02985721, 0.05148756),
        },
        (R1, LDA, 0): {
            'sss': (-0.14220668, 0.53330897),
            'sds': (-0.02100180, 0.04132029),
            'pds': (0.01306082, -0.01713844),
            'pdp': (0.03868762, -0.08647678),
            'dds': (-0.11361757, 0.11424080),
            'ddp': (0.11207245, -0.17544888),
            'ddd': (-0.03072826, 0.05277629),
        },
        (R1, LDA, 1): {
            'sss': (-0.14220577, 0.53330897),
            'sps': (0.12766464, -0.63823570),
            'sds': (-0.06810850, 0.15439422),
            'dds': (-0.11361731, 0.11424080),
            'ddp': (0.11207388, -0.17544888),
            'ddd': (-0.03072854, 0.05277629),
        },
        (R2, PBE_LibXC, 0): {
            'sss': (-0.02623889, 0.08288329),
            'sds': (-0.01056420, 0.02163522),
            'pds': (-0.01325186, 0.02910852),
            'pdp': (0.00890873, -0.01877149),
            'dds': (-0.00605358, 0.00848006),
            'ddp': (0.00195294, -0.00199644),
            'ddd': (-0.00016594, 0.00013093),
        },
        (R2, PBE_LibXC, 1): {
            'sss': (-0.02623929, 0.08288329),
            'sps': (0.06448442, -0.28770206),
            'sds': (-0.01154098, 0.02872762),
            'dds': (-0.00605751, 0.00848006),
            'ddp': (0.00195074, -0.00199644),
            'ddd': (-0.00016662, 0.00013093),
        },
        (R2, LDA, 0): {
            'sss': (-0.02722000, 0.08195823),
            'sds': (-0.01095766, 0.02194262),
            'pds': (-0.01438793, 0.03106099),
            'pdp': (0.00906440, -0.01872350),
            'dds': (-0.00639524, 0.00884368),
            'ddp': (0.00206439, -0.00208836),
            'ddd': (-0.00017539, 0.00013721),
        },
        (R2, LDA, 1): {
            'sss': (-0.02722038, 0.08195823),
            'sps': (0.06642387, -0.27886389),
            'sds': (-0.01206749, 0.02892776),
            'dds': (-0.00639936, 0.00884368),
            'ddp': (0.00206206, -0.00208836),
            'ddd': (-0.00017610, 0.00013721),
        },
    }

    for i in range(2):
        H = off2c.tables[(i, 0, 0)][0, :20]
        S = off2c.tables[(i, 0, 0)][0, 20:41]

        htol = 1e-4
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
    on2c = Onsite2cTable(atom_Mo, atom_Au)
    on2c.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
             smoothen_tails=False, ntheta=300, nr=100)
    H = on2c.tables[(0, 0)]

    H_ref = {
        (R1, PBE_LibXC): {
            'sss': -0.03067385,
            'sps': -0.03820167,
            'sds': -0.01805628,
            'pps': -0.06005515,
            'ppp': -0.01406962,
            'pds': -0.01950260,
            'pdp': -0.01322438,
            'dds': -0.03677451,
            'ddp': -0.03301575,
            'ddd': -0.00867336,
        },
        (R1, LDA): {
            'sss': -0.03268953,
            'sps': -0.04052465,
            'sds': -0.01906964,
            'pps': -0.06312160,
            'ppp': -0.01573455,
            'pds': -0.02079035,
            'pdp': -0.01455391,
            'dds': -0.03969149,
            'ddp': -0.03564724,
            'ddd': -0.00958035,
        },
        (R2, PBE_LibXC): {
            'sss': -0.00141767,
            'sps': -0.00430789,
            'sds': -0.00040590,
            'pps': -0.01723390,
            'ppp': -0.00122340,
            'pds': -0.00095106,
            'pdp': -0.00007429,
            'dds': -0.00018445,
            'ddp': -0.00002978,
            'ddd': -0.00000132,
        },
        (R2, LDA): {
            'sss': -0.00195892,
            'sps': -0.00536204,
            'sds': -0.00066895,
            'pps': -0.01896047,
            'ppp': -0.00150401,
            'pds': -0.00142823,
            'pdp': -0.00017201,
            'dds': -0.00036952,
            'ddp': -0.00007939,
            'ddd': -0.00000628,
        },
    }

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        index = INTEGRALS.index(integral)
        val = H[0, index]
        diff = abs(val - ref)
        assert diff < 1e-4, msg.format(integral, val)


@pytest.mark.parametrize('R', [R1, R2])
@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_rep2c(R, atoms):
    from hotcent.repulsion_twocenter import Repulsion2cTable

    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    for i, (a1, a2) in enumerate([(atom_Mo, atom_Au), (atom_Au, atom_Mo)]):
        rmin, dr, N = R, R, 3
        rep2c = Repulsion2cTable(a1, a2)
        rep2c.run(rmin=rmin, dr=dr, N=N, xc=xc, smoothen_tails=False,
                  ntheta=600, nr=200)
        E = rep2c.erep[0]

        E_ref = {
            (R1, PBE_LibXC): 0.51045937,
            (R1, LDA): 0.51516242,
            (R2, PBE_LibXC): 0.00124821,
            (R2, LDA): 0.00152796,
        }

        etol = 5e-5
        E_diff = abs(E - E_ref[(R, xc)])
        msg = 'Too large error for E_rep (value={0})'
        assert E_diff < etol, msg.format(E)


@pytest.fixture(scope='module')
def grids(request):
    all_grids = {
        R1: (R1, np.array([R1]), np.array([2.]), np.array([0.6*np.pi])),
        R2: (R2, np.array([R2]), np.array([2.]), np.array([0.7*np.pi])),
    }
    return all_grids[request.param]


@pytest.mark.parametrize('nphi', ['adaptive', 13])
@pytest.mark.parametrize('grids', [R1, R2], indirect=True)
@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_off3c(nphi, grids, atoms):
    from hotcent.offsite_threecenter import Offsite3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    off3c = Offsite3cTable(atom_Mo, atom_Au)
    H = off3c.run(atom_S, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, nphi=nphi, xc=xc,
                  write=False)
    H_ref = {
        (R1, PBE_LibXC): {
            's_s': -0.08468788,
            's_dxz': 0.05218667,
            's_dx2-y2': -0.01592966,
            's_dz2': -0.03375516,
            'px_s': -0.05529692,
            'px_dxz': 0.03386836,
            'px_dx2-y2': -0.01534578,
            'px_dz2': -0.00754164,
            'py_dxy': -0.01073400,
            'py_dyz': 0.01679903,
            'pz_s': -0.06606534,
            'pz_dxz': 0.04845204,
            'pz_dx2-y2': -0.01791395,
            'pz_dz2': -0.02418851,
            'dxy_dxy': -0.01195106,
            'dxy_dyz': 0.01994486,
            'dyz_dxy': -0.01576465,
            'dyz_dyz': 0.03870748,
            'dxz_s': -0.07968778,
            'dxz_dxz': 0.05792244,
            'dxz_dx2-y2': -0.01508558,
            'dxz_dz2': -0.04364736,
            'dx2-y2_s': -0.02444355,
            'dx2-y2_dxz': 0.01267106,
            'dx2-y2_dx2-y2': -0.00792957,
            'dx2-y2_dz2': -0.00536486,
            'dz2_s': -0.04455630,
            'dz2_dxz': 0.03951593,
            'dz2_dx2-y2': -0.00844312,
            'dz2_dz2': -0.05892678,
        },
        (R1, LDA): {
            's_s': -0.08500536,
            's_dxz': 0.05272313,
            's_dx2-y2': -0.01603876,
            's_dz2': -0.03424355,
            'px_s': -0.05598552,
            'px_dxz': 0.03476620,
            'px_dx2-y2': -0.01573828,
            'px_dz2': -0.00775206,
            'py_dxy': -0.01108541,
            'py_dyz': 0.01740075,
            'pz_s': -0.06759863,
            'pz_dxz': 0.04978520,
            'pz_dx2-y2': -0.01835800,
            'pz_dz2': -0.02489421,
            'dxy_dxy': -0.01221275,
            'dxy_dyz': 0.02031922,
            'dyz_dxy': -0.01608258,
            'dyz_dyz': 0.03958918,
            'dxz_s': -0.08021103,
            'dxz_dxz': 0.05914145,
            'dxz_dx2-y2': -0.01544342,
            'dxz_dz2': -0.04432465,
            'dx2-y2_s': -0.02435336,
            'dx2-y2_dxz': 0.01301201,
            'dx2-y2_dx2-y2': -0.00816312,
            'dx2-y2_dz2': -0.00550620,
            'dz2_s': -0.04515486,
            'dz2_dxz': 0.04019290,
            'dz2_dx2-y2': -0.00863674,
            'dz2_dz2': -0.05976545,
        },
        (R2, PBE_LibXC): {
            's_s': -0.01592420,
            's_dxz': 0.00158830,
            's_dx2-y2': -0.00018938,
            's_dz2': -0.00366842,
            'px_s': -0.00770899,
            'px_dxz': 0.00129994,
            'px_dx2-y2': -0.00025527,
            'px_dz2': -0.00120900,
            'py_dxy': -0.00018136,
            'py_dyz': 0.00071153,
            'pz_s': -0.02542589,
            'pz_dxz': 0.00335771,
            'pz_dx2-y2': -0.00044975,
            'pz_dz2': -0.00694593,
            'dxy_dxy': -0.00002281,
            'dxy_dyz': 0.00005314,
            'dyz_dxy': -0.00006006,
            'dyz_dyz': 0.00034501,
            'dxz_s': -0.00524514,
            'dxz_dxz': 0.00055157,
            'dxz_dx2-y2': -0.00007736,
            'dxz_dz2': -0.00066516,
            'dx2-y2_s': -0.00050292,
            'dx2-y2_dxz': 0.00007091,
            'dx2-y2_dx2-y2': -0.00002506,
            'dx2-y2_dz2': -0.00004794,
            'dz2_s': -0.01095693,
            'dz2_dxz': 0.00075072,
            'dz2_dx2-y2': -0.00006416,
            'dz2_dz2': -0.00242213,
        },
        (R2, LDA): {
            's_s': -0.01574702,
            's_dxz': 0.00161533,
            's_dx2-y2': -0.00018960,
            's_dz2': -0.00375320,
            'px_s': -0.00764752,
            'px_dxz': 0.00138394,
            'px_dx2-y2': -0.00027354,
            'px_dz2': -0.00121279,
            'py_dxy': -0.00020091,
            'py_dyz': 0.00079519,
            'pz_s': -0.02573815,
            'pz_dxz': 0.00348874,
            'pz_dx2-y2': -0.00046107,
            'pz_dz2': -0.00723642,
            'dxy_dxy': -0.00002386,
            'dxy_dyz': 0.00005365,
            'dyz_dxy': -0.00006089,
            'dyz_dyz': 0.00035923,
            'dxz_s': -0.00513977,
            'dxz_dxz': 0.00057060,
            'dxz_dx2-y2': -0.00007897,
            'dxz_dz2': -0.00067598,
            'dx2-y2_s': -0.00049089,
            'dx2-y2_dxz': 0.00007233,
            'dx2-y2_dx2-y2': -0.00002625,
            'dx2-y2_dz2': -0.00004951,
            'dz2_s': -0.01087125,
            'dz2_dxz': 0.00076447,
            'dz2_dx2-y2': -0.00006534,
            'dz2_dz2': -0.00248706,
        },
    }

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        key = ('S', 0, 0, 0)
        val = H[key][integral][0][1]
        diff = abs(val - ref)
        assert diff < 5e-4, msg.format(integral, val)


@pytest.mark.parametrize('nphi', ['adaptive', 13])
@pytest.mark.parametrize('grids', [R1, R2], indirect=True)
@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_on3c(nphi, grids, atoms):
    from hotcent.onsite_threecenter import Onsite3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    on3c = Onsite3cTable(atom_Mo, atom_Au)
    H = on3c.run(atom_S, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                 nphi=nphi, write=False)

    H_ref = {
        (R1, PBE_LibXC): {
            's_s': 0.00829476,
            's_px': 0.00380302,
            's_pz': 0.01007826,
            's_dxz': 0.00403894,
            's_dx2-y2': 0.00039091,
            's_dz2': 0.00641582,
            'px_s': 0.00380302,
            'px_px': 0.00362617,
            'px_pz': 0.00491640,
            'px_dxz': 0.00290919,
            'px_dx2-y2': 0.00065698,
            'px_dz2': 0.00168967,
            'py_py': 0.00234862,
            'py_dxy': 0.00075126,
            'py_dyz': 0.00213556,
            'pz_s': 0.01007826,
            'pz_px': 0.00491640,
            'pz_pz': 0.01368764,
            'pz_dxz': 0.00433483,
            'pz_dx2-y2': 0.00038682,
            'pz_dz2': 0.00745148,
            'dxy_py': 0.00075126,
            'dxy_dxy': 0.00082895,
            'dxy_dyz': 0.00101041,
            'dyz_py': 0.00213556,
            'dyz_dxy': 0.00101041,
            'dyz_dyz': 0.00344466,
            'dxz_s': 0.00403894,
            'dxz_px': 0.00290919,
            'dxz_pz': 0.00433483,
            'dxz_dxz': 0.00432187,
            'dxz_dx2-y2': 0.00078109,
            'dxz_dz2': 0.00292518,
            'dx2-y2_s': 0.00039091,
            'dx2-y2_px': 0.00065698,
            'dx2-y2_pz': 0.00038682,
            'dx2-y2_dxz': 0.00078109,
            'dx2-y2_dx2-y2': 0.00068368,
            'dx2-y2_dz2': 0.00010353,
            'dz2_s': 0.00641582,
            'dz2_px': 0.00168967,
            'dz2_pz': 0.00745148,
            'dz2_dxz': 0.00292518,
            'dz2_dx2-y2': 0.00010353,
            'dz2_dz2': 0.00776058,
        },
        (R1, LDA): {
            's_s': 0.00898655,
            's_px': 0.00410271,
            's_pz': 0.01078541,
            's_dxz': 0.00421720,
            's_dx2-y2': 0.00039656,
            's_dz2': 0.00665371,
            'px_s': 0.00410271,
            'px_px': 0.00418067,
            'px_pz': 0.00515862,
            'px_dxz': 0.00337527,
            'px_dx2-y2': 0.00081075,
            'px_dz2': 0.00159928,
            'py_py': 0.00283958,
            'py_dxy': 0.00089536,
            'py_dyz': 0.00261154,
            'pz_s': 0.01078541,
            'pz_px': 0.00515862,
            'pz_pz': 0.01456810,
            'pz_dxz': 0.00447615,
            'pz_dx2-y2': 0.00038187,
            'pz_dz2': 0.00785803,
            'dxy_py': 0.00089536,
            'dxy_dxy': 0.00125160,
            'dxy_dyz': 0.00118732,
            'dyz_py': 0.00261154,
            'dyz_dxy': 0.00118732,
            'dyz_dyz': 0.00458366,
            'dxz_s': 0.00421720,
            'dxz_px': 0.00337527,
            'dxz_pz': 0.00447615,
            'dxz_dxz': 0.00545769,
            'dxz_dx2-y2': 0.00101075,
            'dxz_dz2': 0.00293157,
            'dx2-y2_s': 0.00039656,
            'dx2-y2_px': 0.00081075,
            'dx2-y2_pz': 0.00038187,
            'dx2-y2_dxz': 0.00101075,
            'dx2-y2_dx2-y2': 0.00114567,
            'dx2-y2_dz2': 0.00008667,
            'dz2_s': 0.00665371,
            'dz2_px': 0.00159928,
            'dz2_pz': 0.00785803,
            'dz2_dxz': 0.00293157,
            'dz2_dx2-y2': 0.00008667,
            'dz2_dz2': 0.00893071,
        },
        (R2, PBE_LibXC): {
            's_s': 0.00055953,
            's_px': 0.00015916,
            's_pz': 0.00155673,
            's_dxz': 0.00002294,
            's_dx2-y2': 0.00000030,
            's_dz2': 0.00018495,
            'px_s': 0.00015916,
            'px_px': 0.00040336,
            'px_pz': 0.00065236,
            'px_dxz': 0.00003153,
            'px_dx2-y2': 0.00000147,
            'px_dz2': 0.00002673,
            'py_py': 0.00035321,
            'py_dxy': 0.00000145,
            'py_dyz': 0.00003047,
            'pz_s': 0.00155673,
            'pz_px': 0.00065236,
            'pz_pz': 0.00489351,
            'pz_dxz': 0.00004922,
            'pz_dx2-y2': 0.00000053,
            'pz_dz2': 0.00043980,
            'dxy_py': 0.00000145,
            'dxy_dxy': 0.00000009,
            'dxy_dyz': 0.00000095,
            'dyz_py': 0.00003047,
            'dyz_dxy': 0.00000095,
            'dyz_dyz': 0.00000921,
            'dxz_s': 0.00002294,
            'dxz_px': 0.00003153,
            'dxz_pz': 0.00004922,
            'dxz_dxz': 0.00000925,
            'dxz_dx2-y2': 0.00000076,
            'dxz_dz2': 0.00001129,
            'dx2-y2_s': 0.00000030,
            'dx2-y2_px': 0.00000147,
            'dx2-y2_pz': 0.00000053,
            'dx2-y2_dxz': 0.00000076,
            'dx2-y2_dx2-y2': 0.00000003,
            'dx2-y2_dz2': -0.00000004,
            'dz2_s': 0.00018495,
            'dz2_px': 0.00002673,
            'dz2_pz': 0.00043980,
            'dz2_dxz': 0.00001129,
            'dz2_dx2-y2': -0.00000004,
            'dz2_dz2': 0.00007705,
        },
        (R2, LDA): {
            's_s': 0.00090410,
            's_px': 0.00025092,
            's_pz': 0.00216480,
            's_dxz': 0.00004965,
            's_dx2-y2': -0.00000120,
            's_dz2': 0.00038357,
            'px_s': 0.00025092,
            'px_px': 0.00053876,
            'px_pz': 0.00079337,
            'px_dxz': 0.00009482,
            'px_dx2-y2': 0.00000706,
            'px_dz2': 0.00004933,
            'py_py': 0.00049518,
            'py_dxy': 0.00000799,
            'py_dyz': 0.00009912,
            'pz_s': 0.00216480,
            'pz_px': 0.00079337,
            'pz_pz': 0.00585272,
            'pz_dxz': 0.00010049,
            'pz_dx2-y2': -0.00000215,
            'pz_dz2': 0.00078116,
            'dxy_py': 0.00000799,
            'dxy_dxy': 0.00000345,
            'dxy_dyz': 0.00000326,
            'dyz_py': 0.00009912,
            'dyz_dxy': 0.00000326,
            'dyz_dyz': 0.00004833,
            'dxz_s': 0.00004965,
            'dxz_px': 0.00009482,
            'dxz_pz': 0.00010049,
            'dxz_dxz': 0.00004591,
            'dxz_dx2-y2': 0.00000288,
            'dxz_dz2': 0.00002058,
            'dx2-y2_s': -0.00000120,
            'dx2-y2_px': 0.00000706,
            'dx2-y2_pz': -0.00000215,
            'dx2-y2_dxz': 0.00000288,
            'dx2-y2_dx2-y2': 0.00000345,
            'dx2-y2_dz2': -0.00000061,
            'dz2_s': 0.00038357,
            'dz2_px': 0.00004933,
            'dz2_pz': 0.00078116,
            'dz2_dxz': 0.00002058,
            'dz2_dx2-y2': -0.00000061,
            'dz2_dz2': 0.00022791,
        },
    }

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref[(R, xc)].items():
        key = (0, 0, 0)
        val = H[key][integral][0][1]
        diff = abs(val - ref)
        assert diff < 5e-6, msg.format(integral, val)


@pytest.mark.parametrize('nphi', ['adaptive', 13])
@pytest.mark.parametrize('grids', [R1, R2], indirect=True)
@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_rep3c(nphi, grids, atoms):
    from hotcent.repulsion_threecenter import Repulsion3cTable

    R, Rgrid, Sgrid, Tgrid = grids
    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    rep3c = Repulsion3cTable(atom_Mo, atom_Au)
    E = rep3c.run(atom_S, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc, nphi=nphi,
                  write=False)

    E_ref = {
        (R1, PBE_LibXC): -0.01991109,
        (R1, LDA): -0.01891927,
        (R2, PBE_LibXC): -0.00030411,
        (R2, LDA): -0.00061494,
    }

    tol = 1e-5
    val = E[('Mo', 'Au')]['s_s'][0][1]
    diff = abs(val - E_ref[(R, xc)])
    msg = 'Too large error for E_rep (value={0})'
    assert diff < tol, msg.format(val)
