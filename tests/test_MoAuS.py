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
            atom1.pp.build_overlaps(atom2, atom1, rmin=0.5, rmax=5.)
    return atoms


@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_on1c(atoms):
    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    if xc == PBE_LibXC:
        H_ref = {
            '4d': -0.08227156,
            '5s': -0.12145480,
            '5p': -0.01686089,
        }
    elif xc == LDA:
        H_ref = {
            '4d': -0.08259557,
            '5s': -0.12780923,
            '5p': -0.02060831,
        }

    htol = 1e-5
    msg = 'Too large error for H_{0} (value={1})'
    for nl, ref in H_ref.items():
        H = atom_Mo.get_onecenter_integral(nl)
        H_diff = abs(H - ref)
        assert H_diff < htol, msg.format(nl, H)


@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_off2c(atoms):
    from hotcent.slako import SlaterKosterTable

    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    rmin, dr, N = 4.0, 4.0, 2
    off2c = SlaterKosterTable(atom_Mo, atom_Au)
    off2c.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
              smoothen_tails=False, ntheta=300, nr=100)

    for i in range(2):
        H, S = off2c.tables[i][0, :20], off2c.tables[i][0, 20:41]

        if i == 0:
            if xc == PBE_LibXC:
                HS_ref = {
                    'sss': (-0.13632270,  0.53477384),
                    'sds': (-0.02059026,  0.04140705),
                    'pds': ( 0.01275936, -0.01597752),
                    'pdp': ( 0.03713049, -0.08335779),
                    'dds': (-0.11242042,  0.11452732),
                    'ddp': ( 0.10998343, -0.17262074),
                    'ddd': (-0.02992897,  0.05149113),
                }
            elif xc == LDA:
                HS_ref = {
                    'sss': (-0.14225054,  0.53331619),
                    'sds': (-0.02103561,  0.04132978),
                    'pds': ( 0.01303762, -0.01714720),
                    'pdp': ( 0.03901744, -0.08648932),
                    'dds': (-0.11359708,  0.11424404),
                    'ddp': ( 0.11212830, -0.17544303),
                    'ddd': (-0.03080327,  0.05277999),
                }
        elif i == 1:
            if xc == PBE_LibXC:
                HS_ref = {
                    'sss': (-0.13633473,  0.53477384),
                    'sps': ( 0.12040788, -0.63723479),
                    'sds': (-0.06543059,  0.15205108),
                    'dds': (-0.11242073,  0.11452732),
                    'ddp': ( 0.10995152, -0.17262074),
                    'ddd': (-0.02986408,  0.05149113),
                }
            elif xc == LDA:
                HS_ref = {
                    'sss': (-0.14226172,  0.53331619),
                    'sps': ( 0.12767598, -0.63823622),
                    'sds': (-0.06811197,  0.15440267),
                    'dds': (-0.11359779,  0.11424404),
                    'ddp': ( 0.11209514, -0.17544303),
                    'ddd': (-0.03073530,  0.05277999),
                }

        htol = 5e-4
        stol = 1e-4
        msg = 'Too large error for {0}[{1}]_{2} (value={3})'

        for integral, ref in HS_ref.items():
            index = INTEGRALS.index(integral)

            H_diff = abs(H[index] - ref[0])
            assert H_diff < htol, msg.format('H', i, integral, H[index])

            S_diff = abs(S[index] - ref[1])
            assert S_diff < stol, msg.format('S', i, integral, S[index])


@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_on2c(atoms):
    from hotcent.onsite_twocenter import Onsite2cTable

    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    rmin, dr, N = 4.0, 4.0, 2
    on2c = Onsite2cTable(atom_Mo, atom_Mo)
    H = on2c.run(atom_Au, rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
                 smoothen_tails=False, ntheta=300, nr=100, write=False)

    if xc == PBE_LibXC:
        H_ref = {
            'sss': -0.03075806,
            'sps': -0.03831150,
            'sds': -0.01811909,
            'pps': -0.06019212,
            'ppp': -0.01407127,
            'pds': -0.01957250,
            'pdp': -0.01322414,
            'dds': -0.03681505,
            'ddp': -0.03301457,
            'ddd': -0.00867295,
        }
    elif xc == LDA:
        H_ref = {
            'sss': -0.03277479,
            'sps': -0.04063762,
            'sds': -0.01913406,
            'pps': -0.06326507,
            'ppp': -0.01573543,
            'pds': -0.02086328,
            'pdp': -0.01455389,
            'dds': -0.03973310,
            'ddp': -0.03564661,
            'ddd': -0.00958009,
        }

    htol = 2e-4
    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref.items():
        pair = ('Mo', 'Mo')
        val = H[pair][integral][0]
        diff = abs(val - ref)
        assert diff < 5e-4, msg.format(integral, val)


@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_rep2c(atoms):
    from hotcent.slako import SlaterKosterTable

    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    for i, (a1, a2) in enumerate([(atom_Mo, atom_Au), (atom_Au, atom_Mo)]):
        rmin, dr, N = 4.0, 4.0, 3
        off2c = SlaterKosterTable(a1, a2)
        off2c.run_repulsion(rmin=rmin, dr=dr, N=N, xc=xc, smoothen_tails=False,
                            ntheta=600, nr=200)
        E = off2c.erep[0]

        if xc == PBE_LibXC:
            E_ref = 0.51045989
        elif xc == LDA:
            E_ref = 0.515162968

        etol = 5e-5
        E_diff = abs(E - E_ref)
        msg = 'Too large error for E_rep (value={0})'
        assert E_diff < etol, msg.format(E)


@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_off3c(atoms):
    from hotcent.offsite_threecenter import Offsite3cTable

    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    off3c = Offsite3cTable(atom_Mo, atom_Au)
    Rgrid, Sgrid, Tgrid = np.array([4.]), np.array([2.]), np.array([np.pi*0.6])
    H = off3c.run(atom_S, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                  write=False)

    if xc == PBE_LibXC:
        H_ref = {
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
        }
    elif xc == LDA:
        H_ref = {
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
        }

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref.items():
        pair = ('Mo', 'Au')
        val = H[pair][integral][0][1]
        diff = abs(val - ref)
        assert diff < 5e-4, msg.format(integral, val)


@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_on3c(atoms):
    from hotcent.onsite_threecenter import Onsite3cTable

    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    on3c = Onsite3cTable(atom_Mo, atom_Mo)
    Rgrid, Sgrid, Tgrid = np.array([4.]), np.array([2.]), np.array([np.pi*0.6])
    H = on3c.run(atom_Au, atom_S, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                 write=False)

    if xc == PBE_LibXC:
        H_ref = {
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
        }
    elif xc == LDA:
        H_ref = {
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
        }

    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref.items():
        pair = ('Au', 'S')
        val = H[pair][integral][0][1]
        diff = abs(val - ref)
        assert diff < 1e-5, msg.format(integral, val)


@pytest.mark.parametrize('atoms', [PBE_LibXC, LDA], indirect=True)
def test_rep3c(atoms):
    from hotcent.offsite_threecenter import Offsite3cTable

    atom_Mo, atom_Au, atom_S = atoms
    xc = atom_Mo.xcname

    off3c = Offsite3cTable(atom_Mo, atom_Au)
    Rgrid, Sgrid, Tgrid = np.array([4.]), np.array([2.]), np.array([np.pi*0.6])
    E = off3c.run_repulsion(atom_S, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
                            write=False)

    if xc == PBE_LibXC:
        E_ref = -0.01991227
    elif xc == LDA:
        E_ref = -0.01891973

    tol = 1e-5
    val = E[('Mo', 'Au')]['s_s'][0][1]
    diff = abs(val - E_ref)
    msg = 'Too large error for E_rep (value={0})'
    assert diff < tol, msg.format(val)
