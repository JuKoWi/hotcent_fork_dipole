""" Basic tests with three different elements (Mo/Au/S), KB pseudopotentials
and LDA and GGA functionals.

Reference values come from the BeckeHarris tool, unless mentioned otherwise.
"""
import pytest
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
            atom1.pp.build_overlaps(atom2, atom1, rmin=0.05, rmax=12.)
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
    on2c.run(atom_Au, rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
             smoothen_tails=False, ntheta=300, nr=100)

    H, S = on2c.tables[0][0, :20], on2c.tables[0][0, 20:41]

    if xc == PBE_LibXC:
        HS_ref = {
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
        HS_ref = {
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

    for integral, ref in HS_ref.items():
        index = INTEGRALS.index(integral)
        H_diff = abs(H[index] - ref)
        assert H_diff < htol, msg.format(integral, H[index])


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
            E_ref = 0.50983998
        elif xc == LDA:
            E_ref = 0.51443028

        etol = 1e-3
        E_diff = abs(E - E_ref)
        msg = 'Too large error for E_rep (value={0})'
        assert E_diff < etol, msg.format(E)
