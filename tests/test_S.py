""" Basic tests with one element (S), a KB pseudopotential,
and LDA and GGA functionals.

Reference values come from the BeckeHarris tool, unless mentioned otherwise.
"""
import pytest
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT
from hotcent.slako import INTEGRALS


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
    atom.pp.build_overlaps(atom, atom, rmin=0.05, rmax=12.)
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

    if xc == PBE_LibXC:
        H_ref = {
            '3s': -0.60003476,
            '3p': -0.22792440,
        }
    elif xc in [LDA, LDA_LibXC]:
        H_ref = {
            '3s': -0.60021238,
            '3p': -0.22965093,
        }

    htol = 1e-5
    msg = 'Too large error for H_{0} (value={1})'
    for nl, ref in H_ref.items():
        H = atom.get_onecenter_integral(nl)
        H_diff = abs(H - ref)
        assert H_diff < htol, msg.format(nl, H)


@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_off2c(atom):
    from hotcent.slako import SlaterKosterTable

    xc = atom.xcname

    rmin, dr, N = 2.4, 2.4, 2
    off2c = SlaterKosterTable(atom, atom)
    off2c.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
              smoothen_tails=False, ntheta=300, nr=100)
    H, S = off2c.tables[0][0, :20], off2c.tables[0][0, 20:41]

    if xc == PBE_LibXC:
        HS_ref = {
            'sss': (-0.56677916,  0.49949084),
            'sps': ( 0.54239200, -0.56984428),
            'pps': ( 0.23153307, -0.18330357),
            'ppp': (-0.29839091,  0.46900432),
        }
    elif xc in [LDA, LDA_LibXC]:
        HS_ref = {
            'sss': (-0.57099349,  0.50148895),
            'sps': ( 0.54669416, -0.57206251),
            'pps': ( 0.23101831, -0.17645402),
            'ppp': (-0.30160739,  0.47370130),
        }

    htol = 5e-4
    stol = 1e-4
    msg = 'Too large error for {0}_{1} (value={2})'

    for integral, ref in HS_ref.items():
        index = INTEGRALS.index(integral)

        H_diff = abs(H[index] - ref[0])
        assert H_diff < htol, msg.format('H', integral, H[index])

        S_diff = abs(S[index] - ref[1])
        assert S_diff < stol, msg.format('S', integral, S[index])


@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_on2c(atom):
    from hotcent.onsite_twocenter import Onsite2cTable

    xc = atom.xcname

    rmin, dr, N = 2.4, 2.4, 2
    on2c = Onsite2cTable(atom, atom)
    on2c.run(atom, rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
             smoothen_tails=False, ntheta=300, nr=100)
    H = on2c.tables[0][0, :20]

    if xc == PBE_LibXC:
        H_ref = {
            'sss': -0.20686456,
            'sps': -0.23146656,
            'pps': -0.32711570,
            'ppp': -0.12725461,
        }
    elif xc in [LDA, LDA_LibXC]:
        H_ref = {
            'sss': -0.21148071,
            'sps': -0.23413930,
            'pps': -0.33049489,
            'ppp': -0.13063231,
        }

    htol = 5e-4
    msg = 'Too large error for H_{0} (value={1})'

    for integral, ref in H_ref.items():
        index = INTEGRALS.index(integral)
        H_diff = abs(H[index] - ref)
        assert H_diff < htol, msg.format(integral, H[index])


@pytest.mark.parametrize('atom', [PBE_LibXC, LDA, LDA_LibXC], indirect=True)
def test_rep2c(atom):
    from hotcent.slako import SlaterKosterTable

    xc = atom.xcname

    rmin, dr, N = 2.4, 2.4, 3
    off2c = SlaterKosterTable(atom, atom)
    off2c.run_repulsion(rmin=rmin, dr=dr, N=N, xc=xc, smoothen_tails=False,
                        ntheta=600, nr=200)
    E = off2c.erep[0]

    if xc == PBE_LibXC:
        E_ref = 2.40955060
    elif xc in [LDA, LDA_LibXC]:
        E_ref = 2.44177361

    etol = 5e-4
    E_diff = abs(E - E_ref)
    msg = 'Too large error for E_rep (value={0})'
    assert E_diff < etol, msg.format(E)
