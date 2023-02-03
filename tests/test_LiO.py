""" Tests for multipole-related functionality.

Reference values come from the BeckeHarris tool, unless mentioned otherwise.
"""
import pytest
import numpy as np
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT
from hotcent.offsite_chargetransfer import INTEGRALS_2CK


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
def atoms(request):
    size, xc = request.param.split('-')

    configuration = {
        'Li': '[He] 2s1',
        'O': '[He] 2s2 2p4',
    }

    wf_confinements = {
        'O': {
            '2s': SoftConfinement(rc=4.01),
            '2p': SoftConfinement(rc=4.80),
        },
        'Li': {
            '2s': SoftConfinement(rc=7.73),
        },
    }

    pp_setup = {
        'O': {
            'lmax': 1,
            'local_component': 'd',
        },
        'Li': {
            'valence': ['2s'],
            'local_component': 'siesta',
            'rcore': 2.4958,
            'with_polarization': True,
        },
    }

    r_pol = {
        'Li': 2.338,
        'O': 1.125,
    }

    degree = {
        'Li': 3,
        'O': 2,
    }

    atoms = []
    for element in ['Li', 'O']:
        valence = list(wf_confinements[element].keys())
        pp = KleinmanBylanderPP('./pseudos/{0}.psf'.format(element),
                                verbose=True, **pp_setup[element])

        atom = PseudoAtomicDFT(element, pp,
                               xc=xc,
                               nodegpts=1000,
                               valence=valence,
                               configuration=configuration[element],
                               wf_confinement=wf_confinements[element],
                               perturbative_confinement=True,
                               scalarrel=False,
                               timing=False,
                               txt=None,
                               )
        atom.run()
        atom.generate_nonminimal_basis(size=size, tail_norm=0.15,
                                       r_pol=r_pol[element],
                                       degree=degree[element])
        atom.generate_auxiliary_basis(nzeta=2, tail_norm=0.2, lmax=2)
        atoms.append(atom)

    return atoms


@pytest.mark.parametrize('atoms', [SZP_LDA, SZP_PBE, DZP_LDA, DZP_PBE],
                         indirect=True)
def test_on1cU(atoms):
    from hotcent.onsite_chargetransfer import Onsite1cUTable

    atom_Li, atom_O = atoms
    size = atom_O.basis_size
    xc = atom_O.xcname

    chgon1c = Onsite1cUTable(atom_O, use_multipoles=True)
    chgon1c.run(xc=xc)
    U = chgon1c.tables

    U_ref = {
        LDA: {
            (0, 0): np.array([ 9.81308902, 1.73932109, 0.47140535]),
            (0, 1): np.array([10.85796538, 1.90433789, 0.49579248]),
            (1, 0): np.array([10.85796538, 1.90433789, 0.49579248]),
            (1, 1): np.array([12.18469754, 2.13604733, 0.51837480]),
        },
        PBE: {
            (0, 0): np.array([ 9.83795160, 1.70129861, 0.34327386]),
            (0, 1): np.array([10.88073236, 1.85001015, 0.31813791]),
            (1, 0): np.array([10.88073236, 1.85001015, 0.31813791]),
            (1, 1): np.array([12.18662524, 2.03683611, 0.23965404]),
        },
    }
    if size in [SZP]:
        U_ref = {key: {(0, 0): U_ref[key][(0, 0)]} for key in U_ref}

    msg = 'Too large error for U_{0}[{1}] (value={2})'
    tol = 1e-4

    for key, refs in U_ref[xc].items():
        for i, (val, ref) in enumerate(zip(U[key][0, :], refs)):
            U_diff = np.abs(val - ref)
            assert U_diff < tol, msg.format(key, i, val)

    # Simple test of monopole variant
    chgon1c = Onsite1cUTable(atom_Li, use_multipoles=False)
    chgon1c.run()
    return


@pytest.mark.parametrize('atoms', [SZP_LDA, SZP_PBE, DZP_LDA, DZP_PBE],
                         indirect=True)
def test_on1cW(atoms):
    from hotcent.onsite_magnetization import Onsite1cWTable

    atom_Li, atom_O = atoms
    size = atom_O.basis_size
    xc = atom_O.xcname

    magon1c = Onsite1cWTable(atom_O, use_multipoles=True)
    magon1c.run(xc=xc)
    W = magon1c.tables

    W_ref = {
        LDA: {
            (0, 0): np.array([-0.40022678]*3),
            (0, 1): np.array([-0.47828641]*3),
            (1, 0): np.array([-0.47828641]*3),
            (1, 1): np.array([-0.62755370]*3),
        },
        PBE: {
            (0, 0): np.array([-0.42228170, -0.54343398, -0.78573852]),
            (0, 1): np.array([-0.49796319, -0.67465486, -1.02803820]),
            (1, 0): np.array([-0.49796319, -0.67465486, -1.02803820]),
            (1, 1): np.array([-0.65959742, -0.92614369, -1.45923624]),
        },
    }
    if size in [SZP]:
        W_ref = {key: {(0, 0): W_ref[key][(0, 0)]} for key in W_ref}

    msg = 'Too large error for W_{0}[{1}] (value={2})'
    tol = 1e-4

    for key, refs in W_ref[xc].items():
        for i, (val, ref) in enumerate(zip(W[key][0, :], refs)):
            W_diff = np.abs(val - ref)
            assert W_diff < tol, msg.format(key, i, val)

    # Simple test of monopole variant
    chgon1c = Onsite1cWTable(atom_Li, use_multipoles=False)
    chgon1c.run()
    return


@pytest.mark.parametrize('atoms', [DZP_LDA], indirect=True)
def test_on1cM(atoms):
    # Regression test
    from hotcent.onsite_chargetransfer import Onsite1cMTable

    atom_Li, atom_O = atoms
    size = atom_O.basis_size
    xc = atom_O.xcname

    momon1c = Onsite1cMTable(atom_O)
    momon1c.run()
    M = momon1c.tables

    M_ref = {
        (DZP, LDA): {
            (0, 0): np.array([
                [[1.        , 0.98838401, 0.89197069, 0.        ],
                 [0.98838401, 1.        , 0.89339865, 0.        ],
                 [0.89197069, 0.89339865, 1.        , 0.        ],
                 [0.        , 0.        , 0.        , 0.        ]],
                [[1.        , 0.98838401, 0.89197069, 0.        ],
                 [0.98838401, 1.        , 0.89339865, 0.        ],
                 [0.89197069, 0.89339865, 1.        , 0.        ],
                 [0.        , 0.        , 0.        , 0.        ]],
                [[1.        , 0.98838401, 0.89197069, 0.        ],
                 [0.98838401, 1.        , 0.89339865, 0.        ],
                 [0.89197069, 0.89339865, 1.        , 0.        ],
                 [0.        , 0.        , 0.        , 0.        ]],
                 ]),
            (0, 1): np.array([
                [[0.94300449, 0.97578518, 0.        , 0.        ],
                 [0.9021508 , 0.94983381, 0.        , 0.        ],
                 [0.72607128, 0.81161633, 0.        , 0.        ],
                 [0.        , 0.        , 0.        , 0.        ]],
                [[0.94300449, 0.97578518, 0.        , 0.        ],
                 [0.9021508 , 0.94983381, 0.        , 0.        ],
                 [0.72607128, 0.81161633, 0.        , 0.        ],
                 [0.        , 0.        , 0.        , 0.        ]],
                [[0.94300449, 0.97578518, 0.        , 0.        ],
                 [0.9021508 , 0.94983381, 0.        , 0.        ],
                 [0.72607128, 0.81161633, 0.        , 0.        ],
                 [0.        , 0.        , 0.        , 0.        ]],
                ]),
            (1, 0): np.array([
                [[0.94300449, 0.9021508 , 0.72607128, 0.        ],
                 [0.97578518, 0.94983381, 0.81161633, 0.        ],
                 [0.        , 0.        , 0.        , 0.        ],
                 [0.        , 0.        , 0.        , 0.        ]],
                [[0.94300449, 0.9021508 , 0.72607128, 0.        ],
                 [0.97578518, 0.94983381, 0.81161633, 0.        ],
                 [0.        , 0.        , 0.        , 0.        ],
                 [0.        , 0.        , 0.        , 0.        ]],
                [[0.94300449, 0.9021508 , 0.72607128, 0.        ],
                 [0.97578518, 0.94983381, 0.81161633, 0.        ],
                 [0.        , 0.        , 0.        , 0.        ],
                 [0.        , 0.        , 0.        , 0.        ]],
                ]),
            (1, 1): np.array([
                [[1.       , 0.9844906, 0.       , 0.       ],
                 [0.9844906, 1.       , 0.       , 0.       ],
                 [0.       , 0.       , 0.       , 0.       ],
                 [0.       , 0.       , 0.       , 0.       ]],
                [[1.       , 0.9844906, 0.       , 0.       ],
                 [0.9844906, 1.       , 0.       , 0.       ],
                 [0.       , 0.       , 0.       , 0.       ],
                 [0.       , 0.       , 0.       , 0.       ]],
                [[1.       , 0.9844906, 0.       , 0.       ],
                 [0.9844906, 1.       , 0.       , 0.       ],
                 [0.       , 0.       , 0.       , 0.       ],
                 [0.       , 0.       , 0.       , 0.       ]],
                ]),
        },
    }

    msg = 'Too large error for M_{0} (value={1})'
    tol = 1e-8

    for key, ref in M_ref[(size, xc)].items():
        val = M[key]
        M_diff = np.max(np.abs(val - ref))
        assert M_diff < tol, msg.format(key, str(val))

    return


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atoms', [SZP_LDA, SZP_PBE, DZP_LDA, DZP_PBE],
                         indirect=True)
def test_on2cU(R, atoms):
    from hotcent.onsite_chargetransfer import Onsite2cUTable

    atom_Li, atom_O = atoms
    size = atom_O.basis_size
    xc = atom_O.xcname
    rmin, dr, N = R, R, 2

    chgon2c = Onsite2cUTable(atom_O, atom_Li, use_multipoles=True)
    chgon2c.run(rmin=rmin, dr=dr, N=N, xc=xc, ntheta=300,
                nr=100, smoothen_tails=False)
    U = chgon2c.tables

    U_ref = {
        (R1, LDA): {
            (0, 0): {
                'sss': 0.00759117,
                'sps': 0.00488657,
                'sds': 0.00295531,
                'pss': 0.00488657,
                'pps': 0.01023448,
                'ppp': 0.00626952,
                'pds': 0.00647822,
                'pdp': 0.00256833,
                'dss': 0.00295531,
                'dps': 0.00647822,
                'dpp': 0.00256833,
                'dds': 0.01113785,
                'ddp': 0.00742948,
                'ddd': 0.00597954,
            },
            (0, 1): {
                'sss': 0.00485802,
                'sps': 0.00226534,
                'sds': 0.00116507,
                'pss': 0.00226534,
                'pps': 0.00590009,
                'ppp': 0.00433699,
                'pds': 0.00285654,
                'pdp': 0.00127532,
                'dss': 0.00116507,
                'dps': 0.00285654,
                'dpp': 0.00127532,
                'dds': 0.00626046,
                'ddp': 0.00479145,
                'ddd': 0.00422340,
            },
            (1, 0): {
                'sss': 0.00485802,
                'sps': 0.00226534,
                'sds': 0.00116507,
                'pss': 0.00226534,
                'pps': 0.00590009,
                'ppp': 0.00433699,
                'pds': 0.00285654,
                'pdp': 0.00127532,
                'dss': 0.00116507,
                'dps': 0.00285654,
                'dpp': 0.00127532,
                'dds': 0.00626046,
                'ddp': 0.00479145,
                'ddd': 0.00422340,
            },
            (1, 1): {
                'sss': 0.00504803,
                'sps': 0.00190690,
                'sds': 0.00081117,
                'pss': 0.00190690,
                'pps': 0.00577356,
                'ppp': 0.00468526,
                'pds': 0.00226517,
                'pdp': 0.00115400,
                'dss': 0.00081117,
                'dps': 0.00226517,
                'dpp': 0.00115400,
                'dds': 0.00600702,
                'ddp': 0.00501331,
                'ddd': 0.00460329,
            },
        },
        (R1, PBE): {
            (0, 0): {
                'sss': 0.00623540,
                'sps': 0.00298782,
                'sds': 0.00148456,
                'pss': 0.00298782,
                'pps': 0.00852410,
                'ppp': 0.00672596,
                'pds': 0.00450092,
                'pdp': 0.00201793,
                'dss': 0.00148456,
                'dps': 0.00450092,
                'dpp': 0.00201793,
                'dds': 0.01205319,
                'ddp': 0.00922699,
                'ddd': 0.00850963,
            },
            (0, 1): {
                'sss': 0.00351881,
                'sps': 0.00068559,
                'sds': -0.00004175,
                'pss': 0.00065574,
                'pps': 0.00466691,
                'ppp': 0.00488210,
                'pds': 0.00089412,
                'pdp': 0.00085637,
                'dss': -0.00012765,
                'dps': 0.00079329,
                'dpp': 0.00083752,
                'dds': 0.00745886,
                'ddp': 0.00745201,
                'ddd': 0.00730248,
            },
            (1, 0): {
                'sss': 0.00351881,
                'sps': 0.00065574,
                'sds': -0.00012765,
                'pss': 0.00068559,
                'pps': 0.00466691,
                'ppp': 0.00488210,
                'pds': 0.00079329,
                'pdp': 0.00083752,
                'dss': -0.00004175,
                'dps': 0.00089412,
                'dpp': 0.00085637,
                'dds': 0.00745886,
                'ddp': 0.00745201,
                'ddd': 0.00730248,
            },
            (1, 1): {
                'sss': 0.00325535,
                'sps': 0.00009900,
                'sds': -0.00035814,
                'pss': 0.00009900,
                'pps': 0.00468503,
                'ppp': 0.00524606,
                'pds': 0.00012179,
                'pdp': 0.00046698,
                'dss': -0.00035814,
                'dps': 0.00012179,
                'dpp': 0.00046698,
                'dds': 0.00839195,
                'ddp': 0.00871412,
                'ddd': 0.00875627,
            },
        },
    }
    if size in [SZP]:
        U_ref = {key: {(0, 0): U_ref[key][(0, 0)]} for key in U_ref}

    msg = 'Too large error for U_{0}[{1}] (value={2})'
    tol = 1e-6 if xc == 'LDA' else 1e-5

    for key, refs in U_ref[(R, xc)].items():
        for integral, ref in refs.items():
            index = INTEGRALS_2CK.index(integral)
            val = U[key][0, index]
            U_diff = np.abs(val - ref)
            assert U_diff < tol, msg.format(key, integral, val)
    return


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atoms', [SZP_LDA, SZP_PBE, DZP_LDA, DZP_PBE],
                         indirect=True)
def test_on2cW(R, atoms):
    from hotcent.onsite_magnetization import Onsite2cWTable

    atom_Li, atom_O = atoms
    size = atom_O.basis_size
    xc = atom_O.xcname
    rmin, dr, N = R, R, 2

    magon2c = Onsite2cWTable(atom_O, atom_Li, use_multipoles=True)
    magon2c.run(rmin=rmin, dr=dr, N=N, xc=xc, ntheta=300,
                nr=100, smoothen_tails=False)
    W = magon2c.tables

    W_ref = {
        (R1, LDA): {
            (0, 0): {
                'sss': 0.00488524,
                'sps': 0.00306412,
                'sds': 0.00187013,
                'pss': 0.00306412,
                'pps': 0.00655794,
                'ppp': 0.00404889,
                'pds': 0.00408701,
                'pdp': 0.00159612,
                'dss': 0.00187013,
                'dps': 0.00408701,
                'dpp': 0.00159612,
                'dds': 0.00714514,
                'ddp': 0.00477255,
                'ddd': 0.00386798,
            },
            (0, 1): {
                'sss': 0.00342968,
                'sps': 0.00154930,
                'sds': 0.00078064,
                'pss': 0.00154930,
                'pps': 0.00412791,
                'ppp': 0.00308057,
                'pds': 0.00194141,
                'pdp': 0.00087926,
                'dss': 0.00078064,
                'dps': 0.00194141,
                'dpp': 0.00087926,
                'dds': 0.00436947,
                'ddp': 0.00338502,
                'ddd': 0.00300447,
            },
            (1, 0): {
                'sss': 0.00342968,
                'sps': 0.00154930,
                'sds': 0.00078064,
                'pss': 0.00154930,
                'pps': 0.00412791,
                'ppp': 0.00308057,
                'pds': 0.00194141,
                'pdp': 0.00087926,
                'dss': 0.00078064,
                'dps': 0.00194141,
                'dpp': 0.00087926,
                'dds': 0.00436947,
                'ddp': 0.00338502,
                'ddd': 0.00300447,
            },
            (1, 1): {
                'sss': 0.00366059,
                'sps': 0.00134492,
                'sds': 0.00055771,
                'pss': 0.00134492,
                'pps': 0.00415942,
                'ppp': 0.00341117,
                'pds': 0.00158622,
                'pdp': 0.00082048,
                'dss': 0.00055771,
                'dps': 0.00158622,
                'dpp': 0.00082048,
                'dds': 0.00431879,
                'ddp': 0.00363748,
                'ddd': 0.00335462,
            },
        },
        (R1, PBE): {
            (0, 0): {
                'sss': 0.00569793,
                'sps': 0.00233432,
                'sds': 0.00038207,
                'pss': 0.00233432,
                'pps': 0.00713610,
                'ppp': 0.00672668,
                'pds': 0.00302457,
                'pdp': 0.00172639,
                'dss': 0.00038207,
                'dps': 0.00302457,
                'dpp': 0.00172639,
                'dds': 0.01089620,
                'ddp': 0.00854817,
                'ddd': 0.00898777,
            },
            (0, 1): {
                'sss': 0.00339742,
                'sps': 0.00057678,
                'sds': -0.00051286,
                'pss': 0.00068463,
                'pps': 0.00465735,
                'ppp': 0.00524497,
                'pds': 0.00051500,
                'pdp': 0.00073961,
                'dss': -0.00032919,
                'dps': 0.00079143,
                'dpp': 0.00085848,
                'dds': 0.00832634,
                'ddp': 0.00809943,
                'ddd': 0.00861859,
            },
            (1, 0): {
                'sss': 0.00339742,
                'sps': 0.00068463,
                'sds': -0.00032919,
                'pss': 0.00057678,
                'pps': 0.00465735,
                'ppp': 0.00524497,
                'pds': 0.00079143,
                'pdp': 0.00085848,
                'dss': -0.00051286,
                'dps': 0.00051500,
                'dpp': 0.00073961,
                'dds': 0.00832634,
                'ddp': 0.00809943,
                'ddd': 0.00861859,
            },
            (1, 1): {
                'sss': 0.00245796,
                'sps': -0.00035934,
                'sds': -0.00100522,
                'pss': -0.00035934,
                'pps': 0.00402295,
                'ppp': 0.00538533,
                'pds': -0.00062479,
                'pdp': 0.00035269,
                'dss': -0.00100522,
                'dps': -0.00062479,
                'dpp': 0.00035269,
                'dds': 0.00901569,
                'ddp': 0.00972852,
                'ddd': 0.01045807,
            },
        },
    }
    if size in [SZP]:
        W_ref = {key: {(0, 0): W_ref[key][(0, 0)]} for key in W_ref}

    msg = 'Too large error for W_{0}[{1}] (value={2})'
    tol = 1e-6 if xc == 'LDA' else 1e-5
    tol *= 1 if size == SZP else 2

    for key, refs in W_ref[(R, xc)].items():
        for integral, ref in refs.items():
            index = INTEGRALS_2CK.index(integral)
            val = W[key][0, index]
            W_diff = np.abs(val - ref)
            assert W_diff < tol, msg.format(key, integral, val)

    return


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atoms', [SZP_LDA, SZP_PBE, DZP_LDA, DZP_PBE],
                         indirect=True)
def test_off2cU(R, atoms):
    from hotcent.offsite_chargetransfer import Offsite2cUTable

    atom_Li, atom_O = atoms
    size = atom_O.basis_size
    xc = atom_O.xcname
    rmin, dr, N = R, R, 2

    chgoff2c = Offsite2cUTable(atom_O, atom_Li, use_multipoles=True)
    chgoff2c.run(rmin=rmin, dr=dr, N=N, xc=xc, ntheta=300,
                 nr=100, smoothen_tails=False)
    U = chgoff2c.tables

    U_ref = {
        (R1, LDA): {
            (0, 0, 0): {
                'sss': -1.33682398,
                'sps': -1.43962603,
                'sds': 0.70122587,
                'pss': 0.32894411,
                'pps': 0.03409693,
                'ppp': 0.38044666,
                'pds': -0.09723295,
                'pdp': -0.31847979,
                'dss': 0.01671739,
                'dps': 0.04257551,
                'dpp': 0.08194276,
                'dds': -0.00535394,
                'ddp': -0.03105666,
                'ddd': 0.07323604,
            },
            (0, 0, 1): {
                'sss': -0.94235921,
                'sps': -1.67568622,
                'sds': 0.84798314,
                'pss': 0.44503661,
                'pps': -0.04796949,
                'ppp': 0.44251988,
                'pds': -0.05013178,
                'pdp': -0.38494074,
                'dss': 0.01307638,
                'dps': 0.06411796,
                'dpp': 0.11417130,
                'dds': -0.02515111,
                'ddp': -0.06166001,
                'ddd': 0.08721705,
            },
            (0, 1, 0): {
                'sss': -1.25754235,
                'sps': -1.55656186,
                'sds': 0.81216879,
                'pss': 0.27403168,
                'pps': 0.01078211,
                'ppp': 0.33024720,
                'pds': -0.07304950,
                'pdp': -0.30346754,
                'dss': 0.00631275,
                'dps': 0.03022385,
                'dpp': 0.05951443,
                'dds': -0.00582645,
                'ddp': -0.03294586,
                'ddd': 0.05652429,
            },
            (0, 1, 1): {
                'sss': -0.83293033,
                'sps': -1.83703119,
                'sds': 1.00076076,
                'pss': 0.37857350,
                'pps': -0.07198776,
                'ppp': 0.39122972,
                'pds': -0.02131828,
                'pdp': -0.37627233,
                'dss': -0.00142875,
                'dps': 0.05120976,
                'dpp': 0.08621047,
                'dds': -0.02516568,
                'ddp': -0.06296595,
                'ddd': 0.07004524,
            },
            (1, 0, 0): {
                'sss': -1.33682383,
                'sps': -0.32894414,
                'sds': 0.01671807,
                'pss': 1.43962609,
                'pps': 0.03409696,
                'ppp': 0.38044664,
                'pds': -0.04257362,
                'pdp': -0.08194286,
                'dss': 0.70122589,
                'dps': 0.09723301,
                'dpp': 0.31847968,
                'dds': -0.00535135,
                'ddp': -0.03105694,
                'ddd': 0.07323594,
            },
            (1, 0, 1): {
                'sss': -1.25754213,
                'sps': -0.27403215,
                'sds': 0.00631439,
                'pss': 1.55656190,
                'pps': 0.01078201,
                'ppp': 0.33024720,
                'pds': -0.03022053,
                'pdp': -0.05951475,
                'dss': 0.81216874,
                'dps': 0.07304956,
                'dpp': 0.30346737,
                'dds': -0.00582209,
                'ddp': -0.03294643,
                'ddd': 0.05652410,
            },
            (1, 1, 0): {
                'sss': -0.94235905,
                'sps': -0.44503660,
                'sds': 0.01307776,
                'pss': 1.67568630,
                'pps': -0.04796941,
                'ppp': 0.44251982,
                'pds': -0.06411480,
                'pdp': -0.11417145,
                'dss': 0.84798318,
                'dps': 0.05013187,
                'dpp': 0.38494055,
                'dds': -0.02514684,
                'ddp': -0.06166041,
                'ddd': 0.08721689,
            },
            (1, 1, 1): {
                'sss': -0.83293008,
                'sps': -0.37857397,
                'sds': -0.00142594,
                'pss': 1.83703126,
                'pps': -0.07198782,
                'ppp': 0.39122967,
                'pds': -0.05120433,
                'pdp': -0.08621091,
                'dss': 1.00076071,
                'dps': 0.02131839,
                'dpp': 0.37627202,
                'dds': -0.02515857,
                'ddp': -0.06296677,
                'ddd': 0.07004493,
            },
        },
        (R1, PBE): {
            (0, 0, 0): {
                'sss': -1.34354020,
                'sps': -1.43832498,
                'sds': 0.69851735,
                'pss': 0.32442785,
                'pps': 0.03843244,
                'ppp': 0.38064809,
                'pds': -0.10279173,
                'pdp': -0.31724836,
                'dss': 0.01602681,
                'dps': 0.04272861,
                'dpp': 0.08203186,
                'dds': -0.00572780,
                'ddp': -0.02866380,
                'ddd': 0.07327689,
            },
            (0, 0, 1): {
                'sss': -0.96441293,
                'sps': -1.67094515,
                'sds': 0.84344477,
                'pss': 0.42961697,
                'pps': -0.03346929,
                'ppp': 0.44259526,
                'pds': -0.06441986,
                'pdp': -0.38349899,
                'dss': 0.01050001,
                'dps': 0.06466734,
                'dpp': 0.11199058,
                'dds': -0.02479520,
                'ddp': -0.05502933,
                'ddd': 0.08766743,
            },
            (0, 1, 0): {
                'sss': -1.26365976,
                'sps': -1.55844861,
                'sds': 0.81418713,
                'pss': 0.26997820,
                'pps': 0.01386072,
                'ppp': 0.33176268,
                'pds': -0.07678146,
                'pdp': -0.30712876,
                'dss': 0.00505575,
                'dps': 0.03121201,
                'dpp': 0.05980965,
                'dds': -0.00663216,
                'ddp': -0.03366311,
                'ddd': 0.05778530,
            },
            (0, 1, 1): {
                'sss': -0.85596455,
                'sps': -1.83426277,
                'sds': 1.00076424,
                'pss': 0.36448659,
                'pps': -0.05953748,
                'ppp': 0.39241264,
                'pds': -0.03266998,
                'pdp': -0.38052042,
                'dss': -0.00535009,
                'dps': 0.05447756,
                'dpp': 0.08499401,
                'dds': -0.02714789,
                'ddp': -0.06189415,
                'ddd': 0.07173512,
            },
            (1, 0, 0): {
                'sss': -1.34354009,
                'sps': -0.32442788,
                'sds': 0.01602756,
                'pss': 1.43832504,
                'pps': 0.03843247,
                'ppp': 0.38064807,
                'pds': -0.04272675,
                'pdp': -0.08203197,
                'dss': 0.69851737,
                'dps': 0.10279179,
                'dpp': 0.31724826,
                'dds': -0.00572528,
                'ddp': -0.02866407,
                'ddd': 0.07327679,
            },
            (1, 0, 1): {
                'sss': -1.26365963,
                'sps': -0.26997866,
                'sds': 0.00505752,
                'pss': 1.55844863,
                'pps': 0.01386063,
                'ppp': 0.33176268,
                'pds': -0.03120874,
                'pdp': -0.05980998,
                'dss': 0.81418707,
                'dps': 0.07678153,
                'dpp': 0.30712860,
                'dds': -0.00662794,
                'ddp': -0.03366367,
                'ddd': 0.05778510,
            },
            (1, 1, 0): {
                'sss': -0.96441278,
                'sps': -0.42961697,
                'sds': 0.01050144,
                'pss': 1.67094526,
                'pps': -0.03346922,
                'ppp': 0.44259521,
                'pds': -0.06466424,
                'pdp': -0.11199073,
                'dss': 0.84344484,
                'dps': 0.06441996,
                'dpp': 0.38349881,
                'dds': -0.02479106,
                'ddp': -0.05502974,
                'ddd': 0.08766726,
            },
            (1, 1, 1): {
                'sss': -0.85596436,
                'sps': -0.36448706,
                'sds': -0.00534717,
                'pss': 1.83426285,
                'pps': -0.05953753,
                'ppp': 0.39241259,
                'pds': -0.05447226,
                'pdp': -0.08499445,
                'dss': 1.00076422,
                'dps': 0.03267011,
                'dpp': 0.38052013,
                'dds': -0.02714103,
                'ddp': -0.06189497,
                'ddd': 0.07173482,
            },
        },
    }
    if size in [SZP]:
        U_ref = {key1: {key2: U_ref[key1][key2]
                        for key2 in [(0, 0, 0), (1, 0, 0)]}
                 for key1 in U_ref}

    msg = 'Too large error for U_{0}[{1}] (value={2})'
    tol = 1e-4

    for key, refs in U_ref[(R, xc)].items():
        if key[0] == 0:
            sym1, sym2 = chgoff2c.ela.get_symbol(), chgoff2c.elb.get_symbol()
        elif key[0] == 1:
            sym2, sym1 = chgoff2c.ela.get_symbol(), chgoff2c.elb.get_symbol()

        nls = {sym: ['2s', '2s+'] for sym in ['O', 'Li']}
        nl1 = nls[sym1][key[1]]
        nl2 = nls[sym2][key[2]]

        for integral, ref in refs.items():
            index = INTEGRALS_2CK.index(integral)

            val = U[key][0, index]
            if integral != 'sss':
                U_delta = chgoff2c.evaluate_point_multipole_hartree(sym1, sym2,
                                                        nl1, nl2, integral, R)
                val += U_delta

            U_diff = np.abs(val - ref)
            assert U_diff < tol, msg.format(key, integral, val)
    return


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atoms', [SZP_LDA, SZP_PBE, DZP_LDA, DZP_PBE],
                         indirect=True)
def test_off2cW(R, atoms):
    from hotcent.offsite_magnetization import Offsite2cWTable

    atom_Li, atom_O = atoms
    size = atom_O.basis_size
    xc = atom_O.xcname
    rmin, dr, N = R, R, 2

    magoff2c = Offsite2cWTable(atom_O, atom_Li, use_multipoles=True)
    magoff2c.run(rmin=rmin, dr=dr, N=N, xc=xc, ntheta=300,
                 nr=100, smoothen_tails=False)
    W = magoff2c.tables

    W_ref = {
        (R1, LDA): {
            (0, 0, 0): {
                'sss': -0.03630386,
                'sps': 0.05251373,
                'sds': -0.04758354,
                'pss': -0.01040472,
                'pps': 0.01111905,
                'ppp': -0.02166012,
                'pds': -0.00252751,
                'pdp': 0.03744352,
                'dss': 0.00401262,
                'dps': -0.00557701,
                'dpp': -0.01133170,
                'dds': 0.00283175,
                'ddp': 0.01573443,
                'ddd': -0.01354332,
            },
            (0, 0, 1): {
                'sss': -0.04921057,
                'sps': 0.07089680,
                'sds': -0.06350902,
                'pss': -0.01978778,
                'pps': 0.02412935,
                'ppp': -0.03036574,
                'pds': -0.01311128,
                'pdp': 0.05229390,
                'dss': 0.01046154,
                'dps': -0.01523142,
                'dpp': -0.01941781,
                'dds': 0.01200840,
                'ddp': 0.02883417,
                'ddd': -0.01923775,
            },
            (0, 1, 0): {
                'sss': -0.02529087,
                'sps': 0.04059418,
                'sds': -0.04492344,
                'pss': -0.00552057,
                'pps': 0.00796412,
                'ppp': -0.01067995,
                'pds': -0.00669828,
                'pdp': 0.02141384,
                'dss': 0.00157013,
                'dps': -0.00184253,
                'dpp': -0.00423530,
                'dds': 0.00030129,
                'ddp': 0.00784961,
                'ddd': -0.00486869,
            },
            (0, 1, 1): {
                'sss': -0.03719783,
                'sps': 0.05960002,
                'sds': -0.06569982,
                'pss': -0.01105503,
                'pps': 0.01660119,
                'ppp': -0.01621951,
                'pds': -0.01567885,
                'pdp': 0.03247474,
                'dss': 0.00573964,
                'dps': -0.00830425,
                'dpp': -0.00764626,
                'dds': 0.00686101,
                'ddp': 0.01446588,
                'ddd': -0.00750884,
            },
            (1, 0, 0): {
                'sss': -0.03630386,
                'sps': 0.01040472,
                'sds': 0.00401262,
                'pss': -0.05251373,
                'pps': 0.01111905,
                'ppp': -0.02166012,
                'pds': 0.00557701,
                'pdp': 0.01133170,
                'dss': -0.04758354,
                'dps': 0.00252751,
                'dpp': -0.03744352,
                'dds': 0.00283175,
                'ddp': 0.01573443,
                'ddd': -0.01354332,
            },
            (1, 0, 1): {
                'sss': -0.02529087,
                'sps': 0.00552057,
                'sds': 0.00157013,
                'pss': -0.04059418,
                'pps': 0.00796412,
                'ppp': -0.01067995,
                'pds': 0.00184253,
                'pdp': 0.00423530,
                'dss': -0.04492344,
                'dps': 0.00669828,
                'dpp': -0.02141384,
                'dds': 0.00030129,
                'ddp': 0.00784961,
                'ddd': -0.00486869,
            },
            (1, 1, 0): {
                'sss': -0.04921057,
                'sps': 0.01978778,
                'sds': 0.01046154,
                'pss': -0.07089680,
                'pps': 0.02412935,
                'ppp': -0.03036574,
                'pds': 0.01523142,
                'pdp': 0.01941781,
                'dss': -0.06350902,
                'dps': 0.01311128,
                'dpp': -0.05229390,
                'dds': 0.01200840,
                'ddp': 0.02883417,
                'ddd': -0.01923775,
            },
            (1, 1, 1): {
                'sss': -0.03719783,
                'sps': 0.01105503,
                'sds': 0.00573964,
                'pss': -0.05960002,
                'pps': 0.01660119,
                'ppp': -0.01621951,
                'pds': 0.00830425,
                'pdp': 0.00764626,
                'dss': -0.06569982,
                'dps': 0.01567885,
                'dpp': -0.03247474,
                'dds': 0.00686101,
                'ddp': 0.01446588,
                'ddd': -0.00750884,
            },
        },
        (R1, PBE): {
            (0, 0, 0): {
                'sss': -0.04047031,
                'sps': 0.05907080,
                'sds': -0.05258252,
                'pss': -0.01147600,
                'pps': 0.01287661,
                'ppp': -0.02679979,
                'pds': -0.00240487,
                'pdp': 0.04753129,
                'dss': 0.00638664,
                'dps': -0.00826532,
                'dpp': -0.01467610,
                'dds': 0.00393776,
                'ddp': 0.02209740,
                'ddd': -0.01966919,
            },
            (0, 0, 1): {
                'sss': -0.05303103,
                'sps': 0.07716947,
                'sds': -0.06818144,
                'pss': -0.02225110,
                'pps': 0.02764371,
                'ppp': -0.03602343,
                'pds': -0.01358513,
                'pdp': 0.06387365,
                'dss': 0.01455424,
                'dps': -0.02138072,
                'dpp': -0.02556667,
                'dds': 0.01788456,
                'ddp': 0.04004168,
                'ddd': -0.02659412,
            },
            (0, 1, 0): {
                'sss': -0.02458134,
                'sps': 0.03978267,
                'sds': -0.04475557,
                'pss': -0.00511753,
                'pps': 0.00755463,
                'ppp': -0.01122204,
                'pds': -0.00672284,
                'pdp': 0.02274888,
                'dss': 0.00227546,
                'dps': -0.00262071,
                'dpp': -0.00469925,
                'dds': 0.00034738,
                'ddp': 0.00891136,
                'ddd': -0.00592550,
            },
            (0, 1, 1): {
                'sss': -0.03629712,
                'sps': 0.05856157,
                'sds': -0.06549867,
                'pss': -0.01047345,
                'pps': 0.01583812,
                'ppp': -0.01709432,
                'pds': -0.01518842,
                'pdp': 0.03455503,
                'dss': 0.00761756,
                'dps': -0.01104086,
                'dpp': -0.00861842,
                'dds': 0.00920049,
                'ddp': 0.01648641,
                'ddd': -0.00914045,
            },
            (1, 0, 0): {
                'sss': -0.04047031,
                'sps': 0.01147600,
                'sds': 0.00638664,
                'pss': -0.05907080,
                'pps': 0.01287661,
                'ppp': -0.02679979,
                'pds': 0.00826532,
                'pdp': 0.01467610,
                'dss': -0.05258252,
                'dps': 0.00240487,
                'dpp': -0.04753129,
                'dds': 0.00393776,
                'ddp': 0.02209740,
                'ddd': -0.01966919,
            },
            (1, 0, 1): {
                'sss': -0.02458134,
                'sps': 0.00511753,
                'sds': 0.00227546,
                'pss': -0.03978267,
                'pps': 0.00755463,
                'ppp': -0.01122204,
                'pds': 0.00262071,
                'pdp': 0.00469925,
                'dss': -0.04475557,
                'dps': 0.00672284,
                'dpp': -0.02274888,
                'dds': 0.00034738,
                'ddp': 0.00891136,
                'ddd': -0.00592550,
            },
            (1, 1, 0): {
                'sss': -0.05303103,
                'sps': 0.02225110,
                'sds': 0.01455424,
                'pss': -0.07716947,
                'pps': 0.02764371,
                'ppp': -0.03602343,
                'pds': 0.02138072,
                'pdp': 0.02556667,
                'dss': -0.06818144,
                'dps': 0.01358513,
                'dpp': -0.06387365,
                'dds': 0.01788456,
                'ddp': 0.04004168,
                'ddd': -0.02659412,
            },
            (1, 1, 1): {
                'sss': -0.03629712,
                'sps': 0.01047345,
                'sds': 0.00761756,
                'pss': -0.05856157,
                'pps': 0.01583812,
                'ppp': -0.01709432,
                'pds': 0.01104086,
                'pdp': 0.00861842,
                'dss': -0.06549867,
                'dps': 0.01518842,
                'dpp': -0.03455503,
                'dds': 0.00920049,
                'ddp': 0.01648641,
                'ddd': -0.00914045,
            },
        },
    }
    if size in [SZP]:
        W_ref = {key1: {key2: W_ref[key1][key2]
                        for key2 in [(0, 0, 0), (1, 0, 0)]}
                 for key1 in W_ref}

    msg = 'Too large error for W_{0}[{1}] (value={2})'
    tol = 1e-4

    for key, refs in W_ref[(R, xc)].items():
        for integral, ref in refs.items():
            index = INTEGRALS_2CK.index(integral)
            val = W[key][0, index]
            W_diff = np.abs(val - ref)
            assert W_diff < tol, msg.format(key, integral, val)
    return


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atoms', [DZP_LDA], indirect=True)
def test_off2cM(R, atoms):
    # Regression test
    from hotcent.offsite_chargetransfer import Offsite2cMTable

    atom_Li, atom_O = atoms
    size = atom_O.basis_size
    xc = atom_O.xcname
    rmin, dr, N = R, R, 2

    momoff2c = Offsite2cMTable(atom_O, atom_Li)
    momoff2c.run(rmin=rmin, dr=dr, N=N, ntheta=300, nr=100,
                 smoothen_tails=False)
    M = momoff2c.tables

    M_ref = {
        (DZP, LDA): {
            (0, 0, 0): (
                np.array([
                    3.805011e-01, -5.776285e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 7.647023e-02, -1.564089e-01, 2.527447e-01,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -1.682107e-02, 3.099168e-02,
                    1.600786e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -1.803853e-02, 3.332237e-02,
                    6.250976e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    -9.134461e-03, 1.332263e-02, 1.850562e-02, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
                np.array([
                    4.476727e-01, -5.982948e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 1.109324e-01, -1.519095e-01, 3.167438e-01,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -1.582277e-02, 6.088318e-02,
                    2.238881e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -2.315836e-02, 5.802928e-02,
                    9.887938e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    -1.308051e-02, 2.538399e-02, 3.335858e-02, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
                np.array([
                    4.036245e-01, -6.019672e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 8.358031e-02, -1.844798e-01, 2.890860e-01,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -2.161411e-02, 3.068535e-02,
                    1.841500e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -2.164519e-02, 3.520976e-02,
                    6.922550e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    -1.046655e-02, 1.340059e-02, 1.917001e-02, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
            ),
            (0, 0, 1): (
                np.array([
                    4.142357e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 1.194630e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -5.663481e-02, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -6.058223e-02, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    -3.080230e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
                np.array([
                    4.711823e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 1.685270e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -5.434368e-02, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -7.712353e-02, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    -4.437805e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
                np.array([
                    4.315427e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 1.298109e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -7.269208e-02, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -7.274750e-02, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    -3.526901e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
            ),
            (0, 1, 0): (
                np.array([
                    2.475523e-01, -4.513037e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 3.237011e-02, -1.117745e-01, 1.292109e-01,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -9.395511e-03, 3.406289e-03,
                    5.981550e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -6.864092e-03, 7.199881e-03,
                    1.639508e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    -2.674855e-03, 2.130698e-03, 3.299352e-03, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
                np.array([
                    2.937110e-01, -5.066638e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 4.488358e-02, -1.340048e-01, 1.709726e-01,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -1.278268e-02, 9.090049e-03,
                    9.023230e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -1.056874e-02, 1.341284e-02,
                    2.864946e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    -4.533852e-03, 4.514603e-03, 6.755405e-03, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
                np.zeros(55),
            ),
            (0, 1, 1): (
                np.array([
                    2.901818e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 5.375877e-02, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -3.159152e-02, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -2.306920e-02, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    -9.015967e-03, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
                np.array([
                    3.350286e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 7.262014e-02, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -4.297470e-02, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, -3.552688e-02, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    -1.527885e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
                np.zeros(55),
            ),
            (1, 0, 0): (
                np.array([
                    3.805011e-01, -1.109324e-01, -2.161411e-02, 0.000000e+00,
                    0.000000e+00, 5.327943e-01, -2.571821e-02, 2.962018e-01,
                    -3.638999e-02, -1.182601e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 4.712476e-01, 1.172667e-01,
                    4.358033e-01, -6.506054e-03, -1.397278e-01, 1.683735e-01,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 3.396584e-01, 1.785503e-01,
                    4.348469e-01, 7.889641e-02, -5.646419e-02, 3.194126e-01,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    2.176638e-01, 1.693153e-01, 3.667707e-01, 1.817985e-01,
                    7.165367e-02, 4.112596e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
                np.array([
                    4.189509e-01, -2.157434e-01, -9.051354e-03, 0.000000e+00,
                    0.000000e+00, 5.776285e-01, -1.519095e-01, 3.167438e-01,
                    -3.068535e-02, -1.841500e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 4.984536e-01, 1.425732e-02,
                    4.545682e-01, -2.105285e-02, -2.429699e-01, 1.884217e-01,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 3.505796e-01, 9.855065e-02,
                    4.461018e-01, 4.915232e-02, -1.623997e-01, 3.516460e-01,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    2.207951e-01, 1.102091e-01, 3.764076e-01, 1.470665e-01,
                    -1.517166e-02, 4.438207e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
                np.zeros(55),
            ),
            (1, 0, 1): (
                np.array([
                    2.475523e-01, -4.488358e-02, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 3.922706e-01, -4.376527e-02, 1.488135e-01,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 4.233234e-01, 7.859692e-03,
                    2.771905e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 3.815550e-01, 7.472404e-02,
                    3.552012e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    2.990748e-01, 1.229434e-01, 3.696608e-01, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
                np.array([
                    2.859718e-01, -1.053588e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 4.513037e-01, -1.340048e-01, 1.709726e-01,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 4.828968e-01, -7.931034e-02,
                    3.159223e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 4.293102e-01, 6.040742e-03,
                    4.002306e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    3.297311e-01, 7.436962e-02, 4.108985e-01, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
                np.zeros(55),
            ),
            (1, 1, 0): (
                np.array([
                    4.142357e-01, -1.685270e-01, -7.269208e-02, 0.000000e+00,
                    0.000000e+00, 5.810731e-01, -1.087230e-01, 3.189576e-01,
                    -1.080563e-01, -1.585986e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 5.103816e-01, 4.638222e-02,
                    4.662102e-01, -7.393820e-02, -2.077955e-01, 1.929391e-01,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 3.631303e-01, 1.248231e-01,
                    4.625113e-01, 2.585095e-02, -1.297037e-01, 3.642184e-01,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    2.301611e-01, 1.309946e-01, 3.926922e-01, 1.455459e-01,
                    1.173399e-02, 4.647385e-01, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
                np.zeros(55),
                np.zeros(55),
            ),
            (1, 1, 1): (
                np.array([
                    2.901818e-01, -7.262014e-02, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 4.592652e-01, -8.647338e-02, 1.743908e-01,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 4.942337e-01, -3.494271e-02,
                    3.246855e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 4.431292e-01, 4.106555e-02,
                    4.153512e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    3.441135e-01, 1.013372e-01, 4.307519e-01, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                    0.000000e+00, 0.000000e+00, 0.000000e+00,
                ]),
                np.zeros(55),
                np.zeros(55),
            ),
        },
    }

    msg = 'Too large error for M_{0} (value={1})'
    tol = 1e-7

    for key, ref in M_ref[(size, xc)].items():
        val = M[key][:, 0, :]
        M_diff = np.max(np.abs(val - ref))
        assert M_diff < tol, msg.format(key, str(val))

    return
