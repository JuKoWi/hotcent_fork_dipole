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

DZP = 'dzp'

LDA = 'LDA'
PBE = 'GGA_X_PBE+GGA_C_PBE'

# The following will be used to avoid unnecessary fixture rebuilds
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
        atom.generate_nonminimal_basis(size=size, tail_norms=[0.15],
                                       r_pol=r_pol[element],
                                       degree=degree[element])
        atom.generate_auxiliary_basis(nzeta=2, tail_norms=[0.2], lmax=2,
                                      degree=degree[element])
        atoms.append(atom)

    return atoms


@pytest.mark.parametrize('atoms', [DZP_LDA, DZP_PBE], indirect=True)
def test_on1cU(atoms):
    from hotcent.onsite_chargetransfer import Onsite1cUTable

    atom_Li, atom_O = atoms
    xc = atom_O.xcname

    chgon1c = Onsite1cUTable(atom_O, use_multipoles=True)
    chgon1c.run(xc=xc)
    U = chgon1c.tables

    U_ref = {
        LDA: {
            (0, 0): np.array([ 9.81308902, 1.52281194, 0.35847313]),
            (0, 1): np.array([11.04291245, 1.75837856, 0.48564695]),
            (1, 0): np.array([11.04291245, 1.75837856, 0.48564695]),
            (1, 1): np.array([12.68383125, 2.15292813, 0.60966907]),
        },
        PBE: {
            (0, 0): np.array([ 9.83795160, 1.50166505, 0.28309507]),
            (0, 1): np.array([11.06541141, 1.75523409, 0.43972204]),
            (1, 0): np.array([11.06541141, 1.75523409, 0.43972204]),
            (1, 1): np.array([12.67373250, 2.11772207, 0.48241347]),
        },
    }

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


@pytest.mark.parametrize('atoms', [DZP_LDA, DZP_PBE], indirect=True)
def test_on1cW(atoms):
    from hotcent.onsite_magnetization import Onsite1cWTable

    atom_Li, atom_O = atoms
    xc = atom_O.xcname

    magon1c = Onsite1cWTable(atom_O, use_multipoles=True)
    magon1c.run(xc=xc)
    W = magon1c.tables

    W_ref = {
        LDA: {
            (0, 0): np.array([-0.40022678, -0.31719088, -0.29888349]),
            (0, 1): np.array([-0.49165106, -0.35721938, -0.29146248]),
            (1, 0): np.array([-0.49165106, -0.35721938, -0.29146248]),
            (1, 1): np.array([-0.69092883, -0.55428943, -0.52299681]),
        },
        PBE: {
            (0, 0): np.array([-0.42228170, -0.38857272, -0.43599409]),
            (0, 1): np.array([-0.51059659, -0.41967997, -0.41829073]),
            (1, 0): np.array([-0.51059659, -0.41967997, -0.41829073]),
            (1, 1): np.array([-0.73142931, -0.65793380, -0.76139458]),
        },
    }

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
        key1: {
            key2: np.zeros((9, 9, 18))
            for key2 in [(0, 0), (0, 1), (1, 0), (1, 1)]
        }
        for key1 in [(DZP, LDA)]
    }

    M_ref[(DZP, LDA)][(0, 0)][0, 0, 0] = 0.282094791774
    M_ref[(DZP, LDA)][(0, 0)][0, 1, 1] = 0.150292808512
    M_ref[(DZP, LDA)][(0, 0)][0, 1, 10] = 0.127412109831
    M_ref[(DZP, LDA)][(0, 0)][0, 2, 2] = 0.150292808512
    M_ref[(DZP, LDA)][(0, 0)][0, 2, 11] = 0.127412109831
    M_ref[(DZP, LDA)][(0, 0)][0, 3, 3] = 0.150292808512
    M_ref[(DZP, LDA)][(0, 0)][0, 3, 12] = 0.127412109831
    M_ref[(DZP, LDA)][(0, 0)][0, 4, 4] = 0.133099690753
    M_ref[(DZP, LDA)][(0, 0)][0, 4, 13] = 0.110381438627
    M_ref[(DZP, LDA)][(0, 0)][0, 5, 5] = 0.133099690753
    M_ref[(DZP, LDA)][(0, 0)][0, 5, 14] = 0.110381438627
    M_ref[(DZP, LDA)][(0, 0)][0, 6, 6] = 0.133099690753
    M_ref[(DZP, LDA)][(0, 0)][0, 6, 15] = 0.110381438627
    M_ref[(DZP, LDA)][(0, 0)][0, 7, 7] = 0.133099690753
    M_ref[(DZP, LDA)][(0, 0)][0, 7, 16] = 0.110381438627
    M_ref[(DZP, LDA)][(0, 0)][0, 8, 8] = 0.133099690753
    M_ref[(DZP, LDA)][(0, 0)][0, 8, 17] = 0.110381438627
    M_ref[(DZP, LDA)][(0, 0)][1, 0, 1] = 0.150292808512
    M_ref[(DZP, LDA)][(0, 0)][1, 0, 10] = 0.127412109831
    M_ref[(DZP, LDA)][(0, 0)][1, 1, 0] = 0.340855523088
    M_ref[(DZP, LDA)][(0, 0)][1, 1, 7] = 0.101937114974
    M_ref[(DZP, LDA)][(0, 0)][1, 1, 8] = -0.058853420771
    M_ref[(DZP, LDA)][(0, 0)][1, 1, 9] = -0.058760731314
    M_ref[(DZP, LDA)][(0, 0)][1, 1, 16] = 0.111806722257
    M_ref[(DZP, LDA)][(0, 0)][1, 1, 17] = -0.064551641192
    M_ref[(DZP, LDA)][(0, 0)][1, 2, 4] = 0.101937114974
    M_ref[(DZP, LDA)][(0, 0)][1, 2, 13] = 0.111806722257
    M_ref[(DZP, LDA)][(0, 0)][1, 3, 6] = 0.101937114974
    M_ref[(DZP, LDA)][(0, 0)][1, 3, 15] = 0.111806722257
    M_ref[(DZP, LDA)][(0, 0)][1, 4, 2] = 0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][1, 4, 11] = -0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][1, 6, 3] = 0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][1, 6, 12] = -0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][1, 7, 1] = 0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][1, 7, 10] = -0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][1, 8, 1] = -0.122910408171
    M_ref[(DZP, LDA)][(0, 0)][1, 8, 10] = 0.011060784432
    M_ref[(DZP, LDA)][(0, 0)][2, 0, 2] = 0.150292808512
    M_ref[(DZP, LDA)][(0, 0)][2, 0, 11] = 0.127412109831
    M_ref[(DZP, LDA)][(0, 0)][2, 1, 4] = 0.101937114974
    M_ref[(DZP, LDA)][(0, 0)][2, 1, 13] = 0.111806722257
    M_ref[(DZP, LDA)][(0, 0)][2, 2, 0] = 0.340855523088
    M_ref[(DZP, LDA)][(0, 0)][2, 2, 7] = -0.101937114974
    M_ref[(DZP, LDA)][(0, 0)][2, 2, 8] = -0.058853420771
    M_ref[(DZP, LDA)][(0, 0)][2, 2, 9] = -0.058760731314
    M_ref[(DZP, LDA)][(0, 0)][2, 2, 16] = -0.111806722257
    M_ref[(DZP, LDA)][(0, 0)][2, 2, 17] = -0.064551641192
    M_ref[(DZP, LDA)][(0, 0)][2, 3, 5] = 0.101937114974
    M_ref[(DZP, LDA)][(0, 0)][2, 3, 14] = 0.111806722257
    M_ref[(DZP, LDA)][(0, 0)][2, 4, 1] = 0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][2, 4, 10] = -0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][2, 5, 3] = 0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][2, 5, 12] = -0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][2, 7, 2] = -0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][2, 7, 11] = 0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][2, 8, 2] = -0.122910408171
    M_ref[(DZP, LDA)][(0, 0)][2, 8, 11] = 0.011060784432
    M_ref[(DZP, LDA)][(0, 0)][3, 0, 3] = 0.150292808512
    M_ref[(DZP, LDA)][(0, 0)][3, 0, 12] = 0.127412109831
    M_ref[(DZP, LDA)][(0, 0)][3, 1, 6] = 0.101937114974
    M_ref[(DZP, LDA)][(0, 0)][3, 1, 15] = 0.111806722257
    M_ref[(DZP, LDA)][(0, 0)][3, 2, 5] = 0.101937114974
    M_ref[(DZP, LDA)][(0, 0)][3, 2, 14] = 0.111806722257
    M_ref[(DZP, LDA)][(0, 0)][3, 3, 0] = 0.340855523088
    M_ref[(DZP, LDA)][(0, 0)][3, 3, 8] = 0.117706841541
    M_ref[(DZP, LDA)][(0, 0)][3, 3, 9] = -0.058760731314
    M_ref[(DZP, LDA)][(0, 0)][3, 3, 17] = 0.129103282385
    M_ref[(DZP, LDA)][(0, 0)][3, 5, 2] = 0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][3, 5, 11] = -0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][3, 6, 1] = 0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][3, 6, 10] = -0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][3, 8, 3] = 0.245820816343
    M_ref[(DZP, LDA)][(0, 0)][3, 8, 12] = -0.022121568865
    M_ref[(DZP, LDA)][(0, 0)][4, 0, 4] = 0.133099690753
    M_ref[(DZP, LDA)][(0, 0)][4, 0, 13] = 0.110381438627
    M_ref[(DZP, LDA)][(0, 0)][4, 1, 2] = 0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][4, 1, 11] = -0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][4, 2, 1] = 0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][4, 2, 10] = -0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][4, 4, 0] = 0.669295649676
    M_ref[(DZP, LDA)][(0, 0)][4, 4, 8] = -0.132621507813
    M_ref[(DZP, LDA)][(0, 0)][4, 4, 9] = -0.387200857902
    M_ref[(DZP, LDA)][(0, 0)][4, 4, 17] = -0.038366218117
    M_ref[(DZP, LDA)][(0, 0)][4, 5, 6] = 0.114853594854
    M_ref[(DZP, LDA)][(0, 0)][4, 5, 15] = 0.033226119536
    M_ref[(DZP, LDA)][(0, 0)][4, 6, 5] = 0.114853594854
    M_ref[(DZP, LDA)][(0, 0)][4, 6, 14] = 0.033226119536
    M_ref[(DZP, LDA)][(0, 0)][4, 8, 4] = -0.132621507813
    M_ref[(DZP, LDA)][(0, 0)][4, 8, 13] = -0.038366218117
    M_ref[(DZP, LDA)][(0, 0)][5, 0, 5] = 0.133099690753
    M_ref[(DZP, LDA)][(0, 0)][5, 0, 14] = 0.110381438627
    M_ref[(DZP, LDA)][(0, 0)][5, 2, 3] = 0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][5, 2, 12] = -0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][5, 3, 2] = 0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][5, 3, 11] = -0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][5, 4, 6] = 0.114853594854
    M_ref[(DZP, LDA)][(0, 0)][5, 4, 15] = 0.033226119536
    M_ref[(DZP, LDA)][(0, 0)][5, 5, 0] = 0.669295649676
    M_ref[(DZP, LDA)][(0, 0)][5, 5, 7] = -0.114853594854
    M_ref[(DZP, LDA)][(0, 0)][5, 5, 8] = 0.066310753907
    M_ref[(DZP, LDA)][(0, 0)][5, 5, 9] = -0.387200857902
    M_ref[(DZP, LDA)][(0, 0)][5, 5, 16] = -0.033226119536
    M_ref[(DZP, LDA)][(0, 0)][5, 5, 17] = 0.019183109058
    M_ref[(DZP, LDA)][(0, 0)][5, 6, 4] = 0.114853594854
    M_ref[(DZP, LDA)][(0, 0)][5, 6, 13] = 0.033226119536
    M_ref[(DZP, LDA)][(0, 0)][5, 7, 5] = -0.114853594854
    M_ref[(DZP, LDA)][(0, 0)][5, 7, 14] = -0.033226119536
    M_ref[(DZP, LDA)][(0, 0)][5, 8, 5] = 0.066310753907
    M_ref[(DZP, LDA)][(0, 0)][5, 8, 14] = 0.019183109058
    M_ref[(DZP, LDA)][(0, 0)][6, 0, 6] = 0.133099690753
    M_ref[(DZP, LDA)][(0, 0)][6, 0, 15] = 0.110381438627
    M_ref[(DZP, LDA)][(0, 0)][6, 1, 3] = 0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][6, 1, 12] = -0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][6, 3, 1] = 0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][6, 3, 10] = -0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][6, 4, 5] = 0.114853594854
    M_ref[(DZP, LDA)][(0, 0)][6, 4, 14] = 0.033226119536
    M_ref[(DZP, LDA)][(0, 0)][6, 5, 4] = 0.114853594854
    M_ref[(DZP, LDA)][(0, 0)][6, 5, 13] = 0.033226119536
    M_ref[(DZP, LDA)][(0, 0)][6, 6, 0] = 0.669295649676
    M_ref[(DZP, LDA)][(0, 0)][6, 6, 7] = 0.114853594854
    M_ref[(DZP, LDA)][(0, 0)][6, 6, 8] = 0.066310753907
    M_ref[(DZP, LDA)][(0, 0)][6, 6, 9] = -0.387200857902
    M_ref[(DZP, LDA)][(0, 0)][6, 6, 16] = 0.033226119536
    M_ref[(DZP, LDA)][(0, 0)][6, 6, 17] = 0.019183109058
    M_ref[(DZP, LDA)][(0, 0)][6, 7, 6] = 0.114853594854
    M_ref[(DZP, LDA)][(0, 0)][6, 7, 15] = 0.033226119536
    M_ref[(DZP, LDA)][(0, 0)][6, 8, 6] = 0.066310753907
    M_ref[(DZP, LDA)][(0, 0)][6, 8, 15] = 0.019183109058
    M_ref[(DZP, LDA)][(0, 0)][7, 0, 7] = 0.133099690753
    M_ref[(DZP, LDA)][(0, 0)][7, 0, 16] = 0.110381438627
    M_ref[(DZP, LDA)][(0, 0)][7, 1, 1] = 0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][7, 1, 10] = -0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][7, 2, 2] = -0.212887071732
    M_ref[(DZP, LDA)][(0, 0)][7, 2, 11] = 0.019157840608
    M_ref[(DZP, LDA)][(0, 0)][7, 5, 5] = -0.114853594854
    M_ref[(DZP, LDA)][(0, 0)][7, 5, 14] = -0.033226119536
    M_ref[(DZP, LDA)][(0, 0)][7, 6, 6] = 0.114853594854
    M_ref[(DZP, LDA)][(0, 0)][7, 6, 15] = 0.033226119536
    M_ref[(DZP, LDA)][(0, 0)][7, 7, 0] = 0.669295649676
    M_ref[(DZP, LDA)][(0, 0)][7, 7, 8] = -0.132621507813
    M_ref[(DZP, LDA)][(0, 0)][7, 7, 9] = -0.387200857902
    M_ref[(DZP, LDA)][(0, 0)][7, 7, 17] = -0.038366218117
    M_ref[(DZP, LDA)][(0, 0)][7, 8, 7] = -0.132621507813
    M_ref[(DZP, LDA)][(0, 0)][7, 8, 16] = -0.038366218117
    M_ref[(DZP, LDA)][(0, 0)][8, 0, 8] = 0.133099690753
    M_ref[(DZP, LDA)][(0, 0)][8, 0, 17] = 0.110381438627
    M_ref[(DZP, LDA)][(0, 0)][8, 1, 1] = -0.122910408171
    M_ref[(DZP, LDA)][(0, 0)][8, 1, 10] = 0.011060784432
    M_ref[(DZP, LDA)][(0, 0)][8, 2, 2] = -0.122910408171
    M_ref[(DZP, LDA)][(0, 0)][8, 2, 11] = 0.011060784432
    M_ref[(DZP, LDA)][(0, 0)][8, 3, 3] = 0.245820816343
    M_ref[(DZP, LDA)][(0, 0)][8, 3, 12] = -0.022121568865
    M_ref[(DZP, LDA)][(0, 0)][8, 4, 4] = -0.132621507813
    M_ref[(DZP, LDA)][(0, 0)][8, 4, 13] = -0.038366218117
    M_ref[(DZP, LDA)][(0, 0)][8, 5, 5] = 0.066310753907
    M_ref[(DZP, LDA)][(0, 0)][8, 5, 14] = 0.019183109058
    M_ref[(DZP, LDA)][(0, 0)][8, 6, 6] = 0.066310753907
    M_ref[(DZP, LDA)][(0, 0)][8, 6, 15] = 0.019183109058
    M_ref[(DZP, LDA)][(0, 0)][8, 7, 7] = -0.132621507813
    M_ref[(DZP, LDA)][(0, 0)][8, 7, 16] = -0.038366218117
    M_ref[(DZP, LDA)][(0, 0)][8, 8, 0] = 0.669295649676
    M_ref[(DZP, LDA)][(0, 0)][8, 8, 8] = 0.132621507813
    M_ref[(DZP, LDA)][(0, 0)][8, 8, 9] = -0.387200857902
    M_ref[(DZP, LDA)][(0, 0)][8, 8, 17] = 0.038366218117
    M_ref[(DZP, LDA)][(0, 1)][0, 0, 0] = 0.124477860838
    M_ref[(DZP, LDA)][(0, 1)][0, 0, 9] = 0.141538793262
    M_ref[(DZP, LDA)][(0, 1)][0, 1, 1] = 0.054190691356
    M_ref[(DZP, LDA)][(0, 1)][0, 1, 10] = 0.217122866287
    M_ref[(DZP, LDA)][(0, 1)][0, 2, 2] = 0.054190691356
    M_ref[(DZP, LDA)][(0, 1)][0, 2, 11] = 0.217122866287
    M_ref[(DZP, LDA)][(0, 1)][0, 3, 3] = 0.054190691356
    M_ref[(DZP, LDA)][(0, 1)][0, 3, 12] = 0.217122866287
    M_ref[(DZP, LDA)][(0, 1)][1, 0, 1] = 0.005263473627
    M_ref[(DZP, LDA)][(0, 1)][1, 0, 10] = 0.245699308310
    M_ref[(DZP, LDA)][(0, 1)][1, 1, 0] = 0.200750757040
    M_ref[(DZP, LDA)][(0, 1)][1, 1, 7] = 0.017850109854
    M_ref[(DZP, LDA)][(0, 1)][1, 1, 8] = -0.010305765729
    M_ref[(DZP, LDA)][(0, 1)][1, 1, 9] = 0.067192413843
    M_ref[(DZP, LDA)][(0, 1)][1, 1, 16] = 0.177015192454
    M_ref[(DZP, LDA)][(0, 1)][1, 1, 17] = -0.102199769014
    M_ref[(DZP, LDA)][(0, 1)][1, 2, 4] = 0.017850109854
    M_ref[(DZP, LDA)][(0, 1)][1, 2, 13] = 0.177015192454
    M_ref[(DZP, LDA)][(0, 1)][1, 3, 6] = 0.017850109854
    M_ref[(DZP, LDA)][(0, 1)][1, 3, 15] = 0.177015192454
    M_ref[(DZP, LDA)][(0, 1)][2, 0, 2] = 0.005263473627
    M_ref[(DZP, LDA)][(0, 1)][2, 0, 11] = 0.245699308310
    M_ref[(DZP, LDA)][(0, 1)][2, 1, 4] = 0.017850109854
    M_ref[(DZP, LDA)][(0, 1)][2, 1, 13] = 0.177015192454
    M_ref[(DZP, LDA)][(0, 1)][2, 2, 0] = 0.200750757040
    M_ref[(DZP, LDA)][(0, 1)][2, 2, 7] = -0.017850109854
    M_ref[(DZP, LDA)][(0, 1)][2, 2, 8] = -0.010305765729
    M_ref[(DZP, LDA)][(0, 1)][2, 2, 9] = 0.067192413843
    M_ref[(DZP, LDA)][(0, 1)][2, 2, 16] = -0.177015192454
    M_ref[(DZP, LDA)][(0, 1)][2, 2, 17] = -0.102199769014
    M_ref[(DZP, LDA)][(0, 1)][2, 3, 5] = 0.017850109854
    M_ref[(DZP, LDA)][(0, 1)][2, 3, 14] = 0.177015192454
    M_ref[(DZP, LDA)][(0, 1)][3, 0, 3] = 0.005263473627
    M_ref[(DZP, LDA)][(0, 1)][3, 0, 12] = 0.245699308310
    M_ref[(DZP, LDA)][(0, 1)][3, 1, 6] = 0.017850109854
    M_ref[(DZP, LDA)][(0, 1)][3, 1, 15] = 0.177015192454
    M_ref[(DZP, LDA)][(0, 1)][3, 2, 5] = 0.017850109854
    M_ref[(DZP, LDA)][(0, 1)][3, 2, 14] = 0.177015192454
    M_ref[(DZP, LDA)][(0, 1)][3, 3, 0] = 0.200750757040
    M_ref[(DZP, LDA)][(0, 1)][3, 3, 8] = 0.020611531459
    M_ref[(DZP, LDA)][(0, 1)][3, 3, 9] = 0.067192413843
    M_ref[(DZP, LDA)][(0, 1)][3, 3, 17] = 0.204399538028
    M_ref[(DZP, LDA)][(0, 1)][4, 0, 4] = 0.034495715709
    M_ref[(DZP, LDA)][(0, 1)][4, 0, 13] = 0.166050566007
    M_ref[(DZP, LDA)][(0, 1)][4, 1, 2] = 0.134108751706
    M_ref[(DZP, LDA)][(0, 1)][4, 1, 11] = 0.040722834832
    M_ref[(DZP, LDA)][(0, 1)][4, 2, 1] = 0.134108751706
    M_ref[(DZP, LDA)][(0, 1)][4, 2, 10] = 0.040722834832
    M_ref[(DZP, LDA)][(0, 1)][5, 0, 5] = 0.034495715709
    M_ref[(DZP, LDA)][(0, 1)][5, 0, 14] = 0.166050566007
    M_ref[(DZP, LDA)][(0, 1)][5, 2, 3] = 0.134108751706
    M_ref[(DZP, LDA)][(0, 1)][5, 2, 12] = 0.040722834832
    M_ref[(DZP, LDA)][(0, 1)][5, 3, 2] = 0.134108751706
    M_ref[(DZP, LDA)][(0, 1)][5, 3, 11] = 0.040722834832
    M_ref[(DZP, LDA)][(0, 1)][6, 0, 6] = 0.034495715709
    M_ref[(DZP, LDA)][(0, 1)][6, 0, 15] = 0.166050566007
    M_ref[(DZP, LDA)][(0, 1)][6, 1, 3] = 0.134108751706
    M_ref[(DZP, LDA)][(0, 1)][6, 1, 12] = 0.040722834832
    M_ref[(DZP, LDA)][(0, 1)][6, 3, 1] = 0.134108751706
    M_ref[(DZP, LDA)][(0, 1)][6, 3, 10] = 0.040722834832
    M_ref[(DZP, LDA)][(0, 1)][7, 0, 7] = 0.034495715709
    M_ref[(DZP, LDA)][(0, 1)][7, 0, 16] = 0.166050566007
    M_ref[(DZP, LDA)][(0, 1)][7, 1, 1] = 0.134108751706
    M_ref[(DZP, LDA)][(0, 1)][7, 1, 10] = 0.040722834832
    M_ref[(DZP, LDA)][(0, 1)][7, 2, 2] = -0.134108751706
    M_ref[(DZP, LDA)][(0, 1)][7, 2, 11] = -0.040722834832
    M_ref[(DZP, LDA)][(0, 1)][8, 0, 8] = 0.034495715709
    M_ref[(DZP, LDA)][(0, 1)][8, 0, 17] = 0.166050566007
    M_ref[(DZP, LDA)][(0, 1)][8, 1, 1] = -0.077427723898
    M_ref[(DZP, LDA)][(0, 1)][8, 1, 10] = -0.023511339652
    M_ref[(DZP, LDA)][(0, 1)][8, 2, 2] = -0.077427723898
    M_ref[(DZP, LDA)][(0, 1)][8, 2, 11] = -0.023511339652
    M_ref[(DZP, LDA)][(0, 1)][8, 3, 3] = 0.154855447797
    M_ref[(DZP, LDA)][(0, 1)][8, 3, 12] = 0.047022679305
    M_ref[(DZP, LDA)][(1, 0)][0, 0, 0] = 0.124477860838
    M_ref[(DZP, LDA)][(1, 0)][0, 0, 9] = 0.141538793262
    M_ref[(DZP, LDA)][(1, 0)][0, 1, 1] = 0.005263473627
    M_ref[(DZP, LDA)][(1, 0)][0, 1, 10] = 0.245699308310
    M_ref[(DZP, LDA)][(1, 0)][0, 2, 2] = 0.005263473627
    M_ref[(DZP, LDA)][(1, 0)][0, 2, 11] = 0.245699308310
    M_ref[(DZP, LDA)][(1, 0)][0, 3, 3] = 0.005263473627
    M_ref[(DZP, LDA)][(1, 0)][0, 3, 12] = 0.245699308310
    M_ref[(DZP, LDA)][(1, 0)][0, 4, 4] = 0.034495715709
    M_ref[(DZP, LDA)][(1, 0)][0, 4, 13] = 0.166050566007
    M_ref[(DZP, LDA)][(1, 0)][0, 5, 5] = 0.034495715709
    M_ref[(DZP, LDA)][(1, 0)][0, 5, 14] = 0.166050566007
    M_ref[(DZP, LDA)][(1, 0)][0, 6, 6] = 0.034495715709
    M_ref[(DZP, LDA)][(1, 0)][0, 6, 15] = 0.166050566007
    M_ref[(DZP, LDA)][(1, 0)][0, 7, 7] = 0.034495715709
    M_ref[(DZP, LDA)][(1, 0)][0, 7, 16] = 0.166050566007
    M_ref[(DZP, LDA)][(1, 0)][0, 8, 8] = 0.034495715709
    M_ref[(DZP, LDA)][(1, 0)][0, 8, 17] = 0.166050566007
    M_ref[(DZP, LDA)][(1, 0)][1, 0, 1] = 0.054190691356
    M_ref[(DZP, LDA)][(1, 0)][1, 0, 10] = 0.217122866287
    M_ref[(DZP, LDA)][(1, 0)][1, 1, 0] = 0.200750757040
    M_ref[(DZP, LDA)][(1, 0)][1, 1, 7] = 0.017850109854
    M_ref[(DZP, LDA)][(1, 0)][1, 1, 8] = -0.010305765729
    M_ref[(DZP, LDA)][(1, 0)][1, 1, 9] = 0.067192413843
    M_ref[(DZP, LDA)][(1, 0)][1, 1, 16] = 0.177015192454
    M_ref[(DZP, LDA)][(1, 0)][1, 1, 17] = -0.102199769014
    M_ref[(DZP, LDA)][(1, 0)][1, 2, 4] = 0.017850109854
    M_ref[(DZP, LDA)][(1, 0)][1, 2, 13] = 0.177015192454
    M_ref[(DZP, LDA)][(1, 0)][1, 3, 6] = 0.017850109854
    M_ref[(DZP, LDA)][(1, 0)][1, 3, 15] = 0.177015192454
    M_ref[(DZP, LDA)][(1, 0)][1, 4, 2] = 0.134108751706
    M_ref[(DZP, LDA)][(1, 0)][1, 4, 11] = 0.040722834832
    M_ref[(DZP, LDA)][(1, 0)][1, 6, 3] = 0.134108751706
    M_ref[(DZP, LDA)][(1, 0)][1, 6, 12] = 0.040722834832
    M_ref[(DZP, LDA)][(1, 0)][1, 7, 1] = 0.134108751706
    M_ref[(DZP, LDA)][(1, 0)][1, 7, 10] = 0.040722834832
    M_ref[(DZP, LDA)][(1, 0)][1, 8, 1] = -0.077427723898
    M_ref[(DZP, LDA)][(1, 0)][1, 8, 10] = -0.023511339652
    M_ref[(DZP, LDA)][(1, 0)][2, 0, 2] = 0.054190691356
    M_ref[(DZP, LDA)][(1, 0)][2, 0, 11] = 0.217122866287
    M_ref[(DZP, LDA)][(1, 0)][2, 1, 4] = 0.017850109854
    M_ref[(DZP, LDA)][(1, 0)][2, 1, 13] = 0.177015192454
    M_ref[(DZP, LDA)][(1, 0)][2, 2, 0] = 0.200750757040
    M_ref[(DZP, LDA)][(1, 0)][2, 2, 7] = -0.017850109854
    M_ref[(DZP, LDA)][(1, 0)][2, 2, 8] = -0.010305765729
    M_ref[(DZP, LDA)][(1, 0)][2, 2, 9] = 0.067192413843
    M_ref[(DZP, LDA)][(1, 0)][2, 2, 16] = -0.177015192454
    M_ref[(DZP, LDA)][(1, 0)][2, 2, 17] = -0.102199769014
    M_ref[(DZP, LDA)][(1, 0)][2, 3, 5] = 0.017850109854
    M_ref[(DZP, LDA)][(1, 0)][2, 3, 14] = 0.177015192454
    M_ref[(DZP, LDA)][(1, 0)][2, 4, 1] = 0.134108751706
    M_ref[(DZP, LDA)][(1, 0)][2, 4, 10] = 0.040722834832
    M_ref[(DZP, LDA)][(1, 0)][2, 5, 3] = 0.134108751706
    M_ref[(DZP, LDA)][(1, 0)][2, 5, 12] = 0.040722834832
    M_ref[(DZP, LDA)][(1, 0)][2, 7, 2] = -0.134108751706
    M_ref[(DZP, LDA)][(1, 0)][2, 7, 11] = -0.040722834832
    M_ref[(DZP, LDA)][(1, 0)][2, 8, 2] = -0.077427723898
    M_ref[(DZP, LDA)][(1, 0)][2, 8, 11] = -0.023511339652
    M_ref[(DZP, LDA)][(1, 0)][3, 0, 3] = 0.054190691356
    M_ref[(DZP, LDA)][(1, 0)][3, 0, 12] = 0.217122866287
    M_ref[(DZP, LDA)][(1, 0)][3, 1, 6] = 0.017850109854
    M_ref[(DZP, LDA)][(1, 0)][3, 1, 15] = 0.177015192454
    M_ref[(DZP, LDA)][(1, 0)][3, 2, 5] = 0.017850109854
    M_ref[(DZP, LDA)][(1, 0)][3, 2, 14] = 0.177015192454
    M_ref[(DZP, LDA)][(1, 0)][3, 3, 0] = 0.200750757040
    M_ref[(DZP, LDA)][(1, 0)][3, 3, 8] = 0.020611531459
    M_ref[(DZP, LDA)][(1, 0)][3, 3, 9] = 0.067192413843
    M_ref[(DZP, LDA)][(1, 0)][3, 3, 17] = 0.204399538028
    M_ref[(DZP, LDA)][(1, 0)][3, 5, 2] = 0.134108751706
    M_ref[(DZP, LDA)][(1, 0)][3, 5, 11] = 0.040722834832
    M_ref[(DZP, LDA)][(1, 0)][3, 6, 1] = 0.134108751706
    M_ref[(DZP, LDA)][(1, 0)][3, 6, 10] = 0.040722834832
    M_ref[(DZP, LDA)][(1, 0)][3, 8, 3] = 0.154855447797
    M_ref[(DZP, LDA)][(1, 0)][3, 8, 12] = 0.047022679305
    M_ref[(DZP, LDA)][(1, 1)][0, 0, 0] = 0.049624930293
    M_ref[(DZP, LDA)][(1, 1)][0, 0, 9] = 0.232469861481
    M_ref[(DZP, LDA)][(1, 1)][0, 1, 1] = -0.025153625036
    M_ref[(DZP, LDA)][(1, 1)][0, 1, 10] = 0.299417122728
    M_ref[(DZP, LDA)][(1, 1)][0, 2, 2] = -0.025153625036
    M_ref[(DZP, LDA)][(1, 1)][0, 2, 11] = 0.299417122728
    M_ref[(DZP, LDA)][(1, 1)][0, 3, 3] = -0.025153625036
    M_ref[(DZP, LDA)][(1, 1)][0, 3, 12] = 0.299417122728
    M_ref[(DZP, LDA)][(1, 1)][1, 0, 1] = -0.025153625036
    M_ref[(DZP, LDA)][(1, 1)][1, 0, 10] = 0.299417122728
    M_ref[(DZP, LDA)][(1, 1)][1, 1, 0] = 0.139645508352
    M_ref[(DZP, LDA)][(1, 1)][1, 1, 7] = -0.007476786181
    M_ref[(DZP, LDA)][(1, 1)][1, 1, 8] = 0.004316724514
    M_ref[(DZP, LDA)][(1, 1)][1, 1, 9] = 0.142449283422
    M_ref[(DZP, LDA)][(1, 1)][1, 1, 16] = 0.212173572705
    M_ref[(DZP, LDA)][(1, 1)][1, 1, 17] = -0.122498469316
    M_ref[(DZP, LDA)][(1, 1)][1, 2, 4] = -0.007476786181
    M_ref[(DZP, LDA)][(1, 1)][1, 2, 13] = 0.212173572705
    M_ref[(DZP, LDA)][(1, 1)][1, 3, 6] = -0.007476786181
    M_ref[(DZP, LDA)][(1, 1)][1, 3, 15] = 0.212173572705
    M_ref[(DZP, LDA)][(1, 1)][2, 0, 2] = -0.025153625036
    M_ref[(DZP, LDA)][(1, 1)][2, 0, 11] = 0.299417122728
    M_ref[(DZP, LDA)][(1, 1)][2, 1, 4] = -0.007476786181
    M_ref[(DZP, LDA)][(1, 1)][2, 1, 13] = 0.212173572705
    M_ref[(DZP, LDA)][(1, 1)][2, 2, 0] = 0.139645508352
    M_ref[(DZP, LDA)][(1, 1)][2, 2, 7] = 0.007476786181
    M_ref[(DZP, LDA)][(1, 1)][2, 2, 8] = 0.004316724514
    M_ref[(DZP, LDA)][(1, 1)][2, 2, 9] = 0.142449283422
    M_ref[(DZP, LDA)][(1, 1)][2, 2, 16] = -0.212173572705
    M_ref[(DZP, LDA)][(1, 1)][2, 2, 17] = -0.122498469316
    M_ref[(DZP, LDA)][(1, 1)][2, 3, 5] = -0.007476786181
    M_ref[(DZP, LDA)][(1, 1)][2, 3, 14] = 0.212173572705
    M_ref[(DZP, LDA)][(1, 1)][3, 0, 3] = -0.025153625036
    M_ref[(DZP, LDA)][(1, 1)][3, 0, 12] = 0.299417122728
    M_ref[(DZP, LDA)][(1, 1)][3, 1, 6] = -0.007476786181
    M_ref[(DZP, LDA)][(1, 1)][3, 1, 15] = 0.212173572705
    M_ref[(DZP, LDA)][(1, 1)][3, 2, 5] = -0.007476786181
    M_ref[(DZP, LDA)][(1, 1)][3, 2, 14] = 0.212173572705
    M_ref[(DZP, LDA)][(1, 1)][3, 3, 0] = 0.139645508352
    M_ref[(DZP, LDA)][(1, 1)][3, 3, 8] = -0.008633449029
    M_ref[(DZP, LDA)][(1, 1)][3, 3, 9] = 0.142449283422
    M_ref[(DZP, LDA)][(1, 1)][3, 3, 17] = 0.244996938632

    msg = 'Too large error for M_{0} (value={1})'
    tol = 1e-8

    for key, ref in M_ref[(size, xc)].items():
        val = M[key]
        M_diff = np.max(np.abs(val - ref))
        assert M_diff < tol, msg.format(key, str(val))

    return


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atoms', [DZP_LDA, DZP_PBE], indirect=True)
def test_on2cU(R, atoms):
    from hotcent.onsite_chargetransfer import Onsite2cUTable

    atom_Li, atom_O = atoms
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
                'sps': 0.00711285,
                'sds': 0.00572734,
                'pss': 0.00711285,
                'pps': 0.02081951,
                'ppp': 0.01157808,
                'pds': 0.02006952,
                'pdp': 0.00806458,
                'dss': 0.00572734,
                'dps': 0.02006952,
                'dpp': 0.00806458,
                'dds': 0.04323755,
                'ddp': 0.02749947,
                'ddd': 0.02083667,
            },
            (0, 1): {
                'sss': 0.00445332,
                'sps': 0.00264111,
                'sds': 0.00196433,
                'pss': 0.00198877,
                'pps': 0.00683182,
                'ppp': 0.00456657,
                'pds': 0.00547367,
                'pdp': 0.00215382,
                'dss': 0.00105713,
                'dps': 0.00391196,
                'dpp': 0.00153931,
                'dds': 0.00972233,
                'ddp': 0.00632449,
                'ddd': 0.00508863,
            },
            (1, 0): {
                'sss': 0.00445332,
                'sps': 0.00198877,
                'sds': 0.00105713,
                'pss': 0.00264111,
                'pps': 0.00683182,
                'ppp': 0.00456657,
                'pds': 0.00391196,
                'pdp': 0.00153931,
                'dss': 0.00196433,
                'dps': 0.00547367,
                'dpp': 0.00215382,
                'dds': 0.00972233,
                'ddp': 0.00632449,
                'ddd': 0.00508863,
            },
            (1, 1): {
                'sss': 0.00498160,
                'sps': 0.00202983,
                'sds': 0.00110446,
                'pss': 0.00202983,
                'pps': 0.00699064,
                'ppp': 0.00529922,
                'pds': 0.00410613,
                'pdp': 0.00185891,
                'dss': 0.00110446,
                'dps': 0.00410613,
                'dpp': 0.00185891,
                'dds': 0.01046410,
                'ddp': 0.00761714,
                'ddd': 0.00652945,
            },
        },
        (R1, PBE): {
            (0, 0): {
                'sss': 0.00623540,
                'sps': 0.00496394,
                'sds': 0.00395225,
                'pss': 0.00496394,
                'pps': 0.01771203,
                'ppp': 0.01151255,
                'pds': 0.01813133,
                'pdp': 0.00668501,
                'dss': 0.00395225,
                'dps': 0.01813133,
                'dpp': 0.00668501,
                'dds': 0.04330121,
                'ddp': 0.02361661,
                'ddd': 0.01931466,
            },
            (0, 1): {
                'sss': 0.00316980,
                'sps': 0.00071152,
                'sds': 0.00014042,
                'pss': 0.00059966,
                'pps': 0.00452507,
                'ppp': 0.00470413,
                'pds': 0.00262232,
                'pdp': 0.00197753,
                'dss': -0.00017434,
                'dps': 0.00164612,
                'dpp': 0.00170384,
                'dds': 0.00915321,
                'ddp': 0.00869776,
                'ddd': 0.00796291,
            },
            (1, 0): {
                'sss': 0.00316980,
                'sps': 0.00059966,
                'sds': -0.00017434,
                'pss': 0.00071152,
                'pps': 0.00452507,
                'ppp': 0.00470413,
                'pds': 0.00164612,
                'pdp': 0.00170384,
                'dss': 0.00014042,
                'dps': 0.00262232,
                'dpp': 0.00197753,
                'dds': 0.00915321,
                'ddp': 0.00869776,
                'ddd': 0.00796291,
            },
            (1, 1): {
                'sss': 0.00338563,
                'sps': -0.00011976,
                'sds': -0.00046255,
                'pss': -0.00011976,
                'pps': 0.00290030,
                'ppp': 0.00367492,
                'pds': 0.00047458,
                'pdp': 0.00060302,
                'dss': -0.00046255,
                'dps': 0.00047458,
                'dpp': 0.00060302,
                'dds': 0.00673831,
                'ddp': 0.00599253,
                'ddd': 0.00591922,
            },
        },
    }

    msg = 'Too large error for U_{0}[{1}] (value={2})'
    tol = 1e-5

    for key, refs in U_ref[(R, xc)].items():
        for integral, ref in refs.items():
            index = INTEGRALS_2CK.index(integral)
            val = U[key][0, index]
            U_diff = np.abs(val - ref)
            assert U_diff < tol, msg.format(key, integral, val)
    return


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atoms', [DZP_LDA, DZP_PBE], indirect=True)
def test_on2cW(R, atoms):
    from hotcent.onsite_magnetization import Onsite2cWTable

    atom_Li, atom_O = atoms
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
                'sps': 0.00433160,
                'sds': 0.00352359,
                'pss': 0.00433160,
                'pps': 0.01242298,
                'ppp': 0.00673744,
                'pds': 0.01185479,
                'pdp': 0.00449523,
                'dss': 0.00352359,
                'dps': 0.01185479,
                'dpp': 0.00449523,
                'dds': 0.02454772,
                'ddp': 0.01490413,
                'ddd': 0.01100563,
            },
            (0, 1): {
                'sss': 0.00318827,
                'sps': 0.00180628,
                'sds': 0.00131067,
                'pss': 0.00136014,
                'pps': 0.00466753,
                'ppp': 0.00315608,
                'pds': 0.00364288,
                'pdp': 0.00144071,
                'dss': 0.00070536,
                'dps': 0.00260352,
                'dpp': 0.00102966,
                'dds': 0.00646387,
                'ddp': 0.00422246,
                'ddd': 0.00341325,
            },
            (1, 0): {
                'sss': 0.00318827,
                'sps': 0.00136014,
                'sds': 0.00070536,
                'pss': 0.00180628,
                'pps': 0.00466753,
                'ppp': 0.00315608,
                'pds': 0.00260352,
                'pdp': 0.00102966,
                'dss': 0.00131067,
                'dps': 0.00364288,
                'dpp': 0.00144071,
                'dds': 0.00646387,
                'ddp': 0.00422246,
                'ddd': 0.00341325,
            },
            (1, 1): {
                'sss': 0.00365697,
                'sps': 0.00143421,
                'sds': 0.00075818,
                'pss': 0.00143421,
                'pps': 0.00494353,
                'ppp': 0.00378241,
                'pds': 0.00282310,
                'pdp': 0.00129029,
                'dss': 0.00075818,
                'dps': 0.00282310,
                'dpp': 0.00129029,
                'dds': 0.00721201,
                'ddp': 0.00528523,
                'ddd': 0.00454886,
            },
        },
        (R1, PBE): {
            (0, 0): {
                'sss': 0.00569793,
                'sps': 0.00397598,
                'sds': 0.00248800,
                'pss': 0.00397598,
                'pps': 0.01317676,
                'ppp': 0.01040289,
                'pds': 0.01250907,
                'pdp': 0.00579619,
                'dss': 0.00248800,
                'dps': 0.01250907,
                'dpp': 0.00579619,
                'dds': 0.03010348,
                'ddp': 0.01662511,
                'ddd': 0.01471903,
            },
            (0, 1): {
                'sss': 0.00313906,
                'sps': 0.00054571,
                'sds': -0.00103159,
                'pss': 0.00124287,
                'pps': 0.00499915,
                'ppp': 0.00514968,
                'pds': 0.00159237,
                'pdp': 0.00149699,
                'dss': 0.00074671,
                'dps': 0.00368828,
                'dpp': 0.00206000,
                'dds': 0.01071638,
                'ddp': 0.00833367,
                'ddd': 0.00846799,
            },
            (1, 0): {
                'sss': 0.00313906,
                'sps': 0.00124287,
                'sds': 0.00074671,
                'pss': 0.00054571,
                'pps': 0.00499915,
                'ppp': 0.00514968,
                'pds': 0.00368828,
                'pdp': 0.00206000,
                'dss': -0.00103159,
                'dps': 0.00159237,
                'dpp': 0.00149699,
                'dds': 0.01071638,
                'ddp': 0.00833367,
                'ddd': 0.00846799,
            },
            (1, 1): {
                'sss': 0.00243092,
                'sps': -0.00071933,
                'sds': -0.00154043,
                'pss': -0.00071933,
                'pps': 0.00054527,
                'ppp': 0.00317437,
                'pds': -0.00285001,
                'pdp': -0.00018665,
                'dss': -0.00154043,
                'dps': -0.00285001,
                'dpp': -0.00018665,
                'dds': 0.00001360,
                'ddp': 0.00273435,
                'ddd': 0.00513594,
            },
        },
    }

    msg = 'Too large error for W_{0}[{1}] (value={2})'
    tol = 1e-5

    for key, refs in W_ref[(R, xc)].items():
        for integral, ref in refs.items():
            index = INTEGRALS_2CK.index(integral)
            val = W[key][0, index]
            W_diff = np.abs(val - ref)
            assert W_diff < tol, msg.format(key, integral, val)

    return


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atoms', [DZP_LDA, DZP_PBE], indirect=True)
def test_off2cU(R, atoms):
    from hotcent.offsite_chargetransfer import Offsite2cUTable

    atom_Li, atom_O = atoms
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
                'sps': -1.33050789,
                'sds': 0.57711139,
                'pss': 0.38044328,
                'pps': 0.12931266,
                'ppp': 0.39747431,
                'pds': -0.18192880,
                'pdp': -0.28105437,
                'dss': 0.03922170,
                'dps': 0.06548470,
                'dpp': 0.09050758,
                'dds': 0.00858278,
                'ddp': 0.01143261,
                'ddd': 0.08606517,
            },
            (0, 0, 1): {
                'sss': -0.84811618,
                'sps': -1.66041597,
                'sds': 0.82473164,
                'pss': 0.57346979,
                'pps': 0.00168941,
                'ppp': 0.48619331,
                'pds': -0.13609812,
                'pdp': -0.38581498,
                'dss': 0.06851085,
                'dps': 0.08844243,
                'dpp': 0.14933741,
                'dds': -0.02091618,
                'ddp': -0.02701160,
                'ddd': 0.10868355,
            },
            (0, 1, 0): {
                'sss': -1.24721211,
                'sps': -1.43873638,
                'sds': 0.66106493,
                'pss': 0.29777413,
                'pps': 0.09380324,
                'ppp': 0.32752907,
                'pds': -0.16018501,
                'pdp': -0.26001729,
                'dss': 0.00947374,
                'dps': 0.04516779,
                'dpp': 0.05364457,
                'dds': -0.00040194,
                'ddp': 0.00400165,
                'ddd': 0.05944086,
            },
            (0, 1, 1): {
                'sss': -0.71925598,
                'sps': -1.84011302,
                'sds': 0.98726978,
                'pss': 0.47136675,
                'pps': -0.04185113,
                'ppp': 0.41829667,
                'pds': -0.10529874,
                'pdp': -0.38696562,
                'dss': 0.02051172,
                'dps': 0.06926456,
                'dpp': 0.10488547,
                'dds': -0.03424018,
                'ddp': -0.04584462,
                'ddd': 0.08776642,
            },
            (1, 0, 0): {
                'sss': -1.33682383,
                'sps': -0.38044319,
                'sds': 0.03922041,
                'pss': 1.33050795,
                'pps': 0.12931259,
                'ppp': 0.39747438,
                'pds': -0.06548515,
                'pdp': -0.09050744,
                'dss': 0.57711140,
                'dps': 0.18192871,
                'dpp': 0.28105446,
                'dds': 0.00858262,
                'ddp': 0.01143265,
                'ddd': 0.08606520,
            },
            (1, 0, 1): {
                'sss': -1.24721180,
                'sps': -0.29777579,
                'sds': 0.00946954,
                'pss': 1.43873639,
                'pps': 0.09380254,
                'ppp': 0.32752948,
                'pds': -0.04516893,
                'pdp': -0.05364491,
                'dss': 0.66106482,
                'dps': 0.16018469,
                'dpp': 0.26001759,
                'dds': -0.00040212,
                'ddp': 0.00400125,
                'ddd': 0.05944090,
            },
            (1, 1, 0): {
                'sss': -0.84811605,
                'sps': -0.57346966,
                'sds': 0.06850940,
                'pss': 1.66041595,
                'pps': 0.00168934,
                'ppp': 0.48619339,
                'pds': -0.08844302,
                'pdp': -0.14933720,
                'dss': 0.82473153,
                'dps': 0.13609797,
                'dpp': 0.38581509,
                'dds': -0.02091644,
                'ddp': -0.02701150,
                'ddd': 0.10868359,
            },
            (1, 1, 1): {
                'sss': -0.71925564,
                'sps': -0.47136860,
                'sds': 0.02050691,
                'pss': 1.84011297,
                'pps': -0.04185202,
                'ppp': 0.41829720,
                'pds': -0.06926610,
                'pdp': -0.10488586,
                'dss': 0.98726951,
                'dps': 0.10529824,
                'dpp': 0.38696610,
                'dds': -0.03424051,
                'ddp': -0.04584521,
                'ddd': 0.08776651,
            },
        },
        (R1, PBE): {
            (0, 0, 0): {
                'sss': -1.34354021,
                'sps': -1.32940164,
                'sds': 0.57568586,
                'pss': 0.37542713,
                'pps': 0.13150695,
                'ppp': 0.39663114,
                'pds': -0.18335395,
                'pdp': -0.27696122,
                'dss': 0.03982590,
                'dps': 0.06119804,
                'dpp': 0.09259954,
                'dds': 0.01103275,
                'ddp': 0.01100008,
                'ddd': 0.08406806,
            },
            (0, 0, 1): {
                'sss': -0.86187892,
                'sps': -1.65725615,
                'sds': 0.81969122,
                'pss': 0.56026086,
                'pps': 0.01338826,
                'ppp': 0.48626787,
                'pds': -0.14676410,
                'pdp': -0.38048223,
                'dss': 0.06688377,
                'dps': 0.08348504,
                'dpp': 0.14765082,
                'dds': -0.01428304,
                'ddp': -0.01679571,
                'ddd': 0.10825171,
            },
            (0, 1, 0): {
                'sss': -1.25336229,
                'sps': -1.44034761,
                'sds': 0.66343639,
                'pss': 0.29330110,
                'pps': 0.09683455,
                'ppp': 0.32863532,
                'pds': -0.16306380,
                'pdp': -0.26254886,
                'dss': 0.00784511,
                'dps': 0.04603189,
                'dpp': 0.05351057,
                'dds': -0.00056041,
                'ddp': 0.00419837,
                'ddd': 0.06025412,
            },
            (0, 1, 1): {
                'sss': -0.73314234,
                'sps': -1.83925912,
                'sds': 0.98690375,
                'pss': 0.46138334,
                'pps': -0.03362485,
                'ppp': 0.41992155,
                'pds': -0.11234398,
                'pdp': -0.39081357,
                'dss': 0.01685019,
                'dps': 0.07124868,
                'dpp': 0.10426328,
                'dds': -0.03423484,
                'ddp': -0.04530061,
                'ddd': 0.08937824,
            },
            (1, 0, 0): {
                'sss': -1.34354009,
                'sps': -0.37542707,
                'sds': 0.03982464,
                'pss': 1.32940169,
                'pps': 0.13150688,
                'ppp': 0.39663122,
                'pds': -0.06119848,
                'pdp': -0.09259941,
                'dss': 0.57568586,
                'dps': 0.18335386,
                'dpp': 0.27696131,
                'dds': 0.01103260,
                'ddp': 0.01100012,
                'ddd': 0.08406809,
            },
            (1, 0, 1): {
                'sss': -1.25336195,
                'sps': -0.29330334,
                'sds': 0.00784219,
                'pss': 1.44034760,
                'pps': 0.09683379,
                'ppp': 0.32863576,
                'pds': -0.04603283,
                'pdp': -0.05351110,
                'dss': 0.66343627,
                'dps': 0.16306350,
                'dpp': 0.26254916,
                'dds': -0.00056055,
                'ddp': 0.00419791,
                'ddd': 0.06025415,
            },
            (1, 1, 0): {
                'sss': -0.86187882,
                'sps': -0.56026077,
                'sds': 0.06688235,
                'pss': 1.65725615,
                'pps': 0.01338818,
                'ppp': 0.48626796,
                'pds': -0.08348561,
                'pdp': -0.14765063,
                'dss': 0.81969111,
                'dps': 0.14676396,
                'dpp': 0.38048234,
                'dds': -0.01428329,
                'ddp': -0.01679563,
                'ddd': 0.10825176,
            },
            (1, 1, 1): {
                'sss': -0.73314194,
                'sps': -0.46138589,
                'sds': 0.01684688,
                'pss': 1.83925907,
                'pps': -0.03362585,
                'ppp': 0.41992213,
                'pds': -0.07124993,
                'pdp': -0.10426396,
                'dss': 0.98690347,
                'dps': 0.11234350,
                'dpp': 0.39081405,
                'dds': -0.03423512,
                'ddp': -0.04530131,
                'ddd': 0.08937832,
            },
        },
    }

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
@pytest.mark.parametrize('atoms', [DZP_LDA, DZP_PBE], indirect=True)
def test_off2cW(R, atoms):
    from hotcent.offsite_magnetization import Offsite2cWTable

    atom_Li, atom_O = atoms
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
                'sps': 0.04321304,
                'sds': -0.03263552,
                'pss': -0.01519029,
                'pps': -0.00125559,
                'ppp': -0.02672276,
                'pds': 0.01846841,
                'pdp': 0.03592971,
                'dss': 0.00893447,
                'dps': -0.01715703,
                'dpp': -0.01436607,
                'dds': 0.00419396,
                'ddp': 0.00102434,
                'ddd': -0.02249887,
            },
            (0, 0, 1): {
                'sss': -0.05103998,
                'sps': 0.06905533,
                'sds': -0.05904859,
                'pss': -0.03595845,
                'pps': 0.02456823,
                'ppp': -0.04278627,
                'pds': 0.00571473,
                'pdp': 0.06550267,
                'dss': 0.01053004,
                'dps': -0.03096229,
                'dpp': -0.04048147,
                'dds': 0.02473805,
                'ddp': 0.03453890,
                'ddd': -0.04144046,
            },
            (0, 1, 0): {
                'sss': -0.02398122,
                'sps': 0.03044775,
                'sds': -0.02559673,
                'pss': -0.00621976,
                'pps': -0.00016472,
                'ppp': -0.00946047,
                'pds': 0.00727597,
                'pdp': 0.01392206,
                'dss': 0.00235812,
                'dps': -0.00489418,
                'dpp': -0.00273153,
                'dds': 0.00240029,
                'ddp': 0.00033322,
                'ddd': -0.00414353,
            },
            (0, 1, 1): {
                'sss': -0.03677731,
                'sps': 0.05571313,
                'sds': -0.05733277,
                'pss': -0.01651465,
                'pps': 0.01252110,
                'ppp': -0.01763362,
                'pds': 0.00108006,
                'pdp': 0.03203038,
                'dss': 0.00523780,
                'dps': -0.01363968,
                'dpp': -0.00909584,
                'dds': 0.01396355,
                'ddp': 0.00908518,
                'ddd': -0.00971911,
            },
            (1, 0, 0): {
                'sss': -0.03630386,
                'sps': 0.01519029,
                'sds': 0.00893447,
                'pss': -0.04321304,
                'pps': -0.00125559,
                'ppp': -0.02672276,
                'pds': 0.01715703,
                'pdp': 0.01436607,
                'dss': -0.03263552,
                'dps': -0.01846841,
                'dpp': -0.03592971,
                'dds': 0.00419396,
                'ddp': 0.00102434,
                'ddd': -0.02249887,
            },
            (1, 0, 1): {
                'sss': -0.02398122,
                'sps': 0.00621976,
                'sds': 0.00235812,
                'pss': -0.03044775,
                'pps': -0.00016472,
                'ppp': -0.00946047,
                'pds': 0.00489418,
                'pdp': 0.00273153,
                'dss': -0.02559673,
                'dps': -0.00727597,
                'dpp': -0.01392206,
                'dds': 0.00240029,
                'ddp': 0.00033322,
                'ddd': -0.00414353,
            },
            (1, 1, 0): {
                'sss': -0.05103998,
                'sps': 0.03595845,
                'sds': 0.01053004,
                'pss': -0.06905533,
                'pps': 0.02456823,
                'ppp': -0.04278627,
                'pds': 0.03096229,
                'pdp': 0.04048147,
                'dss': -0.05904859,
                'dps': -0.00571473,
                'dpp': -0.06550267,
                'dds': 0.02473805,
                'ddp': 0.03453890,
                'ddd': -0.04144046,
            },
            (1, 1, 1): {
                'sss': -0.03677731,
                'sps': 0.01651465,
                'sds': 0.00523780,
                'pss': -0.05571313,
                'pps': 0.01252110,
                'ppp': -0.01763362,
                'pds': 0.01363968,
                'pdp': 0.00909584,
                'dss': -0.05733277,
                'dps': -0.00108006,
                'dpp': -0.03203038,
                'dds': 0.01396355,
                'ddp': 0.00908518,
                'ddd': -0.00971911,
            },
        },
        (R1, PBE): {
            (0, 0, 0): {
                'sss': -0.04047031,
                'sps': 0.04977486,
                'sds': -0.03701839,
                'pss': -0.01727133,
                'pps': -0.00184452,
                'ppp': -0.03582086,
                'pds': 0.02480336,
                'pdp': 0.05169990,
                'dss': 0.01405461,
                'dps': -0.02730115,
                'dpp': -0.01885305,
                'dds': 0.00706699,
                'ddp': 0.00340811,
                'ddd': -0.03773565,
            },
            (0, 0, 1): {
                'sss': -0.05475492,
                'sps': 0.07540528,
                'sds': -0.06327334,
                'pss': -0.04233472,
                'pps': 0.03106577,
                'ppp': -0.05148989,
                'pds': 0.00757083,
                'pdp': 0.08417260,
                'dss': 0.01210301,
                'dps': -0.04214521,
                'dpp': -0.05253300,
                'dds': 0.03724308,
                'ddp': 0.05412843,
                'ddd': -0.05757351,
            },
            (0, 1, 0): {
                'sss': -0.02253006,
                'sps': 0.02900076,
                'sds': -0.02471028,
                'pss': -0.00575219,
                'pps': -0.00098848,
                'ppp': -0.01005580,
                'pds': 0.00847142,
                'pdp': 0.01500851,
                'dss': 0.00360047,
                'dps': -0.00640818,
                'dpp': -0.00301706,
                'dds': 0.00288850,
                'ddp': -0.00018498,
                'ddd': -0.00539221,
            },
            (0, 1, 1): {
                'sss': -0.03489986,
                'sps': 0.05348673,
                'sds': -0.05584527,
                'pss': -0.01640830,
                'pps': 0.01171184,
                'ppp': -0.01871494,
                'pds': 0.00291640,
                'pdp': 0.03436080,
                'dss': 0.00773220,
                'dps': -0.01846605,
                'dpp': -0.01073048,
                'dds': 0.01849789,
                'ddp': 0.01018712,
                'ddd': -0.01248703,
            },
            (1, 0, 0): {
                'sss': -0.04047031,
                'sps': 0.01727133,
                'sds': 0.01405461,
                'pss': -0.04977486,
                'pps': -0.00184452,
                'ppp': -0.03582086,
                'pds': 0.02730115,
                'pdp': 0.01885305,
                'dss': -0.03701839,
                'dps': -0.02480336,
                'dpp': -0.05169990,
                'dds': 0.00706699,
                'ddp': 0.00340811,
                'ddd': -0.03773565,
            },
            (1, 0, 1): {
                'sss': -0.02253006,
                'sps': 0.00575219,
                'sds': 0.00360047,
                'pss': -0.02900076,
                'pps': -0.00098848,
                'ppp': -0.01005580,
                'pds': 0.00640818,
                'pdp': 0.00301706,
                'dss': -0.02471028,
                'dps': -0.00847142,
                'dpp': -0.01500851,
                'dds': 0.00288850,
                'ddp': -0.00018498,
                'ddd': -0.00539221,
            },
            (1, 1, 0): {
                'sss': -0.05475492,
                'sps': 0.04233472,
                'sds': 0.01210301,
                'pss': -0.07540528,
                'pps': 0.03106577,
                'ppp': -0.05148989,
                'pds': 0.04214521,
                'pdp': 0.05253300,
                'dss': -0.06327334,
                'dps': -0.00757083,
                'dpp': -0.08417260,
                'dds': 0.03724308,
                'ddp': 0.05412843,
                'ddd': -0.05757351,
            },
            (1, 1, 1): {
                'sss': -0.03489986,
                'sps': 0.01640830,
                'sds': 0.00773220,
                'pss': -0.05348673,
                'pps': 0.01171184,
                'ppp': -0.01871494,
                'pds': 0.01846605,
                'pdp': 0.01073048,
                'dss': -0.05584527,
                'dps': -0.00291640,
                'dpp': -0.03436080,
                'dds': 0.01849789,
                'ddp': 0.01018712,
                'ddd': -0.01248703,
            },
        },
    }

    msg = 'Too large error for W_{0}[{1}] (value={2})'
    tol = 1e-5

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
        key1: {
            key2: np.zeros((9, 9, 36))
            for key2 in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                         (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
        }
        for key1 in [(DZP, LDA)]
    }

    M_ref[(DZP, LDA)][(0, 0, 0)][0, 0, 0] = 0.116892249500
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 0, 3] = 0.012423085656
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 0, 8] = 0.004289171495
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 0, 9] = -0.046656558049
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 0, 12] = -0.003185900663
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 0, 17] = -0.002878634260
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 0, 18] = 0.016291550919
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 0, 21] = -0.013983829816
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 0, 26] = 0.010260462319
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 0, 27] = 0.020809860507
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 0, 30] = -0.029020197580
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 0, 35] = 0.014919786828
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 1, 1] = 0.085526420257
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 1, 6] = 0.044377965368
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 1, 10] = -0.031061937422
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 1, 15] = -0.012443807201
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 1, 19] = -0.005171372052
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 1, 24] = -0.001954855393
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 1, 28] = 0.021943478512
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 1, 33] = -0.011733443295
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 2, 2] = 0.085526420257
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 2, 5] = 0.044377965368
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 2, 11] = -0.031061937422
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 2, 14] = -0.012443807201
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 2, 20] = -0.005171372052
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 2, 23] = -0.001954855393
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 2, 29] = 0.021943478512
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 2, 32] = -0.011733443295
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 3, 0] = -0.212143329421
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 3, 3] = -0.047930886028
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 3, 8] = 0.002604559609
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 3, 9] = 0.072891732967
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 3, 12] = 0.008505008550
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 3, 17] = -0.002317754497
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 3, 18] = -0.014126141334
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 3, 21] = 0.007161917579
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 3, 26] = -0.005775770125
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 3, 27] = -0.009568053159
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 3, 30] = 0.024840603362
    M_ref[(DZP, LDA)][(0, 0, 0)][0, 3, 35] = -0.014357557049
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 0, 1] = 0.104452783883
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 0, 6] = 0.007267185775
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 0, 10] = -0.016922134403
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 0, 15] = 0.001291581122
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 0, 19] = 0.012874265105
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 0, 24] = -0.016658653481
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 0, 28] = 0.022932161340
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 0, 33] = -0.020415987096
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 0] = 0.038062868448
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 3] = 0.017788323916
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 7] = 0.058876894279
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 8] = -0.001259984863
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 9] = -0.023346771023
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 12] = -0.007729124357
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 16] = -0.011877181895
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 17] = -0.006200393793
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 18] = 0.011985577343
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 21] = -0.002793200822
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 25] = -0.007993685622
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 26] = 0.003371896718
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 27] = 0.062649722770
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 30] = -0.076697351406
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 34] = 0.030459286709
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 1, 35] = 0.033227023172
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 2, 4] = 0.058876894279
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 2, 13] = -0.011877181895
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 2, 22] = -0.007993685622
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 2, 31] = 0.030459286709
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 3, 1] = -0.149647225370
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 3, 6] = -0.030094472124
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 3, 10] = -0.000682273091
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 3, 15] = -0.003392679034
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 3, 19] = -0.011062991459
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 3, 24] = 0.008405859410
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 3, 28] = -0.008531442518
    M_ref[(DZP, LDA)][(0, 0, 0)][1, 3, 33] = 0.019569284843
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 0, 2] = 0.104452783883
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 0, 5] = 0.007267185775
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 0, 11] = -0.016922134403
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 0, 14] = 0.001291581122
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 0, 20] = 0.012874265105
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 0, 23] = -0.016658653481
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 0, 29] = 0.022932161340
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 0, 32] = -0.020415987096
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 1, 4] = 0.058876894279
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 1, 13] = -0.011877181895
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 1, 22] = -0.007993685622
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 1, 31] = 0.030459286709
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 0] = 0.038062868448
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 3] = 0.017788323916
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 7] = -0.058876894279
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 8] = -0.001259984863
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 9] = -0.023346771023
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 12] = -0.007729124357
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 16] = 0.011877181895
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 17] = -0.006200393793
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 18] = 0.011985577343
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 21] = -0.002793200822
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 25] = 0.007993685622
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 26] = 0.003371896718
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 27] = 0.062649722770
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 30] = -0.076697351406
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 34] = -0.030459286709
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 2, 35] = 0.033227023172
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 3, 2] = -0.149647225370
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 3, 5] = -0.030094472124
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 3, 11] = -0.000682273091
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 3, 14] = -0.003392679034
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 3, 20] = -0.011062991459
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 3, 23] = 0.008405859410
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 3, 29] = -0.008531442518
    M_ref[(DZP, LDA)][(0, 0, 0)][2, 3, 32] = 0.019569284843
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 0, 0] = -0.016040487143
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 0, 3] = 0.084191792482
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 0, 8] = 0.028192280652
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 0, 9] = 0.007411131515
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 0, 12] = -0.007303408442
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 0, 17] = -0.007938410324
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 0, 18] = -0.037509605632
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 0, 21] = 0.035188384883
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 0, 26] = -0.024863050075
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 0, 27] = 0.077432373116
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 0, 30] = -0.064050269503
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 0, 35] = 0.029202308231
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 1, 1] = 0.033553999957
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 1, 6] = 0.070630219774
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 1, 10] = -0.016504889577
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 1, 15] = -0.015941011740
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 1, 19] = -0.024119615963
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 1, 24] = 0.000312114558
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 1, 28] = 0.061644996035
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 1, 33] = -0.016028568462
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 2, 2] = 0.033553999957
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 2, 5] = 0.070630219774
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 2, 11] = -0.016504889577
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 2, 14] = -0.015941011740
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 2, 20] = -0.024119615963
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 2, 23] = 0.000312114558
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 2, 29] = 0.061644996035
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 2, 32] = -0.016028568462
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 3, 0] = -0.048424632051
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 3, 3] = -0.123286295556
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 3, 8] = -0.036370920553
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 3, 9] = 0.025048731589
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 3, 12] = -0.016662480759
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 3, 17] = -0.004092629969
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 3, 18] = 0.008258536094
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 3, 21] = -0.028173788377
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 3, 26] = 0.015733884344
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 3, 27] = -0.027735380548
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 3, 30] = 0.060446011875
    M_ref[(DZP, LDA)][(0, 0, 0)][3, 3, 35] = -0.028479718067
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 0, 4] = 0.141082255130
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 0, 13] = -0.024358900389
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 0, 22] = -0.004951878011
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 0, 31] = 0.008524402149
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 1, 2] = 0.096885641651
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 1, 5] = 0.035060396288
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 1, 11] = -0.045466863924
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 1, 14] = -0.012121242574
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 1, 20] = -0.000279880599
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 1, 23] = 0.001357391385
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 1, 29] = 0.006660569592
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 1, 32] = -0.010216034982
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 2, 1] = 0.096885641651
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 2, 6] = 0.035060396288
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 2, 10] = -0.045466863924
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 2, 15] = -0.012121242574
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 2, 19] = -0.000279880599
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 2, 24] = 0.001357391385
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 2, 28] = 0.006660569592
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 2, 33] = -0.010216034982
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 3, 4] = -0.190037191673
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 3, 13] = 0.010409772262
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 3, 22] = 0.009689290553
    M_ref[(DZP, LDA)][(0, 0, 0)][4, 3, 31] = -0.011506586367
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 0, 2] = 0.034338964326
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 0, 5] = 0.134087872353
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 0, 11] = -0.017238824670
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 0, 14] = -0.022177101717
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 0, 20] = -0.002000766298
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 0, 23] = -0.002802428136
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 0, 29] = 0.006410785151
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 0, 32] = -0.000909688228
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 1, 4] = 0.032098689973
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 1, 13] = -0.010634798472
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 1, 22] = -0.015997889686
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 1, 31] = 0.031968976805
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 0] = 0.019157889266
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 3] = 0.091141034852
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 7] = -0.032098689973
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 8] = 0.052361208839
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 9] = -0.020550834118
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 12] = -0.042346312124
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 16] = 0.010634798472
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 17] = -0.020023228972
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 18] = -0.016949694970
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 21] = 0.023489578177
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 25] = 0.015997889686
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 26] = -0.008892889163
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 27] = 0.070289996210
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 30] = -0.089699370255
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 34] = -0.031968976805
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 2, 35] = 0.047244154617
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 3, 2] = -0.089896532435
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 3, 5] = -0.190216906917
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 3, 11] = 0.040367739314
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 3, 14] = 0.010327337910
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 3, 20] = -0.003764441896
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 3, 23] = 0.010947558521
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 3, 29] = 0.007862778725
    M_ref[(DZP, LDA)][(0, 0, 0)][5, 3, 32] = -0.015257742488
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 0, 1] = 0.034338964326
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 0, 6] = 0.134087872353
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 0, 10] = -0.017238824670
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 0, 15] = -0.022177101717
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 0, 19] = -0.002000766298
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 0, 24] = -0.002802428136
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 0, 28] = 0.006410785151
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 0, 33] = -0.000909688228
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 0] = 0.019157889266
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 3] = 0.091141034852
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 7] = 0.032098689973
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 8] = 0.052361208839
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 9] = -0.020550834118
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 12] = -0.042346312124
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 16] = -0.010634798472
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 17] = -0.020023228972
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 18] = -0.016949694970
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 21] = 0.023489578177
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 25] = -0.015997889686
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 26] = -0.008892889163
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 27] = 0.070289996210
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 30] = -0.089699370255
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 34] = 0.031968976805
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 1, 35] = 0.047244154617
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 2, 4] = 0.032098689973
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 2, 13] = -0.010634798472
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 2, 22] = -0.015997889686
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 2, 31] = 0.031968976805
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 3, 1] = -0.089896532435
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 3, 6] = -0.190216906917
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 3, 10] = 0.040367739314
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 3, 15] = 0.010327337910
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 3, 19] = -0.003764441896
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 3, 24] = 0.010947558521
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 3, 28] = 0.007862778725
    M_ref[(DZP, LDA)][(0, 0, 0)][6, 3, 33] = -0.015257742488
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 0, 7] = 0.141082255130
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 0, 16] = -0.024358900389
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 0, 25] = -0.004951878011
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 0, 34] = 0.008524402149
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 1, 1] = 0.096885641651
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 1, 6] = 0.035060396288
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 1, 10] = -0.045466863924
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 1, 15] = -0.012121242574
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 1, 19] = -0.000279880599
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 1, 24] = 0.001357391385
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 1, 28] = 0.006660569592
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 1, 33] = -0.010216034982
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 2, 2] = -0.096885641651
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 2, 5] = -0.035060396288
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 2, 11] = 0.045466863924
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 2, 14] = 0.012121242574
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 2, 20] = 0.000279880599
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 2, 23] = -0.001357391385
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 2, 29] = -0.006660569592
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 2, 32] = 0.010216034982
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 3, 7] = -0.190037191673
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 3, 16] = 0.010409772262
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 3, 25] = 0.009689290553
    M_ref[(DZP, LDA)][(0, 0, 0)][7, 3, 34] = -0.011506586367
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 0, 0] = -0.000733993722
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 0, 3] = 0.034092310901
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 0, 8] = 0.123802210771
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 0, 9] = 0.001644761275
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 0, 12] = -0.017263129652
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 0, 17] = -0.018609237434
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 0, 18] = -0.000958695225
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 0, 21] = -0.002136416058
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 0, 26] = -0.000965587189
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 0, 27] = -0.006049204743
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 0, 30] = 0.013640044122
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 0, 35] = -0.008333451762
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 1, 1] = -0.030212103920
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 1, 6] = 0.034462339282
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 1, 10] = 0.011944935513
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 1, 15] = -0.013773444142
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 1, 19] = -0.008780031995
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 1, 24] = -0.009541761521
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 1, 28] = 0.008744003146
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 1, 33] = 0.022178148715
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 2, 2] = -0.030212103920
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 2, 5] = 0.034462339282
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 2, 11] = 0.011944935513
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 2, 14] = -0.013773444142
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 2, 20] = -0.008780031995
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 2, 23] = -0.009541761521
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 2, 29] = 0.008744003146
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 2, 32] = 0.022178148715
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 3, 0] = -0.029380575004
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 3, 3] = -0.097693053375
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 3, 8] = -0.160244527089
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 3, 9] = 0.014246949863
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 3, 12] = 0.043113720229
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 3, 17] = -0.002972815099
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 3, 18] = -0.020164906653
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 3, 21] = 0.018034810613
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 3, 26] = -0.005210142118
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 3, 27] = 0.043954486899
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 3, 30] = -0.047136729129
    M_ref[(DZP, LDA)][(0, 0, 0)][8, 3, 35] = 0.026476787770
    M_ref[(DZP, LDA)][(0, 0, 1)][0, 0, 0] = 0.122044526464
    M_ref[(DZP, LDA)][(0, 0, 1)][0, 0, 3] = 0.026368210453
    M_ref[(DZP, LDA)][(0, 0, 1)][0, 0, 8] = -0.000806006844
    M_ref[(DZP, LDA)][(0, 0, 1)][0, 0, 9] = -0.043035540215
    M_ref[(DZP, LDA)][(0, 0, 1)][0, 0, 12] = -0.006933327081
    M_ref[(DZP, LDA)][(0, 0, 1)][0, 0, 17] = -0.002179715405
    M_ref[(DZP, LDA)][(0, 0, 1)][0, 0, 18] = 0.001714004145
    M_ref[(DZP, LDA)][(0, 0, 1)][0, 0, 21] = 0.001857826540
    M_ref[(DZP, LDA)][(0, 0, 1)][0, 0, 26] = 0.000772534596
    M_ref[(DZP, LDA)][(0, 0, 1)][0, 0, 27] = 0.037289792149
    M_ref[(DZP, LDA)][(0, 0, 1)][0, 0, 30] = -0.046427947584
    M_ref[(DZP, LDA)][(0, 0, 1)][0, 0, 35] = 0.023513457009
    M_ref[(DZP, LDA)][(0, 0, 1)][1, 0, 1] = 0.106240645482
    M_ref[(DZP, LDA)][(0, 0, 1)][1, 0, 6] = 0.026094274398
    M_ref[(DZP, LDA)][(0, 0, 1)][1, 0, 10] = -0.007716843380
    M_ref[(DZP, LDA)][(0, 0, 1)][1, 0, 15] = -0.003258403542
    M_ref[(DZP, LDA)][(0, 0, 1)][1, 0, 19] = 0.000984104111
    M_ref[(DZP, LDA)][(0, 0, 1)][1, 0, 24] = -0.002824377664
    M_ref[(DZP, LDA)][(0, 0, 1)][1, 0, 28] = 0.035004325624
    M_ref[(DZP, LDA)][(0, 0, 1)][1, 0, 33] = -0.032240829129
    M_ref[(DZP, LDA)][(0, 0, 1)][2, 0, 2] = 0.106240645482
    M_ref[(DZP, LDA)][(0, 0, 1)][2, 0, 5] = 0.026094274398
    M_ref[(DZP, LDA)][(0, 0, 1)][2, 0, 11] = -0.007716843380
    M_ref[(DZP, LDA)][(0, 0, 1)][2, 0, 14] = -0.003258403542
    M_ref[(DZP, LDA)][(0, 0, 1)][2, 0, 20] = 0.000984104111
    M_ref[(DZP, LDA)][(0, 0, 1)][2, 0, 23] = -0.002824377664
    M_ref[(DZP, LDA)][(0, 0, 1)][2, 0, 29] = 0.035004325624
    M_ref[(DZP, LDA)][(0, 0, 1)][2, 0, 32] = -0.032240829129
    M_ref[(DZP, LDA)][(0, 0, 1)][3, 0, 0] = -0.003669431490
    M_ref[(DZP, LDA)][(0, 0, 1)][3, 0, 3] = 0.071003107597
    M_ref[(DZP, LDA)][(0, 0, 1)][3, 0, 8] = 0.041969902969
    M_ref[(DZP, LDA)][(0, 0, 1)][3, 0, 9] = 0.000675781450
    M_ref[(DZP, LDA)][(0, 0, 1)][3, 0, 12] = 0.008992444203
    M_ref[(DZP, LDA)][(0, 0, 1)][3, 0, 17] = -0.011081101718
    M_ref[(DZP, LDA)][(0, 0, 1)][3, 0, 18] = -0.026695040817
    M_ref[(DZP, LDA)][(0, 0, 1)][3, 0, 21] = 0.019763722709
    M_ref[(DZP, LDA)][(0, 0, 1)][3, 0, 26] = -0.008379515414
    M_ref[(DZP, LDA)][(0, 0, 1)][3, 0, 27] = 0.083346382897
    M_ref[(DZP, LDA)][(0, 0, 1)][3, 0, 30] = -0.068752441835
    M_ref[(DZP, LDA)][(0, 0, 1)][3, 0, 35] = 0.025607323767
    M_ref[(DZP, LDA)][(0, 0, 1)][4, 0, 4] = 0.150352318485
    M_ref[(DZP, LDA)][(0, 0, 1)][4, 0, 13] = -0.019407504706
    M_ref[(DZP, LDA)][(0, 0, 1)][4, 0, 22] = -0.009949048547
    M_ref[(DZP, LDA)][(0, 0, 1)][4, 0, 31] = 0.016158149064
    M_ref[(DZP, LDA)][(0, 0, 1)][5, 0, 2] = 0.067124086978
    M_ref[(DZP, LDA)][(0, 0, 1)][5, 0, 5] = 0.137862477756
    M_ref[(DZP, LDA)][(0, 0, 1)][5, 0, 11] = -0.033286438760
    M_ref[(DZP, LDA)][(0, 0, 1)][5, 0, 14] = -0.016268766249
    M_ref[(DZP, LDA)][(0, 0, 1)][5, 0, 20] = -0.003372101401
    M_ref[(DZP, LDA)][(0, 0, 1)][5, 0, 23] = -0.004982549219
    M_ref[(DZP, LDA)][(0, 0, 1)][5, 0, 29] = 0.010749045704
    M_ref[(DZP, LDA)][(0, 0, 1)][5, 0, 32] = -0.001455692551
    M_ref[(DZP, LDA)][(0, 0, 1)][6, 0, 1] = 0.067124086978
    M_ref[(DZP, LDA)][(0, 0, 1)][6, 0, 6] = 0.137862477756
    M_ref[(DZP, LDA)][(0, 0, 1)][6, 0, 10] = -0.033286438760
    M_ref[(DZP, LDA)][(0, 0, 1)][6, 0, 15] = -0.016268766249
    M_ref[(DZP, LDA)][(0, 0, 1)][6, 0, 19] = -0.003372101401
    M_ref[(DZP, LDA)][(0, 0, 1)][6, 0, 24] = -0.004982549219
    M_ref[(DZP, LDA)][(0, 0, 1)][6, 0, 28] = 0.010749045704
    M_ref[(DZP, LDA)][(0, 0, 1)][6, 0, 33] = -0.001455692551
    M_ref[(DZP, LDA)][(0, 0, 1)][7, 0, 7] = 0.150352318485
    M_ref[(DZP, LDA)][(0, 0, 1)][7, 0, 16] = -0.019407504706
    M_ref[(DZP, LDA)][(0, 0, 1)][7, 0, 25] = -0.009949048547
    M_ref[(DZP, LDA)][(0, 0, 1)][7, 0, 34] = 0.016158149064
    M_ref[(DZP, LDA)][(0, 0, 1)][8, 0, 0] = -0.009137219003
    M_ref[(DZP, LDA)][(0, 0, 1)][8, 0, 3] = 0.054287404622
    M_ref[(DZP, LDA)][(0, 0, 1)][8, 0, 8] = 0.113695454587
    M_ref[(DZP, LDA)][(0, 0, 1)][8, 0, 9] = 0.007796495148
    M_ref[(DZP, LDA)][(0, 0, 1)][8, 0, 12] = -0.026538494433
    M_ref[(DZP, LDA)][(0, 0, 1)][8, 0, 17] = -0.007714625648
    M_ref[(DZP, LDA)][(0, 0, 1)][8, 0, 18] = 0.000132783762
    M_ref[(DZP, LDA)][(0, 0, 1)][8, 0, 21] = -0.005088487319
    M_ref[(DZP, LDA)][(0, 0, 1)][8, 0, 26] = -0.000153424088
    M_ref[(DZP, LDA)][(0, 0, 1)][8, 0, 27] = -0.011775994852
    M_ref[(DZP, LDA)][(0, 0, 1)][8, 0, 30] = 0.023712003742
    M_ref[(DZP, LDA)][(0, 0, 1)][8, 0, 35] = -0.015206953101
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 0, 0] = 0.104092553849
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 0, 3] = 0.012964030852
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 0, 8] = -0.004892345292
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 0, 9] = -0.025106041229
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 0, 12] = -0.001864886551
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 0, 17] = 0.000395061278
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 0, 18] = -0.005988034409
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 0, 21] = 0.005179692649
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 0, 26] = -0.003371083581
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 0, 27] = -0.003165193979
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 0, 30] = 0.005962558715
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 0, 35] = -0.003902812294
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 1, 1] = 0.044979101358
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 1, 6] = 0.019773463722
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 1, 10] = -0.003160893193
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 1, 15] = 0.000840549708
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 1, 19] = 0.000404964364
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 1, 24] = -0.000787914547
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 1, 28] = -0.004988953600
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 1, 33] = 0.006443597474
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 2, 2] = 0.044979101358
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 2, 5] = 0.019773463722
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 2, 11] = -0.003160893193
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 2, 14] = 0.000840549708
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 2, 20] = 0.000404964364
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 2, 23] = -0.000787914547
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 2, 29] = -0.004988953600
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 2, 32] = 0.006443597474
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 3, 0] = -0.171326192092
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 3, 3] = -0.035798656464
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 3, 8] = 0.007668147678
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 3, 9] = 0.025905265404
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 3, 12] = -0.001626124901
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 3, 17] = -0.003067299480
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 3, 18] = 0.007295135531
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 3, 21] = -0.005474850302
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 3, 26] = 0.003457471511
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 3, 27] = 0.010815233355
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 3, 30] = -0.016413812799
    M_ref[(DZP, LDA)][(0, 1, 0)][0, 3, 35] = 0.010599434556
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 0, 1] = 0.072891230434
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 0, 6] = 0.011030365807
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 0, 10] = 0.010724152005
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 0, 15] = 0.000161454672
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 0, 19] = -0.000686265826
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 0, 24] = 0.001252413087
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 0, 28] = 0.000551651639
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 0, 33] = -0.001912888933
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 0] = 0.082641954458
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 3] = 0.023443840094
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 7] = 0.030669639595
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 8] = -0.007009184032
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 9] = -0.042797079359
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 12] = -0.008058140014
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 16] = 0.005183103339
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 17] = -0.006395326481
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 18] = -0.004265863476
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 21] = 0.005859435924
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 25] = -0.003674072235
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 26] = -0.002132419304
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 27] = 0.012651383629
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 30] = -0.017766114669
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 34] = 0.005442690489
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 1, 35] = 0.010514386545
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 2, 4] = 0.030669639595
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 2, 13] = 0.005183103339
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 2, 22] = -0.003674072235
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 2, 31] = 0.005442690489
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 3, 1] = -0.097563210699
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 3, 6] = -0.023508881669
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 3, 10] = -0.045877456256
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 3, 15] = -0.008544607201
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 3, 19] = 0.000319621998
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 3, 24] = -0.001064083934
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 3, 28] = 0.000876610607
    M_ref[(DZP, LDA)][(0, 1, 0)][1, 3, 33] = 0.001020848012
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 0, 2] = 0.072891230434
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 0, 5] = 0.011030365807
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 0, 11] = 0.010724152005
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 0, 14] = 0.000161454672
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 0, 20] = -0.000686265826
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 0, 23] = 0.001252413087
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 0, 29] = 0.000551651639
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 0, 32] = -0.001912888933
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 1, 4] = 0.030669639595
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 1, 13] = 0.005183103339
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 1, 22] = -0.003674072235
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 1, 31] = 0.005442690489
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 0] = 0.082641954458
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 3] = 0.023443840094
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 7] = -0.030669639595
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 8] = -0.007009184032
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 9] = -0.042797079359
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 12] = -0.008058140014
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 16] = -0.005183103339
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 17] = -0.006395326481
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 18] = -0.004265863476
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 21] = 0.005859435924
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 25] = 0.003674072235
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 26] = -0.002132419304
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 27] = 0.012651383629
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 30] = -0.017766114669
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 34] = -0.005442690489
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 2, 35] = 0.010514386545
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 3, 2] = -0.097563210699
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 3, 5] = -0.023508881669
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 3, 11] = -0.045877456256
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 3, 14] = -0.008544607201
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 3, 20] = 0.000319621998
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 3, 23] = -0.001064083934
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 3, 29] = 0.000876610607
    M_ref[(DZP, LDA)][(0, 1, 0)][2, 3, 32] = 0.001020848012
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 0, 0] = 0.016735416074
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 0, 3] = 0.060828087033
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 0, 8] = 0.010196193196
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 0, 9] = -0.008732644138
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 0, 12] = 0.015908601677
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 0, 17] = 0.000069039558
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 0, 18] = -0.001314840909
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 0, 21] = 0.002481721526
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 0, 26] = -0.001084863595
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 0, 27] = 0.005973442919
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 0, 30] = -0.009274957810
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 0, 35] = 0.005861386154
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 1, 1] = 0.037245633981
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 1, 6] = 0.046497508852
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 1, 10] = -0.015742184031
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 1, 15] = -0.000302334443
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 1, 19] = -0.000500050387
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 1, 24] = -0.004063595927
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 1, 28] = -0.000695852110
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 1, 33] = 0.007192850821
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 2, 2] = 0.037245633981
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 2, 5] = 0.046497508852
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 2, 11] = -0.015742184031
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 2, 14] = -0.000302334443
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 2, 20] = -0.000500050387
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 2, 23] = -0.004063595927
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 2, 29] = -0.000695852110
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 2, 32] = 0.007192850821
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 3, 0] = -0.070392589363
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 3, 3] = -0.087465797453
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 3, 8] = -0.019840814272
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 3, 9] = 0.032888103163
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 3, 12] = -0.051860167809
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 3, 17] = -0.012279836344
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 3, 18] = 0.001014508410
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 3, 21] = -0.002887340741
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 3, 26] = 0.001776020333
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 3, 27] = -0.001312057587
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 3, 30] = 0.005087779452
    M_ref[(DZP, LDA)][(0, 1, 0)][3, 3, 35] = -0.003743525355
    M_ref[(DZP, LDA)][(0, 1, 1)][0, 0, 0] = 0.113115195900
    M_ref[(DZP, LDA)][(0, 1, 1)][0, 0, 3] = 0.021171131822
    M_ref[(DZP, LDA)][(0, 1, 1)][0, 0, 8] = -0.007567603320
    M_ref[(DZP, LDA)][(0, 1, 1)][0, 0, 9] = -0.021409551323
    M_ref[(DZP, LDA)][(0, 1, 1)][0, 0, 12] = -0.000966974577
    M_ref[(DZP, LDA)][(0, 1, 1)][0, 0, 17] = -0.000696305205
    M_ref[(DZP, LDA)][(0, 1, 1)][0, 0, 18] = -0.006746945452
    M_ref[(DZP, LDA)][(0, 1, 1)][0, 0, 21] = 0.005691634868
    M_ref[(DZP, LDA)][(0, 1, 1)][0, 0, 26] = -0.003191086457
    M_ref[(DZP, LDA)][(0, 1, 1)][0, 0, 27] = -0.003228964603
    M_ref[(DZP, LDA)][(0, 1, 1)][0, 0, 30] = 0.006075146499
    M_ref[(DZP, LDA)][(0, 1, 1)][0, 0, 35] = -0.004212002724
    M_ref[(DZP, LDA)][(0, 1, 1)][1, 0, 1] = 0.077725992830
    M_ref[(DZP, LDA)][(0, 1, 1)][1, 0, 6] = 0.021732208415
    M_ref[(DZP, LDA)][(0, 1, 1)][1, 0, 10] = 0.019727510494
    M_ref[(DZP, LDA)][(0, 1, 1)][1, 0, 15] = 0.000755049874
    M_ref[(DZP, LDA)][(0, 1, 1)][1, 0, 19] = -0.000525881957
    M_ref[(DZP, LDA)][(0, 1, 1)][1, 0, 24] = 0.001422880394
    M_ref[(DZP, LDA)][(0, 1, 1)][1, 0, 28] = 0.000247025462
    M_ref[(DZP, LDA)][(0, 1, 1)][1, 0, 33] = -0.002449422593
    M_ref[(DZP, LDA)][(0, 1, 1)][2, 0, 2] = 0.077725992830
    M_ref[(DZP, LDA)][(0, 1, 1)][2, 0, 5] = 0.021732208415
    M_ref[(DZP, LDA)][(0, 1, 1)][2, 0, 11] = 0.019727510494
    M_ref[(DZP, LDA)][(0, 1, 1)][2, 0, 14] = 0.000755049874
    M_ref[(DZP, LDA)][(0, 1, 1)][2, 0, 20] = -0.000525881957
    M_ref[(DZP, LDA)][(0, 1, 1)][2, 0, 23] = 0.001422880394
    M_ref[(DZP, LDA)][(0, 1, 1)][2, 0, 29] = 0.000247025462
    M_ref[(DZP, LDA)][(0, 1, 1)][2, 0, 32] = -0.002449422593
    M_ref[(DZP, LDA)][(0, 1, 1)][3, 0, 0] = 0.031781379715
    M_ref[(DZP, LDA)][(0, 1, 1)][3, 0, 3] = 0.051480711611
    M_ref[(DZP, LDA)][(0, 1, 1)][3, 0, 8] = 0.015698322785
    M_ref[(DZP, LDA)][(0, 1, 1)][3, 0, 9] = -0.016009745730
    M_ref[(DZP, LDA)][(0, 1, 1)][3, 0, 12] = 0.030695715073
    M_ref[(DZP, LDA)][(0, 1, 1)][3, 0, 17] = 0.001901617640
    M_ref[(DZP, LDA)][(0, 1, 1)][3, 0, 18] = -0.004123647352
    M_ref[(DZP, LDA)][(0, 1, 1)][3, 0, 21] = 0.006215465344
    M_ref[(DZP, LDA)][(0, 1, 1)][3, 0, 26] = -0.002996021350
    M_ref[(DZP, LDA)][(0, 1, 1)][3, 0, 27] = 0.012398999376
    M_ref[(DZP, LDA)][(0, 1, 1)][3, 0, 30] = -0.018027029838
    M_ref[(DZP, LDA)][(0, 1, 1)][3, 0, 35] = 0.010884383943
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 0, 0] = 0.016293279574
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 0, 3] = 0.013993921213
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 0, 8] = 0.010275052946
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 0, 9] = 0.020786751279
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 0, 12] = 0.028979051243
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 0, 17] = 0.014874250038
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 0, 18] = 0.116942384782
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 0, 21] = -0.012438513721
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 0, 26] = 0.004278917472
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 0, 27] = -0.046685009529
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 0, 30] = 0.003193658810
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 0, 35] = -0.002874214005
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 1, 1] = 0.012874392252
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 1, 6] = 0.016658826261
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 1, 10] = 0.022932067054
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 1, 15] = 0.020415866949
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 1, 19] = 0.104452783039
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 1, 24] = -0.007267213316
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 1, 28] = -0.016922126752
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 1, 33] = -0.001291571980
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 2, 2] = 0.012874392252
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 2, 5] = 0.016658826261
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 2, 11] = 0.022932067054
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 2, 14] = 0.020415866949
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 2, 20] = 0.104452783039
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 2, 23] = -0.007267213316
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 2, 29] = -0.016922126752
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 2, 32] = -0.001291571980
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 3, 0] = 0.037510284580
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 3, 3] = 0.035196303825
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 3, 8] = 0.024872920077
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 3, 9] = -0.077453070240
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 3, 12] = -0.064087338761
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 3, 17] = -0.029237394935
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 3, 18] = 0.016085982922
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 3, 21] = 0.084176830235
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 3, 26] = -0.028207560745
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 3, 27] = -0.007436448321
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 3, 30] = -0.007295367589
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 3, 35] = 0.007943581120
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 4, 4] = -0.004951687014
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 4, 13] = 0.008524285556
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 4, 22] = 0.141082253544
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 4, 31] = -0.024358898332
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 5, 2] = 0.002000696549
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 5, 5] = -0.002802316226
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 5, 11] = -0.006410742365
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 5, 14] = -0.000909758122
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 5, 20] = -0.034338949848
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 5, 23] = 0.134087866595
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 5, 29] = 0.017238815928
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 5, 32] = -0.022177099049
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 6, 1] = 0.002000696549
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 6, 6] = -0.002802316226
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 6, 10] = -0.006410742365
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 6, 15] = -0.000909758122
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 6, 19] = -0.034338949848
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 6, 24] = 0.134087866595
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 6, 28] = 0.017238815928
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 6, 33] = -0.022177099049
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 7, 7] = -0.004951687014
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 7, 16] = 0.008524285556
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 7, 25] = 0.141082253544
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 7, 34] = -0.024358898332
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 8, 0] = -0.000969318486
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 8, 3] = 0.002124074300
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 8, 8] = -0.000975022094
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 8, 9] = -0.006022448126
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 8, 12] = -0.013604459028
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 8, 17] = -0.008305687948
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 8, 18] = -0.000775007356
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 8, 21] = -0.034074359721
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 8, 26] = 0.123810544600
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 8, 27] = 0.001668471071
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 8, 30] = 0.017253391248
    M_ref[(DZP, LDA)][(1, 0, 0)][0, 8, 35] = -0.018612367879
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 0, 1] = -0.005171193554
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 0, 6] = 0.001955005903
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 0, 10] = 0.021943351981
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 0, 15] = 0.011733338740
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 0, 19] = 0.085526415213
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 0, 24] = -0.044377998905
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 0, 28] = -0.031061929018
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 0, 33] = 0.012443820783
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 0] = 0.011988387633
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 3] = 0.002789607723
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 7] = -0.007993658738
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 8] = 0.003367291389
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 9] = 0.062658976401
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 12] = 0.076718338236
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 16] = 0.030459270051
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 17] = 0.033245760476
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 18] = 0.038038290167
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 21] = -0.017779546812
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 25] = 0.058876882662
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 26] = -0.001246746889
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 27] = -0.023333781829
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 30] = 0.007723998246
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 34] = -0.011877173461
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 1, 35] = -0.006204392325
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 2, 4] = -0.007993658738
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 2, 13] = 0.030459270051
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 2, 22] = 0.058876882662
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 2, 31] = -0.011877173461
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 3, 1] = 0.024119722195
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 3, 6] = 0.000312182794
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 3, 10] = -0.061645087351
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 3, 15] = -0.016028646391
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 3, 19] = -0.033553914586
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 3, 24] = 0.070630174166
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 3, 28] = 0.016504835300
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 3, 33] = -0.015940984496
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 4, 2] = -0.000279631851
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 4, 5] = -0.001357186598
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 4, 11] = 0.006660385138
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 4, 14] = 0.010215878958
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 4, 20] = 0.096885668254
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 4, 23] = -0.035060455244
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 4, 29] = -0.045466874404
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 4, 32] = 0.012121269788
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 5, 4] = 0.015997851978
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 5, 13] = -0.031968955557
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 5, 22] = -0.032098672648
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 5, 31] = 0.010634786941
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 0] = 0.016951520592
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 3] = 0.023502171901
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 7] = 0.015997851978
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 8] = 0.008906483491
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 9] = -0.070320529244
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 12] = -0.089753142996
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 16] = -0.031968955557
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 17] = -0.047292606514
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 18] = -0.019093714148
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 21] = 0.091117288785
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 25] = -0.032098672648
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 26] = -0.052382760981
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 27] = 0.020515295089
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 30] = -0.042333161648
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 34] = 0.010634786941
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 6, 35] = 0.020030572549
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 7, 1] = -0.000279631851
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 7, 6] = -0.001357186598
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 7, 10] = 0.006660385138
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 7, 15] = 0.010215878958
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 7, 19] = 0.096885668254
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 7, 24] = -0.035060455244
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 7, 28] = -0.045466874404
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 7, 33] = 0.012121269788
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 8, 1] = -0.008780109422
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 8, 6] = 0.009541639729
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 8, 10] = 0.008744067540
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 8, 15] = -0.022178053859
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 8, 19] = -0.030212135815
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 8, 24] = -0.034462308099
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 8, 28] = 0.011944952245
    M_ref[(DZP, LDA)][(1, 0, 0)][1, 8, 33] = 0.013773429547
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 0, 2] = -0.005171193554
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 0, 5] = 0.001955005903
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 0, 11] = 0.021943351981
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 0, 14] = 0.011733338740
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 0, 20] = 0.085526415213
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 0, 23] = -0.044377998905
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 0, 29] = -0.031061929018
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 0, 32] = 0.012443820783
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 1, 4] = -0.007993658738
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 1, 13] = 0.030459270051
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 1, 22] = 0.058876882662
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 1, 31] = -0.011877173461
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 0] = 0.011988387633
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 3] = 0.002789607723
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 7] = 0.007993658738
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 8] = 0.003367291389
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 9] = 0.062658976401
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 12] = 0.076718338236
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 16] = -0.030459270051
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 17] = 0.033245760476
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 18] = 0.038038290167
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 21] = -0.017779546812
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 25] = -0.058876882662
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 26] = -0.001246746889
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 27] = -0.023333781829
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 30] = 0.007723998246
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 34] = 0.011877173461
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 2, 35] = -0.006204392325
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 3, 2] = 0.024119722195
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 3, 5] = 0.000312182794
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 3, 11] = -0.061645087351
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 3, 14] = -0.016028646391
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 3, 20] = -0.033553914586
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 3, 23] = 0.070630174166
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 3, 29] = 0.016504835300
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 3, 32] = -0.015940984496
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 4, 1] = -0.000279631851
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 4, 6] = -0.001357186598
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 4, 10] = 0.006660385138
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 4, 15] = 0.010215878958
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 4, 19] = 0.096885668254
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 4, 24] = -0.035060455244
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 4, 28] = -0.045466874404
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 4, 33] = 0.012121269788
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 0] = 0.016951520592
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 3] = 0.023502171901
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 7] = -0.015997851978
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 8] = 0.008906483491
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 9] = -0.070320529244
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 12] = -0.089753142996
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 16] = 0.031968955557
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 17] = -0.047292606514
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 18] = -0.019093714148
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 21] = 0.091117288785
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 25] = 0.032098672648
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 26] = -0.052382760981
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 27] = 0.020515295089
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 30] = -0.042333161648
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 34] = -0.010634786941
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 5, 35] = 0.020030572549
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 6, 4] = 0.015997851978
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 6, 13] = -0.031968955557
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 6, 22] = -0.032098672648
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 6, 31] = 0.010634786941
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 7, 2] = 0.000279631851
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 7, 5] = 0.001357186598
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 7, 11] = -0.006660385138
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 7, 14] = -0.010215878958
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 7, 20] = -0.096885668254
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 7, 23] = 0.035060455244
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 7, 29] = 0.045466874404
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 7, 32] = -0.012121269788
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 8, 2] = -0.008780109422
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 8, 5] = 0.009541639729
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 8, 11] = 0.008744067540
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 8, 14] = -0.022178053859
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 8, 20] = -0.030212135815
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 8, 23] = -0.034462308099
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 8, 29] = 0.011944952245
    M_ref[(DZP, LDA)][(1, 0, 0)][2, 8, 32] = 0.013773429547
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 0, 0] = 0.014128740919
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 0, 3] = 0.007187800855
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 0, 8] = 0.005811521324
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 0, 9] = 0.009509218569
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 0, 12] = 0.024733288336
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 0, 17] = 0.014243332899
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 0, 18] = 0.212272514902
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 0, 21] = -0.047972478970
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 0, 26] = -0.002637097039
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 0, 27] = -0.072964363990
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 0, 30] = 0.008526588521
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 0, 35] = 0.002330503933
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 1, 1] = 0.011063477404
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 1, 6] = 0.008406294504
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 1, 10] = 0.008531098207
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 1, 15] = 0.019568981229
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 1, 19] = 0.149647256672
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 1, 24] = -0.030094570234
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 1, 28] = 0.000682264885
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 1, 33] = -0.003392635390
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 2, 2] = 0.011063477404
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 2, 5] = 0.008406294504
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 2, 11] = 0.008531098207
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 2, 14] = 0.019568981229
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 2, 20] = 0.149647256672
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 2, 23] = -0.030094570234
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 2, 29] = 0.000682264885
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 2, 32] = -0.003392635390
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 3, 0] = 0.008260501183
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 3, 3] = 0.028178963486
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 3, 8] = 0.015738156841
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 3, 9] = -0.027752689860
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 3, 12] = -0.060473352860
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 3, 17] = -0.028501380313
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 3, 18] = -0.048388315234
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 3, 21] = 0.123274032699
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 3, 26] = -0.036380936745
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 3, 27] = 0.025028019966
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 3, 30] = 0.016669248296
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 3, 35] = -0.004089204053
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 4, 4] = -0.009689003506
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 4, 13] = 0.011506414766
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 4, 22] = 0.190037186746
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 4, 31] = -0.010409767959
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 5, 2] = -0.003764725159
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 5, 5] = -0.010947525573
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 5, 11] = 0.007862971975
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 5, 14] = 0.015257740159
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 5, 20] = -0.089896544783
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 5, 23] = 0.190216938697
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 5, 29] = 0.040367745614
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 5, 32] = -0.010327354678
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 6, 1] = -0.003764725159
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 6, 6] = -0.010947525573
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 6, 10] = 0.007862971975
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 6, 15] = 0.015257740159
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 6, 19] = -0.089896544783
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 6, 24] = 0.190216938697
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 6, 28] = 0.040367745614
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 6, 33] = -0.010327354678
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 7, 7] = -0.009689003506
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 7, 16] = 0.011506414766
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 7, 25] = 0.190037186746
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 7, 34] = -0.010409767959
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 8, 0] = 0.020148588116
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 8, 3] = 0.018019699012
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 8, 8] = 0.005200924779
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 8, 9] = -0.043920333481
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 8, 12] = -0.047095849097
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 8, 17] = -0.026448628288
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 8, 18] = 0.029332247380
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 8, 21] = -0.097671270540
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 8, 26] = 0.160251434188
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 8, 27] = -0.014218335323
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 8, 30] = 0.043101916637
    M_ref[(DZP, LDA)][(1, 0, 0)][3, 8, 35] = 0.002970030863
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 0, 0] = -0.005986865440
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 0, 3] = -0.005160963127
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 0, 8] = -0.003346246768
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 0, 9] = -0.003208814121
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 0, 12] = -0.006042594546
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 0, 17] = -0.003985226019
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 0, 18] = 0.104189227354
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 0, 21] = -0.012995816779
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 0, 26] = -0.004918684289
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 0, 27] = -0.025160233149
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 0, 30] = 0.001881686969
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 0, 35] = 0.000404997284
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 1, 1] = -0.000685918314
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 1, 6] = -0.001252099998
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 1, 10] = 0.000551408482
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 1, 15] = 0.001912675977
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 1, 19] = 0.072891250436
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 1, 24] = -0.011030436626
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 1, 28] = 0.010724145736
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 1, 33] = -0.000161421828
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 2, 2] = -0.000685918314
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 2, 5] = -0.001252099998
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 2, 11] = 0.000551408482
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 2, 14] = 0.001912675977
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 2, 20] = 0.072891250436
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 2, 23] = -0.011030436626
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 2, 29] = 0.010724145736
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 2, 32] = -0.000161421828
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 3, 0] = 0.001314596057
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 3, 3] = 0.002482722989
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 3, 8] = 0.001085825197
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 3, 9] = -0.005979351497
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 3, 12] = -0.009284873434
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 3, 17] = -0.005869671555
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 3, 18] = -0.016720559218
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 3, 21] = 0.060824208391
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 3, 26] = -0.010199723747
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 3, 27] = 0.008723989597
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 3, 30] = 0.015910679941
    M_ref[(DZP, LDA)][(1, 0, 1)][0, 3, 35] = -0.000067773086
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 0, 1] = 0.000405179359
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 0, 6] = 0.000788079835
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 0, 10] = -0.004989105668
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 0, 15] = -0.006443715183
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 0, 19] = 0.044979121396
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 0, 24] = -0.019773507878
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 0, 28] = -0.003160902612
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 0, 33] = -0.000840528351
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 0] = -0.004264471769
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 3] = -0.005852654546
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 7] = -0.003674021071
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 8] = -0.002123366528
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 9] = 0.012635509258
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 12] = 0.017738334818
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 16] = 0.005442661088
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 17] = 0.010485157927
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 18] = 0.082676058276
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 21] = -0.023454555459
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 25] = 0.030669636067
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 26] = -0.007017055245
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 27] = -0.042816444642
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 30] = 0.008063659545
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 34] = 0.005183105749
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 1, 35] = -0.006392208173
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 2, 4] = -0.003674021071
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 2, 13] = 0.005442661088
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 2, 22] = 0.030669636067
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 2, 31] = 0.005183105749
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 3, 1] = 0.000499927770
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 3, 6] = -0.004063635922
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 3, 10] = 0.000695936861
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 3, 15] = 0.007192883262
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 3, 19] = -0.037245639881
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 3, 24] = 0.046497527396
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 3, 28] = 0.015742186484
    M_ref[(DZP, LDA)][(1, 0, 1)][1, 3, 33] = -0.000302343584
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 0, 2] = 0.000405179359
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 0, 5] = 0.000788079835
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 0, 11] = -0.004989105668
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 0, 14] = -0.006443715183
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 0, 20] = 0.044979121396
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 0, 23] = -0.019773507878
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 0, 29] = -0.003160902612
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 0, 32] = -0.000840528351
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 1, 4] = -0.003674021071
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 1, 13] = 0.005442661088
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 1, 22] = 0.030669636067
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 1, 31] = 0.005183105749
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 0] = -0.004264471769
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 3] = -0.005852654546
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 7] = 0.003674021071
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 8] = -0.002123366528
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 9] = 0.012635509258
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 12] = 0.017738334818
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 16] = -0.005442661088
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 17] = 0.010485157927
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 18] = 0.082676058276
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 21] = -0.023454555459
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 25] = -0.030669636067
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 26] = -0.007017055245
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 27] = -0.042816444642
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 30] = 0.008063659545
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 34] = -0.005183105749
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 2, 35] = -0.006392208173
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 3, 2] = 0.000499927770
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 3, 5] = -0.004063635922
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 3, 11] = 0.000695936861
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 3, 14] = 0.007192883262
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 3, 20] = -0.037245639881
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 3, 23] = 0.046497527396
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 3, 29] = 0.015742186484
    M_ref[(DZP, LDA)][(1, 0, 1)][2, 3, 32] = -0.000302343584
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 0, 0] = -0.007292418538
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 0, 3] = -0.005438905853
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 0, 8] = -0.003410203602
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 0, 9] = -0.010897198592
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 0, 12] = -0.016564176621
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 0, 17] = -0.010754135290
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 0, 18] = 0.171506114135
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 0, 21] = -0.035858838375
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 0, 26] = -0.007717733510
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 0, 27] = -0.026005928837
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 0, 30] = -0.001594256868
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 0, 35] = 0.003085992373
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 1, 1] = -0.000319065239
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 1, 6] = -0.001063598595
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 1, 10] = -0.000876992353
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 1, 15] = 0.001020527474
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 1, 19] = 0.097563231347
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 1, 24] = -0.023508985804
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 1, 28] = 0.045877452299
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 1, 33] = -0.008544559144
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 2, 2] = -0.000319065239
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 2, 5] = -0.001063598595
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 2, 11] = -0.000876992353
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 2, 14] = 0.001020527474
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 2, 20] = 0.097563231347
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 2, 23] = -0.023508985804
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 2, 29] = 0.045877452299
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 2, 32] = -0.008544559144
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 3, 0] = 0.001015624647
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 3, 3] = 0.002886002773
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 3, 8] = 0.001772358945
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 3, 9] = -0.001314280154
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 3, 12] = -0.005088100364
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 3, 17] = -0.003739202682
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 3, 18] = -0.070387801867
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 3, 21] = 0.087465171570
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 3, 26] = -0.019840444424
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 3, 27] = 0.032884700622
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 3, 30] = 0.051860561793
    M_ref[(DZP, LDA)][(1, 0, 1)][3, 3, 35] = -0.012280017786
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 0, 0] = 0.001716335051
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 0, 3] = -0.001845395509
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 0, 8] = 0.000789848046
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 0, 9] = 0.037261452818
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 0, 12] = 0.046377795029
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 0, 17] = 0.023458859080
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 0, 18] = 0.122105582920
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 0, 21] = -0.026387344411
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 0, 26] = -0.000818672000
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 0, 27] = -0.043070241882
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 0, 30] = 0.006943043150
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 0, 35] = -0.002174315819
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 1, 1] = 0.000984291153
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 1, 6] = 0.002824610691
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 1, 10] = 0.035004194180
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 1, 15] = 0.032240676540
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 1, 19] = 0.106240633606
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 1, 24] = -0.026094309082
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 1, 28] = -0.007716827883
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 1, 33] = 0.003258414873
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 2, 2] = 0.000984291153
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 2, 5] = 0.002824610691
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 2, 11] = 0.035004194180
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 2, 14] = 0.032240676540
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 2, 20] = 0.106240633606
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 2, 23] = -0.026094309082
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 2, 29] = -0.007716827883
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 2, 32] = 0.003258414873
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 3, 0] = 0.026695836994
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 3, 3] = 0.019773451292
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 3, 8] = 0.008390863619
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 3, 9] = -0.083371630788
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 3, 12] = -0.068797646584
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 3, 17] = -0.025648970269
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 3, 18] = 0.003724969178
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 3, 21] = 0.070984466863
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 3, 26] = -0.041988959223
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 3, 27] = -0.000706736641
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 3, 30] = 0.009002585464
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 3, 35] = 0.011087470866
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 4, 4] = -0.009948835252
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 4, 13] = 0.016158020485
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 4, 22] = 0.150352312373
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 4, 31] = -0.019407499697
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 5, 2] = 0.003371954056
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 5, 5] = -0.004982501586
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 5, 11] = -0.010748950799
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 5, 14] = -0.001455721289
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 5, 20] = -0.067124064962
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 5, 23] = 0.137862486283
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 5, 29] = 0.033286423653
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 5, 32] = -0.016268769992
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 6, 1] = 0.003371954056
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 6, 6] = -0.004982501586
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 6, 10] = -0.010748950799
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 6, 15] = -0.001455721289
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 6, 19] = -0.067124064962
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 6, 24] = 0.137862486283
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 6, 28] = 0.033286423653
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 6, 33] = -0.016268769992
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 7, 7] = -0.009948835252
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 7, 16] = 0.016158020485
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 7, 25] = 0.150352312373
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 7, 34] = -0.019407499697
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 8, 0] = 0.000122235281
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 8, 3] = 0.005076980231
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 8, 8] = -0.000162041930
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 8, 9] = -0.011750237359
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 8, 12] = -0.023678484421
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 8, 17] = -0.015180949082
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 8, 18] = -0.009176685043
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 8, 21] = -0.054270692056
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 8, 26] = 0.113702658350
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 8, 27] = 0.007819508906
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 8, 30] = 0.026529505027
    M_ref[(DZP, LDA)][(1, 1, 0)][0, 8, 35] = -0.007717393728
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 0, 0] = -0.006744893274
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 0, 3] = -0.005668943816
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 0, 8] = -0.003161436476
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 0, 9] = -0.003280999024
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 0, 12] = -0.006170016806
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 0, 17] = -0.004309340747
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 0, 18] = 0.113229045609
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 0, 21] = -0.021209116350
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 0, 26] = -0.007598624580
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 0, 27] = -0.021473341965
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 0, 30] = 0.000987073759
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 0, 35] = -0.000684593615
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 1, 1] = -0.000525488424
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 1, 6] = -0.001422540481
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 1, 10] = 0.000246752737
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 1, 15] = 0.002449193894
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 1, 19] = 0.077726010705
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 1, 24] = -0.021732285010
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 1, 28] = 0.019727506010
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 1, 33] = -0.000755014400
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 2, 2] = -0.000525488424
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 2, 5] = -0.001422540481
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 2, 11] = 0.000246752737
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 2, 14] = 0.002449193894
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 2, 20] = 0.077726010705
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 2, 23] = -0.021732285010
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 2, 29] = 0.019727506010
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 2, 32] = -0.000755014400
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 3, 0] = 0.004124003376
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 3, 3] = 0.006216929406
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 3, 8] = 0.002996877945
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 3, 9] = -0.012405598901
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 3, 12] = -0.018037488313
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 3, 17] = -0.010892102742
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 3, 18] = -0.031766325139
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 3, 21] = 0.051476134079
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 3, 26] = -0.015702236395
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 3, 27] = 0.016001019824
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 3, 30] = 0.030698243909
    M_ref[(DZP, LDA)][(1, 1, 1)][0, 3, 35] = -0.001900274146

    msg = 'Too large error for M_{0} (value={1})'
    tol = 1e-7

    for key, ref in M_ref[(size, xc)].items():
        val = M[key][:, :, 0, :]
        M_diff = np.max(np.abs(val - ref))
        assert M_diff < tol, msg.format(key, str(val))

    return
