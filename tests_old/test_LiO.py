""" Tests for multipole-related functionality (and repulsion).

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
                                txt='-', **pp_setup[element])

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

    chgon1c = Onsite1cUTable(atom_O, basis='auxiliary')
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
    chgon1c = Onsite1cUTable(atom_Li, basis='main')
    chgon1c.run()
    return


@pytest.mark.parametrize('atoms', [DZP_LDA, DZP_PBE], indirect=True)
def test_on1cW(atoms):
    from hotcent.onsite_magnetization import Onsite1cWTable

    atom_Li, atom_O = atoms
    xc = atom_O.xcname

    magon1c = Onsite1cWTable(atom_O, basis='auxiliary')
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
    chgon1c = Onsite1cWTable(atom_Li, basis='main')
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

    chgon2c = Onsite2cUTable(atom_O, atom_Li, basis='auxiliary')
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

    magon2c = Onsite2cWTable(atom_O, atom_Li, basis='auxiliary')
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

    chgoff2c = Offsite2cUTable(atom_O, atom_Li, basis='auxiliary')
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

    magoff2c = Offsite2cWTable(atom_O, atom_Li, basis='auxiliary')
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
    from hotcent.offsite_chargetransfer import Offsite2cMTable

    atom_Li, atom_O = atoms
    size = atom_O.basis_size
    xc = atom_O.xcname
    rmin, dr, N = R, R, 2

    M_ref = {
        key1: {
            key2: np.zeros((9, 9, 36, 2))
            for key2 in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                         (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
        }
        for key1 in [(DZP, LDA)]
    }

    k = (DZP, LDA)
    M_ref[k][(0, 0, 0)][0, 0, 0, :] = [0.127291207746, 0.116886212436]
    M_ref[k][(0, 0, 0)][0, 0, 3, :] = [0.000224813780, 0.012427056550]
    M_ref[k][(0, 0, 0)][0, 0, 8, :] = [0.011729463349, 0.004296840428]
    M_ref[k][(0, 0, 0)][0, 0, 9, :] = [-0.053719803634, -0.046654414765]
    M_ref[k][(0, 0, 0)][0, 0, 12, :] = [0.004402480545, -0.003188514451]
    M_ref[k][(0, 0, 0)][0, 0, 17, :] = [-0.007158129423, -0.002882012275]
    M_ref[k][(0, 0, 0)][0, 0, 18, :] = [0.007321916195, 0.016292248912]
    M_ref[k][(0, 0, 0)][0, 0, 21, :] = [-0.003662900114, -0.013981311067]
    M_ref[k][(0, 0, 0)][0, 0, 26, :] = [0.002069892587, 0.010258785596]
    M_ref[k][(0, 0, 0)][0, 0, 27, :] = [0.026447803380, 0.020817074456]
    M_ref[k][(0, 0, 0)][0, 0, 30, :] = [-0.033810963202, -0.029033005424]
    M_ref[k][(0, 0, 0)][0, 0, 35, :] = [0.016204568638, 0.014927605449]
    M_ref[k][(0, 0, 0)][0, 1, 1, :] = [0.098765864796, 0.085534492610]
    M_ref[k][(0, 0, 0)][0, 1, 6, :] = [0.033375186201, 0.044388882335]
    M_ref[k][(0, 0, 0)][0, 1, 10, :] = [-0.039430745393, -0.031066976049]
    M_ref[k][(0, 0, 0)][0, 1, 15, :] = [-0.006041282237, -0.012449380831]
    M_ref[k][(0, 0, 0)][0, 1, 19, :] = [-0.009198056655, -0.005171776572]
    M_ref[k][(0, 0, 0)][0, 1, 24, :] = [0.002369022926, -0.001955119831]
    M_ref[k][(0, 0, 0)][0, 1, 28, :] = [0.023548840710, 0.021945937577]
    M_ref[k][(0, 0, 0)][0, 1, 33, :] = [-0.010823544592, -0.011735376302]
    M_ref[k][(0, 0, 0)][0, 2, 2, :] = [0.098765864796, 0.085534492610]
    M_ref[k][(0, 0, 0)][0, 2, 5, :] = [0.033375186201, 0.044388882335]
    M_ref[k][(0, 0, 0)][0, 2, 11, :] = [-0.039430745393, -0.031066976049]
    M_ref[k][(0, 0, 0)][0, 2, 14, :] = [-0.006041282237, -0.012449380831]
    M_ref[k][(0, 0, 0)][0, 2, 20, :] = [-0.009198056655, -0.005171776572]
    M_ref[k][(0, 0, 0)][0, 2, 23, :] = [0.002369022926, -0.001955119831]
    M_ref[k][(0, 0, 0)][0, 2, 29, :] = [0.023548840710, 0.021945937577]
    M_ref[k][(0, 0, 0)][0, 2, 32, :] = [-0.010823544592, -0.011735376302]
    M_ref[k][(0, 0, 0)][0, 3, 0, :] = [-0.220275203061, -0.212143504028]
    M_ref[k][(0, 0, 0)][0, 3, 3, :] = [-0.038403254903, -0.047939536543]
    M_ref[k][(0, 0, 0)][0, 3, 8, :] = [-0.003212836665, 0.002595897585]
    M_ref[k][(0, 0, 0)][0, 3, 9, :] = [0.078415649570, 0.072893915623]
    M_ref[k][(0, 0, 0)][0, 3, 12, :] = [0.002578174629, 0.008510679625]
    M_ref[k][(0, 0, 0)][0, 3, 17, :] = [0.001029841887, -0.002312023884]
    M_ref[k][(0, 0, 0)][0, 3, 18, :] = [-0.007115468011, -0.014125951313]
    M_ref[k][(0, 0, 0)][0, 3, 21, :] = [-0.000904325070, 0.007159706663]
    M_ref[k][(0, 0, 0)][0, 3, 26, :] = [0.000624760493, -0.005775013018]
    M_ref[k][(0, 0, 0)][0, 3, 27, :] = [-0.013976014520, -0.009575494233]
    M_ref[k][(0, 0, 0)][0, 3, 30, :] = [0.028586167812, 0.024852104174]
    M_ref[k][(0, 0, 0)][0, 3, 35, :] = [-0.015361112583, -0.014363141895]
    M_ref[k][(0, 0, 0)][1, 0, 1, :] = [0.145453348332, 0.104458013201]
    M_ref[k][(0, 0, 0)][1, 0, 6, :] = [-0.026849160750, 0.007275052347]
    M_ref[k][(0, 0, 0)][1, 0, 10, :] = [-0.042839300155, -0.016925470330]
    M_ref[k][(0, 0, 0)][1, 0, 15, :] = [0.021141889277, 0.001287399077]
    M_ref[k][(0, 0, 0)][1, 0, 19, :] = [0.000398857852, 0.012873656308]
    M_ref[k][(0, 0, 0)][1, 0, 24, :] = [-0.003260653770, -0.016658333089]
    M_ref[k][(0, 0, 0)][1, 0, 28, :] = [0.027902606838, 0.022936262442]
    M_ref[k][(0, 0, 0)][1, 0, 33, :] = [-0.017594961463, -0.020420129255]
    M_ref[k][(0, 0, 0)][1, 1, 0, :] = [0.043312753343, 0.038048435461]
    M_ref[k][(0, 0, 0)][1, 1, 3, :] = [0.011615391467, 0.017789011243]
    M_ref[k][(0, 0, 0)][1, 1, 7, :] = [0.082042556545, 0.058882059919]
    M_ref[k][(0, 0, 0)][1, 1, 8, :] = [0.002514467159, -0.001246004479]
    M_ref[k][(0, 0, 0)][1, 1, 9, :] = [-0.026914806412, -0.023340133685]
    M_ref[k][(0, 0, 0)][1, 1, 12, :] = [-0.003889679383, -0.007730278009]
    M_ref[k][(0, 0, 0)][1, 1, 16, :] = [-0.025172109001, -0.011879841799]
    M_ref[k][(0, 0, 0)][1, 1, 17, :] = [-0.008367055625, -0.006203590933]
    M_ref[k][(0, 0, 0)][1, 1, 18, :] = [0.007448948770, 0.011987411453]
    M_ref[k][(0, 0, 0)][1, 1, 21, :] = [0.002432463441, -0.002788047830]
    M_ref[k][(0, 0, 0)][1, 1, 25, :] = [-0.008088405596, -0.007995552957]
    M_ref[k][(0, 0, 0)][1, 1, 26, :] = [-0.000775362321, 0.003367737662]
    M_ref[k][(0, 0, 0)][1, 1, 27, :] = [0.065511902834, 0.062663083966]
    M_ref[k][(0, 0, 0)][1, 1, 30, :] = [-0.079139573371, -0.076722206730]
    M_ref[k][(0, 0, 0)][1, 1, 34, :] = [0.025229318092, 0.030464229281]
    M_ref[k][(0, 0, 0)][1, 1, 35, :] = [0.033888880952, 0.033242812410]
    M_ref[k][(0, 0, 0)][1, 2, 4, :] = [0.082042627411, 0.058882223478]
    M_ref[k][(0, 0, 0)][1, 2, 13, :] = [-0.025172175898, -0.011879961395]
    M_ref[k][(0, 0, 0)][1, 2, 22, :] = [-0.008088460083, -0.007995592685]
    M_ref[k][(0, 0, 0)][1, 2, 31, :] = [0.025229418446, 0.030464281718]
    M_ref[k][(0, 0, 0)][1, 3, 1, :] = [-0.178325181664, -0.149657894192]
    M_ref[k][(0, 0, 0)][1, 3, 6, :] = [-0.006245670111, -0.030108107715]
    M_ref[k][(0, 0, 0)][1, 3, 10, :] = [0.017445168313, -0.000675898554]
    M_ref[k][(0, 0, 0)][1, 3, 15, :] = [-0.017269424761, -0.003385543184]
    M_ref[k][(0, 0, 0)][1, 3, 19, :] = [-0.002339423418, -0.011062821638]
    M_ref[k][(0, 0, 0)][1, 3, 24, :] = [-0.000962220219, 0.008406531696]
    M_ref[k][(0, 0, 0)][1, 3, 28, :] = [-0.012006930215, -0.008534056484]
    M_ref[k][(0, 0, 0)][1, 3, 33, :] = [0.017595337647, 0.019570925772]
    M_ref[k][(0, 0, 0)][2, 0, 2, :] = [0.145453348332, 0.104458013201]
    M_ref[k][(0, 0, 0)][2, 0, 5, :] = [-0.026849160750, 0.007275052347]
    M_ref[k][(0, 0, 0)][2, 0, 11, :] = [-0.042839300155, -0.016925470330]
    M_ref[k][(0, 0, 0)][2, 0, 14, :] = [0.021141889277, 0.001287399077]
    M_ref[k][(0, 0, 0)][2, 0, 20, :] = [0.000398857852, 0.012873656308]
    M_ref[k][(0, 0, 0)][2, 0, 23, :] = [-0.003260653770, -0.016658333089]
    M_ref[k][(0, 0, 0)][2, 0, 29, :] = [0.027902606838, 0.022936262442]
    M_ref[k][(0, 0, 0)][2, 0, 32, :] = [-0.017594961463, -0.020420129255]
    M_ref[k][(0, 0, 0)][2, 1, 4, :] = [0.082042627411, 0.058882223478]
    M_ref[k][(0, 0, 0)][2, 1, 13, :] = [-0.025172175898, -0.011879961395]
    M_ref[k][(0, 0, 0)][2, 1, 22, :] = [-0.008088460083, -0.007995592685]
    M_ref[k][(0, 0, 0)][2, 1, 31, :] = [0.025229418446, 0.030464281718]
    M_ref[k][(0, 0, 0)][2, 2, 0, :] = [0.043312753343, 0.038048435461]
    M_ref[k][(0, 0, 0)][2, 2, 3, :] = [0.011615391467, 0.017789011243]
    M_ref[k][(0, 0, 0)][2, 2, 7, :] = [-0.082042556545, -0.058882059919]
    M_ref[k][(0, 0, 0)][2, 2, 8, :] = [0.002514467159, -0.001246004479]
    M_ref[k][(0, 0, 0)][2, 2, 9, :] = [-0.026914806412, -0.023340133685]
    M_ref[k][(0, 0, 0)][2, 2, 12, :] = [-0.003889679383, -0.007730278009]
    M_ref[k][(0, 0, 0)][2, 2, 16, :] = [0.025172109001, 0.011879841799]
    M_ref[k][(0, 0, 0)][2, 2, 17, :] = [-0.008367055625, -0.006203590933]
    M_ref[k][(0, 0, 0)][2, 2, 18, :] = [0.007448948770, 0.011987411453]
    M_ref[k][(0, 0, 0)][2, 2, 21, :] = [0.002432463441, -0.002788047830]
    M_ref[k][(0, 0, 0)][2, 2, 25, :] = [0.008088405596, 0.007995552957]
    M_ref[k][(0, 0, 0)][2, 2, 26, :] = [-0.000775362321, 0.003367737662]
    M_ref[k][(0, 0, 0)][2, 2, 27, :] = [0.065511902834, 0.062663083966]
    M_ref[k][(0, 0, 0)][2, 2, 30, :] = [-0.079139573371, -0.076722206730]
    M_ref[k][(0, 0, 0)][2, 2, 34, :] = [-0.025229318092, -0.030464229281]
    M_ref[k][(0, 0, 0)][2, 2, 35, :] = [0.033888880952, 0.033242812410]
    M_ref[k][(0, 0, 0)][2, 3, 2, :] = [-0.178325181664, -0.149657894192]
    M_ref[k][(0, 0, 0)][2, 3, 5, :] = [-0.006245670111, -0.030108107715]
    M_ref[k][(0, 0, 0)][2, 3, 11, :] = [0.017445168313, -0.000675898554]
    M_ref[k][(0, 0, 0)][2, 3, 14, :] = [-0.017269424761, -0.003385543184]
    M_ref[k][(0, 0, 0)][2, 3, 20, :] = [-0.002339423418, -0.011062821638]
    M_ref[k][(0, 0, 0)][2, 3, 23, :] = [-0.000962220219, 0.008406531696]
    M_ref[k][(0, 0, 0)][2, 3, 29, :] = [-0.012006930215, -0.008534056484]
    M_ref[k][(0, 0, 0)][2, 3, 32, :] = [0.017595337647, 0.019570925772]
    M_ref[k][(0, 0, 0)][3, 0, 0, :] = [-0.035638735314, -0.016065647804]
    M_ref[k][(0, 0, 0)][3, 0, 3, :] = [0.107139488911, 0.084185555878]
    M_ref[k][(0, 0, 0)][3, 0, 8, :] = [0.014227156946, 0.028208843100]
    M_ref[k][(0, 0, 0)][3, 0, 9, :] = [0.020714998420, 0.007424125095]
    M_ref[k][(0, 0, 0)][3, 0, 12, :] = [-0.021581651754, -0.007302048073]
    M_ref[k][(0, 0, 0)][3, 0, 17, :] = [0.000100571369, -0.007943335682]
    M_ref[k][(0, 0, 0)][3, 0, 18, :] = [-0.020637555765, -0.037511864885]
    M_ref[k][(0, 0, 0)][3, 0, 21, :] = [0.015786396912, 0.035196608500]
    M_ref[k][(0, 0, 0)][3, 0, 26, :] = [-0.009465245436, -0.024869569484]
    M_ref[k][(0, 0, 0)][3, 0, 27, :] = [0.066860976386, 0.077453076304]
    M_ref[k][(0, 0, 0)][3, 0, 30, :] = [-0.055095837931, -0.064083769489]
    M_ref[k][(0, 0, 0)][3, 0, 35, :] = [0.026824077529, 0.029226203766]
    M_ref[k][(0, 0, 0)][3, 1, 1, :] = [0.043997900883, 0.033562796673]
    M_ref[k][(0, 0, 0)][3, 1, 6, :] = [0.061957175129, 0.070643278526]
    M_ref[k][(0, 0, 0)][3, 1, 10, :] = [-0.023106751107, -0.016510549032]
    M_ref[k][(0, 0, 0)][3, 1, 15, :] = [-0.010893726068, -0.015947561428]
    M_ref[k][(0, 0, 0)][3, 1, 19, :] = [-0.027295835964, -0.024120454632]
    M_ref[k][(0, 0, 0)][3, 1, 24, :] = [0.003722463169, 0.000312168329]
    M_ref[k][(0, 0, 0)][3, 1, 28, :] = [0.062913139818, 0.061648988149]
    M_ref[k][(0, 0, 0)][3, 1, 33, :] = [-0.015313162368, -0.016032291026]
    M_ref[k][(0, 0, 0)][3, 2, 2, :] = [0.043997900883, 0.033562796673]
    M_ref[k][(0, 0, 0)][3, 2, 5, :] = [0.061957175129, 0.070643278526]
    M_ref[k][(0, 0, 0)][3, 2, 11, :] = [-0.023106751107, -0.016510549032]
    M_ref[k][(0, 0, 0)][3, 2, 14, :] = [-0.010893726068, -0.015947561428]
    M_ref[k][(0, 0, 0)][3, 2, 20, :] = [-0.027295835964, -0.024120454632]
    M_ref[k][(0, 0, 0)][3, 2, 23, :] = [0.003722463169, 0.000312168329]
    M_ref[k][(0, 0, 0)][3, 2, 29, :] = [0.062913139818, 0.061648988149]
    M_ref[k][(0, 0, 0)][3, 2, 32, :] = [-0.015313162368, -0.016032291026]
    M_ref[k][(0, 0, 0)][3, 3, 0, :] = [-0.039544407064, -0.048388662010]
    M_ref[k][(0, 0, 0)][3, 3, 3, :] = [-0.133650996302, -0.123279079472]
    M_ref[k][(0, 0, 0)][3, 3, 8, :] = [-0.030074144669, -0.036391880532]
    M_ref[k][(0, 0, 0)][3, 3, 9, :] = [0.019024619606, 0.025030206210]
    M_ref[k][(0, 0, 0)][3, 3, 12, :] = [-0.010210330803, -0.016662683066]
    M_ref[k][(0, 0, 0)][3, 3, 17, :] = [-0.007719631345, -0.004084928098]
    M_ref[k][(0, 0, 0)][3, 3, 18, :] = [0.000639355009, 0.008264145519]
    M_ref[k][(0, 0, 0)][3, 3, 21, :] = [-0.019415363297, -0.028186021466]
    M_ref[k][(0, 0, 0)][3, 3, 26, :] = [0.008782436457, 0.015743002546]
    M_ref[k][(0, 0, 0)][3, 3, 27, :] = [-0.022980390051, -0.027766514471]
    M_ref[k][(0, 0, 0)][3, 3, 30, :] = [0.056431921777, 0.060493189990]
    M_ref[k][(0, 0, 0)][3, 3, 35, :] = [-0.027428475523, -0.028513895330]
    M_ref[k][(0, 0, 0)][4, 0, 4, :] = [0.127723830867, 0.141088287484]
    M_ref[k][(0, 0, 0)][4, 0, 13, :] = [-0.016691696152, -0.024361822485]
    M_ref[k][(0, 0, 0)][4, 0, 22, :] = [-0.004900117981, -0.004953706093]
    M_ref[k][(0, 0, 0)][4, 0, 31, :] = [0.011549059259, 0.008528338926]
    M_ref[k][(0, 0, 0)][4, 1, 2, :] = [0.102112270499, 0.096892769341]
    M_ref[k][(0, 0, 0)][4, 1, 5, :] = [0.030724750478, 0.035069424554]
    M_ref[k][(0, 0, 0)][4, 1, 11, :] = [-0.048770595902, -0.045471262836]
    M_ref[k][(0, 0, 0)][4, 1, 14, :] = [-0.009597976785, -0.012125838412]
    M_ref[k][(0, 0, 0)][4, 1, 20, :] = [-0.001868459122, -0.000280175363]
    M_ref[k][(0, 0, 0)][4, 1, 23, :] = [0.003062795508, 0.001357011119]
    M_ref[k][(0, 0, 0)][4, 1, 29, :] = [0.007294383550, 0.006662071599]
    M_ref[k][(0, 0, 0)][4, 1, 32, :] = [-0.009857127678, -0.010216826324]
    M_ref[k][(0, 0, 0)][4, 2, 1, :] = [0.102112270499, 0.096892769341]
    M_ref[k][(0, 0, 0)][4, 2, 6, :] = [0.030724750478, 0.035069424554]
    M_ref[k][(0, 0, 0)][4, 2, 10, :] = [-0.048770595902, -0.045471262836]
    M_ref[k][(0, 0, 0)][4, 2, 15, :] = [-0.009597976785, -0.012125838412]
    M_ref[k][(0, 0, 0)][4, 2, 19, :] = [-0.001868459122, -0.000280175363]
    M_ref[k][(0, 0, 0)][4, 2, 24, :] = [0.003062795508, 0.001357011119]
    M_ref[k][(0, 0, 0)][4, 2, 28, :] = [0.007294383550, 0.006662071599]
    M_ref[k][(0, 0, 0)][4, 2, 33, :] = [-0.009857127678, -0.010216826324]
    M_ref[k][(0, 0, 0)][4, 3, 4, :] = [-0.148003794840, -0.190045739946]
    M_ref[k][(0, 0, 0)][4, 3, 13, :] = [-0.013714923985, 0.010413780942]
    M_ref[k][(0, 0, 0)][4, 3, 22, :] = [0.009523227410, 0.009691805042]
    M_ref[k][(0, 0, 0)][4, 3, 31, :] = [-0.021014202178, -0.011511611682]
    M_ref[k][(0, 0, 0)][5, 0, 2, :] = [0.045973372512, 0.034352944141]
    M_ref[k][(0, 0, 0)][5, 0, 5, :] = [0.124434058420, 0.134106816863]
    M_ref[k][(0, 0, 0)][5, 0, 11, :] = [-0.024592991179, -0.017247526083]
    M_ref[k][(0, 0, 0)][5, 0, 14, :] = [-0.016558632225, -0.022186533070]
    M_ref[k][(0, 0, 0)][5, 0, 20, :] = [-0.005537167881, -0.002001094778]
    M_ref[k][(0, 0, 0)][5, 0, 23, :] = [0.000994207255, -0.002803463196]
    M_ref[k][(0, 0, 0)][5, 0, 29, :] = [0.007820214806, 0.006412468068]
    M_ref[k][(0, 0, 0)][5, 0, 32, :] = [-0.000109204920, -0.000910019448]
    M_ref[k][(0, 0, 0)][5, 1, 4, :] = [0.012927477535, 0.032107117464]
    M_ref[k][(0, 0, 0)][5, 1, 13, :] = [0.000368468623, -0.010639106580]
    M_ref[k][(0, 0, 0)][5, 1, 22, :] = [-0.015923878210, -0.016000783745]
    M_ref[k][(0, 0, 0)][5, 1, 31, :] = [0.036309795569, 0.031974690149]
    M_ref[k][(0, 0, 0)][5, 2, 0, :] = [0.024659182741, 0.019129717610]
    M_ref[k][(0, 0, 0)][5, 2, 3, :] = [0.084658250209, 0.091142815929]
    M_ref[k][(0, 0, 0)][5, 2, 7, :] = [-0.012927475172, -0.032107121819]
    M_ref[k][(0, 0, 0)][5, 2, 8, :] = [0.056333469691, 0.052383594843]
    M_ref[k][(0, 0, 0)][5, 2, 9, :] = [-0.024292276347, -0.020537558520]
    M_ref[k][(0, 0, 0)][5, 2, 12, :] = [-0.038314477076, -0.042348514659]
    M_ref[k][(0, 0, 0)][5, 2, 16, :] = [-0.000368474173, 0.010639104473]
    M_ref[k][(0, 0, 0)][5, 2, 17, :] = [-0.022302530529, -0.020030098879]
    M_ref[k][(0, 0, 0)][5, 2, 18, :] = [-0.021719472421, -0.016952421550]
    M_ref[k][(0, 0, 0)][5, 2, 21, :] = [0.028983771386, 0.023500319269]
    M_ref[k][(0, 0, 0)][5, 2, 25, :] = [0.015923876170, 0.016000769202]
    M_ref[k][(0, 0, 0)][5, 2, 26, :] = [-0.013252624079, -0.008900849090]
    M_ref[k][(0, 0, 0)][5, 2, 27, :] = [0.073309752619, 0.070317447643]
    M_ref[k][(0, 0, 0)][5, 2, 30, :] = [-0.092283157927, -0.089744036078]
    M_ref[k][(0, 0, 0)][5, 2, 34, :] = [-0.036309791937, -0.031974662668]
    M_ref[k][(0, 0, 0)][5, 2, 35, :] = [0.047954648593, 0.047276039597]
    M_ref[k][(0, 0, 0)][5, 3, 2, :] = [-0.112131684689, -0.089920460338]
    M_ref[k][(0, 0, 0)][5, 3, 5, :] = [-0.171759438745, -0.190247897260]
    M_ref[k][(0, 0, 0)][5, 3, 11, :] = [0.054422530411, 0.040382447894]
    M_ref[k][(0, 0, 0)][5, 3, 14, :] = [-0.000414383049, 0.010342756386]
    M_ref[k][(0, 0, 0)][5, 3, 20, :] = [0.002994143886, -0.003764687243]
    M_ref[k][(0, 0, 0)][5, 3, 23, :] = [0.003691317403, 0.010950164298]
    M_ref[k][(0, 0, 0)][5, 3, 29, :] = [0.005171795658, 0.007862555209]
    M_ref[k][(0, 0, 0)][5, 3, 32, :] = [-0.016791014649, -0.015260342062]
    M_ref[k][(0, 0, 0)][6, 0, 1, :] = [0.045973372512, 0.034352944141]
    M_ref[k][(0, 0, 0)][6, 0, 6, :] = [0.124434058420, 0.134106816863]
    M_ref[k][(0, 0, 0)][6, 0, 10, :] = [-0.024592991179, -0.017247526083]
    M_ref[k][(0, 0, 0)][6, 0, 15, :] = [-0.016558632225, -0.022186533070]
    M_ref[k][(0, 0, 0)][6, 0, 19, :] = [-0.005537167881, -0.002001094778]
    M_ref[k][(0, 0, 0)][6, 0, 24, :] = [0.000994207255, -0.002803463196]
    M_ref[k][(0, 0, 0)][6, 0, 28, :] = [0.007820214806, 0.006412468068]
    M_ref[k][(0, 0, 0)][6, 0, 33, :] = [-0.000109204920, -0.000910019448]
    M_ref[k][(0, 0, 0)][6, 1, 0, :] = [0.024659182741, 0.019129717610]
    M_ref[k][(0, 0, 0)][6, 1, 3, :] = [0.084658250209, 0.091142815929]
    M_ref[k][(0, 0, 0)][6, 1, 7, :] = [0.012927475172, 0.032107121819]
    M_ref[k][(0, 0, 0)][6, 1, 8, :] = [0.056333469691, 0.052383594843]
    M_ref[k][(0, 0, 0)][6, 1, 9, :] = [-0.024292276347, -0.020537558520]
    M_ref[k][(0, 0, 0)][6, 1, 12, :] = [-0.038314477076, -0.042348514659]
    M_ref[k][(0, 0, 0)][6, 1, 16, :] = [0.000368474173, -0.010639104473]
    M_ref[k][(0, 0, 0)][6, 1, 17, :] = [-0.022302530529, -0.020030098879]
    M_ref[k][(0, 0, 0)][6, 1, 18, :] = [-0.021719472421, -0.016952421550]
    M_ref[k][(0, 0, 0)][6, 1, 21, :] = [0.028983771386, 0.023500319269]
    M_ref[k][(0, 0, 0)][6, 1, 25, :] = [-0.015923876170, -0.016000769202]
    M_ref[k][(0, 0, 0)][6, 1, 26, :] = [-0.013252624079, -0.008900849090]
    M_ref[k][(0, 0, 0)][6, 1, 27, :] = [0.073309752619, 0.070317447643]
    M_ref[k][(0, 0, 0)][6, 1, 30, :] = [-0.092283157927, -0.089744036078]
    M_ref[k][(0, 0, 0)][6, 1, 34, :] = [0.036309791937, 0.031974662668]
    M_ref[k][(0, 0, 0)][6, 1, 35, :] = [0.047954648593, 0.047276039597]
    M_ref[k][(0, 0, 0)][6, 2, 4, :] = [0.012927477535, 0.032107117464]
    M_ref[k][(0, 0, 0)][6, 2, 13, :] = [0.000368468623, -0.010639106580]
    M_ref[k][(0, 0, 0)][6, 2, 22, :] = [-0.015923878210, -0.016000783745]
    M_ref[k][(0, 0, 0)][6, 2, 31, :] = [0.036309795569, 0.031974690149]
    M_ref[k][(0, 0, 0)][6, 3, 1, :] = [-0.112131684689, -0.089920460338]
    M_ref[k][(0, 0, 0)][6, 3, 6, :] = [-0.171759438745, -0.190247897260]
    M_ref[k][(0, 0, 0)][6, 3, 10, :] = [0.054422530411, 0.040382447894]
    M_ref[k][(0, 0, 0)][6, 3, 15, :] = [-0.000414383049, 0.010342756386]
    M_ref[k][(0, 0, 0)][6, 3, 19, :] = [0.002994143886, -0.003764687243]
    M_ref[k][(0, 0, 0)][6, 3, 24, :] = [0.003691317403, 0.010950164298]
    M_ref[k][(0, 0, 0)][6, 3, 28, :] = [0.005171795658, 0.007862555209]
    M_ref[k][(0, 0, 0)][6, 3, 33, :] = [-0.016791014649, -0.015260342062]
    M_ref[k][(0, 0, 0)][7, 0, 7, :] = [0.127723864728, 0.141088467151]
    M_ref[k][(0, 0, 0)][7, 0, 16, :] = [-0.016691773864, -0.024361983589]
    M_ref[k][(0, 0, 0)][7, 0, 25, :] = [-0.004900087127, -0.004953667092]
    M_ref[k][(0, 0, 0)][7, 0, 34, :] = [0.011549004439, 0.008528235589]
    M_ref[k][(0, 0, 0)][7, 1, 1, :] = [0.102112318530, 0.096892841021]
    M_ref[k][(0, 0, 0)][7, 1, 6, :] = [0.030724779635, 0.035069434025]
    M_ref[k][(0, 0, 0)][7, 1, 10, :] = [-0.048770638260, -0.045471320143]
    M_ref[k][(0, 0, 0)][7, 1, 15, :] = [-0.009597988937, -0.012125839111]
    M_ref[k][(0, 0, 0)][7, 1, 19, :] = [-0.001868452798, -0.000280176235]
    M_ref[k][(0, 0, 0)][7, 1, 24, :] = [0.003062789637, 0.001357012977]
    M_ref[k][(0, 0, 0)][7, 1, 28, :] = [0.007294366605, 0.006662057519]
    M_ref[k][(0, 0, 0)][7, 1, 33, :] = [-0.009857109833, -0.010216806850]
    M_ref[k][(0, 0, 0)][7, 2, 2, :] = [-0.102112318530, -0.096892841021]
    M_ref[k][(0, 0, 0)][7, 2, 5, :] = [-0.030724779635, -0.035069434025]
    M_ref[k][(0, 0, 0)][7, 2, 11, :] = [0.048770638260, 0.045471320143]
    M_ref[k][(0, 0, 0)][7, 2, 14, :] = [0.009597988937, 0.012125839111]
    M_ref[k][(0, 0, 0)][7, 2, 20, :] = [0.001868452798, 0.000280176235]
    M_ref[k][(0, 0, 0)][7, 2, 23, :] = [-0.003062789637, -0.001357012977]
    M_ref[k][(0, 0, 0)][7, 2, 29, :] = [-0.007294366605, -0.006662057519]
    M_ref[k][(0, 0, 0)][7, 2, 32, :] = [0.009857109833, 0.010216806850]
    M_ref[k][(0, 0, 0)][7, 3, 7, :] = [-0.148003856872, -0.190046026743]
    M_ref[k][(0, 0, 0)][7, 3, 16, :] = [-0.013714782743, 0.010414050274]
    M_ref[k][(0, 0, 0)][7, 3, 25, :] = [0.009523222553, 0.009691773620]
    M_ref[k][(0, 0, 0)][7, 3, 34, :] = [-0.021014193571, -0.011511503323]
    M_ref[k][(0, 0, 0)][8, 0, 0, :] = [-0.008217981712, -0.000788690567]
    M_ref[k][(0, 0, 0)][8, 0, 3, :] = [0.042793759114, 0.034081211779]
    M_ref[k][(0, 0, 0)][8, 0, 8, :] = [0.118506388852, 0.123813370547]
    M_ref[k][(0, 0, 0)][8, 0, 9, :] = [0.006720330263, 0.001675558075]
    M_ref[k][(0, 0, 0)][8, 0, 12, :] = [-0.022677330305, -0.017257269053]
    M_ref[k][(0, 0, 0)][8, 0, 17, :] = [-0.015562537193, -0.018615735999]
    M_ref[k][(0, 0, 0)][8, 0, 18, :] = [0.005441629460, -0.000963295427]
    M_ref[k][(0, 0, 0)][8, 0, 21, :] = [-0.009494165386, -0.002126696508]
    M_ref[k][(0, 0, 0)][8, 0, 26, :] = [0.004872249764, -0.000974717822]
    M_ref[k][(0, 0, 0)][8, 0, 27, :] = [-0.010041600455, -0.006021192633]
    M_ref[k][(0, 0, 0)][8, 0, 30, :] = [0.017008338642, 0.013596819622]
    M_ref[k][(0, 0, 0)][8, 0, 35, :] = [-0.009207945866, -0.008296178871]
    M_ref[k][(0, 0, 0)][8, 1, 1, :] = [-0.020380333713, -0.030205729267]
    M_ref[k][(0, 0, 0)][8, 1, 6, :] = [0.026291083922, 0.034469670353]
    M_ref[k][(0, 0, 0)][8, 1, 10, :] = [0.005730437051, 0.011941232195]
    M_ref[k][(0, 0, 0)][8, 1, 15, :] = [-0.009018199601, -0.013776746375]
    M_ref[k][(0, 0, 0)][8, 1, 19, :] = [-0.011769423841, -0.008779575633]
    M_ref[k][(0, 0, 0)][8, 1, 24, :] = [-0.006331765554, -0.009542801600]
    M_ref[k][(0, 0, 0)][8, 1, 28, :] = [0.009932603439, 0.008742314398]
    M_ref[k][(0, 0, 0)][8, 1, 33, :] = [0.022857720059, 0.022180609091]
    M_ref[k][(0, 0, 0)][8, 2, 2, :] = [-0.020380333713, -0.030205729267]
    M_ref[k][(0, 0, 0)][8, 2, 5, :] = [0.026291083922, 0.034469670353]
    M_ref[k][(0, 0, 0)][8, 2, 11, :] = [0.005730437051, 0.011941232195]
    M_ref[k][(0, 0, 0)][8, 2, 14, :] = [-0.009018199601, -0.013776746375]
    M_ref[k][(0, 0, 0)][8, 2, 20, :] = [-0.011769423841, -0.008779575633]
    M_ref[k][(0, 0, 0)][8, 2, 23, :] = [-0.006331765554, -0.009542801600]
    M_ref[k][(0, 0, 0)][8, 2, 29, :] = [0.009932603439, 0.008742314398]
    M_ref[k][(0, 0, 0)][8, 2, 32, :] = [0.022857720059, 0.022180609091]
    M_ref[k][(0, 0, 0)][8, 3, 0, :] = [-0.017253659389, -0.029295220245]
    M_ref[k][(0, 0, 0)][8, 3, 3, :] = [-0.111794616089, -0.097673124967]
    M_ref[k][(0, 0, 0)][8, 3, 8, :] = [-0.151662783287, -0.160264457632]
    M_ref[k][(0, 0, 0)][8, 3, 9, :] = [0.006022765057, 0.014199444382]
    M_ref[k][(0, 0, 0)][8, 3, 12, :] = [0.051888689046, 0.043103732833]
    M_ref[k][(0, 0, 0)][8, 3, 17, :] = [-0.007909028552, -0.002960335800]
    M_ref[k][(0, 0, 0)][8, 3, 18, :] = [-0.030538133794, -0.020156888623]
    M_ref[k][(0, 0, 0)][8, 3, 21, :] = [0.029960208473, 0.018018850526]
    M_ref[k][(0, 0, 0)][8, 3, 26, :] = [-0.014672728097, -0.005195832570]
    M_ref[k][(0, 0, 0)][8, 3, 27, :] = [0.050425005354, 0.043908638649]
    M_ref[k][(0, 0, 0)][8, 3, 30, :] = [-0.052596175436, -0.047066709287]
    M_ref[k][(0, 0, 0)][8, 3, 35, :] = [0.027895283306, 0.026417471016]
    M_ref[k][(0, 0, 1)][0, 0, 0, :] = [0.127077223203, 0.122039872250]
    M_ref[k][(0, 0, 1)][0, 0, 3, :] = [0.020465811709, 0.026373260718]
    M_ref[k][(0, 0, 1)][0, 0, 8, :] = [0.002801862981, -0.000796478876]
    M_ref[k][(0, 0, 1)][0, 0, 9, :] = [-0.046455122723, -0.043034569202]
    M_ref[k][(0, 0, 1)][0, 0, 12, :] = [-0.003261721785, -0.006936736019]
    M_ref[k][(0, 0, 1)][0, 0, 17, :] = [-0.004254077497, -0.002183888886]
    M_ref[k][(0, 0, 1)][0, 0, 18, :] = [-0.002628189648, 0.001714600782]
    M_ref[k][(0, 0, 1)][0, 0, 21, :] = [0.006856165311, 0.001860732282]
    M_ref[k][(0, 0, 1)][0, 0, 26, :] = [-0.003193729058, 0.000770744425]
    M_ref[k][(0, 0, 1)][0, 0, 27, :] = [0.040024193511, 0.037298199230]
    M_ref[k][(0, 0, 1)][0, 0, 30, :] = [-0.048755649472, -0.046442505694]
    M_ref[k][(0, 0, 1)][0, 0, 35, :] = [0.024140022841, 0.023521809037]
    M_ref[k][(0, 0, 1)][1, 0, 1, :] = [0.127149317219, 0.106247427482]
    M_ref[k][(0, 0, 1)][1, 0, 6, :] = [0.008705980329, 0.026104558416]
    M_ref[k][(0, 0, 1)][1, 0, 10, :] = [-0.020933590520, -0.007721160083]
    M_ref[k][(0, 0, 1)][1, 0, 15, :] = [0.006859163445, -0.003263850884]
    M_ref[k][(0, 0, 1)][1, 0, 19, :] = [-0.005377111328, 0.000983291898]
    M_ref[k][(0, 0, 1)][1, 0, 24, :] = [0.004006943881, -0.002823999566]
    M_ref[k][(0, 0, 1)][1, 0, 28, :] = [0.037541464989, 0.035009323646]
    M_ref[k][(0, 0, 1)][1, 0, 33, :] = [-0.030805379731, -0.032245820345]
    M_ref[k][(0, 0, 1)][2, 0, 2, :] = [0.127149317219, 0.106247427482]
    M_ref[k][(0, 0, 1)][2, 0, 5, :] = [0.008705980329, 0.026104558416]
    M_ref[k][(0, 0, 1)][2, 0, 11, :] = [-0.020933590520, -0.007721160083]
    M_ref[k][(0, 0, 1)][2, 0, 14, :] = [0.006859163445, -0.003263850884]
    M_ref[k][(0, 0, 1)][2, 0, 20, :] = [-0.005377111328, 0.000983291898]
    M_ref[k][(0, 0, 1)][2, 0, 23, :] = [0.004006943881, -0.002823999566]
    M_ref[k][(0, 0, 1)][2, 0, 29, :] = [0.037541464989, 0.035009323646]
    M_ref[k][(0, 0, 1)][2, 0, 32, :] = [-0.030805379732, -0.032245820345]
    M_ref[k][(0, 0, 1)][3, 0, 0, :] = [-0.007094638597, -0.003703966694]
    M_ref[k][(0, 0, 1)][3, 0, 3, :] = [0.074972948193, 0.070996607937]
    M_ref[k][(0, 0, 1)][3, 0, 8, :] = [0.039568207303, 0.041990273335]
    M_ref[k][(0, 0, 1)][3, 0, 9, :] = [0.002996194886, 0.000693799276]
    M_ref[k][(0, 0, 1)][3, 0, 12, :] = [0.006519891094, 0.008993565774]
    M_ref[k][(0, 0, 1)][3, 0, 17, :] = [-0.009693717291, -0.011087173976]
    M_ref[k][(0, 0, 1)][3, 0, 18, :] = [-0.023774674729, -0.026697833673]
    M_ref[k][(0, 0, 1)][3, 0, 21, :] = [0.016412189090, 0.019774645781]
    M_ref[k][(0, 0, 1)][3, 0, 26, :] = [-0.005719837580, -0.008388349060]
    M_ref[k][(0, 0, 1)][3, 0, 27, :] = [0.081538280613, 0.083373164127]
    M_ref[k][(0, 0, 1)][3, 0, 30, :] = [-0.067239203588, -0.068796194890]
    M_ref[k][(0, 0, 1)][3, 0, 35, :] = [0.025222951433, 0.025639074946]
    M_ref[k][(0, 0, 1)][4, 0, 4, :] = [0.121030080822, 0.150359902762]
    M_ref[k][(0, 0, 1)][4, 0, 13, :] = [-0.002578230430, -0.019411196111]
    M_ref[k][(0, 0, 1)][4, 0, 22, :] = [-0.009833838166, -0.009951443380]
    M_ref[k][(0, 0, 1)][4, 0, 31, :] = [0.022792563169, 0.016163248409]
    M_ref[k][(0, 0, 1)][5, 0, 2, :] = [0.087850270321, 0.067142177357]
    M_ref[k][(0, 0, 1)][5, 0, 5, :] = [0.120649339028, 0.137886602118]
    M_ref[k][(0, 0, 1)][5, 0, 11, :] = [-0.046387623837, -0.033297695548]
    M_ref[k][(0, 0, 1)][5, 0, 14, :] = [-0.006251691335, -0.016280847761]
    M_ref[k][(0, 0, 1)][5, 0, 20, :] = [-0.009674044136, -0.003372612880]
    M_ref[k][(0, 0, 1)][5, 0, 23, :] = [0.001783754837, -0.004983853915]
    M_ref[k][(0, 0, 1)][5, 0, 29, :] = [0.013259946416, 0.010751282416]
    M_ref[k][(0, 0, 1)][5, 0, 32, :] = [-0.000029032756, -0.001456117987]
    M_ref[k][(0, 0, 1)][6, 0, 1, :] = [0.087850270321, 0.067142177357]
    M_ref[k][(0, 0, 1)][6, 0, 6, :] = [0.120649339028, 0.137886602118]
    M_ref[k][(0, 0, 1)][6, 0, 10, :] = [-0.046387623837, -0.033297695548]
    M_ref[k][(0, 0, 1)][6, 0, 15, :] = [-0.006251691335, -0.016280847761]
    M_ref[k][(0, 0, 1)][6, 0, 19, :] = [-0.009674044136, -0.003372612880]
    M_ref[k][(0, 0, 1)][6, 0, 24, :] = [0.001783754837, -0.004983853915]
    M_ref[k][(0, 0, 1)][6, 0, 28, :] = [0.013259946416, 0.010751282416]
    M_ref[k][(0, 0, 1)][6, 0, 33, :] = [-0.000029032756, -0.001456117987]
    M_ref[k][(0, 0, 1)][7, 0, 7, :] = [0.121030120056, 0.150360106550]
    M_ref[k][(0, 0, 1)][7, 0, 16, :] = [-0.002578322824, -0.019411382313]
    M_ref[k][(0, 0, 1)][7, 0, 25, :] = [-0.009833821318, -0.009951408030]
    M_ref[k][(0, 0, 1)][7, 0, 34, :] = [0.022792533393, 0.016163147291]
    M_ref[k][(0, 0, 1)][8, 0, 0, :] = [-0.020614576788, -0.009191747426]
    M_ref[k][(0, 0, 1)][8, 0, 3, :] = [0.067671066877, 0.054275180301]
    M_ref[k][(0, 0, 1)][8, 0, 8, :] = [0.105549894215, 0.113709588747]
    M_ref[k][(0, 0, 1)][8, 0, 9, :] = [0.015583355020, 0.007826817990]
    M_ref[k][(0, 0, 1)][8, 0, 12, :] = [-0.034865760551, -0.026532201724]
    M_ref[k][(0, 0, 1)][8, 0, 17, :] = [-0.003028331664, -0.007722745747]
    M_ref[k][(0, 0, 1)][8, 0, 18, :] = [0.009975695284, 0.000127869611]
    M_ref[k][(0, 0, 1)][8, 0, 21, :] = [-0.016406072557, -0.005078297379]
    M_ref[k][(0, 0, 1)][8, 0, 26, :] = [0.008827190731, -0.000162753495]
    M_ref[k][(0, 0, 1)][8, 0, 27, :] = [-0.017928088143, -0.011746551894]
    M_ref[k][(0, 0, 1)][8, 0, 30, :] = [0.028911899134, 0.023666553531]
    M_ref[k][(0, 0, 1)][8, 0, 35, :] = [-0.016569971709, -0.015168093845]
    M_ref[k][(0, 1, 0)][0, 0, 0, :] = [0.100801908177, 0.104100686579]
    M_ref[k][(0, 1, 0)][0, 0, 3, :] = [0.016837925628, 0.012969351573]
    M_ref[k][(0, 1, 0)][0, 0, 8, :] = [-0.007249644339, -0.004893220792]
    M_ref[k][(0, 1, 0)][0, 0, 9, :] = [-0.022871321220, -0.025111317636]
    M_ref[k][(0, 1, 0)][0, 0, 12, :] = [-0.004274440076, -0.001867806570]
    M_ref[k][(0, 1, 0)][0, 0, 17, :] = [0.001748991034, 0.000393299612]
    M_ref[k][(0, 1, 0)][0, 0, 18, :] = [-0.003143434077, -0.005987369989]
    M_ref[k][(0, 1, 0)][0, 0, 21, :] = [0.001906298924, 0.005177626798]
    M_ref[k][(0, 1, 0)][0, 0, 26, :] = [-0.000772878450, -0.003369068318]
    M_ref[k][(0, 1, 0)][0, 0, 27, :] = [-0.004952936269, -0.003167781504]
    M_ref[k][(0, 1, 0)][0, 0, 30, :] = [0.007482287796, 0.005967493848]
    M_ref[k][(0, 1, 0)][0, 0, 35, :] = [-0.004312549642, -0.003907703848]
    M_ref[k][(0, 1, 0)][0, 1, 1, :] = [0.042500521189, 0.044983664248]
    M_ref[k][(0, 1, 0)][0, 1, 6, :] = [0.021845280715, 0.019778330838]
    M_ref[k][(0, 1, 0)][0, 1, 10, :] = [-0.001593931771, -0.003163567595]
    M_ref[k][(0, 1, 0)][0, 1, 15, :] = [-0.000364586378, 0.000838027025]
    M_ref[k][(0, 1, 0)][0, 1, 19, :] = [0.001161080596, 0.000405465131]
    M_ref[k][(0, 1, 0)][0, 1, 24, :] = [-0.001600302983, -0.000788787373]
    M_ref[k][(0, 1, 0)][0, 1, 28, :] = [-0.005290604423, -0.004989786206]
    M_ref[k][(0, 1, 0)][0, 1, 33, :] = [0.006273906659, 0.006445030905]
    M_ref[k][(0, 1, 0)][0, 2, 2, :] = [0.042500521189, 0.044983664248]
    M_ref[k][(0, 1, 0)][0, 2, 5, :] = [0.021845280715, 0.019778330838]
    M_ref[k][(0, 1, 0)][0, 2, 11, :] = [-0.001593931771, -0.003163567595]
    M_ref[k][(0, 1, 0)][0, 2, 14, :] = [-0.000364586378, 0.000838027025]
    M_ref[k][(0, 1, 0)][0, 2, 20, :] = [0.001161080596, 0.000405465131]
    M_ref[k][(0, 1, 0)][0, 2, 23, :] = [-0.001600302983, -0.000788787373]
    M_ref[k][(0, 1, 0)][0, 2, 29, :] = [-0.005290604423, -0.004989786206]
    M_ref[k][(0, 1, 0)][0, 2, 32, :] = [0.006273906659, 0.006445030905]
    M_ref[k][(0, 1, 0)][0, 3, 0, :] = [-0.166803391818, -0.171337895442]
    M_ref[k][(0, 1, 0)][0, 3, 3, :] = [-0.041124177670, -0.035806432451]
    M_ref[k][(0, 1, 0)][0, 3, 8, :] = [0.010907692322, 0.007668550490]
    M_ref[k][(0, 1, 0)][0, 3, 9, :] = [0.022833948748, 0.025913049745]
    M_ref[k][(0, 1, 0)][0, 3, 12, :] = [0.001686735087, -0.001621425396]
    M_ref[k][(0, 1, 0)][0, 3, 17, :] = [-0.004926796927, -0.003063262333]
    M_ref[k][(0, 1, 0)][0, 3, 18, :] = [0.003385745856, 0.007295022603]
    M_ref[k][(0, 1, 0)][0, 3, 21, :] = [-0.000975656507, -0.005472426613]
    M_ref[k][(0, 1, 0)][0, 3, 26, :] = [-0.000113866439, 0.003454858382]
    M_ref[k][(0, 1, 0)][0, 3, 27, :] = [0.013271976682, 0.010818101409]
    M_ref[k][(0, 1, 0)][0, 3, 30, :] = [-0.018502621613, -0.016420384551]
    M_ref[k][(0, 1, 0)][0, 3, 35, :] = [0.011162874734, 0.010606373358]
    M_ref[k][(0, 1, 0)][1, 0, 1, :] = [0.072589559149, 0.072896960649]
    M_ref[k][(0, 1, 0)][1, 0, 6, :] = [0.011292217072, 0.011036338342]
    M_ref[k][(0, 1, 0)][1, 0, 10, :] = [0.010915221467, 0.010720907891]
    M_ref[k][(0, 1, 0)][1, 0, 15, :] = [0.000009478106, 0.000158356022]
    M_ref[k][(0, 1, 0)][1, 0, 19, :] = [-0.000592298925, -0.000685840587]
    M_ref[k][(0, 1, 0)][1, 0, 24, :] = [0.001150931017, 0.001251392856]
    M_ref[k][(0, 1, 0)][1, 0, 28, :] = [0.000513859770, 0.000551099659]
    M_ref[k][(0, 1, 0)][1, 0, 33, :] = [-0.001932614467, -0.001911430086]
    M_ref[k][(0, 1, 0)][1, 1, 0, :] = [0.085351744152, 0.082650724354]
    M_ref[k][(0, 1, 0)][1, 1, 3, :] = [0.020283013679, 0.023450578711]
    M_ref[k][(0, 1, 0)][1, 1, 7, :] = [0.019928135432, 0.030672101937]
    M_ref[k][(0, 1, 0)][1, 1, 8, :] = [-0.005077567402, -0.007006992747]
    M_ref[k][(0, 1, 0)][1, 1, 9, :] = [-0.044637023614, -0.042802928100]
    M_ref[k][(0, 1, 0)][1, 1, 12, :] = [-0.006091667994, -0.008062204949]
    M_ref[k][(0, 1, 0)][1, 1, 16, :] = [0.011348145294, 0.005181970367]
    M_ref[k][(0, 1, 0)][1, 1, 17, :] = [-0.007506120386, -0.006396088451]
    M_ref[k][(0, 1, 0)][1, 1, 18, :] = [-0.006594245289, -0.004265647770]
    M_ref[k][(0, 1, 0)][1, 1, 21, :] = [0.008537670707, 0.005859127234]
    M_ref[k][(0, 1, 0)][1, 1, 25, :] = [-0.003631746393, -0.003674819979]
    M_ref[k][(0, 1, 0)][1, 1, 26, :] = [-0.004257217815, -0.002131473258]
    M_ref[k][(0, 1, 0)][1, 1, 27, :] = [0.014112867670, 0.012651193748]
    M_ref[k][(0, 1, 0)][1, 1, 30, :] = [-0.019005836948, -0.017765532827]
    M_ref[k][(0, 1, 0)][1, 1, 34, :] = [0.007872563897, 0.005444131063]
    M_ref[k][(0, 1, 0)][1, 1, 35, :] = [0.010843561858, 0.010512076571]
    M_ref[k][(0, 1, 0)][1, 2, 4, :] = [0.019928205244, 0.030672324490]
    M_ref[k][(0, 1, 0)][1, 2, 13, :] = [0.011348072822, 0.005181810001]
    M_ref[k][(0, 1, 0)][1, 2, 22, :] = [-0.003631765674, -0.003674846891]
    M_ref[k][(0, 1, 0)][1, 2, 31, :] = [0.007872604140, 0.005444149292]
    M_ref[k][(0, 1, 0)][1, 3, 1, :] = [-0.097744553299, -0.097572304788]
    M_ref[k][(0, 1, 0)][1, 3, 6, :] = [-0.023375316686, -0.023518695070]
    M_ref[k][(0, 1, 0)][1, 3, 10, :] = [-0.045763516696, -0.045872397830]
    M_ref[k][(0, 1, 0)][1, 3, 15, :] = [-0.008622902257, -0.008539480415]
    M_ref[k][(0, 1, 0)][1, 3, 19, :] = [0.000371272397, 0.000318857520]
    M_ref[k][(0, 1, 0)][1, 3, 24, :] = [-0.001118705339, -0.001062412829]
    M_ref[k][(0, 1, 0)][1, 3, 28, :] = [0.000857073421, 0.000877940318]
    M_ref[k][(0, 1, 0)][1, 3, 33, :] = [0.001006123243, 0.001017993641]
    M_ref[k][(0, 1, 0)][2, 0, 2, :] = [0.072589559149, 0.072896960649]
    M_ref[k][(0, 1, 0)][2, 0, 5, :] = [0.011292217072, 0.011036338342]
    M_ref[k][(0, 1, 0)][2, 0, 11, :] = [0.010915221467, 0.010720907891]
    M_ref[k][(0, 1, 0)][2, 0, 14, :] = [0.000009478106, 0.000158356022]
    M_ref[k][(0, 1, 0)][2, 0, 20, :] = [-0.000592298925, -0.000685840587]
    M_ref[k][(0, 1, 0)][2, 0, 23, :] = [0.001150931017, 0.001251392856]
    M_ref[k][(0, 1, 0)][2, 0, 29, :] = [0.000513859770, 0.000551099659]
    M_ref[k][(0, 1, 0)][2, 0, 32, :] = [-0.001932614467, -0.001911430086]
    M_ref[k][(0, 1, 0)][2, 1, 4, :] = [0.019928205244, 0.030672324490]
    M_ref[k][(0, 1, 0)][2, 1, 13, :] = [0.011348072822, 0.005181810001]
    M_ref[k][(0, 1, 0)][2, 1, 22, :] = [-0.003631765674, -0.003674846891]
    M_ref[k][(0, 1, 0)][2, 1, 31, :] = [0.007872604140, 0.005444149292]
    M_ref[k][(0, 1, 0)][2, 2, 0, :] = [0.085351744152, 0.082650724354]
    M_ref[k][(0, 1, 0)][2, 2, 3, :] = [0.020283013679, 0.023450578711]
    M_ref[k][(0, 1, 0)][2, 2, 7, :] = [-0.019928135432, -0.030672101937]
    M_ref[k][(0, 1, 0)][2, 2, 8, :] = [-0.005077567402, -0.007006992747]
    M_ref[k][(0, 1, 0)][2, 2, 9, :] = [-0.044637023614, -0.042802928100]
    M_ref[k][(0, 1, 0)][2, 2, 12, :] = [-0.006091667994, -0.008062204949]
    M_ref[k][(0, 1, 0)][2, 2, 16, :] = [-0.011348145294, -0.005181970367]
    M_ref[k][(0, 1, 0)][2, 2, 17, :] = [-0.007506120386, -0.006396088451]
    M_ref[k][(0, 1, 0)][2, 2, 18, :] = [-0.006594245289, -0.004265647770]
    M_ref[k][(0, 1, 0)][2, 2, 21, :] = [0.008537670707, 0.005859127234]
    M_ref[k][(0, 1, 0)][2, 2, 25, :] = [0.003631746393, 0.003674819979]
    M_ref[k][(0, 1, 0)][2, 2, 26, :] = [-0.004257217815, -0.002131473258]
    M_ref[k][(0, 1, 0)][2, 2, 27, :] = [0.014112867670, 0.012651193748]
    M_ref[k][(0, 1, 0)][2, 2, 30, :] = [-0.019005836948, -0.017765532827]
    M_ref[k][(0, 1, 0)][2, 2, 34, :] = [-0.007872563897, -0.005444131063]
    M_ref[k][(0, 1, 0)][2, 2, 35, :] = [0.010843561858, 0.010512076571]
    M_ref[k][(0, 1, 0)][2, 3, 2, :] = [-0.097744553299, -0.097572304788]
    M_ref[k][(0, 1, 0)][2, 3, 5, :] = [-0.023375316686, -0.023518695070]
    M_ref[k][(0, 1, 0)][2, 3, 11, :] = [-0.045763516696, -0.045872397830]
    M_ref[k][(0, 1, 0)][2, 3, 14, :] = [-0.008622902257, -0.008539480415]
    M_ref[k][(0, 1, 0)][2, 3, 20, :] = [0.000371272397, 0.000318857520]
    M_ref[k][(0, 1, 0)][2, 3, 23, :] = [-0.001118705339, -0.001062412829]
    M_ref[k][(0, 1, 0)][2, 3, 29, :] = [0.000857073421, 0.000877940318]
    M_ref[k][(0, 1, 0)][2, 3, 32, :] = [0.001006123243, 0.001017993641]
    M_ref[k][(0, 1, 0)][3, 0, 0, :] = [0.017697559035, 0.016725526429]
    M_ref[k][(0, 1, 0)][3, 0, 3, :] = [0.059686061990, 0.060825993099]
    M_ref[k][(0, 1, 0)][3, 0, 8, :] = [0.010898526186, 0.010204172021]
    M_ref[k][(0, 1, 0)][3, 0, 9, :] = [-0.009387746781, -0.008727699549]
    M_ref[k][(0, 1, 0)][3, 0, 12, :] = [0.016616879553, 0.015907730301]
    M_ref[k][(0, 1, 0)][3, 0, 17, :] = [-0.000333048040, 0.000066425979]
    M_ref[k][(0, 1, 0)][3, 0, 18, :] = [-0.002156006517, -0.001317999807]
    M_ref[k][(0, 1, 0)][3, 0, 21, :] = [0.003450808121, 0.002486864209]
    M_ref[k][(0, 1, 0)][3, 0, 26, :] = [-0.001853425509, -0.001088420743]
    M_ref[k][(0, 1, 0)][3, 0, 27, :] = [0.006510587443, 0.005984565859]
    M_ref[k][(0, 1, 0)][3, 0, 30, :] = [-0.009737535064, -0.009291179192]
    M_ref[k][(0, 1, 0)][3, 0, 35, :] = [0.005992186606, 0.005872892957]
    M_ref[k][(0, 1, 0)][3, 1, 1, :] = [0.043886318623, 0.037253349636]
    M_ref[k][(0, 1, 0)][3, 1, 6, :] = [0.040985963041, 0.046507197260]
    M_ref[k][(0, 1, 0)][3, 1, 10, :] = [-0.019939671791, -0.015746862316]
    M_ref[k][(0, 1, 0)][3, 1, 15, :] = [0.002905287039, -0.000307132553]
    M_ref[k][(0, 1, 0)][3, 1, 19, :] = [-0.002518349818, -0.000499950626]
    M_ref[k][(0, 1, 0)][3, 1, 24, :] = [-0.001896735560, -0.004064455159]
    M_ref[k][(0, 1, 0)][3, 1, 28, :] = [0.000107492201, -0.000696053086]
    M_ref[k][(0, 1, 0)][3, 1, 33, :] = [0.007651109421, 0.007194002526]
    M_ref[k][(0, 1, 0)][3, 2, 2, :] = [0.043886318623, 0.037253349636]
    M_ref[k][(0, 1, 0)][3, 2, 5, :] = [0.040985963041, 0.046507197260]
    M_ref[k][(0, 1, 0)][3, 2, 11, :] = [-0.019939671791, -0.015746862316]
    M_ref[k][(0, 1, 0)][3, 2, 14, :] = [0.002905287039, -0.000307132553]
    M_ref[k][(0, 1, 0)][3, 2, 20, :] = [-0.002518349818, -0.000499950626]
    M_ref[k][(0, 1, 0)][3, 2, 23, :] = [-0.001896735560, -0.004064455159]
    M_ref[k][(0, 1, 0)][3, 2, 29, :] = [0.000107492201, -0.000696053086]
    M_ref[k][(0, 1, 0)][3, 2, 32, :] = [0.007651109421, 0.007194002526]
    M_ref[k][(0, 1, 0)][3, 3, 0, :] = [-0.069986900633, -0.070368342781]
    M_ref[k][(0, 1, 0)][3, 3, 3, :] = [-0.087908196856, -0.087460868476]
    M_ref[k][(0, 1, 0)][3, 3, 8, :] = [-0.019582175623, -0.019854652021]
    M_ref[k][(0, 1, 0)][3, 3, 9, :] = [0.032616354834, 0.032875368608]
    M_ref[k][(0, 1, 0)][3, 3, 12, :] = [-0.051580747437, -0.051859029680]
    M_ref[k][(0, 1, 0)][3, 3, 17, :] = [-0.012430735489, -0.012273975081]
    M_ref[k][(0, 1, 0)][3, 3, 18, :] = [0.000691204021, 0.001020052125]
    M_ref[k][(0, 1, 0)][3, 3, 21, :] = [-0.002518094303, -0.002896362310]
    M_ref[k][(0, 1, 0)][3, 3, 26, :] = [0.001482338613, 0.001782539511]
    M_ref[k][(0, 1, 0)][3, 3, 27, :] = [-0.001126957954, -0.001333377781]
    M_ref[k][(0, 1, 0)][3, 3, 30, :] = [0.004943768264, 0.005118925909]
    M_ref[k][(0, 1, 0)][3, 3, 35, :] = [-0.003719792417, -0.003766605276]
    M_ref[k][(0, 1, 1)][0, 0, 0, :] = [0.110566876018, 0.113125891930]
    M_ref[k][(0, 1, 1)][0, 0, 3, :] = [0.024178637919, 0.021177605004]
    M_ref[k][(0, 1, 1)][0, 0, 8, :] = [-0.009396334525, -0.007568347126]
    M_ref[k][(0, 1, 1)][0, 0, 9, :] = [-0.019678747920, -0.021416417375]
    M_ref[k][(0, 1, 1)][0, 0, 12, :] = [-0.002837612120, -0.000970674520]
    M_ref[k][(0, 1, 1)][0, 0, 17, :] = [0.000353347730, -0.000698325192]
    M_ref[k][(0, 1, 1)][0, 0, 18, :] = [-0.004540383803, -0.006746557229]
    M_ref[k][(0, 1, 1)][0, 0, 21, :] = [0.003151873138, 0.005689594398]
    M_ref[k][(0, 1, 1)][0, 0, 26, :] = [-0.001174947840, -0.003188933131]
    M_ref[k][(0, 1, 1)][0, 0, 27, :] = [-0.004616815523, -0.003231987901]
    M_ref[k][(0, 1, 1)][0, 0, 30, :] = [0.007256211986, 0.006081115831]
    M_ref[k][(0, 1, 1)][0, 0, 35, :] = [-0.004532146351, -0.004218088630]
    M_ref[k][(0, 1, 1)][1, 0, 1, :] = [0.077964469621, 0.077732947146]
    M_ref[k][(0, 1, 1)][1, 0, 6, :] = [0.021547182014, 0.021739899606]
    M_ref[k][(0, 1, 1)][1, 0, 10, :] = [0.019577220185, 0.019723569373]
    M_ref[k][(0, 1, 1)][1, 0, 15, :] = [0.000863191321, 0.000751062449]
    M_ref[k][(0, 1, 1)][1, 0, 19, :] = [-0.000595959162, -0.000525507336]
    M_ref[k][(0, 1, 1)][1, 0, 24, :] = [0.001497469566, 0.001421805740]
    M_ref[k][(0, 1, 1)][1, 0, 28, :] = [0.000274566383, 0.000246518794]
    M_ref[k][(0, 1, 1)][1, 0, 33, :] = [-0.002431882270, -0.002447837496]
    M_ref[k][(0, 1, 1)][2, 0, 2, :] = [0.077964469621, 0.077732947146]
    M_ref[k][(0, 1, 1)][2, 0, 5, :] = [0.021547182014, 0.021739899606]
    M_ref[k][(0, 1, 1)][2, 0, 11, :] = [0.019577220185, 0.019723569373]
    M_ref[k][(0, 1, 1)][2, 0, 14, :] = [0.000863191321, 0.000751062449]
    M_ref[k][(0, 1, 1)][2, 0, 20, :] = [-0.000595959162, -0.000525507336]
    M_ref[k][(0, 1, 1)][2, 0, 23, :] = [0.001497469566, 0.001421805740]
    M_ref[k][(0, 1, 1)][2, 0, 29, :] = [0.000274566383, 0.000246518794]
    M_ref[k][(0, 1, 1)][2, 0, 32, :] = [-0.002431882270, -0.002447837496]
    M_ref[k][(0, 1, 1)][3, 0, 0, :] = [0.032648060706, 0.031766316011]
    M_ref[k][(0, 1, 1)][3, 0, 3, :] = [0.050444760560, 0.051478808390]
    M_ref[k][(0, 1, 1)][3, 0, 8, :] = [0.016338253026, 0.015708394422]
    M_ref[k][(0, 1, 1)][3, 0, 9, :] = [-0.016600756733, -0.016002018439]
    M_ref[k][(0, 1, 1)][3, 0, 12, :] = [0.031337764810, 0.030694485369]
    M_ref[k][(0, 1, 1)][3, 0, 17, :] = [0.001535680340, 0.001898048943]
    M_ref[k][(0, 1, 1)][3, 0, 18, :] = [-0.004887054413, -0.004126886528]
    M_ref[k][(0, 1, 1)][3, 0, 21, :] = [0.007095888846, 0.006221481521]
    M_ref[k][(0, 1, 1)][3, 0, 26, :] = [-0.003694267674, -0.003000320896]
    M_ref[k][(0, 1, 1)][3, 0, 27, :] = [0.012890408160, 0.012413246450]
    M_ref[k][(0, 1, 1)][3, 0, 30, :] = [-0.018453287355, -0.018048391553]
    M_ref[k][(0, 1, 1)][3, 0, 35, :] = [0.011008152891, 0.010899939914]
    M_ref[k][(1, 0, 0)][0, 0, 0, :] = [0.007321916195, 0.016292248912]
    M_ref[k][(1, 0, 0)][0, 0, 3, :] = [0.003662900114, 0.013981311067]
    M_ref[k][(1, 0, 0)][0, 0, 8, :] = [0.002069892587, 0.010258785596]
    M_ref[k][(1, 0, 0)][0, 0, 9, :] = [0.026447803380, 0.020817074456]
    M_ref[k][(1, 0, 0)][0, 0, 12, :] = [0.033810963203, 0.029033005425]
    M_ref[k][(1, 0, 0)][0, 0, 17, :] = [0.016204568638, 0.014927605450]
    M_ref[k][(1, 0, 0)][0, 0, 18, :] = [0.127291207745, 0.116886212435]
    M_ref[k][(1, 0, 0)][0, 0, 21, :] = [-0.000224813780, -0.012427056550]
    M_ref[k][(1, 0, 0)][0, 0, 26, :] = [0.011729463349, 0.004296840428]
    M_ref[k][(1, 0, 0)][0, 0, 27, :] = [-0.053719803634, -0.046654414765]
    M_ref[k][(1, 0, 0)][0, 0, 30, :] = [-0.004402480545, 0.003188514451]
    M_ref[k][(1, 0, 0)][0, 0, 35, :] = [-0.007158129424, -0.002882012275]
    M_ref[k][(1, 0, 0)][0, 1, 1, :] = [0.000398857852, 0.012873656308]
    M_ref[k][(1, 0, 0)][0, 1, 6, :] = [0.003260653770, 0.016658333089]
    M_ref[k][(1, 0, 0)][0, 1, 10, :] = [0.027902606838, 0.022936262442]
    M_ref[k][(1, 0, 0)][0, 1, 15, :] = [0.017594961463, 0.020420129255]
    M_ref[k][(1, 0, 0)][0, 1, 19, :] = [0.145453348332, 0.104458013201]
    M_ref[k][(1, 0, 0)][0, 1, 24, :] = [0.026849160750, -0.007275052347]
    M_ref[k][(1, 0, 0)][0, 1, 28, :] = [-0.042839300155, -0.016925470330]
    M_ref[k][(1, 0, 0)][0, 1, 33, :] = [-0.021141889277, -0.001287399077]
    M_ref[k][(1, 0, 0)][0, 2, 2, :] = [0.000398857852, 0.012873656308]
    M_ref[k][(1, 0, 0)][0, 2, 5, :] = [0.003260653770, 0.016658333089]
    M_ref[k][(1, 0, 0)][0, 2, 11, :] = [0.027902606838, 0.022936262442]
    M_ref[k][(1, 0, 0)][0, 2, 14, :] = [0.017594961463, 0.020420129255]
    M_ref[k][(1, 0, 0)][0, 2, 20, :] = [0.145453348332, 0.104458013201]
    M_ref[k][(1, 0, 0)][0, 2, 23, :] = [0.026849160750, -0.007275052347]
    M_ref[k][(1, 0, 0)][0, 2, 29, :] = [-0.042839300155, -0.016925470330]
    M_ref[k][(1, 0, 0)][0, 2, 32, :] = [-0.021141889277, -0.001287399077]
    M_ref[k][(1, 0, 0)][0, 3, 0, :] = [0.020637555765, 0.037511864885]
    M_ref[k][(1, 0, 0)][0, 3, 3, :] = [0.015786396912, 0.035196608500]
    M_ref[k][(1, 0, 0)][0, 3, 8, :] = [0.009465245436, 0.024869569485]
    M_ref[k][(1, 0, 0)][0, 3, 9, :] = [-0.066860976386, -0.077453076304]
    M_ref[k][(1, 0, 0)][0, 3, 12, :] = [-0.055095837931, -0.064083769489]
    M_ref[k][(1, 0, 0)][0, 3, 17, :] = [-0.026824077529, -0.029226203766]
    M_ref[k][(1, 0, 0)][0, 3, 18, :] = [0.035638735314, 0.016065647804]
    M_ref[k][(1, 0, 0)][0, 3, 21, :] = [0.107139488911, 0.084185555878]
    M_ref[k][(1, 0, 0)][0, 3, 26, :] = [-0.014227156947, -0.028208843100]
    M_ref[k][(1, 0, 0)][0, 3, 27, :] = [-0.020714998420, -0.007424125095]
    M_ref[k][(1, 0, 0)][0, 3, 30, :] = [-0.021581651754, -0.007302048073]
    M_ref[k][(1, 0, 0)][0, 3, 35, :] = [-0.000100571369, 0.007943335682]
    M_ref[k][(1, 0, 0)][0, 4, 4, :] = [-0.004900117981, -0.004953706093]
    M_ref[k][(1, 0, 0)][0, 4, 13, :] = [0.011549059259, 0.008528338926]
    M_ref[k][(1, 0, 0)][0, 4, 22, :] = [0.127723830867, 0.141088287484]
    M_ref[k][(1, 0, 0)][0, 4, 31, :] = [-0.016691696152, -0.024361822485]
    M_ref[k][(1, 0, 0)][0, 5, 2, :] = [0.005537167881, 0.002001094778]
    M_ref[k][(1, 0, 0)][0, 5, 5, :] = [0.000994207255, -0.002803463196]
    M_ref[k][(1, 0, 0)][0, 5, 11, :] = [-0.007820214806, -0.006412468068]
    M_ref[k][(1, 0, 0)][0, 5, 14, :] = [-0.000109204920, -0.000910019448]
    M_ref[k][(1, 0, 0)][0, 5, 20, :] = [-0.045973372512, -0.034352944141]
    M_ref[k][(1, 0, 0)][0, 5, 23, :] = [0.124434058420, 0.134106816863]
    M_ref[k][(1, 0, 0)][0, 5, 29, :] = [0.024592991179, 0.017247526083]
    M_ref[k][(1, 0, 0)][0, 5, 32, :] = [-0.016558632225, -0.022186533070]
    M_ref[k][(1, 0, 0)][0, 6, 1, :] = [0.005537167881, 0.002001094778]
    M_ref[k][(1, 0, 0)][0, 6, 6, :] = [0.000994207255, -0.002803463196]
    M_ref[k][(1, 0, 0)][0, 6, 10, :] = [-0.007820214806, -0.006412468068]
    M_ref[k][(1, 0, 0)][0, 6, 15, :] = [-0.000109204920, -0.000910019448]
    M_ref[k][(1, 0, 0)][0, 6, 19, :] = [-0.045973372512, -0.034352944141]
    M_ref[k][(1, 0, 0)][0, 6, 24, :] = [0.124434058420, 0.134106816863]
    M_ref[k][(1, 0, 0)][0, 6, 28, :] = [0.024592991179, 0.017247526083]
    M_ref[k][(1, 0, 0)][0, 6, 33, :] = [-0.016558632225, -0.022186533070]
    M_ref[k][(1, 0, 0)][0, 7, 7, :] = [-0.004900087127, -0.004953667092]
    M_ref[k][(1, 0, 0)][0, 7, 16, :] = [0.011549004439, 0.008528235589]
    M_ref[k][(1, 0, 0)][0, 7, 25, :] = [0.127723864728, 0.141088467151]
    M_ref[k][(1, 0, 0)][0, 7, 34, :] = [-0.016691773864, -0.024361983589]
    M_ref[k][(1, 0, 0)][0, 8, 0, :] = [0.005441629460, -0.000963295427]
    M_ref[k][(1, 0, 0)][0, 8, 3, :] = [0.009494165386, 0.002126696508]
    M_ref[k][(1, 0, 0)][0, 8, 8, :] = [0.004872249764, -0.000974717822]
    M_ref[k][(1, 0, 0)][0, 8, 9, :] = [-0.010041600455, -0.006021192633]
    M_ref[k][(1, 0, 0)][0, 8, 12, :] = [-0.017008338643, -0.013596819622]
    M_ref[k][(1, 0, 0)][0, 8, 17, :] = [-0.009207945866, -0.008296178871]
    M_ref[k][(1, 0, 0)][0, 8, 18, :] = [-0.008217981712, -0.000788690567]
    M_ref[k][(1, 0, 0)][0, 8, 21, :] = [-0.042793759114, -0.034081211779]
    M_ref[k][(1, 0, 0)][0, 8, 26, :] = [0.118506388852, 0.123813370547]
    M_ref[k][(1, 0, 0)][0, 8, 27, :] = [0.006720330263, 0.001675558075]
    M_ref[k][(1, 0, 0)][0, 8, 30, :] = [0.022677330305, 0.017257269053]
    M_ref[k][(1, 0, 0)][0, 8, 35, :] = [-0.015562537193, -0.018615735999]
    M_ref[k][(1, 0, 0)][1, 0, 1, :] = [-0.009198056655, -0.005171776572]
    M_ref[k][(1, 0, 0)][1, 0, 6, :] = [-0.002369022926, 0.001955119831]
    M_ref[k][(1, 0, 0)][1, 0, 10, :] = [0.023548840710, 0.021945937577]
    M_ref[k][(1, 0, 0)][1, 0, 15, :] = [0.010823544592, 0.011735376302]
    M_ref[k][(1, 0, 0)][1, 0, 19, :] = [0.098765864796, 0.085534492610]
    M_ref[k][(1, 0, 0)][1, 0, 24, :] = [-0.033375186200, -0.044388882335]
    M_ref[k][(1, 0, 0)][1, 0, 28, :] = [-0.039430745393, -0.031066976049]
    M_ref[k][(1, 0, 0)][1, 0, 33, :] = [0.006041282237, 0.012449380830]
    M_ref[k][(1, 0, 0)][1, 1, 0, :] = [0.007448948770, 0.011987411453]
    M_ref[k][(1, 0, 0)][1, 1, 3, :] = [-0.002432463441, 0.002788047830]
    M_ref[k][(1, 0, 0)][1, 1, 7, :] = [-0.008088405596, -0.007995552957]
    M_ref[k][(1, 0, 0)][1, 1, 8, :] = [-0.000775362321, 0.003367737662]
    M_ref[k][(1, 0, 0)][1, 1, 9, :] = [0.065511902834, 0.062663083966]
    M_ref[k][(1, 0, 0)][1, 1, 12, :] = [0.079139573372, 0.076722206730]
    M_ref[k][(1, 0, 0)][1, 1, 16, :] = [0.025229318092, 0.030464229281]
    M_ref[k][(1, 0, 0)][1, 1, 17, :] = [0.033888880952, 0.033242812411]
    M_ref[k][(1, 0, 0)][1, 1, 18, :] = [0.043312753343, 0.038048435461]
    M_ref[k][(1, 0, 0)][1, 1, 21, :] = [-0.011615391467, -0.017789011243]
    M_ref[k][(1, 0, 0)][1, 1, 25, :] = [0.082042556545, 0.058882059919]
    M_ref[k][(1, 0, 0)][1, 1, 26, :] = [0.002514467159, -0.001246004479]
    M_ref[k][(1, 0, 0)][1, 1, 27, :] = [-0.026914806412, -0.023340133685]
    M_ref[k][(1, 0, 0)][1, 1, 30, :] = [0.003889679383, 0.007730278009]
    M_ref[k][(1, 0, 0)][1, 1, 34, :] = [-0.025172109001, -0.011879841799]
    M_ref[k][(1, 0, 0)][1, 1, 35, :] = [-0.008367055625, -0.006203590933]
    M_ref[k][(1, 0, 0)][1, 2, 4, :] = [-0.008088460083, -0.007995592685]
    M_ref[k][(1, 0, 0)][1, 2, 13, :] = [0.025229418446, 0.030464281718]
    M_ref[k][(1, 0, 0)][1, 2, 22, :] = [0.082042627411, 0.058882223478]
    M_ref[k][(1, 0, 0)][1, 2, 31, :] = [-0.025172175898, -0.011879961395]
    M_ref[k][(1, 0, 0)][1, 3, 1, :] = [0.027295835964, 0.024120454632]
    M_ref[k][(1, 0, 0)][1, 3, 6, :] = [0.003722463169, 0.000312168329]
    M_ref[k][(1, 0, 0)][1, 3, 10, :] = [-0.062913139818, -0.061648988149]
    M_ref[k][(1, 0, 0)][1, 3, 15, :] = [-0.015313162368, -0.016032291026]
    M_ref[k][(1, 0, 0)][1, 3, 19, :] = [-0.043997900882, -0.033562796673]
    M_ref[k][(1, 0, 0)][1, 3, 24, :] = [0.061957175129, 0.070643278526]
    M_ref[k][(1, 0, 0)][1, 3, 28, :] = [0.023106751107, 0.016510549031]
    M_ref[k][(1, 0, 0)][1, 3, 33, :] = [-0.010893726068, -0.015947561428]
    M_ref[k][(1, 0, 0)][1, 4, 2, :] = [-0.001868459122, -0.000280175363]
    M_ref[k][(1, 0, 0)][1, 4, 5, :] = [-0.003062795508, -0.001357011119]
    M_ref[k][(1, 0, 0)][1, 4, 11, :] = [0.007294383550, 0.006662071599]
    M_ref[k][(1, 0, 0)][1, 4, 14, :] = [0.009857127678, 0.010216826324]
    M_ref[k][(1, 0, 0)][1, 4, 20, :] = [0.102112270499, 0.096892769341]
    M_ref[k][(1, 0, 0)][1, 4, 23, :] = [-0.030724750478, -0.035069424554]
    M_ref[k][(1, 0, 0)][1, 4, 29, :] = [-0.048770595902, -0.045471262836]
    M_ref[k][(1, 0, 0)][1, 4, 32, :] = [0.009597976784, 0.012125838412]
    M_ref[k][(1, 0, 0)][1, 5, 4, :] = [0.015923878210, 0.016000783745]
    M_ref[k][(1, 0, 0)][1, 5, 13, :] = [-0.036309795569, -0.031974690149]
    M_ref[k][(1, 0, 0)][1, 5, 22, :] = [-0.012927477535, -0.032107117464]
    M_ref[k][(1, 0, 0)][1, 5, 31, :] = [-0.000368468623, 0.010639106580]
    M_ref[k][(1, 0, 0)][1, 6, 0, :] = [0.021719472421, 0.016952421550]
    M_ref[k][(1, 0, 0)][1, 6, 3, :] = [0.028983771386, 0.023500319269]
    M_ref[k][(1, 0, 0)][1, 6, 7, :] = [0.015923876170, 0.016000769202]
    M_ref[k][(1, 0, 0)][1, 6, 8, :] = [0.013252624079, 0.008900849090]
    M_ref[k][(1, 0, 0)][1, 6, 9, :] = [-0.073309752619, -0.070317447643]
    M_ref[k][(1, 0, 0)][1, 6, 12, :] = [-0.092283157927, -0.089744036078]
    M_ref[k][(1, 0, 0)][1, 6, 16, :] = [-0.036309791937, -0.031974662668]
    M_ref[k][(1, 0, 0)][1, 6, 17, :] = [-0.047954648593, -0.047276039597]
    M_ref[k][(1, 0, 0)][1, 6, 18, :] = [-0.024659182740, -0.019129717610]
    M_ref[k][(1, 0, 0)][1, 6, 21, :] = [0.084658250209, 0.091142815929]
    M_ref[k][(1, 0, 0)][1, 6, 25, :] = [-0.012927475172, -0.032107121819]
    M_ref[k][(1, 0, 0)][1, 6, 26, :] = [-0.056333469691, -0.052383594843]
    M_ref[k][(1, 0, 0)][1, 6, 27, :] = [0.024292276346, 0.020537558520]
    M_ref[k][(1, 0, 0)][1, 6, 30, :] = [-0.038314477076, -0.042348514659]
    M_ref[k][(1, 0, 0)][1, 6, 34, :] = [-0.000368474173, 0.010639104473]
    M_ref[k][(1, 0, 0)][1, 6, 35, :] = [0.022302530529, 0.020030098879]
    M_ref[k][(1, 0, 0)][1, 7, 1, :] = [-0.001868452798, -0.000280176235]
    M_ref[k][(1, 0, 0)][1, 7, 6, :] = [-0.003062789637, -0.001357012977]
    M_ref[k][(1, 0, 0)][1, 7, 10, :] = [0.007294366605, 0.006662057519]
    M_ref[k][(1, 0, 0)][1, 7, 15, :] = [0.009857109833, 0.010216806850]
    M_ref[k][(1, 0, 0)][1, 7, 19, :] = [0.102112318530, 0.096892841021]
    M_ref[k][(1, 0, 0)][1, 7, 24, :] = [-0.030724779635, -0.035069434025]
    M_ref[k][(1, 0, 0)][1, 7, 28, :] = [-0.048770638260, -0.045471320143]
    M_ref[k][(1, 0, 0)][1, 7, 33, :] = [0.009597988937, 0.012125839111]
    M_ref[k][(1, 0, 0)][1, 8, 1, :] = [-0.011769423841, -0.008779575633]
    M_ref[k][(1, 0, 0)][1, 8, 6, :] = [0.006331765554, 0.009542801600]
    M_ref[k][(1, 0, 0)][1, 8, 10, :] = [0.009932603439, 0.008742314398]
    M_ref[k][(1, 0, 0)][1, 8, 15, :] = [-0.022857720059, -0.022180609091]
    M_ref[k][(1, 0, 0)][1, 8, 19, :] = [-0.020380333713, -0.030205729267]
    M_ref[k][(1, 0, 0)][1, 8, 24, :] = [-0.026291083922, -0.034469670353]
    M_ref[k][(1, 0, 0)][1, 8, 28, :] = [0.005730437051, 0.011941232195]
    M_ref[k][(1, 0, 0)][1, 8, 33, :] = [0.009018199600, 0.013776746375]
    M_ref[k][(1, 0, 0)][2, 0, 2, :] = [-0.009198056655, -0.005171776572]
    M_ref[k][(1, 0, 0)][2, 0, 5, :] = [-0.002369022926, 0.001955119831]
    M_ref[k][(1, 0, 0)][2, 0, 11, :] = [0.023548840710, 0.021945937577]
    M_ref[k][(1, 0, 0)][2, 0, 14, :] = [0.010823544592, 0.011735376302]
    M_ref[k][(1, 0, 0)][2, 0, 20, :] = [0.098765864796, 0.085534492610]
    M_ref[k][(1, 0, 0)][2, 0, 23, :] = [-0.033375186200, -0.044388882335]
    M_ref[k][(1, 0, 0)][2, 0, 29, :] = [-0.039430745393, -0.031066976049]
    M_ref[k][(1, 0, 0)][2, 0, 32, :] = [0.006041282237, 0.012449380830]
    M_ref[k][(1, 0, 0)][2, 1, 4, :] = [-0.008088460083, -0.007995592685]
    M_ref[k][(1, 0, 0)][2, 1, 13, :] = [0.025229418446, 0.030464281718]
    M_ref[k][(1, 0, 0)][2, 1, 22, :] = [0.082042627411, 0.058882223478]
    M_ref[k][(1, 0, 0)][2, 1, 31, :] = [-0.025172175898, -0.011879961395]
    M_ref[k][(1, 0, 0)][2, 2, 0, :] = [0.007448948770, 0.011987411453]
    M_ref[k][(1, 0, 0)][2, 2, 3, :] = [-0.002432463441, 0.002788047830]
    M_ref[k][(1, 0, 0)][2, 2, 7, :] = [0.008088405596, 0.007995552957]
    M_ref[k][(1, 0, 0)][2, 2, 8, :] = [-0.000775362321, 0.003367737662]
    M_ref[k][(1, 0, 0)][2, 2, 9, :] = [0.065511902834, 0.062663083966]
    M_ref[k][(1, 0, 0)][2, 2, 12, :] = [0.079139573372, 0.076722206730]
    M_ref[k][(1, 0, 0)][2, 2, 16, :] = [-0.025229318092, -0.030464229281]
    M_ref[k][(1, 0, 0)][2, 2, 17, :] = [0.033888880952, 0.033242812411]
    M_ref[k][(1, 0, 0)][2, 2, 18, :] = [0.043312753343, 0.038048435461]
    M_ref[k][(1, 0, 0)][2, 2, 21, :] = [-0.011615391467, -0.017789011243]
    M_ref[k][(1, 0, 0)][2, 2, 25, :] = [-0.082042556545, -0.058882059919]
    M_ref[k][(1, 0, 0)][2, 2, 26, :] = [0.002514467159, -0.001246004479]
    M_ref[k][(1, 0, 0)][2, 2, 27, :] = [-0.026914806412, -0.023340133685]
    M_ref[k][(1, 0, 0)][2, 2, 30, :] = [0.003889679383, 0.007730278009]
    M_ref[k][(1, 0, 0)][2, 2, 34, :] = [0.025172109001, 0.011879841799]
    M_ref[k][(1, 0, 0)][2, 2, 35, :] = [-0.008367055625, -0.006203590933]
    M_ref[k][(1, 0, 0)][2, 3, 2, :] = [0.027295835964, 0.024120454632]
    M_ref[k][(1, 0, 0)][2, 3, 5, :] = [0.003722463169, 0.000312168329]
    M_ref[k][(1, 0, 0)][2, 3, 11, :] = [-0.062913139818, -0.061648988149]
    M_ref[k][(1, 0, 0)][2, 3, 14, :] = [-0.015313162368, -0.016032291026]
    M_ref[k][(1, 0, 0)][2, 3, 20, :] = [-0.043997900882, -0.033562796673]
    M_ref[k][(1, 0, 0)][2, 3, 23, :] = [0.061957175129, 0.070643278526]
    M_ref[k][(1, 0, 0)][2, 3, 29, :] = [0.023106751107, 0.016510549031]
    M_ref[k][(1, 0, 0)][2, 3, 32, :] = [-0.010893726068, -0.015947561428]
    M_ref[k][(1, 0, 0)][2, 4, 1, :] = [-0.001868459122, -0.000280175363]
    M_ref[k][(1, 0, 0)][2, 4, 6, :] = [-0.003062795508, -0.001357011119]
    M_ref[k][(1, 0, 0)][2, 4, 10, :] = [0.007294383550, 0.006662071599]
    M_ref[k][(1, 0, 0)][2, 4, 15, :] = [0.009857127678, 0.010216826324]
    M_ref[k][(1, 0, 0)][2, 4, 19, :] = [0.102112270499, 0.096892769341]
    M_ref[k][(1, 0, 0)][2, 4, 24, :] = [-0.030724750478, -0.035069424554]
    M_ref[k][(1, 0, 0)][2, 4, 28, :] = [-0.048770595902, -0.045471262836]
    M_ref[k][(1, 0, 0)][2, 4, 33, :] = [0.009597976784, 0.012125838412]
    M_ref[k][(1, 0, 0)][2, 5, 0, :] = [0.021719472421, 0.016952421550]
    M_ref[k][(1, 0, 0)][2, 5, 3, :] = [0.028983771386, 0.023500319269]
    M_ref[k][(1, 0, 0)][2, 5, 7, :] = [-0.015923876170, -0.016000769202]
    M_ref[k][(1, 0, 0)][2, 5, 8, :] = [0.013252624079, 0.008900849090]
    M_ref[k][(1, 0, 0)][2, 5, 9, :] = [-0.073309752619, -0.070317447643]
    M_ref[k][(1, 0, 0)][2, 5, 12, :] = [-0.092283157927, -0.089744036078]
    M_ref[k][(1, 0, 0)][2, 5, 16, :] = [0.036309791937, 0.031974662668]
    M_ref[k][(1, 0, 0)][2, 5, 17, :] = [-0.047954648593, -0.047276039597]
    M_ref[k][(1, 0, 0)][2, 5, 18, :] = [-0.024659182740, -0.019129717610]
    M_ref[k][(1, 0, 0)][2, 5, 21, :] = [0.084658250209, 0.091142815929]
    M_ref[k][(1, 0, 0)][2, 5, 25, :] = [0.012927475172, 0.032107121819]
    M_ref[k][(1, 0, 0)][2, 5, 26, :] = [-0.056333469691, -0.052383594843]
    M_ref[k][(1, 0, 0)][2, 5, 27, :] = [0.024292276346, 0.020537558520]
    M_ref[k][(1, 0, 0)][2, 5, 30, :] = [-0.038314477076, -0.042348514659]
    M_ref[k][(1, 0, 0)][2, 5, 34, :] = [0.000368474173, -0.010639104473]
    M_ref[k][(1, 0, 0)][2, 5, 35, :] = [0.022302530529, 0.020030098879]
    M_ref[k][(1, 0, 0)][2, 6, 4, :] = [0.015923878210, 0.016000783745]
    M_ref[k][(1, 0, 0)][2, 6, 13, :] = [-0.036309795569, -0.031974690149]
    M_ref[k][(1, 0, 0)][2, 6, 22, :] = [-0.012927477535, -0.032107117464]
    M_ref[k][(1, 0, 0)][2, 6, 31, :] = [-0.000368468623, 0.010639106580]
    M_ref[k][(1, 0, 0)][2, 7, 2, :] = [0.001868452798, 0.000280176235]
    M_ref[k][(1, 0, 0)][2, 7, 5, :] = [0.003062789637, 0.001357012977]
    M_ref[k][(1, 0, 0)][2, 7, 11, :] = [-0.007294366605, -0.006662057519]
    M_ref[k][(1, 0, 0)][2, 7, 14, :] = [-0.009857109833, -0.010216806850]
    M_ref[k][(1, 0, 0)][2, 7, 20, :] = [-0.102112318530, -0.096892841021]
    M_ref[k][(1, 0, 0)][2, 7, 23, :] = [0.030724779635, 0.035069434025]
    M_ref[k][(1, 0, 0)][2, 7, 29, :] = [0.048770638260, 0.045471320143]
    M_ref[k][(1, 0, 0)][2, 7, 32, :] = [-0.009597988937, -0.012125839111]
    M_ref[k][(1, 0, 0)][2, 8, 2, :] = [-0.011769423841, -0.008779575633]
    M_ref[k][(1, 0, 0)][2, 8, 5, :] = [0.006331765554, 0.009542801600]
    M_ref[k][(1, 0, 0)][2, 8, 11, :] = [0.009932603439, 0.008742314398]
    M_ref[k][(1, 0, 0)][2, 8, 14, :] = [-0.022857720059, -0.022180609091]
    M_ref[k][(1, 0, 0)][2, 8, 20, :] = [-0.020380333713, -0.030205729267]
    M_ref[k][(1, 0, 0)][2, 8, 23, :] = [-0.026291083922, -0.034469670353]
    M_ref[k][(1, 0, 0)][2, 8, 29, :] = [0.005730437051, 0.011941232195]
    M_ref[k][(1, 0, 0)][2, 8, 32, :] = [0.009018199601, 0.013776746375]
    M_ref[k][(1, 0, 0)][3, 0, 0, :] = [0.007115468011, 0.014125951313]
    M_ref[k][(1, 0, 0)][3, 0, 3, :] = [-0.000904325071, 0.007159706663]
    M_ref[k][(1, 0, 0)][3, 0, 8, :] = [-0.000624760493, 0.005775013018]
    M_ref[k][(1, 0, 0)][3, 0, 9, :] = [0.013976014520, 0.009575494233]
    M_ref[k][(1, 0, 0)][3, 0, 12, :] = [0.028586167813, 0.024852104174]
    M_ref[k][(1, 0, 0)][3, 0, 17, :] = [0.015361112583, 0.014363141896]
    M_ref[k][(1, 0, 0)][3, 0, 18, :] = [0.220275203060, 0.212143504027]
    M_ref[k][(1, 0, 0)][3, 0, 21, :] = [-0.038403254903, -0.047939536543]
    M_ref[k][(1, 0, 0)][3, 0, 26, :] = [0.003212836665, -0.002595897584]
    M_ref[k][(1, 0, 0)][3, 0, 27, :] = [-0.078415649570, -0.072893915622]
    M_ref[k][(1, 0, 0)][3, 0, 30, :] = [0.002578174629, 0.008510679625]
    M_ref[k][(1, 0, 0)][3, 0, 35, :] = [-0.001029841887, 0.002312023884]
    M_ref[k][(1, 0, 0)][3, 1, 1, :] = [0.002339423418, 0.011062821638]
    M_ref[k][(1, 0, 0)][3, 1, 6, :] = [-0.000962220219, 0.008406531696]
    M_ref[k][(1, 0, 0)][3, 1, 10, :] = [0.012006930215, 0.008534056484]
    M_ref[k][(1, 0, 0)][3, 1, 15, :] = [0.017595337647, 0.019570925772]
    M_ref[k][(1, 0, 0)][3, 1, 19, :] = [0.178325181664, 0.149657894191]
    M_ref[k][(1, 0, 0)][3, 1, 24, :] = [-0.006245670111, -0.030108107714]
    M_ref[k][(1, 0, 0)][3, 1, 28, :] = [-0.017445168313, 0.000675898554]
    M_ref[k][(1, 0, 0)][3, 1, 33, :] = [-0.017269424761, -0.003385543184]
    M_ref[k][(1, 0, 0)][3, 2, 2, :] = [0.002339423418, 0.011062821638]
    M_ref[k][(1, 0, 0)][3, 2, 5, :] = [-0.000962220219, 0.008406531696]
    M_ref[k][(1, 0, 0)][3, 2, 11, :] = [0.012006930215, 0.008534056484]
    M_ref[k][(1, 0, 0)][3, 2, 14, :] = [0.017595337647, 0.019570925772]
    M_ref[k][(1, 0, 0)][3, 2, 20, :] = [0.178325181664, 0.149657894191]
    M_ref[k][(1, 0, 0)][3, 2, 23, :] = [-0.006245670111, -0.030108107714]
    M_ref[k][(1, 0, 0)][3, 2, 29, :] = [-0.017445168313, 0.000675898554]
    M_ref[k][(1, 0, 0)][3, 2, 32, :] = [-0.017269424761, -0.003385543184]
    M_ref[k][(1, 0, 0)][3, 3, 0, :] = [0.000639355009, 0.008264145519]
    M_ref[k][(1, 0, 0)][3, 3, 3, :] = [0.019415363297, 0.028186021466]
    M_ref[k][(1, 0, 0)][3, 3, 8, :] = [0.008782436457, 0.015743002546]
    M_ref[k][(1, 0, 0)][3, 3, 9, :] = [-0.022980390052, -0.027766514471]
    M_ref[k][(1, 0, 0)][3, 3, 12, :] = [-0.056431921777, -0.060493189990]
    M_ref[k][(1, 0, 0)][3, 3, 17, :] = [-0.027428475523, -0.028513895330]
    M_ref[k][(1, 0, 0)][3, 3, 18, :] = [-0.039544407063, -0.048388662010]
    M_ref[k][(1, 0, 0)][3, 3, 21, :] = [0.133650996302, 0.123279079472]
    M_ref[k][(1, 0, 0)][3, 3, 26, :] = [-0.030074144669, -0.036391880532]
    M_ref[k][(1, 0, 0)][3, 3, 27, :] = [0.019024619606, 0.025030206210]
    M_ref[k][(1, 0, 0)][3, 3, 30, :] = [0.010210330803, 0.016662683066]
    M_ref[k][(1, 0, 0)][3, 3, 35, :] = [-0.007719631345, -0.004084928098]
    M_ref[k][(1, 0, 0)][3, 4, 4, :] = [-0.009523227410, -0.009691805042]
    M_ref[k][(1, 0, 0)][3, 4, 13, :] = [0.021014202178, 0.011511611682]
    M_ref[k][(1, 0, 0)][3, 4, 22, :] = [0.148003794840, 0.190045739946]
    M_ref[k][(1, 0, 0)][3, 4, 31, :] = [0.013714923985, -0.010413780942]
    M_ref[k][(1, 0, 0)][3, 5, 2, :] = [0.002994143886, -0.003764687243]
    M_ref[k][(1, 0, 0)][3, 5, 5, :] = [-0.003691317403, -0.010950164298]
    M_ref[k][(1, 0, 0)][3, 5, 11, :] = [0.005171795658, 0.007862555209]
    M_ref[k][(1, 0, 0)][3, 5, 14, :] = [0.016791014649, 0.015260342062]
    M_ref[k][(1, 0, 0)][3, 5, 20, :] = [-0.112131684688, -0.089920460338]
    M_ref[k][(1, 0, 0)][3, 5, 23, :] = [0.171759438745, 0.190247897260]
    M_ref[k][(1, 0, 0)][3, 5, 29, :] = [0.054422530410, 0.040382447894]
    M_ref[k][(1, 0, 0)][3, 5, 32, :] = [0.000414383049, -0.010342756386]
    M_ref[k][(1, 0, 0)][3, 6, 1, :] = [0.002994143886, -0.003764687243]
    M_ref[k][(1, 0, 0)][3, 6, 6, :] = [-0.003691317403, -0.010950164298]
    M_ref[k][(1, 0, 0)][3, 6, 10, :] = [0.005171795658, 0.007862555209]
    M_ref[k][(1, 0, 0)][3, 6, 15, :] = [0.016791014649, 0.015260342062]
    M_ref[k][(1, 0, 0)][3, 6, 19, :] = [-0.112131684688, -0.089920460338]
    M_ref[k][(1, 0, 0)][3, 6, 24, :] = [0.171759438745, 0.190247897260]
    M_ref[k][(1, 0, 0)][3, 6, 28, :] = [0.054422530410, 0.040382447894]
    M_ref[k][(1, 0, 0)][3, 6, 33, :] = [0.000414383049, -0.010342756386]
    M_ref[k][(1, 0, 0)][3, 7, 7, :] = [-0.009523222553, -0.009691773620]
    M_ref[k][(1, 0, 0)][3, 7, 16, :] = [0.021014193571, 0.011511503323]
    M_ref[k][(1, 0, 0)][3, 7, 25, :] = [0.148003856872, 0.190046026743]
    M_ref[k][(1, 0, 0)][3, 7, 34, :] = [0.013714782743, -0.010414050274]
    M_ref[k][(1, 0, 0)][3, 8, 0, :] = [0.030538133794, 0.020156888623]
    M_ref[k][(1, 0, 0)][3, 8, 3, :] = [0.029960208473, 0.018018850526]
    M_ref[k][(1, 0, 0)][3, 8, 8, :] = [0.014672728097, 0.005195832570]
    M_ref[k][(1, 0, 0)][3, 8, 9, :] = [-0.050425005354, -0.043908638649]
    M_ref[k][(1, 0, 0)][3, 8, 12, :] = [-0.052596175436, -0.047066709287]
    M_ref[k][(1, 0, 0)][3, 8, 17, :] = [-0.027895283306, -0.026417471016]
    M_ref[k][(1, 0, 0)][3, 8, 18, :] = [0.017253659389, 0.029295220245]
    M_ref[k][(1, 0, 0)][3, 8, 21, :] = [-0.111794616089, -0.097673124967]
    M_ref[k][(1, 0, 0)][3, 8, 26, :] = [0.151662783287, 0.160264457632]
    M_ref[k][(1, 0, 0)][3, 8, 27, :] = [-0.006022765057, -0.014199444382]
    M_ref[k][(1, 0, 0)][3, 8, 30, :] = [0.051888689046, 0.043103732833]
    M_ref[k][(1, 0, 0)][3, 8, 35, :] = [0.007909028552, 0.002960335801]
    M_ref[k][(1, 0, 1)][0, 0, 0, :] = [-0.003143434077, -0.005987369989]
    M_ref[k][(1, 0, 1)][0, 0, 3, :] = [-0.001906298924, -0.005177626798]
    M_ref[k][(1, 0, 1)][0, 0, 8, :] = [-0.000772878450, -0.003369068318]
    M_ref[k][(1, 0, 1)][0, 0, 9, :] = [-0.004952936269, -0.003167781504]
    M_ref[k][(1, 0, 1)][0, 0, 12, :] = [-0.007482287796, -0.005967493848]
    M_ref[k][(1, 0, 1)][0, 0, 17, :] = [-0.004312549641, -0.003907703848]
    M_ref[k][(1, 0, 1)][0, 0, 18, :] = [0.100801908177, 0.104100686578]
    M_ref[k][(1, 0, 1)][0, 0, 21, :] = [-0.016837925628, -0.012969351572]
    M_ref[k][(1, 0, 1)][0, 0, 26, :] = [-0.007249644339, -0.004893220792]
    M_ref[k][(1, 0, 1)][0, 0, 27, :] = [-0.022871321220, -0.025111317635]
    M_ref[k][(1, 0, 1)][0, 0, 30, :] = [0.004274440076, 0.001867806570]
    M_ref[k][(1, 0, 1)][0, 0, 35, :] = [0.001748991034, 0.000393299612]
    M_ref[k][(1, 0, 1)][0, 1, 1, :] = [-0.000592298925, -0.000685840587]
    M_ref[k][(1, 0, 1)][0, 1, 6, :] = [-0.001150931017, -0.001251392856]
    M_ref[k][(1, 0, 1)][0, 1, 10, :] = [0.000513859771, 0.000551099659]
    M_ref[k][(1, 0, 1)][0, 1, 15, :] = [0.001932614467, 0.001911430086]
    M_ref[k][(1, 0, 1)][0, 1, 19, :] = [0.072589559149, 0.072896960649]
    M_ref[k][(1, 0, 1)][0, 1, 24, :] = [-0.011292217072, -0.011036338342]
    M_ref[k][(1, 0, 1)][0, 1, 28, :] = [0.010915221467, 0.010720907891]
    M_ref[k][(1, 0, 1)][0, 1, 33, :] = [-0.000009478106, -0.000158356022]
    M_ref[k][(1, 0, 1)][0, 2, 2, :] = [-0.000592298925, -0.000685840587]
    M_ref[k][(1, 0, 1)][0, 2, 5, :] = [-0.001150931017, -0.001251392856]
    M_ref[k][(1, 0, 1)][0, 2, 11, :] = [0.000513859771, 0.000551099659]
    M_ref[k][(1, 0, 1)][0, 2, 14, :] = [0.001932614467, 0.001911430086]
    M_ref[k][(1, 0, 1)][0, 2, 20, :] = [0.072589559149, 0.072896960649]
    M_ref[k][(1, 0, 1)][0, 2, 23, :] = [-0.011292217072, -0.011036338342]
    M_ref[k][(1, 0, 1)][0, 2, 29, :] = [0.010915221467, 0.010720907891]
    M_ref[k][(1, 0, 1)][0, 2, 32, :] = [-0.000009478106, -0.000158356022]
    M_ref[k][(1, 0, 1)][0, 3, 0, :] = [0.002156006517, 0.001317999807]
    M_ref[k][(1, 0, 1)][0, 3, 3, :] = [0.003450808122, 0.002486864209]
    M_ref[k][(1, 0, 1)][0, 3, 8, :] = [0.001853425510, 0.001088420743]
    M_ref[k][(1, 0, 1)][0, 3, 9, :] = [-0.006510587443, -0.005984565859]
    M_ref[k][(1, 0, 1)][0, 3, 12, :] = [-0.009737535064, -0.009291179192]
    M_ref[k][(1, 0, 1)][0, 3, 17, :] = [-0.005992186606, -0.005872892957]
    M_ref[k][(1, 0, 1)][0, 3, 18, :] = [-0.017697559034, -0.016725526429]
    M_ref[k][(1, 0, 1)][0, 3, 21, :] = [0.059686061990, 0.060825993098]
    M_ref[k][(1, 0, 1)][0, 3, 26, :] = [-0.010898526186, -0.010204172021]
    M_ref[k][(1, 0, 1)][0, 3, 27, :] = [0.009387746780, 0.008727699549]
    M_ref[k][(1, 0, 1)][0, 3, 30, :] = [0.016616879554, 0.015907730301]
    M_ref[k][(1, 0, 1)][0, 3, 35, :] = [0.000333048040, -0.000066425979]
    M_ref[k][(1, 0, 1)][1, 0, 1, :] = [0.001161080596, 0.000405465131]
    M_ref[k][(1, 0, 1)][1, 0, 6, :] = [0.001600302983, 0.000788787373]
    M_ref[k][(1, 0, 1)][1, 0, 10, :] = [-0.005290604423, -0.004989786206]
    M_ref[k][(1, 0, 1)][1, 0, 15, :] = [-0.006273906659, -0.006445030905]
    M_ref[k][(1, 0, 1)][1, 0, 19, :] = [0.042500521189, 0.044983664248]
    M_ref[k][(1, 0, 1)][1, 0, 24, :] = [-0.021845280715, -0.019778330838]
    M_ref[k][(1, 0, 1)][1, 0, 28, :] = [-0.001593931771, -0.003163567595]
    M_ref[k][(1, 0, 1)][1, 0, 33, :] = [0.000364586378, -0.000838027025]
    M_ref[k][(1, 0, 1)][1, 1, 0, :] = [-0.006594245289, -0.004265647770]
    M_ref[k][(1, 0, 1)][1, 1, 3, :] = [-0.008537670707, -0.005859127234]
    M_ref[k][(1, 0, 1)][1, 1, 7, :] = [-0.003631746393, -0.003674819979]
    M_ref[k][(1, 0, 1)][1, 1, 8, :] = [-0.004257217815, -0.002131473258]
    M_ref[k][(1, 0, 1)][1, 1, 9, :] = [0.014112867670, 0.012651193748]
    M_ref[k][(1, 0, 1)][1, 1, 12, :] = [0.019005836948, 0.017765532828]
    M_ref[k][(1, 0, 1)][1, 1, 16, :] = [0.007872563897, 0.005444131063]
    M_ref[k][(1, 0, 1)][1, 1, 17, :] = [0.010843561858, 0.010512076572]
    M_ref[k][(1, 0, 1)][1, 1, 18, :] = [0.085351744152, 0.082650724353]
    M_ref[k][(1, 0, 1)][1, 1, 21, :] = [-0.020283013678, -0.023450578711]
    M_ref[k][(1, 0, 1)][1, 1, 25, :] = [0.019928135432, 0.030672101937]
    M_ref[k][(1, 0, 1)][1, 1, 26, :] = [-0.005077567402, -0.007006992747]
    M_ref[k][(1, 0, 1)][1, 1, 27, :] = [-0.044637023614, -0.042802928100]
    M_ref[k][(1, 0, 1)][1, 1, 30, :] = [0.006091667994, 0.008062204949]
    M_ref[k][(1, 0, 1)][1, 1, 34, :] = [0.011348145294, 0.005181970367]
    M_ref[k][(1, 0, 1)][1, 1, 35, :] = [-0.007506120386, -0.006396088451]
    M_ref[k][(1, 0, 1)][1, 2, 4, :] = [-0.003631765674, -0.003674846891]
    M_ref[k][(1, 0, 1)][1, 2, 13, :] = [0.007872604140, 0.005444149292]
    M_ref[k][(1, 0, 1)][1, 2, 22, :] = [0.019928205244, 0.030672324490]
    M_ref[k][(1, 0, 1)][1, 2, 31, :] = [0.011348072822, 0.005181810001]
    M_ref[k][(1, 0, 1)][1, 3, 1, :] = [0.002518349818, 0.000499950626]
    M_ref[k][(1, 0, 1)][1, 3, 6, :] = [-0.001896735560, -0.004064455159]
    M_ref[k][(1, 0, 1)][1, 3, 10, :] = [-0.000107492202, 0.000696053086]
    M_ref[k][(1, 0, 1)][1, 3, 15, :] = [0.007651109421, 0.007194002526]
    M_ref[k][(1, 0, 1)][1, 3, 19, :] = [-0.043886318623, -0.037253349636]
    M_ref[k][(1, 0, 1)][1, 3, 24, :] = [0.040985963041, 0.046507197260]
    M_ref[k][(1, 0, 1)][1, 3, 28, :] = [0.019939671791, 0.015746862316]
    M_ref[k][(1, 0, 1)][1, 3, 33, :] = [0.002905287039, -0.000307132553]
    M_ref[k][(1, 0, 1)][2, 0, 2, :] = [0.001161080596, 0.000405465131]
    M_ref[k][(1, 0, 1)][2, 0, 5, :] = [0.001600302983, 0.000788787373]
    M_ref[k][(1, 0, 1)][2, 0, 11, :] = [-0.005290604423, -0.004989786206]
    M_ref[k][(1, 0, 1)][2, 0, 14, :] = [-0.006273906659, -0.006445030905]
    M_ref[k][(1, 0, 1)][2, 0, 20, :] = [0.042500521189, 0.044983664248]
    M_ref[k][(1, 0, 1)][2, 0, 23, :] = [-0.021845280715, -0.019778330838]
    M_ref[k][(1, 0, 1)][2, 0, 29, :] = [-0.001593931771, -0.003163567595]
    M_ref[k][(1, 0, 1)][2, 0, 32, :] = [0.000364586378, -0.000838027025]
    M_ref[k][(1, 0, 1)][2, 1, 4, :] = [-0.003631765674, -0.003674846891]
    M_ref[k][(1, 0, 1)][2, 1, 13, :] = [0.007872604140, 0.005444149292]
    M_ref[k][(1, 0, 1)][2, 1, 22, :] = [0.019928205244, 0.030672324490]
    M_ref[k][(1, 0, 1)][2, 1, 31, :] = [0.011348072822, 0.005181810001]
    M_ref[k][(1, 0, 1)][2, 2, 0, :] = [-0.006594245289, -0.004265647770]
    M_ref[k][(1, 0, 1)][2, 2, 3, :] = [-0.008537670707, -0.005859127234]
    M_ref[k][(1, 0, 1)][2, 2, 7, :] = [0.003631746393, 0.003674819979]
    M_ref[k][(1, 0, 1)][2, 2, 8, :] = [-0.004257217815, -0.002131473258]
    M_ref[k][(1, 0, 1)][2, 2, 9, :] = [0.014112867671, 0.012651193749]
    M_ref[k][(1, 0, 1)][2, 2, 12, :] = [0.019005836948, 0.017765532828]
    M_ref[k][(1, 0, 1)][2, 2, 16, :] = [-0.007872563897, -0.005444131063]
    M_ref[k][(1, 0, 1)][2, 2, 17, :] = [0.010843561858, 0.010512076572]
    M_ref[k][(1, 0, 1)][2, 2, 18, :] = [0.085351744152, 0.082650724353]
    M_ref[k][(1, 0, 1)][2, 2, 21, :] = [-0.020283013678, -0.023450578711]
    M_ref[k][(1, 0, 1)][2, 2, 25, :] = [-0.019928135432, -0.030672101937]
    M_ref[k][(1, 0, 1)][2, 2, 26, :] = [-0.005077567402, -0.007006992747]
    M_ref[k][(1, 0, 1)][2, 2, 27, :] = [-0.044637023614, -0.042802928100]
    M_ref[k][(1, 0, 1)][2, 2, 30, :] = [0.006091667994, 0.008062204949]
    M_ref[k][(1, 0, 1)][2, 2, 34, :] = [-0.011348145294, -0.005181970367]
    M_ref[k][(1, 0, 1)][2, 2, 35, :] = [-0.007506120386, -0.006396088451]
    M_ref[k][(1, 0, 1)][2, 3, 2, :] = [0.002518349818, 0.000499950626]
    M_ref[k][(1, 0, 1)][2, 3, 5, :] = [-0.001896735560, -0.004064455159]
    M_ref[k][(1, 0, 1)][2, 3, 11, :] = [-0.000107492202, 0.000696053086]
    M_ref[k][(1, 0, 1)][2, 3, 14, :] = [0.007651109421, 0.007194002526]
    M_ref[k][(1, 0, 1)][2, 3, 20, :] = [-0.043886318623, -0.037253349636]
    M_ref[k][(1, 0, 1)][2, 3, 23, :] = [0.040985963041, 0.046507197260]
    M_ref[k][(1, 0, 1)][2, 3, 29, :] = [0.019939671791, 0.015746862316]
    M_ref[k][(1, 0, 1)][2, 3, 32, :] = [0.002905287039, -0.000307132553]
    M_ref[k][(1, 0, 1)][3, 0, 0, :] = [-0.003385745856, -0.007295022603]
    M_ref[k][(1, 0, 1)][3, 0, 3, :] = [-0.000975656508, -0.005472426613]
    M_ref[k][(1, 0, 1)][3, 0, 8, :] = [0.000113866439, -0.003454858382]
    M_ref[k][(1, 0, 1)][3, 0, 9, :] = [-0.013271976682, -0.010818101409]
    M_ref[k][(1, 0, 1)][3, 0, 12, :] = [-0.018502621612, -0.016420384551]
    M_ref[k][(1, 0, 1)][3, 0, 17, :] = [-0.011162874734, -0.010606373358]
    M_ref[k][(1, 0, 1)][3, 0, 18, :] = [0.166803391817, 0.171337895441]
    M_ref[k][(1, 0, 1)][3, 0, 21, :] = [-0.041124177670, -0.035806432451]
    M_ref[k][(1, 0, 1)][3, 0, 26, :] = [-0.010907692321, -0.007668550490]
    M_ref[k][(1, 0, 1)][3, 0, 27, :] = [-0.022833948748, -0.025913049744]
    M_ref[k][(1, 0, 1)][3, 0, 30, :] = [0.001686735087, -0.001621425397]
    M_ref[k][(1, 0, 1)][3, 0, 35, :] = [0.004926796927, 0.003063262333]
    M_ref[k][(1, 0, 1)][3, 1, 1, :] = [-0.000371272397, -0.000318857520]
    M_ref[k][(1, 0, 1)][3, 1, 6, :] = [-0.001118705339, -0.001062412829]
    M_ref[k][(1, 0, 1)][3, 1, 10, :] = [-0.000857073421, -0.000877940318]
    M_ref[k][(1, 0, 1)][3, 1, 15, :] = [0.001006123243, 0.001017993641]
    M_ref[k][(1, 0, 1)][3, 1, 19, :] = [0.097744553299, 0.097572304788]
    M_ref[k][(1, 0, 1)][3, 1, 24, :] = [-0.023375316686, -0.023518695070]
    M_ref[k][(1, 0, 1)][3, 1, 28, :] = [0.045763516696, 0.045872397830]
    M_ref[k][(1, 0, 1)][3, 1, 33, :] = [-0.008622902257, -0.008539480415]
    M_ref[k][(1, 0, 1)][3, 2, 2, :] = [-0.000371272397, -0.000318857520]
    M_ref[k][(1, 0, 1)][3, 2, 5, :] = [-0.001118705339, -0.001062412829]
    M_ref[k][(1, 0, 1)][3, 2, 11, :] = [-0.000857073421, -0.000877940318]
    M_ref[k][(1, 0, 1)][3, 2, 14, :] = [0.001006123243, 0.001017993641]
    M_ref[k][(1, 0, 1)][3, 2, 20, :] = [0.097744553299, 0.097572304788]
    M_ref[k][(1, 0, 1)][3, 2, 23, :] = [-0.023375316686, -0.023518695070]
    M_ref[k][(1, 0, 1)][3, 2, 29, :] = [0.045763516696, 0.045872397830]
    M_ref[k][(1, 0, 1)][3, 2, 32, :] = [-0.008622902257, -0.008539480415]
    M_ref[k][(1, 0, 1)][3, 3, 0, :] = [0.000691204021, 0.001020052125]
    M_ref[k][(1, 0, 1)][3, 3, 3, :] = [0.002518094303, 0.002896362310]
    M_ref[k][(1, 0, 1)][3, 3, 8, :] = [0.001482338613, 0.001782539511]
    M_ref[k][(1, 0, 1)][3, 3, 9, :] = [-0.001126957954, -0.001333377782]
    M_ref[k][(1, 0, 1)][3, 3, 12, :] = [-0.004943768264, -0.005118925909]
    M_ref[k][(1, 0, 1)][3, 3, 17, :] = [-0.003719792417, -0.003766605276]
    M_ref[k][(1, 0, 1)][3, 3, 18, :] = [-0.069986900633, -0.070368342781]
    M_ref[k][(1, 0, 1)][3, 3, 21, :] = [0.087908196855, 0.087460868476]
    M_ref[k][(1, 0, 1)][3, 3, 26, :] = [-0.019582175623, -0.019854652021]
    M_ref[k][(1, 0, 1)][3, 3, 27, :] = [0.032616354833, 0.032875368608]
    M_ref[k][(1, 0, 1)][3, 3, 30, :] = [0.051580747437, 0.051859029680]
    M_ref[k][(1, 0, 1)][3, 3, 35, :] = [-0.012430735489, -0.012273975081]
    M_ref[k][(1, 1, 0)][0, 0, 0, :] = [-0.002628189648, 0.001714600782]
    M_ref[k][(1, 1, 0)][0, 0, 3, :] = [-0.006856165311, -0.001860732282]
    M_ref[k][(1, 1, 0)][0, 0, 8, :] = [-0.003193729059, 0.000770744424]
    M_ref[k][(1, 1, 0)][0, 0, 9, :] = [0.040024193511, 0.037298199231]
    M_ref[k][(1, 1, 0)][0, 0, 12, :] = [0.048755649472, 0.046442505694]
    M_ref[k][(1, 1, 0)][0, 0, 17, :] = [0.024140022841, 0.023521809037]
    M_ref[k][(1, 1, 0)][0, 0, 18, :] = [0.127077223202, 0.122039872250]
    M_ref[k][(1, 1, 0)][0, 0, 21, :] = [-0.020465811708, -0.026373260718]
    M_ref[k][(1, 1, 0)][0, 0, 26, :] = [0.002801862981, -0.000796478876]
    M_ref[k][(1, 1, 0)][0, 0, 27, :] = [-0.046455122722, -0.043034569201]
    M_ref[k][(1, 1, 0)][0, 0, 30, :] = [0.003261721785, 0.006936736019]
    M_ref[k][(1, 1, 0)][0, 0, 35, :] = [-0.004254077497, -0.002183888886]
    M_ref[k][(1, 1, 0)][0, 1, 1, :] = [-0.005377111329, 0.000983291898]
    M_ref[k][(1, 1, 0)][0, 1, 6, :] = [-0.004006943881, 0.002823999566]
    M_ref[k][(1, 1, 0)][0, 1, 10, :] = [0.037541464989, 0.035009323646]
    M_ref[k][(1, 1, 0)][0, 1, 15, :] = [0.030805379732, 0.032245820345]
    M_ref[k][(1, 1, 0)][0, 1, 19, :] = [0.127149317219, 0.106247427482]
    M_ref[k][(1, 1, 0)][0, 1, 24, :] = [-0.008705980329, -0.026104558416]
    M_ref[k][(1, 1, 0)][0, 1, 28, :] = [-0.020933590520, -0.007721160082]
    M_ref[k][(1, 1, 0)][0, 1, 33, :] = [-0.006859163445, 0.003263850883]
    M_ref[k][(1, 1, 0)][0, 2, 2, :] = [-0.005377111329, 0.000983291898]
    M_ref[k][(1, 1, 0)][0, 2, 5, :] = [-0.004006943881, 0.002823999566]
    M_ref[k][(1, 1, 0)][0, 2, 11, :] = [0.037541464989, 0.035009323646]
    M_ref[k][(1, 1, 0)][0, 2, 14, :] = [0.030805379732, 0.032245820345]
    M_ref[k][(1, 1, 0)][0, 2, 20, :] = [0.127149317219, 0.106247427482]
    M_ref[k][(1, 1, 0)][0, 2, 23, :] = [-0.008705980329, -0.026104558416]
    M_ref[k][(1, 1, 0)][0, 2, 29, :] = [-0.020933590520, -0.007721160082]
    M_ref[k][(1, 1, 0)][0, 2, 32, :] = [-0.006859163445, 0.003263850883]
    M_ref[k][(1, 1, 0)][0, 3, 0, :] = [0.023774674729, 0.026697833673]
    M_ref[k][(1, 1, 0)][0, 3, 3, :] = [0.016412189090, 0.019774645781]
    M_ref[k][(1, 1, 0)][0, 3, 8, :] = [0.005719837580, 0.008388349060]
    M_ref[k][(1, 1, 0)][0, 3, 9, :] = [-0.081538280613, -0.083373164127]
    M_ref[k][(1, 1, 0)][0, 3, 12, :] = [-0.067239203588, -0.068796194890]
    M_ref[k][(1, 1, 0)][0, 3, 17, :] = [-0.025222951433, -0.025639074946]
    M_ref[k][(1, 1, 0)][0, 3, 18, :] = [0.007094638597, 0.003703966694]
    M_ref[k][(1, 1, 0)][0, 3, 21, :] = [0.074972948193, 0.070996607937]
    M_ref[k][(1, 1, 0)][0, 3, 26, :] = [-0.039568207303, -0.041990273335]
    M_ref[k][(1, 1, 0)][0, 3, 27, :] = [-0.002996194886, -0.000693799276]
    M_ref[k][(1, 1, 0)][0, 3, 30, :] = [0.006519891094, 0.008993565774]
    M_ref[k][(1, 1, 0)][0, 3, 35, :] = [0.009693717291, 0.011087173976]
    M_ref[k][(1, 1, 0)][0, 4, 4, :] = [-0.009833838166, -0.009951443380]
    M_ref[k][(1, 1, 0)][0, 4, 13, :] = [0.022792563169, 0.016163248409]
    M_ref[k][(1, 1, 0)][0, 4, 22, :] = [0.121030080822, 0.150359902762]
    M_ref[k][(1, 1, 0)][0, 4, 31, :] = [-0.002578230430, -0.019411196111]
    M_ref[k][(1, 1, 0)][0, 5, 2, :] = [0.009674044136, 0.003372612880]
    M_ref[k][(1, 1, 0)][0, 5, 5, :] = [0.001783754837, -0.004983853915]
    M_ref[k][(1, 1, 0)][0, 5, 11, :] = [-0.013259946416, -0.010751282416]
    M_ref[k][(1, 1, 0)][0, 5, 14, :] = [-0.000029032757, -0.001456117987]
    M_ref[k][(1, 1, 0)][0, 5, 20, :] = [-0.087850270320, -0.067142177357]
    M_ref[k][(1, 1, 0)][0, 5, 23, :] = [0.120649339028, 0.137886602118]
    M_ref[k][(1, 1, 0)][0, 5, 29, :] = [0.046387623837, 0.033297695547]
    M_ref[k][(1, 1, 0)][0, 5, 32, :] = [-0.006251691335, -0.016280847761]
    M_ref[k][(1, 1, 0)][0, 6, 1, :] = [0.009674044136, 0.003372612880]
    M_ref[k][(1, 1, 0)][0, 6, 6, :] = [0.001783754837, -0.004983853915]
    M_ref[k][(1, 1, 0)][0, 6, 10, :] = [-0.013259946416, -0.010751282416]
    M_ref[k][(1, 1, 0)][0, 6, 15, :] = [-0.000029032757, -0.001456117987]
    M_ref[k][(1, 1, 0)][0, 6, 19, :] = [-0.087850270320, -0.067142177357]
    M_ref[k][(1, 1, 0)][0, 6, 24, :] = [0.120649339028, 0.137886602118]
    M_ref[k][(1, 1, 0)][0, 6, 28, :] = [0.046387623837, 0.033297695547]
    M_ref[k][(1, 1, 0)][0, 6, 33, :] = [-0.006251691335, -0.016280847761]
    M_ref[k][(1, 1, 0)][0, 7, 7, :] = [-0.009833821318, -0.009951408030]
    M_ref[k][(1, 1, 0)][0, 7, 16, :] = [0.022792533393, 0.016163147291]
    M_ref[k][(1, 1, 0)][0, 7, 25, :] = [0.121030120056, 0.150360106550]
    M_ref[k][(1, 1, 0)][0, 7, 34, :] = [-0.002578322824, -0.019411382313]
    M_ref[k][(1, 1, 0)][0, 8, 0, :] = [0.009975695284, 0.000127869611]
    M_ref[k][(1, 1, 0)][0, 8, 3, :] = [0.016406072557, 0.005078297379]
    M_ref[k][(1, 1, 0)][0, 8, 8, :] = [0.008827190731, -0.000162753495]
    M_ref[k][(1, 1, 0)][0, 8, 9, :] = [-0.017928088143, -0.011746551895]
    M_ref[k][(1, 1, 0)][0, 8, 12, :] = [-0.028911899134, -0.023666553531]
    M_ref[k][(1, 1, 0)][0, 8, 17, :] = [-0.016569971709, -0.015168093845]
    M_ref[k][(1, 1, 0)][0, 8, 18, :] = [-0.020614576788, -0.009191747426]
    M_ref[k][(1, 1, 0)][0, 8, 21, :] = [-0.067671066877, -0.054275180301]
    M_ref[k][(1, 1, 0)][0, 8, 26, :] = [0.105549894215, 0.113709588747]
    M_ref[k][(1, 1, 0)][0, 8, 27, :] = [0.015583355020, 0.007826817990]
    M_ref[k][(1, 1, 0)][0, 8, 30, :] = [0.034865760552, 0.026532201724]
    M_ref[k][(1, 1, 0)][0, 8, 35, :] = [-0.003028331664, -0.007722745747]
    M_ref[k][(1, 1, 1)][0, 0, 0, :] = [-0.004540383803, -0.006746557229]
    M_ref[k][(1, 1, 1)][0, 0, 3, :] = [-0.003151873139, -0.005689594399]
    M_ref[k][(1, 1, 1)][0, 0, 8, :] = [-0.001174947840, -0.003188933131]
    M_ref[k][(1, 1, 1)][0, 0, 9, :] = [-0.004616815523, -0.003231987901]
    M_ref[k][(1, 1, 1)][0, 0, 12, :] = [-0.007256211985, -0.006081115831]
    M_ref[k][(1, 1, 1)][0, 0, 17, :] = [-0.004532146351, -0.004218088629]
    M_ref[k][(1, 1, 1)][0, 0, 18, :] = [0.110566876018, 0.113125891929]
    M_ref[k][(1, 1, 1)][0, 0, 21, :] = [-0.024178637919, -0.021177605004]
    M_ref[k][(1, 1, 1)][0, 0, 26, :] = [-0.009396334525, -0.007568347126]
    M_ref[k][(1, 1, 1)][0, 0, 27, :] = [-0.019678747919, -0.021416417375]
    M_ref[k][(1, 1, 1)][0, 0, 30, :] = [0.002837612120, 0.000970674520]
    M_ref[k][(1, 1, 1)][0, 0, 35, :] = [0.000353347729, -0.000698325192]
    M_ref[k][(1, 1, 1)][0, 1, 1, :] = [-0.000595959162, -0.000525507336]
    M_ref[k][(1, 1, 1)][0, 1, 6, :] = [-0.001497469566, -0.001421805740]
    M_ref[k][(1, 1, 1)][0, 1, 10, :] = [0.000274566383, 0.000246518794]
    M_ref[k][(1, 1, 1)][0, 1, 15, :] = [0.002431882270, 0.002447837496]
    M_ref[k][(1, 1, 1)][0, 1, 19, :] = [0.077964469621, 0.077732947146]
    M_ref[k][(1, 1, 1)][0, 1, 24, :] = [-0.021547182014, -0.021739899606]
    M_ref[k][(1, 1, 1)][0, 1, 28, :] = [0.019577220185, 0.019723569373]
    M_ref[k][(1, 1, 1)][0, 1, 33, :] = [-0.000863191322, -0.000751062449]
    M_ref[k][(1, 1, 1)][0, 2, 2, :] = [-0.000595959162, -0.000525507336]
    M_ref[k][(1, 1, 1)][0, 2, 5, :] = [-0.001497469566, -0.001421805740]
    M_ref[k][(1, 1, 1)][0, 2, 11, :] = [0.000274566383, 0.000246518794]
    M_ref[k][(1, 1, 1)][0, 2, 14, :] = [0.002431882270, 0.002447837496]
    M_ref[k][(1, 1, 1)][0, 2, 20, :] = [0.077964469621, 0.077732947146]
    M_ref[k][(1, 1, 1)][0, 2, 23, :] = [-0.021547182014, -0.021739899606]
    M_ref[k][(1, 1, 1)][0, 2, 29, :] = [0.019577220185, 0.019723569373]
    M_ref[k][(1, 1, 1)][0, 2, 32, :] = [-0.000863191322, -0.000751062449]
    M_ref[k][(1, 1, 1)][0, 3, 0, :] = [0.004887054414, 0.004126886528]
    M_ref[k][(1, 1, 1)][0, 3, 3, :] = [0.007095888846, 0.006221481521]
    M_ref[k][(1, 1, 1)][0, 3, 8, :] = [0.003694267674, 0.003000320896]
    M_ref[k][(1, 1, 1)][0, 3, 9, :] = [-0.012890408160, -0.012413246451]
    M_ref[k][(1, 1, 1)][0, 3, 12, :] = [-0.018453287355, -0.018048391553]
    M_ref[k][(1, 1, 1)][0, 3, 17, :] = [-0.011008152891, -0.010899939914]
    M_ref[k][(1, 1, 1)][0, 3, 18, :] = [-0.032648060706, -0.031766316011]
    M_ref[k][(1, 1, 1)][0, 3, 21, :] = [0.050444760560, 0.051478808390]
    M_ref[k][(1, 1, 1)][0, 3, 26, :] = [-0.016338253026, -0.015708394422]
    M_ref[k][(1, 1, 1)][0, 3, 27, :] = [0.016600756733, 0.016002018438]
    M_ref[k][(1, 1, 1)][0, 3, 30, :] = [0.031337764810, 0.030694485369]
    M_ref[k][(1, 1, 1)][0, 3, 35, :] = [-0.001535680340, -0.001898048943]

    Naux1 = atom_O.aux_basis.get_size()
    Naux2 = atom_Li.aux_basis.get_size()

    msg = 'Too large error for M_{0}({1}) (value={2})'
    tol = 1e-4

    for index, constraint_method in enumerate(['original', 'reduced']):
        momoff2c = Offsite2cMTable(atom_O, atom_Li,
                                   constraint_method=constraint_method)
        momoff2c.run(rmin=rmin, dr=dr, N=N, ntheta=600, nr=200,
                    smoothen_tails=False)
        M = momoff2c.tables

        for key, ref in M_ref[(size, xc)].items():
            p, bas1, bas2 = key
            Naux = Naux1 if p == 0 else Naux2
            val = M[key][:, :, 0, :]

            M_diff = np.max(np.abs(val - ref[:, :, :Naux, index]))
            assert M_diff < tol, msg.format(key, constraint_method, str(val))

    return


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atoms', [DZP_PBE], indirect=True)
def test_rep2c(atoms, R):
    from hotcent.repulsion_twocenter import Repulsion2cTable

    atom_Li, atom_O = atoms
    xc = atom_O.xcname
    rmin, dr, N = R, R, 2

    Erep = {}
    for pair in [(atom_Li, atom_O), (atom_O, atom_Li)]:
        label = (pair[0].symbol, pair[1].symbol)
        rep2c = Repulsion2cTable(*pair)
        rep2c.run(rmin=rmin, dr=dr, N=N, xc=xc, smoothen_tails=False,
                  shift=False, ntheta=900, nr=300)
        Erep[label] = rep2c.erep[0]

    Erep_ref = 0.64333297

    tol = 2e-5
    msg = 'Too large error for E_rep {0} (value={1})'

    for label, val in Erep.items():
        diff = abs(val - Erep_ref)
        assert diff < tol, msg.format(label, val)


@pytest.fixture(scope='module')
def grids(request):
    all_grids = {
        R1: (R1, np.array([R1]), np.array([R1*np.sqrt(3.)/2]),
             np.array([np.pi/2])),
    }
    return all_grids[request.param]


@pytest.mark.parametrize('nphi', ['adaptive'])
@pytest.mark.parametrize('grids', [R1], indirect=True)
@pytest.mark.parametrize('atoms', [DZP_PBE], indirect=True)
def test_rep3c(nphi, grids, atoms):
    from hotcent.repulsion_threecenter import Repulsion3cTable

    atom_Li, atom_O = atoms
    xc = atom_O.xcname
    R, Rgrid, Sgrid, Tgrid = grids

    triplets = [
        (atom_Li, atom_Li, atom_O),
        (atom_Li, atom_O, atom_Li),
        (atom_Li, atom_O, atom_O),
        (atom_O, atom_Li, atom_Li),
        (atom_O, atom_Li, atom_O),
        (atom_O, atom_O, atom_Li),
    ]

    Erep = {}
    for atom1, atom2, atom3 in triplets:
        label = (atom1.symbol, atom2.symbol, atom3.symbol)
        rep3c = Repulsion3cTable(atom1, atom2)
        Erep[(R, label)] = rep3c.run(atom3, Rgrid, Sgrid=Sgrid, Tgrid=Tgrid,
                                     nphi=nphi, xc=xc, ntheta=300, nr=100,
                                     write=False)

    Erep_ref = {
        ('Li', 'Li', 'O'): -0.00201423,
        ('Li', 'O', 'Li'): -0.00201636,
        ('Li', 'O', 'O'): -0.00396291,
        ('O', 'Li', 'Li'): -0.00201636,
        ('O', 'Li', 'O'): -0.00396291,
        ('O', 'O', 'Li'): -0.00392420,
    }

    tol = 6e-5
    msg = 'Too large error for E_rep {0} (value={1})'

    for label, ref in Erep_ref.items():
        val = Erep[(R, label)][(label[0], label[1])]['s_s'][0][1]
        diff = abs(val - ref)
        assert diff < tol, msg.format(label, val)
