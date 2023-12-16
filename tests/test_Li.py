""" Tests for a pseudopotential with non-linear core corrections
by comparing with all-electron (frozen core) results (unless otherwise
mentioned). The core corrections are needed for a close agreement.
"""
import os
import pytest
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT
from hotcent.siesta_ion import write_ion


R1 = 4.0
PBE_LibXC = 'GGA_X_PBE+GGA_C_PBE'
LDA = 'LDA'


@pytest.fixture(scope='module')
def atoms(request):
    xcname = request.param

    wf_confinement = {'2s': SoftConfinement(rc=20.)}
    valence = list(wf_confinement.keys())

    kwargs = {
        'xc': xcname,
        'configuration': '[He] 2s1',
        'valence': valence,
        'confinement': None,
        'wf_confinement': wf_confinement,
        'perturbative_confinement': True,
        'scalarrel': False,
        'txt': None,
    }

    atom_ae = AtomicDFT('Li', **kwargs)
    atom_ae.run()
    atom_ae.pp.build_projectors(atom_ae)
    atom_ae.pp.build_overlaps(atom_ae, atom_ae, rmin=3., rmax=5., N=100)

    pp = KleinmanBylanderPP('./pseudos/Li.psf', valence=valence,
                            with_polarization=False, local_component='siesta',
                            rcore=2.4958, txt='-')
    atom_pp = PseudoAtomicDFT('Li', pp, **kwargs)
    atom_pp.run()
    atom_pp.pp.build_projectors(atom_pp)
    atom_pp.pp.build_overlaps(atom_pp, atom_pp, rmin=3., rmax=5., N=100)

    return (atom_ae, atom_pp)


@pytest.mark.parametrize('atoms', [PBE_LibXC], indirect=True)
def test_on1c(atoms):
    atom = atoms[1]
    assert atom.pp.lmax == 1

    # Check that we can write an .ion file
    write_ion(atom)
    filename = 'Li.gga.ion'
    assert os.path.exists(filename)
    os.remove(filename)

    # Reference (ref. energy, KB energy, KB cosine) values (Siesta v4.1.5):
    ref = {
        '2s': (-0.211265/2., 1.857598/2., 0.213116),
        '2p': (-0.080116/2., -0.951969/2., -0.127364),
    }
    msg = 'Too large difference for {0} ({1}) (PP: {2}, ref: {3})'
    labels = ['ref. energy', 'KB energy', 'KB cosine']

    for nl, references in ref.items():
        tol = {'2s': 1e-4, '2p': 1e-3}[nl]
        values = (atom.enl_free[nl], atom.pp.energies[nl], atom.pp.cosines[nl])

        for label, ref, val in zip(labels, references, values):
            diff = abs(val - ref)
            assert diff < tol, msg.format(label, nl, val, ref)

    # Check one-center integrals w.r.t. AE
    labels = ['AE', 'PP']
    H = {label: {} for label in labels}
    for atom, label in zip(atoms, labels):
        for nl in atom.valence:
            H[label][nl], S = atom.get_onecenter_integrals(nl, nl)

    msg = 'Too large difference for H_{0} (AE: {1}, PP: {2})'
    tol = 5e-5

    for nl, val_ae in H['AE'].items():
        val_pp = H['PP'][nl]
        diff = abs(val_ae - val_pp)
        assert diff < tol, msg.format(nl, val_ae, val_pp)


@pytest.mark.parametrize('atoms', [PBE_LibXC], indirect=True)
def test_hubbard(atoms):
    labels = ['AE', 'PP']
    U = {label: {} for label in labels}
    for atom, label in zip(atoms, labels):
        for nl in atom.valence:
            U[label][nl] = atom.get_hubbard_value(nl)

    msg = 'Too large difference for U_{0} (AE: {1}, PP: {2})'
    tol = 5e-4

    for nl, val_ae in U['AE'].items():
        val_pp = U['PP'][nl]
        diff = abs(val_ae - val_pp)
        assert diff < tol, msg.format(nl, val_ae, val_pp)


@pytest.mark.parametrize('atoms', [LDA, PBE_LibXC], indirect=True)
def test_hubbard_analytical(atoms):
    for atom, ae_or_pp in zip(atoms, ['AE', 'PP']):
        labels = ['analytical', 'numerical']
        U = {label: {} for label in labels}
        for label in labels:
            for nl in atom.valence:
                if label == 'analytical':
                    U[label][nl] = atom.get_analytical_hubbard_value(nl)
                elif label == 'numerical':
                    U[label][nl] = atom.get_hubbard_value(nl, maxstep=0.2)

        msg = 'Too large diff. for U_{0}-{1} (analytical: {2}, numerical: {3})'
        tol = 5e-4

        for nl, val_ana in U['analytical'].items():
            val_num = U['numerical'][nl]
            diff = abs(val_ana - val_num)
            assert diff < tol, msg.format(nl, ae_or_pp, val_ana, val_num)


@pytest.mark.parametrize('atoms', [PBE_LibXC], indirect=True)
def test_spin(atoms):
    labels = ['AE', 'PP']
    W = {label: {} for label in labels}
    for atom, label in zip(atoms, labels):
        for nl in atom.valence:
            W[label][nl] = atom.get_spin_constant(nl, nl)

    msg = 'Too large difference for W_{0} (AE: {1}, PP: {2})'
    tol = 5e-4

    for nl, val_ae in W['AE'].items():
        val_pp = W['PP'][nl]
        diff = abs(val_ae - val_pp)
        assert diff < tol, msg.format(nl, val_ae, val_pp)


@pytest.mark.parametrize('atoms', [LDA, PBE_LibXC], indirect=True)
def test_spin_analytical(atoms):
    for atom, ae_or_pp in zip(atoms, ['AE', 'PP']):
        labels = ['analytical', 'numerical']
        W = {label: {} for label in labels}
        for label in labels:
            for nl in atom.valence:
                if label == 'analytical':
                    W[label][nl] = atom.get_analytical_spin_constant(nl, nl)
                elif label == 'numerical':
                    W[label][nl] = atom.get_spin_constant(nl, nl, maxstep=0.2)

        msg = 'Too large diff. for W_{0}-{1} (analytical: {2}, numerical: {3})'
        tol = 5e-4

        for nl, val_ana in W['analytical'].items():
            val_num = W['numerical'][nl]
            diff = abs(val_ana - val_num)
            assert diff < tol, msg.format(nl, ae_or_pp, val_ana, val_num)


@pytest.mark.parametrize('R', [R1])
@pytest.mark.parametrize('atoms', [PBE_LibXC], indirect=True)
def test_rep2c(atoms, R):
    from hotcent.repulsion_twocenter import Repulsion2cTable

    xc = atoms[0].xcname
    rmin, dr, N = R, R, 3

    labels = ['AE', 'PP']
    Erep = {}
    for atom, label in zip(atoms, labels):
        rep2c = Repulsion2cTable(atom, atom)
        rep2c.run(rmin=rmin, dr=dr, N=N, xc=xc, smoothen_tails=False,
                  shift=False, ntheta=600, nr=200)
        Erep[label] = rep2c.erep[0]

    msg = 'Too large difference for E_rep (AE: {0}, PP: {1})'
    etol = 3e-4
    E_diff = abs(Erep['AE'] - Erep['PP'])
    assert E_diff < etol, msg.format(Erep['AE'], Erep['PP'])
