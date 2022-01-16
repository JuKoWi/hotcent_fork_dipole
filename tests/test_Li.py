""" Tests for a pseudopotential with non-linear core corrections
by comparing with all-electron (frozen core) results. The core
corrections are needed for a close agreement.
"""
import pytest
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT


R1 = 4.0
PBE_LibXC = 'GGA_X_PBE+GGA_C_PBE'


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

    pp = KleinmanBylanderPP('./pseudos/Li.psf', valence)
    atom_pp = PseudoAtomicDFT('Li', pp, **kwargs)
    atom_pp.run()
    atom_pp.pp.build_projectors(atom_pp)
    atom_pp.pp.build_overlaps(atom_pp, atom_pp, rmin=3., rmax=5., N=100)

    return (atom_ae, atom_pp)


@pytest.mark.parametrize('atoms', [PBE_LibXC], indirect=True)
def test_on1c(atoms):
    labels = ['AE', 'PP']
    HU = {label: {} for label in labels}
    for atom, label in zip(atoms, labels):
        for nl in atom.valence:
            H, S = atom.get_onecenter_integrals(nl, nl)
            U = atom.get_hubbard_value(nl)
            HU[label][nl] = (H, U)

    msg = 'Too large difference for {0}_{1} (AE: {2}, PP: {3})'
    tol = {'H': 5e-5, 'U': 5e-4}

    for nl, vals_ae in HU['AE'].items():
        vals_pp = HU['PP'][nl]
        for key, val_ae, val_pp in zip(['H', 'U'], vals_ae, vals_pp):
            diff = abs(val_ae - val_pp)
            assert diff < tol[key], msg.format(key, nl, val_ae, val_pp)


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
