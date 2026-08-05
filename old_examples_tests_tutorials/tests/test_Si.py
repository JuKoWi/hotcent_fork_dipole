""" Test of soft confinement (Si, all-electron) with an LDA functional
and scalar relativistic corrections.

Reference values come from the GPAW's gpaw-basis and gpaw-setup tools.
"""
from ase.units import Ha
from hotcent.confinement import SoftConfinement
from hotcent.atomic_dft import AtomicDFT


def test_on1c():
    # Checking eigenvalues and their shifts upon confinement

    kwargs = {
        'xc': 'LDA',
        'configuration': '[Ne] 3s2 3p2',
        'valence': ['3s', '3p'],
        'confinement': None,
        'scalarrel': True,
        'timing': True,
        'txt': '-',
    }

    atom = AtomicDFT('Si',
                     wf_confinement=None,
                     perturbative_confinement=False,
                     **kwargs)
    atom.run()
    eps_free = {nl: atom.get_eigenvalue(nl) for nl in atom.valence}

    wf_confinement = {
        '3s': SoftConfinement(amp=12., rc=6.74, x_ri=0.6),
        '3p': SoftConfinement(amp=12., rc=8.70, x_ri=0.6),
    }
    atom = AtomicDFT('Si',
                     wf_confinement=wf_confinement,
                     perturbative_confinement=True,
                     **kwargs)
    atom.run()
    eps_conf = {nl: atom.get_eigenvalue(nl) for nl in atom.valence}

    # gpaw-setup Si -f LDA -a
    eps_ref = {
        '3s': -0.399754,
        '3p': -0.152954,
    }
    # gpaw-basis Si -f LDA -t sz
    shift_ref = {
        '3s': 0.104 / Ha,
        '3p': 0.103 / Ha,
    }

    msg = 'Too large difference for {0}_{1}: {2} (ref: {3})'

    for nl in atom.valence:
        diff = abs(eps_free[nl] - eps_ref[nl])
        assert diff < 1e-4, msg.format(nl, 'free', eps_free[nl], eps_ref[nl])

        shift = eps_conf[nl] - eps_free[nl]
        diff = abs(shift - shift_ref[nl])
        assert diff < 1e-4, msg.format(nl, 'shift', shift, shift_ref[nl])

    # Now check whether find_cutoff_radius() returns a similar cutoff
    for nl in atom.valence:
        rc_ref = wf_confinement[nl].rc
        rc = atom.find_cutoff_radius(nl, energy_shift=shift_ref[nl]*Ha,
                                     tolerance=1e-3)
        diff = abs(rc - rc_ref)
        assert diff < 5e-2, msg.format(nl, 'rc', rc, rc_ref)

    # Also check the characteristic radius for polarization functions
    rpol = atom.find_polarization_radius()
    rpol_ref = 1.838  # gpaw-basis Si -f LDA -t szp
    diff = abs(rpol - rpol_ref)
    assert diff < 2e-2, msg.format(nl, 'rpol', rpol, rpol_ref)
