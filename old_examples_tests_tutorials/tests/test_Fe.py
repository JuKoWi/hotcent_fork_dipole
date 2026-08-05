""" Test of soft confinement (Fe, all-electron) with a GGA functional
and scalar relativistic corrections.

Reference values come from the GPAW's gpaw-basis and gpaw-setup tools.
"""
from ase.units import Ha
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import SoftConfinement


def test_on1c():
    # Checking eigenvalues and their shifts upon confinement

    kwargs = {
        'xc': 'GGA_X_PBE+GGA_C_PBE',
        'configuration': '[Ar] 3d6 4s2 4p0',
        'valence': ['3d', '4s'],
        'confinement': None,
        'scalarrel': True,
        'timing': True,
        'txt': '-',
    }

    atom = AtomicDFT('Fe',
                     wf_confinement=None,
                     perturbative_confinement=False,
                     **kwargs)
    atom.run()
    eps_free = {nl: atom.get_eigenvalue(nl) for nl in atom.valence}

    wf_confinement = {
        '3d': SoftConfinement(amp=12., rc=5.11, x_ri=0.6),
        '4s': SoftConfinement(amp=12., rc=8.85, x_ri=0.6),
    }
    atom = AtomicDFT('Fe',
                     wf_confinement=wf_confinement,
                     perturbative_confinement=True,
                     **kwargs)
    atom.run()
    eps_conf = {nl: atom.get_eigenvalue(nl) for nl in atom.valence}

    # gpaw-setup Fe -f PBE -a
    eps_ref = {
        '4s': -0.194442,
        '3d': -0.275800,
    }
    # gpaw-basis Fe -f PBE -t sz
    shift_ref = {
        '4s': 0.098 / Ha,
        '3d': 0.100 / Ha,
    }

    for nl in atom.valence:
        diff = abs(eps_free[nl] - eps_ref[nl])
        assert diff < 5e-4, 'Too large error for {0}_free'.format(nl)

        shift = eps_conf[nl] - eps_free[nl]
        diff = abs(shift - shift_ref[nl])
        assert diff < 1e-4, 'Too large error for {0}_shift'.format(nl)


def test_spin():
    # Checking spin constants
    wf_confinement = {
        '3d': SoftConfinement(amp=12., rc=40., x_ri=0.6),
        '4s': SoftConfinement(amp=12., rc=40., x_ri=0.6),
        '4p': SoftConfinement(amp=12., rc=40., x_ri=0.6),
    }

    atom = AtomicDFT('Fe',
                     xc='GGA_X_PBE+GGA_C_PBE',
                     configuration='[Ar] 3d7 4s1 4p0',
                     valence=['3d', '4s', '4p'],
                     perturbative_confinement=True,
                     wf_confinement=wf_confinement,
                     scalarrel=True,
                     txt='-',
                     )
    atom.run()

    # Regression test; values compare well with those
    # from e.g. the pbc-0-3 DFTB parameter set
    W_ref = {
        ('3d', '3d'): -0.014418,
        ('3d', '4s'): -0.004625,
        ('3d', '4p'): -0.001953,
        ('4s', '3d'): -0.004584,
        ('4s', '4s'): -0.014979,
        ('4s', '4p'): -0.010887,
        ('4p', '3d'): -0.001987,
        ('4p', '4s'): -0.012411,
        ('4p', '4p'): -0.016399,
    }

    msg = 'Too large error for W_{0}-{1} (value={2})'
    tol = 1e-4

    for nl1 in atom.valence:
        for nl2 in atom.valence:
            W = atom.get_spin_constant(nl1, nl2, scheme=None, maxstep=0.5)
            W_diff = abs(W - W_ref[(nl1, nl2)])
            assert W_diff < tol, msg.format(nl1, nl2, W)
