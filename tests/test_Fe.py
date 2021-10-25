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
