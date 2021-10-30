""" Old and very basic regression tests. """
import pytest
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import PowerConfinement


@pytest.fixture(scope='module')
def atoms():
    configurations = {
        'H': '1s1',
        'B': '[He] 2s2 2p1',
    }

    valences = {
        'H': ['1s'],
        'B': ['2s', '2p'],
    }

    confinements = {
        'H': PowerConfinement(r0=1.1, s=2),
        'B': PowerConfinement(r0=2.9, s=2),
    }

    atoms = []
    for element in ['B', 'H']:
        atom = AtomicDFT(element,
                         xc='LDA',
                         configuration=configurations[element],
                         valence=valences[element],
                         confinement=confinements[element],
                         txt=None,
                         )
        atom.run()
        atoms.append(atom)
    return atoms


msg = 'Too large error for {0} (value={1})'


def test_on1c(atoms):
    atom_B, atom_H = atoms
    eps = 1e-7

    E_tot = atom_B.get_energy()
    diff = abs(E_tot - -23.079723850586106)
    assert diff < eps, msg.format('Boron E_tot', E_tot)

    E_2s = atom_B.get_epsilon('2s')
    diff = abs(E_2s - 0.24990807910273996)
    assert diff < eps, msg.format('Boron E_2s', E_2s)

    E_2p = atom_B.get_epsilon('2p')
    diff = abs(E_2p - 0.47362603289831301)
    assert diff < eps, msg.format('Boron E_2p', E_2p)

    E_tot = atom_H.get_energy()
    diff = abs(E_tot - 0.58885808402033557)
    assert diff < eps, msg.format('Hydrogen E_tot', E_tot)

    E_1s = atom_H.get_epsilon('1s')
    diff = abs(E_1s - 1.00949638195278960)
    assert diff < eps, msg.format('Hydrogen E_1s', E_1s)


def test_off2c(atoms):
    from hotcent.slako import SlaterKosterTable

    atom_B, atom_H = atoms
    eps = 1e-7

    sk = SlaterKosterTable(atom_B, atom_H, txt=None)
    R, nt, nr = 2.0, 150, 50
    wf_range = sk.get_range(1e-7)
    grid, area = sk.make_grid(R, wf_range, nt=nt, nr=nr)

    # B-H sss integrals
    out = sk.calculate_mels([('sss', '2s', '1s')], atom_B, atom_H,
                            R, grid, area)
    results = {key: out[i][-1] for i, key in enumerate(['S', 'H', 'H2'])}
    references = {
        'S': -0.34627316,
        'H': 0.29069894,
        'H2': 0.29077536,
    }
    for key, ref in references.items():
        diff = abs(results[key] - ref)
        assert diff < eps, msg.format('B-H {0}_sss'.format(key), results[key])

    # H-B sps integrals
    out = sk.calculate_mels([('sps', '1s', '2p')], atom_H, atom_B,
                            R, grid, area)
    results = {key: out[i][-2] for i, key in enumerate(['S', 'H', 'H2'])}
    references = {
        'S': -0.47147340,
        'H': 0.33033262,
        'H2': 0.33028882,
    }
    for key, ref in references.items():
        diff = abs(results[key] - ref)
        assert diff < eps, msg.format('H-B {0}_sps'.format(key), results[key])
