#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import os
import yaml
from argparse import ArgumentParser
from datetime import datetime
from ase.units import Ha
from hotcent import __version__
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT
from hotcent.siesta_ion import write_ion
from hotcent.utils import verify_chemical_symbols


def parse_arguments():
    description = """
    Generate a numerical basis set for the given element.

    Note: existing output files will be overwritten.
    """
    parser = ArgumentParser(description=description)
    parser.add_argument('symbol', type=str, help='Element symbol (e.g. "Si").')
    parser.add_argument('--configuration', '-c', type=str, required=True,
                        help='Electronic configuration, e.g. "[Ne],3s2,3p2".')
    parser.add_argument('--energy-shift', '-E', type=str, default='0.2',
                        help='Comma-separated energy shifts in eV for use with'
                             ' --rcut_model=energy_shift_user (one value for '
                             'each valence subshell). If only a single value '
                             'is given (as in the default value of 0.2), it '
                             'is applied to all subshells.')
    parser.add_argument('-f', '--xcfunctional', type=str, default='LDA',
                        help='Exchange-correlation functional. Default: LDA.')
    parser.add_argument('--label', help='Label to use in the output file '
                        'names, which will be "<Symbol>[.<label>].ion/json".')
    parser.add_argument('--pseudo-label', help='Label to use when searching '
                        'for pseudopotential files. The expected file name '
                        'corresponds to "<Symbol>[.<pseudo-label>].psf".')
    parser.add_argument('--pseudo-path', default='.', help='Path to the '
                        'directory where the pseudopotential files are stored '
                        '(default: ".").')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Do not print to standard output.')
    parser.add_argument('--rchar-pol', help='Characteristic radius '
                        'in Bohr radii to use for polarization functions.')
    parser.add_argument('--rcut', type=str, help='Comma-separated cutoff radii'
                        ' in Bohr radii for every valence subshell. Only used '
                        '(and required) for --rcut-approach=user.')
    parser.add_argument('--rcut-approach', choices=['energy_shift_hubbard',
                        'energy_shift_user', 'user'],
                        default='energy_shift_user', help='Approach for '
                        'obtaining the cutoff radii for the (minimal) basis. '
                        'Choices are "energy_shift_hubbard" (shifts calculated'
                        ' from Hubbard parameters), "energy_shift_user" ('
                        'shifts supplied by the user) and "user" (cutoff radii'
                        ' supplied by the user). Default: energy_shift_user.')
    parser.add_argument('--rmin', help='Smallest radius in the radial grid, '
                        'to be used in case the default setting does not '
                        'yield well-behaved results.')
    parser.add_argument('--tail-norm', '-T', type=float, default=0.16,
                        help='Tail norm to use for the second-zeta functions.'
                             ' (default: 0.16).')
    parser.add_argument('--type', '-t', type=str, default='dzp',
                        choices=['sz', 'szp', 'dz', 'dzp'],
                        help='Basis set type (default: dzp).')
    parser.add_argument('--valence', '-v', type=str, required=True,
                        help='Comma-separated subshells to include in the '
                             '(minimal) basis (e.g. "3s,3p").')
    parser.add_argument('--vconf-amplitude', type=float, default=12.0,
                        help='Soft confinement amplitude in Hartree (default: '
                             '12.0).')
    parser.add_argument('--vconf-rstart-rel', type=float, default=0.6,
                        help='Soft confinement start radius, relative to the '
                             'cutoff radius (default: 0.6).')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    args = parser.parse_args()
    return args


def find_energy_shifts(atom, model='hubbard'):
    shifts = {}
    for nl in atom.valence:
        if model == 'hubbard':
            U = atom.get_analytical_hubbard_value(nl)

            if nl[1] in ['s', 'p']:
                shift = (0.003/U**2 + 0.003/U)
            elif nl[1] in ['d']:
                shift = 0.006
            else:
                raise NotImplementedError(nl)
        else:
            raise NotImplementedError(model)

        shifts[nl] = shift * Ha

    return shifts


def find_cutoff_radii(symbol, pp, rcut_approach, amp, x_ri, atom_kwargs,
                      shifts=None, verbose=True):
    assert rcut_approach in ['energy_shift_hubbard', 'energy_shift_user']

    conf_kwargs = dict(amp=amp, x_ri=x_ri, rcuts=50.)
    atom = get_atom(symbol, pp, atom_kwargs, verbose=verbose, **conf_kwargs)
    atom.run()

    if rcut_approach == 'energy_shift_user':
        assert shifts is not None
    elif rcut_approach == 'energy_shift_hubbard':
        shifts = find_energy_shifts(atom, model='hubbard')

    rcuts = {}
    for nl in atom_kwargs['valence']:
        rcut = atom.find_cutoff_radius(nl, energy_shift=shifts[nl],
                                       amp=amp, x_ri=x_ri)
        rcuts[nl] = float(rcut)

    return rcuts


def find_polarization_radius(symbol, pp, amp, x_ri, atom_kwargs, verbose=True):
    conf_kwargs = dict(amp=amp, x_ri=x_ri, rcuts=50.)
    atom = get_atom(symbol, pp, atom_kwargs, verbose=verbose, **conf_kwargs)
    r_pol = float(atom.find_polarization_radius())
    return r_pol


def generate_basis(symbol, pp, atom_kwargs, basis_kwargs, conf_kwargs,
                   verbose=True):
    atom = get_atom(symbol, pp, atom_kwargs, verbose=verbose, **conf_kwargs)
    atom.run()
    atom.generate_nonminimal_basis(**basis_kwargs)
    atom.pp.build_projectors(atom)
    return atom


def get_atom(symbol, pp, atom_kwargs, verbose, amp, x_ri, rcuts):
    wf_conf = {}
    for nl in atom_kwargs['valence']:
        rc = rcuts if isinstance(rcuts, (int, float)) else rcuts[nl]
        wf_conf[nl] = SoftConfinement(amp=amp, x_ri=x_ri, rc=rc)

    atom = PseudoAtomicDFT(symbol, pp, txt='-' if verbose else None,
                           wf_confinement=wf_conf, **atom_kwargs)
    return atom


def sanity_check_atom(symbol, pp, amp, x_ri, atom_kwargs, verbose=True):
    conf_kwargs = dict(amp=amp, x_ri=x_ri, rcuts=50.)
    atom = get_atom(symbol, pp, atom_kwargs, verbose=verbose, **conf_kwargs)
    atom.run()

    for nl in atom_kwargs['valence']:
        H1c, S1c = atom.get_onecenter_integrals(nl, nl)
        msg = 'Suspicious eigenvalue {0:.3f} for {1}_{2}. Consider ' + \
              're-running with a smaller rmin parameter (e.g. 1e-4).'
        assert abs(H1c) < 50., msg.format(H1c, symbol, nl)
    return


def write_yaml(symbol, atom_kwargs, basis_kwargs, conf_kwargs, pp_kwargs,
               stem):
    timestamp = datetime.now().replace(microsecond=0).isoformat()
    meta_kwargs = dict(
        symbol=symbol,
        timestamp=timestamp,
        hotcent_version=__version__,
    )

    setup = [
        dict(meta=meta_kwargs),
        dict(atom=atom_kwargs),
        dict(basis=basis_kwargs),
        dict(confinement=conf_kwargs),
        dict(pseudopotential=pp_kwargs),
    ]

    filename = '{0}.yaml'.format(stem)

    with open(filename, 'w') as f:
        yaml.dump(setup, f)
    return


def main():
    args = parse_arguments()

    symbol = args.symbol
    verify_chemical_symbols(symbol)

    configuration = args.configuration.replace(',', ' ')
    pseudo_label = '' if args.pseudo_label is None else '.' + args.pseudo_label
    rmin = None if args.rmin is None else float(args.rmin)
    r_pol = None if args.rchar_pol is None else float(args.rchar_pol)
    verbose = not args.quiet
    valence = args.valence.split(',')
    amp = args.vconf_amplitude
    x_ri = args.vconf_rstart_rel

    with_polarization = args.type.endswith('p')

    if args.rcut_approach == 'user':
        assert args.rcut is not None, 'For --rcut-approach=user, ' + \
               'cutoff radii need to be supplied via --rcut.'

        rcuts = list(map(float, args.rcut.split(',')))
        if len(rcuts) == 1:
            rcuts *= len(valence)
        else:
            assert len(rcuts) == len(valence), 'When supplying more than ' + \
                   'one cutoff radius, the number of radii must be equal ' + \
                   'the number of valence subshells.'
        rcuts = {nl: rcut for nl, rcut in zip(valence, rcuts)}
    else:
        rcuts = None

        if args.rcut_approach == 'energy_shift_user':
            shifts = list(map(float, args.energy_shift.split(',')))
            if len(shifts) == 1:
                shifts *= len(valence)
            else:
                assert len(shifts) == len(valence), 'When supplying more ' + \
                       'than one energy shift, the number of shifts must ' + \
                       'be equal to the number of valence subshells.'
            shifts = {nl: shift for nl, shift in zip(valence, shifts)}
        else:
            shifts = None

    pp_kwargs = dict(
        valence=valence,
        with_polarization=with_polarization,
        local_component='siesta',
    )

    pp_filename = '{0}{1}.psf'.format(symbol, pseudo_label)
    pp_path = os.path.join(args.pseudo_path, pp_filename)
    if verbose:
        print('Using pseudopotential file {0}'.format(pp_path))
    pp = KleinmanBylanderPP(pp_path, verbose=verbose, **pp_kwargs)

    atom_kwargs = dict(
        configuration=configuration,
        nodegpts=1000,
        perturbative_confinement=True,
        rmin=rmin,
        scalarrel=True,
        valence=valence,
        xc=args.xcfunctional,
    )

    sanity_check_atom(symbol, pp, amp, x_ri, atom_kwargs, verbose=verbose)

    if with_polarization and r_pol is None:
        r_pol = find_polarization_radius(symbol, pp, amp, x_ri, atom_kwargs,
                                         verbose=verbose)

    basis_kwargs = dict(
        size=args.type,
        r_pol=r_pol,
        tail_norm=args.tail_norm,
    )

    if rcuts is None:
        rcuts = find_cutoff_radii(symbol, pp, args.rcut_approach, amp, x_ri,
                                  atom_kwargs, shifts=shifts, verbose=verbose)

    conf_kwargs = dict(amp=amp, x_ri=x_ri, rcuts=rcuts)

    atom = generate_basis(symbol, pp, atom_kwargs, basis_kwargs, conf_kwargs,
                          verbose=verbose)

    stem = symbol
    if args.label is not None:
        stem += '.' + args.label

    write_ion(atom, label=stem)

    pp_kwargs.update(filename=pp_filename)
    write_yaml(symbol, atom_kwargs, basis_kwargs, conf_kwargs, pp_kwargs, stem)
    return


if __name__ == '__main__':
    main()
