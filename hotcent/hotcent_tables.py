#-----------------------------------------------------------------------------#
#   Hotcent: calculating one- and two-center Slater-Koster integrals,         #
#            based on parts of the Hotbit code                                #
#   Copyright 2018-2021 Maxime Van den Bossche                                #
#   SPDX-License-Identifier: GPL-3.0-or-later                                 #
#-----------------------------------------------------------------------------#
import os
import numpy as np
import sys
import traceback
import yaml
from argparse import ArgumentParser
from itertools import combinations_with_replacement, product
from multiprocessing import Pool
from time import time
from ase.data import atomic_numbers, covalent_radii
from ase.units import Bohr
from hotcent.confinement import SoftConfinement
from hotcent.kleinman_bylander import KleinmanBylanderPP
from hotcent.offsite_chargetransfer import Offsite2cMTable, Offsite2cUTable
from hotcent.offsite_magnetization import Offsite2cWTable
from hotcent.offsite_threecenter import Offsite3cTable
from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.onsite_chargetransfer import (Onsite1cMTable, Onsite1cUTable,
                                           Onsite2cUTable)
from hotcent.onsite_magnetization import Onsite1cWTable, Onsite2cWTable
from hotcent.onsite_threecenter import Onsite3cTable
from hotcent.onsite_twocenter import Onsite2cTable
from hotcent.pseudo_atomic_dft import PseudoAtomicDFT
from hotcent.repulsion_twocenter import Repulsion2cTable
from hotcent.repulsion_threecenter import Repulsion3cTable
from hotcent.utils import verify_chemical_symbols


def parse_arguments():
    description = """
    Generate integral tables for all selected element combinations.

    Note: existing output files will be overwritten.
    """
    parser = ArgumentParser(description=description)
    parser.add_argument('include', nargs='*', help='Element combinations '
                        'to consider. To e.g. include all combinations that '
                        'involve H and/or Si, as well as all combinations '
                        'involving only O, write "H,Si O".')
    parser.add_argument('--dry-run', action='store_true', help='Exit after '
                        'printing the task overview, without executing them.')
    parser.add_argument('--exclude', help='Element combinations to exclude. '
                        'To e.g. skip those involving only H or just H and '
                        'Si, write "H,H+Si". By default no combinations get '
                        'excluded.')
    parser.add_argument('--label', help='Label to use when searching for the '
                        'input YAML files. The expected file names correspond '
                        'to "<Symbol>[.<label>].yaml".')
    parser.add_argument('--multipole-subshells', help='Subshells to use for '
                        'fluctuation-type tasks (chg*, mag*) with multipoles, '
                        'instead of the default choice (subshells with lowest '
                        'angular momentum in every basis subset). To e.g. '
                        'select p-type subshells for Si in a DZ or DZP basis, '
                        'specify --multipole-subshells=Si_3p-3p+. Entries for '
                        'multiple elements need to be separated by commas.')
    parser.add_argument('--opts-2c', help='Option for controlling the two-'
                        'center integration grids. The default settings are '
                        'rather tight and correspond to --opts-2c=nr_200,'
                        'ntheta_600.')
    parser.add_argument('--opts-3c', help='Option for controlling the three-'
                        'center integration grids. The default settings are '
                        'rather tight and correspond to --opts-2c=nr_50,'
                        'ntheta_150,nphi_13.')
    parser.add_argument('--processes', type=int, default=1, help='Number of '
                        'processes to use for multiprocessing (default: 1).')
    parser.add_argument('--pseudo-path', default='.', help='Path to the '
                        'directory where the pseudopotential files are stored '
                        '(default: ".").')
    parser.add_argument('--tasks', default='all', help='Comma-separated task '
                        'types to perform. The following types can be chosen: '
                        + ', '.join(TaskGenerator.all_task_types) + '. The '
                        'default "all" selects every type. Types can be '
                        'deselected by prepending a caret (^). For example, '
                        '--tasks=all,^rep3c selects all available task types '
                        'except 3-center repulsion.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--without-monopoles', action='store_true',
                       help='For fluctuation-type tasks (chg*, mag*), do not '
                       'generate tables for monopole-only self-consistency.')
    group.add_argument('--without-multipoles', action='store_true',
                       help='For fluctuation-type tasks (chg*, mag*), do not '
                       'generate tables for multipolar self-consistency.')

    args = parser.parse_args()
    return args


class Task:
    def __init__(self, task_type, elements, workdir, task_kwargs):
        self.task_type = task_type
        self.elements = elements
        self.workdir = workdir
        self.task_kwargs = task_kwargs

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class TaskGenerator:
    all_task_types = [
        'chgoff2c', 'chgon1c', 'chgon2c',
        'magoff2c', 'magon1c', 'magon2c',
        'off2c', 'off3c',
        'on1c', 'on2c', 'on3c',
        'rep2c', 'rep3c',
    ]

    def __init__(self, included, excluded, task_types, task_kwargs):
        self.included = included
        self.excluded = excluded
        self.set_task_types(task_types)
        self.task_kwargs = task_kwargs
        self.generate_tasks()

    def get_task_types(self):
        task_types = [task_type for task_type in self.task_types]
        return task_types

    def set_task_types(self, task_types):
        negation = '^'

        for task_type in task_types:
            if task_type == 'all':
                self.task_types = [task for task in self.all_task_types]
                break
        else:
            self.task_types = []

        msg = 'Unknown task type: {0}'

        for task_type in task_types:
            if task_type != 'all' and not task_type.startswith(negation):
                assert task_type in self.all_task_types, msg.format(task_type)
                if task_type not in self.task_types:
                    self.task_types.append(task_type)

        for task_type in task_types:
            if task_type.startswith(negation):
                assert task_type[1:] in self.all_task_types, \
                       msg.format(task_type)
                if task_type[1:] in self.task_types:
                    index = self.task_types.index(task_type[1:])
                    self.task_types.pop(index)
        return

    def need_combination(self, elements):
        needed = True

        for combination in self.excluded:
            cond1 = all([el in combination for el in elements])
            cond2 = all([el in elements for el in combination])
            if cond1 and cond2:
                needed = False
                break

        return needed

    def need_task_type(self, task_type):
        needed = task_type in self.task_types
        return needed

    def need_task(self, task):
        needed = True

        if not self.need_task_type(task.task_type):
            needed = False

        if not self.need_combination(task.elements):
            needed = False

        if task in self.tasks:
            needed = False

        return needed

    def get_workdir(self, *elements, rootdir='.', prefix='tables_'):
        subdir = prefix + '_'.join(sorted(elements))
        workdir = os.path.join(rootdir, subdir)
        return workdir

    def generate_tasks(self):
        self.tasks = []

        for task in self.yield_all_tasks():
            if self.need_task(task):
                self.tasks.append(task)

        return

    def get_tasks(self):
        priority = [
            'off3c', 'rep3c', 'on3c', 'chgoff2c', 'chgon2c', 'magoff2c',
            'magon2c', 'off2c', 'rep2c', 'on2c', 'chgon1c', 'magon1c',
        ]
        tasks = sorted([task for task in self.tasks],
                       key=lambda x: priority.index(x.task_type))
        return tasks

    def yield_all_tasks(self):
        for elements in self.included:
            for el1 in elements:
                workdir = self.get_workdir(el1)
                for task_type in ['chgon1c', 'magon1c']:
                    yield Task(task_type, [el1], workdir, self.task_kwargs)

            for el1, el2 in combinations_with_replacement(elements, r=2):
                workdir = self.get_workdir(el1, el2)
                for task_type in ['chgoff2c', 'magoff2c', 'off2c', 'rep2c']:
                    yield Task(task_type, [el1, el2], workdir,
                               self.task_kwargs)

            for el1, el2 in product(elements, repeat=2):
                workdir = self.get_workdir(el1, el2)
                for task_type in ['chgon2c', 'magon2c', 'on2c']:
                    yield Task(task_type, [el1, el2], workdir,
                               self.task_kwargs)

            for el1, el2 in combinations_with_replacement(elements, r=2):
                for el3 in elements:
                    workdir = self.get_workdir(el1, el2, el3)
                    for task_type in ['off3c', 'rep3c']:
                        yield Task(task_type, [el1, el2, el3], workdir,
                                   self.task_kwargs)

            for el1, el2, el3 in product(elements, repeat=3):
                workdir = self.get_workdir(el1, el2, el3)
                for task_type in ['on3c']:
                    yield Task(task_type, [el1, el2, el3], workdir,
                               self.task_kwargs)
        return

    def print_task_overview(self):
        ntasks = len(self.tasks)
        print('Number of tasks: {0}'.format(ntasks))
        if ntasks > 0:
            print('Generated tasks (in order of execution):')

            fmt = '{0:<12} {1:<12} {2}'
            print(fmt.format('Task', 'Elements', 'Working directory'))
            print(fmt.format('='*12, '='*12, '='*17))

            for task in self.get_tasks():
                element_str = ' '.join(['{:<2}'.format(el)
                                        for el in task.elements])
                print(fmt.format(task.task_type, element_str, task.workdir))
        return

    def create_working_directories(self):
        for task in self.tasks:
            if not os.path.exists(task.workdir):
                os.mkdir(task.workdir)
        return


def main():
    args = parse_arguments()

    included = []
    for item in args.include:
        elements = tuple(set(item.split(',')))
        verify_chemical_symbols(*elements)
        included.append(elements)

    excluded = []
    if args.exclude is not None:
        for item in args.exclude.split(','):
            elements = tuple(set(item.split('+')))
            verify_chemical_symbols(*elements)
            excluded.append(elements)

    task_types = list(set(args.tasks.split(',')))

    parse_err = 'Cannot parse {0} entry: {1}'

    if args.multipole_subshells is None:
        multipole_subshells = None
    else:
        multipole_subshells = {}

        for entry in args.multipole_subshells.split(','):
            assert '_' in entry, parse_err.format('multipole-subshells', entry)
            symbol, value = entry.split('_', maxsplit=1)
            verify_chemical_symbols(symbol)
            multipole_subshells[symbol] = tuple(value.split('-'))

        for elements in included:
            for element in elements:
                if element not in multipole_subshells:
                    multipole_subshells[element] = None

    opts_2c = dict(nr=200, ntheta=600, smoothen_tails=True)
    if args.opts_2c is not None:
        for entry in args.opts_2c.split(','):
            assert '_' in entry, parse_err.format('opts-2c', entry)
            key, val = entry.split('_', maxsplit=1)
            assert key in opts_2c, 'Unknown opts-2c key: {0}'.format(key)
            opts_2c[key] = int(val)

    opts_3c = dict(nr=50, ntheta=150, nphi=13)
    if args.opts_3c is not None:
        for entry in args.opts_3c.split(','):
            assert '_' in entry, parse_err.format('opts-3c', entry)
            key, val = entry.split('_', maxsplit=1)
            assert key in opts_3c, 'Unknown opts-3c key: {0}'.format(key)
            opts_3c[key] = int(val)

    use_multipoles = []
    if not args.without_monopoles:
        use_multipoles.append(False)
    if not args.without_multipoles:
        use_multipoles.append(True)

    task_kwargs = dict(
        label=args.label,
        multipole_subshells=multipole_subshells,
        opts_2c=opts_2c,
        opts_3c=opts_3c,
        pseudo_path=args.pseudo_path,
        shift=True,
        use_multipoles=use_multipoles,
    )

    generator = TaskGenerator(included, excluded, task_types, task_kwargs)
    generator.print_task_overview()

    if args.dry_run:
        return

    generator.create_working_directories()
    tasks = generator.get_tasks()

    print('\nStarting pool with %d processes' % args.processes, flush=True)
    pool = Pool(processes=args.processes)
    pool.map_async(wrapper, tasks, chunksize=1, callback=callback)
    pool.close()
    pool.join()
    print('All done')
    return


def callback(messages):
    for message in messages:
        print(message, flush=True)
    return


def wrapper(task):
    element_str = '_'.join(task.elements)

    suffix = '{0}-{1}'.format(task.task_type, element_str)
    stdout = os.path.join(task.workdir, 'out_{0}.txt'.format(suffix))
    sys.stdout = open(stdout, 'w')
    stderr = os.path.join(task.workdir, 'err_{0}.txt'.format(suffix))
    sys.stderr = open(stderr, 'w')

    cwd = os.getcwd()
    os.chdir(task.workdir)
    message = 'Task {0} for elements {1}: '.format(task.task_type, element_str)

    try:
        function = globals()[task.task_type]

        t_start = time()
        function(*task.elements, **task.task_kwargs)
        t_end = time()

        t_delta = int(np.ceil(t_end - t_start))
        message += 'completed in {0} seconds'.format(t_delta)
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        message += 'failed (check {0})'.format(stderr)

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr.close()
    sys.stderr = sys.__stderr__

    os.chdir(cwd)
    return message


def read_yaml(filename):
    with open(filename, 'r') as f:
        setup = yaml.safe_load(f)

    dicts = []
    for key in ['meta', 'atom', 'basis', 'confinement', 'pseudopotential']:
        for section in setup:
            if key in section:
                dicts.append(section[key])
                break
        else:
            raise ValueError('No "{0}" section in {1}'.format(key, filename))
    return dicts


def get_atoms(*elements, label=None, only_1c=False, pseudo_path='.', txt='-',
              yaml_path='.'):
    atoms = {}
    eigenvalues, hubbardvalues, occupations = {}, {}, {}
    offdiagonal_H, offdiagonal_S = {}, {}

    xc = None

    for el in set(elements):
        if label is None:
            filename = '{0}.yaml'.format(el)
        else:
            filename = '{0}.{1}.yaml'.format(el, label)
        filename = os.path.join(yaml_path, filename)

        meta_kwargs, atom_kwargs, basis_kwargs, conf_kwargs, pp_kwargs = \
            read_yaml(filename)

        if xc is None:
            xc = atom_kwargs['xc']
        else:
            assert xc == atom_kwargs['xc'], \
                   'The XC functional must be the same in all YAML files.'

        wf_conf = {}
        for nl, rc in conf_kwargs['rcuts'].items():
            wf_conf[nl] = SoftConfinement(amp=conf_kwargs['amp'],
                                          x_ri=conf_kwargs['x_ri'], rc=rc)

        pp_kwargs['filename'] = os.path.join(pseudo_path,
                                             pp_kwargs['filename'])
        pp = KleinmanBylanderPP(verbose=True, **pp_kwargs)

        atom = PseudoAtomicDFT(el, pp, txt=txt, wf_confinement=wf_conf,
                               **atom_kwargs)
        atom.run()
        atom.generate_nonminimal_basis(**basis_kwargs)

        eigenvalues[el], hubbardvalues[el], occupations[el] = {}, {}, {}
        offdiagonal_H[el], offdiagonal_S[el] = {}, {}

        for i, valence1 in enumerate(atom.basis_sets):
            for nl1 in valence1:
                if i == 0 and nl1 in atom.valence:
                    occupations[el][nl1] = atom.configuration[nl1]
                else:
                    occupations[el][nl1] = 0.

                H1c, S1c = atom.get_onecenter_integrals(nl1, nl1)
                eigenvalues[el][nl1] = H1c

                for j, valence2 in enumerate(atom.basis_sets):
                    if i != j:
                        for nl2 in valence2:
                            H1c, S1c = atom.get_onecenter_integrals(nl1, nl2)
                            offdiagonal_H[el][(nl1, nl2)] = H1c
                            offdiagonal_S[el][(nl1, nl2)] = S1c

        atoms[el] = atom

        for i, valence in enumerate(atom.basis_sets):
            for nl in valence:
                hubbardvalues[el][nl] = atom.get_analytical_hubbard_value(nl)

    properties = dict(
        occupations=occupations,
        hubbardvalues=hubbardvalues,
        eigenvalues=eigenvalues,
        offdiagonal_H=offdiagonal_H,
        offdiagonal_S=offdiagonal_S,
    )

    for el1 in set(elements):
        atoms[el1].pp.build_projectors(atoms[el1])

    if not only_1c:
        dr = 0.02
        cov_frac = 0.75
        rmin_halves = {}
        for el in set(elements):
            rcov = cov_frac * covalent_radii[atomic_numbers[el]] / Bohr
            rmin_halves[el] = dr * np.floor(rcov/dr)

        wf_ranges = {}
        numr = {}
        for el1 in set(elements):
            wf_range = max([atoms[el1].wf_confinement[nl].rc
                            for nl in atoms[el1].valence])
            wf_ranges[el1] = wf_range
            numr[el1] = int(np.floor(wf_range/dr))

        for el1 in set(elements):
            for el2 in set(elements):
                rmax = wf_ranges[el1] + wf_ranges[el2] + 1.
                atoms[el1].pp.build_overlaps(atoms[el2], atoms[el1], rmin=1e-2,
                                             rmax=rmax, N=300)
    else:
        dr, rmin_halves, wf_ranges, numr = None, None, None, None

    return (atoms, xc, properties, dr, rmin_halves, wf_ranges, numr)


def get_3c_grids(el1, el2, rmin_halves, wf_ranges, verbose=True):
    min_rAB = rmin_halves[el1] + rmin_halves[el2]
    max_rAB = 0.98 * (wf_ranges[el1] + wf_ranges[el2])
    num_rAB = 6 + int(np.ceil((max_rAB - min_rAB) / 0.4))
    min_rCM = 0.2
    max_rCM = max_rAB
    num_rCM = 6 + int(np.ceil((max_rCM - min_rCM) / 0.4))
    num_theta = 24

    Rgrid = np.exp(np.linspace(np.log(min_rAB), np.log(max_rAB), num=num_rAB,
                   endpoint=True))
    Sgrid = np.exp(np.linspace(np.log(min_rCM), np.log(max_rCM), num=num_rCM,
                   endpoint=True))
    Tgrid = np.linspace(0., np.pi, num=num_theta)

    if verbose:
        print('numRST:', num_rAB, num_rCM, num_theta)
        print('Rgrid:', Rgrid)
        print('Sgrid:', Sgrid)

    return (Rgrid, Sgrid, Tgrid)


def chgoff2c(el1, el2, **kwargs):
    atoms, xc, properties, dr, rmin_halves, wf_ranges, numr = \
            get_atoms(el1, el2, label=kwargs['label'], only_1c=False,
                      pseudo_path=kwargs['pseudo_path'], yaml_path='..')

    rmin = rmin_halves[el1] + rmin_halves[el2]
    N = numr[el1] + numr[el2] - int(np.round(rmin/dr))

    for use_multipoles in kwargs['use_multipoles']:
        run_kwargs = dict(rmin=rmin, dr=dr, N=N, **kwargs['opts_2c'])

        if use_multipoles:
            calc = Offsite2cMTable(atoms[el1], atoms[el2], timing=False)
            calc.run(**run_kwargs)
            calc.write()

        calc = Offsite2cUTable(atoms[el1], atoms[el2],
                               use_multipoles=use_multipoles, timing=False)

        run_kwargs.update(shift=kwargs['shift'], xc=xc)

        if use_multipoles:
            if kwargs['multipole_subshells'] is None:
                subshells = None
            else:
                subshells = {el: kwargs['multipole_subshells'][el]
                             for el in [el1, el2]}
            run_kwargs.update(subshells=subshells)

        calc.run(**run_kwargs)
        calc.write()
    return


def chgon1c(el1, **kwargs):
    atoms, xc, properties, dr, rmin_halves, wf_ranges, numr = \
            get_atoms(el1, label=kwargs['label'], only_1c=True,
                      pseudo_path=kwargs['pseudo_path'], yaml_path='..')

    for use_multipoles in kwargs['use_multipoles']:
        if use_multipoles:
            calc = Onsite1cMTable(atoms[el1])
            calc.run()
            calc.write()

        calc = Onsite1cUTable(atoms[el1], use_multipoles=use_multipoles)

        run_kwargs = dict()

        if use_multipoles:
            if kwargs['multipole_subshells'] is None:
                subshells = None
            else:
                subshells = kwargs['multipole_subshells'][el1]
            run_kwargs.update(subshells=subshells, xc=xc)
        else:
            maxstep = 0.125 if el1 == 'H' else 0.25
            run_kwargs.update(maxstep=maxstep)

        calc.run(**run_kwargs)
        calc.write()
    return


def chgon2c(el1, el2, **kwargs):
    atoms, xc, properties, dr, rmin_halves, wf_ranges, numr = \
            get_atoms(el1, el2, label=kwargs['label'], only_1c=False,
                      pseudo_path=kwargs['pseudo_path'], yaml_path='..')

    rmin = rmin_halves[el1] + rmin_halves[el2]
    N = numr[el1] + numr[el1] - int(np.round(rmin/dr))

    for use_multipoles in kwargs['use_multipoles']:
        calc = Onsite2cUTable(atoms[el1], atoms[el2],
                              use_multipoles=use_multipoles, timing=False)

        run_kwargs = dict(rmin=rmin, dr=dr, N=N, xc=xc, **kwargs['opts_2c'])

        if use_multipoles:
            if kwargs['multipole_subshells'] is None:
                subshells = None
            else:
                subshells = kwargs['multipole_subshells'][el1]
            run_kwargs.update(subshells=subshells)

        calc.run(**run_kwargs)
        calc.write()
    return


def magoff2c(el1, el2, **kwargs):
    atoms, xc, properties, dr, rmin_halves, wf_ranges, numr = \
            get_atoms(el1, el2, label=kwargs['label'], only_1c=False,
                      pseudo_path=kwargs['pseudo_path'], yaml_path='..')

    rmin = rmin_halves[el1] + rmin_halves[el2]
    N = numr[el1] + numr[el2] - int(np.round(rmin/dr))

    for use_multipoles in kwargs['use_multipoles']:
        calc = Offsite2cWTable(atoms[el1], atoms[el2],
                               use_multipoles=use_multipoles, timing=False)

        run_kwargs = dict(rmin=rmin, dr=dr, N=N, xc=xc, **kwargs['opts_2c'])

        if use_multipoles:
            if kwargs['multipole_subshells'] is None:
                subshells = None
            else:
                subshells = {el: kwargs['multipole_subshells'][el]
                             for el in [el1, el2]}
            run_kwargs.update(subshells=subshells)

        calc.run(**run_kwargs)
        calc.write()
    return


def magon1c(el1, **kwargs):
    atoms, xc, properties, dr, rmin_halves, wf_ranges, numr = \
            get_atoms(el1, label=kwargs['label'], only_1c=True,
                      pseudo_path=kwargs['pseudo_path'], yaml_path='..')

    for use_multipoles in kwargs['use_multipoles']:
        if use_multipoles:
            calc = Onsite1cMTable(atoms[el1])
            calc.run()
            calc.write()

        calc = Onsite1cWTable(atoms[el1], use_multipoles=use_multipoles)

        run_kwargs = dict()

        if use_multipoles:
            if kwargs['multipole_subshells'] is None:
                subshells = None
            else:
                subshells = kwargs['multipole_subshells'][el1]
            run_kwargs.update(subshells=subshells, xc=xc)
        else:
            maxstep = 0.125 if el1 == 'H' else 0.25
            run_kwargs.update(maxstep=maxstep)

        calc.run(**run_kwargs)
        calc.write()
    return


def magon2c(el1, el2, **kwargs):
    atoms, xc, properties, dr, rmin_halves, wf_ranges, numr = \
            get_atoms(el1, el2, label=kwargs['label'], only_1c=False,
                      pseudo_path=kwargs['pseudo_path'], yaml_path='..')

    rmin = rmin_halves[el1] + rmin_halves[el2]
    N = numr[el1] + numr[el1] - int(np.round(rmin/dr))

    for use_multipoles in kwargs['use_multipoles']:
        calc = Onsite2cWTable(atoms[el1], atoms[el2],
                              use_multipoles=use_multipoles, timing=False)

        run_kwargs = dict(rmin=rmin, dr=dr, N=N, xc=xc, **kwargs['opts_2c'])

        if use_multipoles:
            if kwargs['multipole_subshells'] is None:
                subshells = None
            else:
                subshells = kwargs['multipole_subshells'][el1]
            run_kwargs.update(subshells=subshells)

        calc.run(**run_kwargs)
        calc.write()
    return


def off2c(el1, el2, **kwargs):
    atoms, xc, properties, dr, rmin_halves, wf_ranges, numr = \
            get_atoms(el1, el2, label=kwargs['label'], only_1c=False,
                      pseudo_path=kwargs['pseudo_path'], yaml_path='..')

    rmin = rmin_halves[el1] + rmin_halves[el2]
    N = numr[el1] + numr[el2] - int(np.round(rmin/dr))

    calc = Offsite2cTable(atoms[el1], atoms[el2], timing=False)
    calc.run(rmin=rmin, dr=dr, N=N, superposition='density', xc=xc,
             **kwargs['opts_2c'])

    write_kwargs = dict()
    if el1 == el2:
        write_kwargs.update(
                eigenvalues=properties['eigenvalues'][el1],
                hubbardvalues=properties['hubbardvalues'][el1],
                occupations=properties['occupations'][el1],
                offdiagonal_H=properties['offdiagonal_H'][el1],
                offdiagonal_S=properties['offdiagonal_S'][el1],
        )
    calc.write(**write_kwargs)
    return


def off3c(el1, el2, el3, **kwargs):
    atoms, xc, properties, dr, rmin_halves, wf_ranges, numr = \
            get_atoms(el1, el2, el3, label=kwargs['label'], only_1c=False,
                      pseudo_path=kwargs['pseudo_path'], yaml_path='..')

    Rgrid, Sgrid, Tgrid = get_3c_grids(el1, el2, rmin_halves, wf_ranges)
    calc = Offsite3cTable(atoms[el1], atoms[el2], timing=False)
    calc.run(atoms[el3], Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
             **kwargs['opts_3c'])
    return


def on2c(el1, el2, **kwargs):
    atoms, xc, properties, dr, rmin_halves, wf_ranges, numr = \
            get_atoms(el1, el2, label=kwargs['label'], only_1c=False,
                      pseudo_path=kwargs['pseudo_path'], yaml_path='..')

    rmin = rmin_halves[el1] + rmin_halves[el2]
    N = numr[el1] + numr[el1] - int(np.round(rmin/dr))

    calc = Onsite2cTable(atoms[el1], atoms[el2], timing=False)
    calc.run(rmin=rmin, dr=dr, N=N, xc=xc, shift=kwargs['shift'],
             **kwargs['opts_2c'])
    calc.write()
    return


def on3c(el1, el2, el3, **kwargs):
    atoms, xc, properties, dr, rmin_halves, wf_ranges, numr = \
            get_atoms(el1, el2, el3, label=kwargs['label'], only_1c=False,
                      pseudo_path=kwargs['pseudo_path'], yaml_path='..')

    Rgrid, Sgrid, Tgrid = get_3c_grids(el1, el2, rmin_halves, wf_ranges)

    on3c = Onsite3cTable(atoms[el1], atoms[el2], timing=False)
    on3c.run(atoms[el3], Rgrid=Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
             **kwargs['opts_3c'])
    return


def rep2c(el1, el2, **kwargs):
    atoms, xc, properties, dr, rmin_halves, wf_ranges, numr = \
            get_atoms(el1, el2, label=kwargs['label'], only_1c=False,
                      pseudo_path=kwargs['pseudo_path'], yaml_path='..')

    rmin = rmin_halves[el1] + rmin_halves[el2]
    N = numr[el1] + numr[el2] - int(np.round(rmin/dr))

    calc = Repulsion2cTable(atoms[el1], atoms[el2], timing=False)
    calc.run(rmin=rmin, dr=dr, N=N, xc=xc, shift=kwargs['shift'],
             **kwargs['opts_2c'])
    calc.write()
    return


def rep3c(el1, el2, el3, **kwargs):
    atoms, xc, properties, dr, rmin_halves, wf_ranges, numr = \
            get_atoms(el1, el2, el3, label=kwargs['label'], only_1c=False,
                      pseudo_path=kwargs['pseudo_path'], yaml_path='..')

    Rgrid, Sgrid, Tgrid = get_3c_grids(el1, el2, rmin_halves, wf_ranges)

    calc = Repulsion3cTable(atoms[el1], atoms[el2], timing=False)
    calc.run(atoms[el3], Rgrid, Sgrid=Sgrid, Tgrid=Tgrid, xc=xc,
             **kwargs['opts_3c'])
    return


if __name__ == '__main__':
    main()
