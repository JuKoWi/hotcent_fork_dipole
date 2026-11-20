import numpy as np
from grid_reference import position_from_grid
from ase.build import graphene, bulk
from ase.visualize import view
from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator
from hotcent.pos_op.utils import angstrom_to_bohr, dim_atom_basis
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import PowerConfinement
import matplotlib.pyplot as plt
COMPONENTS = "xyz"


def _make_atom(symbol, configuration, valence, rcov, s, xc="GGA_X_PBE+GGA_C_PBE"):
    """Confined pseudo-atom for the grid reference."""
    atom = AtomicDFT(
        symbol,
        xc=xc,
        perturbative_confinement=False,
        confinement=PowerConfinement(r0=50, s=4),
        configuration=configuration,
        valence=valence,
        scalarrel=True,
        maxiter=2500,
        timing=False,
        nodegpts=150,
        mix=0.2,
        txt="-",
        rmax=500,
    )
    conf = {}
    for v in valence:
        conf[v] = PowerConfinement(r0=rcov, s=s)
    atom.set_wf_confinement(wf_confinement=conf)
    atom.run()
    return atom


def _plot_comparison(data, title):
    """First entry is the reference; the others are plotted as differences to it."""
    (ref_label, ref), rest = data[0], data[1:]
    diffs = [(f"{label} − {ref_label}", r - ref) for label, r in rest]

    lim_ref = np.abs(ref).max() or 1e-12
    lim_diff = max(np.abs(d).max() for _, d in diffs) or 1e-12

    fig, axs = plt.subplots(ncols=len(data), figsize=(3.2 * len(data), 3.4),
                            constrained_layout=True)
    im_ref = axs[0].imshow(ref, cmap="RdBu_r", vmin=-lim_ref, vmax=lim_ref)
    axs[0].set_title(ref_label)
    axs[0].set_ylabel("orbital index center 1")
    axs[0].set_xlabel("orbital index center 2")

    fig.colorbar(im_ref, ax=axs[0], label=r"r [$a_0$]", shrink=0.8)

    for ax, (label, d) in zip(axs[1:], diffs):
        im = ax.imshow(d, cmap="RdBu_r", vmin=-lim_diff, vmax=lim_diff)
        ax.set_title(label)
        ax.set_ylabel("orbital index center 1")
        ax.set_xlabel("orbital index center 2")
    fig.colorbar(im, ax=axs[1:], label=r"$\Delta r$ [$a_0$]", shrink=0.8)

    fig.suptitle(title)
    return fig


### Specific systems ###

def BN_plot(component: int, grid_reference: bool):
    max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2, "B": 1, "N": 1,"Zn":2, "O": 1}
    ase_atoms = graphene("BN", a=2.504, size=(1, 1, 1), vacuum=10)
    ase_atoms.set_positions(ase_atoms.get_positions() - ase_atoms.get_positions()[0])  # first atom at origin
    sk_integrator = SlaterKosterIntegrator(atoms_unit_cell=ase_atoms, skpath="sk_hdf5",
                                           maxl_dict=max_l, format="hdf5")
    n1 = dim_atom_basis(maxl=max_l[ase_atoms.get_chemical_symbols()[0]])
    print(np.linalg.norm(ase_atoms.positions[1]-ase_atoms.positions[0]))

    methods = {"Slater-Koster": "sk",
               "Onsite expansion": "onsite_expansion",
               "Mulliken": "mulliken"}
    data = []
    for label, approx in methods.items():
        sk_integrator.posop_approx = approx
        r = sk_integrator._calculate_lattice_dict("r")[(0, 0, 0)]
        data.append((label, np.real(angstrom_to_bohr(r))))

    if grid_reference:
        atom_dict = {
            "B": _make_atom(symbol="B", configuration="[He] 2s2 2p1 ", valence=['2s','2p'], rcov=3.0, s=10.4),
            "N": _make_atom(symbol="N", configuration="[He] 2s2 2p3 ", valence=['2s','2p'], rcov=3.4, s=13.4),
        }
        r_grid = np.real(position_from_grid(outfile="BN_pair_r.hdf5", atom_dict=atom_dict, ase_atoms=ase_atoms, max_l_dict=max_l))
        data.insert(0, ("Grid reference", r_grid))

    c = COMPONENTS[component]
    full = [(label, r[component]) for label, r in data]
    offsite = [(label, r[component, :n1, n1:]) for label, r in data]

    # _plot_comparison(full, f"Full matrix ⟨μ|{c}|ν⟩")
    _plot_comparison(offsite, f"Offsite B–N block ⟨μ|{c}|ν⟩")
    plt.savefig("rcomparisonB-N.pdf")
    plt.show()

def C_plot(component: int, grid_reference: bool):
    max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2, "B": 1, "N": 1,"Zn":2, "O": 1}
    ase_atoms = graphene("CC", size=(1, 1, 1), vacuum=10)
    ase_atoms.pbc = [False, False, False]
    # ase_atoms.set_positions(ase_atoms.get_positions() - ase_atoms.get_positions()[0])  # first atom at origin
    # ase_atoms.positions = [[0.0, 0,0], [1.42, 0,0]]
    view(ase_atoms)
    print(np.linalg.norm(ase_atoms.get_positions()[1]-ase_atoms.get_positions()[0]))
    sk_integrator = SlaterKosterIntegrator(atoms_unit_cell=ase_atoms, skpath="sk_hdf5",
                                           maxl_dict=max_l, format="hdf5")
    n1 = dim_atom_basis(maxl=max_l[ase_atoms.get_chemical_symbols()[0]])

    methods = {"Slater-Koster": "sk",
               "Onsite expansion": "onsite_expansion",
               "Mulliken": "mulliken"}
    data = []
    for label, approx in methods.items():
        sk_integrator.posop_approx = approx
        r = sk_integrator._calculate_lattice_dict("r")[(0, 0, 0)]
        data.append((label, np.real(angstrom_to_bohr(r))))

    if grid_reference:
        atom_dict = {
            "C": _make_atom(symbol="C", configuration="[He] 2s2 2p2 ", valence=['2s', '2p'], rcov=3.2, s=8.2),
        }
        r_grid = np.real(position_from_grid(outfile="CC_pair_r.hdf5", atom_dict=atom_dict, ase_atoms=ase_atoms, max_l_dict=max_l))
        data.insert(0, ("Grid reference", r_grid))

    c = COMPONENTS[component]
    full = [(label, r[component]) for label, r in data]
    offsite = [(label, r[component, :n1, n1:]) for label, r in data]

    # _plot_comparison(full, f"Full matrix ⟨μ|{c}|ν⟩")
    _plot_comparison(offsite, f"Offsite C–C block ⟨μ|{c}|ν⟩")
    plt.savefig("rcomparisonC-C.pdf")
    plt.show()

def ZnO_plot(component: int, grid_reference: bool):
    max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2, "B": 1, "N": 1,"Zn":2, "O": 1}
    ase_atoms = bulk('ZnO', crystalstructure='zincblende', a=4.62)
    ase_atoms.pbc = [False, False, False]
    # ase_atoms.set_positions(ase_atoms.get_positions() - ase_atoms.get_positions()[0])  # first atom at origin
    # ase_atoms.positions = [[0.0, 0,0], [1.42, 0,0]]
    view(ase_atoms)
    print(np.linalg.norm(ase_atoms.get_positions()[1]-ase_atoms.get_positions()[0]))
    sk_integrator = SlaterKosterIntegrator(atoms_unit_cell=ase_atoms, skpath="sk_hdf5",
                                           maxl_dict=max_l, format="hdf5")
    n1 = dim_atom_basis(maxl=max_l[ase_atoms.get_chemical_symbols()[0]])

    methods = {"Slater-Koster": "sk",
               "Onsite expansion": "onsite_expansion",
               "Mulliken": "mulliken"}
    data = []
    for label, approx in methods.items():
        sk_integrator.posop_approx = approx
        r = sk_integrator._calculate_lattice_dict("r")[(0, 0, 0)]
        data.append((label, np.real(angstrom_to_bohr(r))))

    if grid_reference:
        atom_dict = {
            "Zn": _make_atom(symbol="Zn", configuration="[Ar] 4s2 4p0 3d10", valence=['4s', '4p','3d'], rcov=4.6, s=2.2),
            "O": _make_atom(symbol="O", configuration="[He] 2s2 2p4", valence=['2s', '2p'], rcov=3.1, s=12.4)
        }
        r_grid = np.real(position_from_grid(outfile="ZnO_pair_r.hdf5", atom_dict=atom_dict, ase_atoms=ase_atoms, max_l_dict=max_l))
        data.insert(0, ("Grid reference", r_grid))

    c = COMPONENTS[component]
    full = [(label, r[component]) for label, r in data]
    offsite = [(label, r[component, :n1, n1:]) for label, r in data]

    # _plot_comparison(full, f"Full matrix ⟨μ|{c}|ν⟩")
    _plot_comparison(offsite, f"Offsite Zn–O block ⟨μ|{c}|ν⟩")
    plt.savefig("rcomparisonZn-O.pdf")
    plt.show()

def plot_distance_error(ase_atoms, rmin_ang, rmax_ang):
    max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2, "B": 1, "N": 1,"Zn":2, "O": 1}
    if len(ase_atoms.positions) != 2:
        raise ValueError("Object can only have two atoms for this analysis")
    orientation = (ase_atoms.positions[1] - ase_atoms.positions[0]) 
    orientation /= np.linalg.norm(orientation) 
    dist = np.linspace(rmin_ang, rmax_ang, 100)
    methods = {"Slater-Koster": "sk",
               "Onsite expansion": "onsite_expansion",
               "Mulliken": "mulliken"}
    n1 = dim_atom_basis(maxl=max_l[ase_atoms.get_chemical_symbols()[0]])
    mae_data = [[],[]]
    labels = list(methods.keys())

    for d in dist:
        ase_atoms.positions = [[0,0,0], d*orientation]
        sk_integrator = SlaterKosterIntegrator(atoms_unit_cell=ase_atoms, skpath="sk_hdf5",
                                               maxl_dict=max_l, format="hdf5")

        r_at_distance = []
        for label, approx in methods.items():
            sk_integrator.posop_approx = approx
            r = sk_integrator._calculate_lattice_dict("r")[(0, 0, 0)]
            r_at_distance.append(np.real(angstrom_to_bohr(r))[:,:n1, n1:])
        for i in range(len(r_at_distance)-1):
            mae_data[i].append(np.sum(np.abs(r_at_distance[i+1] - r_at_distance[0]))/r_at_distance[0].size)
    fix, ax = plt.subplots()
    for i, data in enumerate(mae_data):
        ax.plot(dist, data, label=labels[i+1])
    ax.set_xlabel(r"d [$\AA$]")
    ax.set_ylabel(r"MAE(r) [$a_0$]")
    plt.legend()
    plt.savefig(f"distance_rapprox{ase_atoms.get_chemical_symbols()[0]}-{ase_atoms.get_chemical_symbols()[1]}.pdf")
    plt.show()
        
        

        


        

    






if __name__ == "__main__":
    # C_plot(component=1, grid_reference=False)
    # BN_plot(component=1, grid_reference=False)
    ZnO_plot(component=1, grid_reference=True)
    # ase_atoms = graphene('CC')
    # ase_atoms.pbc = [False, False, False]
    # plot_distance_error(ase_atoms=ase_atoms, rmin_ang=0.5, rmax_ang=6)
    # ase_atoms = bulk('ZnO', crystalstructure='zincblende', a=4.62)
    # ase_atoms.pbc = [False, False, False]
    # plot_distance_error(ase_atoms=ase_atoms, rmin_ang=0.5, rmax_ang=6)
    # ase_atoms = graphene('HC')
    # ase_atoms.pbc = [False, False, False]
    # plot_distance_error(ase_atoms=ase_atoms, rmin_ang=0.5, rmax_ang=6)


