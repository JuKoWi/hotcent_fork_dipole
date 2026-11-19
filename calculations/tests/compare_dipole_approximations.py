import numpy as np
from grid_reference import position_from_grid
from ase.build import graphene
from ase.visualize import view
from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator
from hotcent.pos_op.utils import angstrom_to_bohr, dim_atom_basis
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import PowerConfinement
import matplotlib.pyplot as plt


def BN_plot(component:int, grid_reference:bool):
    max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2, "B": 1, "N":1}
    hBN = graphene("BN", a=2.504, size=(1, 1, 1), vacuum=10)
    new_positions = hBN.get_positions() - hBN.get_positions()[0] #put first atom to 0
    hBN.set_positions(new_positions)
    sk_integrator = SlaterKosterIntegrator(atoms_unit_cell=hBN, skpath="sk_hdf5", maxl_dict=max_l, format='hdf5')

    dim_center1 = dim_atom_basis(maxl=max_l[hBN.get_chemical_symbols()[0]])

    sk_integrator.posop_approx = 'sk'
    r_sk = np.real(angstrom_to_bohr(sk_integrator._calculate_lattice_dict('r')[(0,0,0)]))
    sk_integrator.posop_approx = 'onsite_expansion'
    r_onsite = np.real(angstrom_to_bohr(sk_integrator._calculate_lattice_dict('r')[(0,0,0)]))
    sk_integrator.posop_approx = 'mulliken'
    r_mulliken = np.real(angstrom_to_bohr(sk_integrator._calculate_lattice_dict('r')[(0,0,0)]))

    data_to_compare = [("Slater-Koster",r_sk), ("Onsite expansion", r_onsite), ("Mulliken",r_mulliken)]

    if grid_reference:
        xc = "GGA_X_PBE+GGA_C_PBE"
        atomN = AtomicDFT(
            "N",
            xc=xc,
            perturbative_confinement=False,
            confinement=PowerConfinement(r0=50, s=4),
            configuration="[He] 2s2 2p3 ",
            valence=["2s", "2p"],
            scalarrel=True,
            maxiter=2500,
            timing=False,
            nodegpts=2500,
            mix=0.2,
            txt="-",
            rmax=500,
        )
        atomN.run()
        eigenvaluesN = dict(atomN.enl)
        atomB = AtomicDFT(
            "B",
            xc=xc,
            perturbative_confinement=False,
            confinement=PowerConfinement(r0=50, s=4),
            configuration="[He] 2s2 2p1 ",
            valence=["2s", "2p"],
            scalarrel=True,
            maxiter=2500,
            timing=False,
            nodegpts=2500,
            mix=0.2,
            txt="-",
            rmax=500,
        )
        atomB.run()
        eigenvaluesB = dict(atomB.enl)
        rcovB = 3.0
        rcovN = 3.4
        wfconfN = {
            "2s": PowerConfinement(r0=rcovN, s=13.4),
            "2p": PowerConfinement(r0=rcovN, s=13.4),
        }
        wfconfB = {
            "2s": PowerConfinement(r0=rcovB, s=10.4),
            "2p": PowerConfinement(r0=rcovB, s=10.4),
        }
        atomB.set_wf_confinement(wf_confinement=wfconfB)
        atomB.run()
        atomN.set_wf_confinement(wf_confinement=wfconfN)
        atomN.run()
        atom_dict = {'N': atomN, 'B':atomB}

        r_grid = np.real(position_from_grid(outfile='BN_pair_r.hdf5', atom_dict=''))
        data_to_compare.insert(0, ("Grid reference",r_grid))
    fig, axs = plt.subplots(ncols=len(data_to_compare))
    for i, data in enumerate(data_to_compare):
        label, r = data
        if i == 0:
            ax = axs[i].imshow(r[component])
        else:
            ax = axs[i].imshow(r[component] - data_to_compare[0][1][component])
    plt.show()
    fig, axs = plt.subplots(ncols=len(data_to_compare))
    ref_label, r_ref = data_to_compare[0]
    shown = [r_ref[component,:dim_center1, dim_center1:]] + [r[component, :dim_center1, dim_center1:] - r_ref[component, :dim_center1, dim_center1:] for _, r in data_to_compare[1:]]
    vmin, vmax = min(s.min() for s in shown), max(s.max() for s in shown)
    for i, data in enumerate(data_to_compare):
        label, r = data
        cmap = 'RdBu_r'
        if i == 0:
            ax = axs[i].imshow(r[component,:dim_center1, dim_center1:], vmin=vmin, vmax=vmax, cmap=cmap)
            axs[i].set_title(label)
        else:
            ax = axs[i].imshow(r[component,:dim_center1, dim_center1:] - data_to_compare[0][1][component,:dim_center1, dim_center1:], vmin=vmin, vmax=vmax, cmap=cmap)
            axs[i].set_title(f"{label} − {ref_label}")
    fig.colorbar(ax, ax=axs)
    plt.show()




if __name__ == "__main__":
    BN_plot(component=1, grid_reference=False)
