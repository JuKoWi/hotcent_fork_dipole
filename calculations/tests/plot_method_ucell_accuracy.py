import numpy as np
from ase.build import graphene
import matplotlib.pyplot as plt
from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator

max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2}

graphene = graphene("CC", vacuum=10)
# graphene.set_positions([[0,0,0], [1,1,1]])
# pos = graphene.positions
# pos[:,2] =0
# graphene.positions = pos
# print(graphene.positions)

sk_integrals = SlaterKosterIntegrator(
    graphene,
    skpath="skfiles_consistent/sk_unique",
    maxl_dict=max_l,
    skpath_posop="skfiles_consistent/sk_posop_unique",
    format="unique",
    path_p_onsite="skfiles_consistent/onsite_momentum/",
)

approx = 'sk'
sk_integrals.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'onsite_expansion'
sk_integrals.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'mulliken'
sk_integrals.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'onsite_only'
sk_integrals.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)
approx = 'shift_only'
sk_integrals.write_tb_hdf5(filename=f"graphene_{approx}.hdf5", momentum=True, posop_approx=approx)

lattice_vec = (0,0,0)
sk_integrals.posop_approx = 'sk'
r_sk = np.real(sk_integrals._calculate_lattice_dict(operator='r')[lattice_vec])
sk_integrals.posop_approx = "onsite_expansion"
r_onsite_expand = np.real(sk_integrals._calculate_lattice_dict(operator='r')[lattice_vec])
sk_integrals.posop_approx = 'mulliken'
r_mulliken = np.real(sk_integrals._calculate_lattice_dict(operator='r')[lattice_vec])
sk_integrals.posop_approx = 'onsite_only'
r_onsite = np.real(sk_integrals._calculate_lattice_dict(operator='r')[lattice_vec])
sk_integrals.posop_approx = "shift_only" 
r_shift = np.real(sk_integrals._calculate_lattice_dict(operator='r')[lattice_vec])


print(r_onsite_expand[2,6,1])
print(r_mulliken[2,6,1])

rows = [
    (np.real(r_sk), r"$r^{\mathrm{exact}}_{\mu\nu}$ (Å)"),
    (np.real(r_onsite_expand - r_sk), r"$r^{\mathrm{onsite\,exp.}}_{\mu\nu} - r^{\mathrm{exact}}_{\mu\nu}$ (Å)"),
    (np.real(r_mulliken - r_sk), r"$r^{\mathrm{Mulliken}}_{\mu\nu} - r^{\mathrm{exact}}_{\mu\nu}$ (Å)"),
    (np.real(r_onsite - r_sk), r"$r^{\mathrm{onsite\,only}}_{\mu\nu} - r^{\mathrm{exact}}_{\mu\nu}$ (Å)"),
    (np.real(r_shift - r_sk), r"$r^{\mathrm{shift\,only}}_{\mu\nu} - r^{\mathrm{exact}}_{\mu\nu} (Å)$"),
]

fig, axs = plt.subplots(nrows=5, ncols=3, layout="constrained")
for i, (data, label) in enumerate(rows):
    vmax = np.abs(data).max()
    for c, comp in enumerate("xyz"):
        im = axs[i, c].imshow(data[c], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        if i == 0:
            axs[i, c].set_title(f"${comp}$")
    cbar = fig.colorbar(im, ax=axs[i, :], location="right", shrink=0.9)
    cbar.set_label(label)
plt.show()
