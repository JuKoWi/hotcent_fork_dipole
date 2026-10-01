import matplotlib.pyplot as plt
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.pos_op.onsite_momentum import onsite_momentum, R_spline
from hotcent.pos_op.symbolic_integrals import first_center, theta1, phi
from hotcent.pos_op.rotation_transform import to_spherical
from hotcent.pos_op.utils import dim_atom_basis
import grid 
import numpy as np
import sympy as sp

def evaluate_psi(Ynl, rspline, cart_grid):
    """evaluate psi at a set of grid points"""
    spherical_coord = to_spherical(cart_grid)
    r_evaluated = rspline(spherical_coord[:,0])
    var_theta = spherical_coord[:,1]
    var_phi = spherical_coord[:,2]
    Y_evaluated = Ynl(var_theta, var_phi)
    return r_evaluated * Y_evaluated

def onsite_norm_grid(atom, nl, Ynl):
    """overlap between two basis states centered at the same atom"""
    hotcent_rgrid = atom.rgrid
    hotcent_spline = R_spline(atom=atom, nl=nl)
    oned_grid = grid.onedgrid.GaussLegendre(npoints=100)
    rgrid = grid.rtransform.BeckeRTransform(0.0, R=atom.rmax).transform_1d_grid(oned_grid)
    at_grid = grid.atomgrid.AtomGrid(rgrid, degrees=[20])
    psi = evaluate_psi(Ynl=Ynl, rspline=hotcent_spline, cart_grid=at_grid.points)
    dens = psi**2
    return at_grid.integrate(dens)

def onsite_posop_grid(atom, nl1, nl2, Ynl1, Ynl2, cart_component):
    """position operator matrix elements between basis states centered at the same atom"""
    hotcent_rgrid = atom.rgrid
    hotcent_spline1 = R_spline(atom=atom, nl=nl1)
    hotcent_spline2 = R_spline(atom=atom, nl=nl2)
    oned_grid = grid.onedgrid.GaussLegendre(npoints=100)
    rgrid = grid.rtransform.BeckeRTransform(0.0, R=atom.rmax).transform_1d_grid(oned_grid)
    at_grid = grid.atomgrid.AtomGrid(rgrid, degrees=[20])
    psi1 = evaluate_psi(Ynl=Ynl1, rspline=hotcent_spline1, cart_grid=at_grid.points)
    psi2 = evaluate_psi(Ynl=Ynl2, rspline=hotcent_spline2, cart_grid=at_grid.points)
    posop = psi1 * psi2 * at_grid.points[:,cart_component]
    return at_grid.integrate(posop)

def onsite_momentum_grid(atom, nl1, nl2, Ynl1, Ynl2, cart_component):
    """momentum in a.u. between two basis functions centered at the same atom"""
    hotcent_spline1 = R_spline(atom=atom, nl=nl1)
    hotcent_spline2 = R_spline(atom=atom, nl=nl2)
    oned_grid = grid.onedgrid.GaussLegendre(npoints=500)
    rgrid = grid.rtransform.BeckeRTransform(0.0, R=atom.rmax).transform_1d_grid(oned_grid)
    at_grid = grid.atomgrid.AtomGrid(rgrid, degrees=[20])
    psi1 = evaluate_psi(Ynl=Ynl1, rspline=hotcent_spline1, cart_grid=at_grid.points)
    psi2 = evaluate_psi(Ynl=Ynl2, rspline=hotcent_spline2, cart_grid=at_grid.points)
    psi2_interpolator = at_grid.interpolate(psi2)
    partial_psi2 = psi2_interpolator(at_grid.points, deriv=1)[:,cart_component]
    momentum = psi1 * partial_psi2
    return -1j*at_grid.integrate(momentum)

def twocenter_overlap(atom1, atom2, nl1, nl2, Ynl1, Ynl2, pos_au_1, pos_au_2):
    """overlap between two basis functions centered at different atoms"""
    oned_grid = grid.onedgrid.GaussLegendre(npoints=500)
    rgrid = grid.rtransform.BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned_grid)
    mgrid = grid.MolGrid.from_preset(
        atnums=np.array([atom1.Z, atom2.Z]),
        atcoords=np.array([pos_au_1, pos_au_2]),
        rgrid=rgrid,
        preset="fine",
        aim_weights=grid.BeckeWeights(),
        store=True,
    )
    cart_grid1 = mgrid.points - pos_au_1
    cart_grid2 = mgrid.points - pos_au_2
    hotcent_spline1 = R_spline(atom=atom1, nl=nl1)
    hotcent_spline2 = R_spline(atom=atom2, nl=nl2)
    psi1 = evaluate_psi(Ynl=Ynl1, rspline=hotcent_spline1, cart_grid=cart_grid1)
    psi2 = evaluate_psi(Ynl=Ynl2, rspline=hotcent_spline2, cart_grid=cart_grid2)
    return mgrid.integrate(psi1 * psi2)

def twocenter_position(atom1, atom2, nl1, nl2, Ynl1, Ynl2, pos_au_1, pos_au_2, cart_component):
    """position matrix elements between two basis functions centered at different atoms"""
    oned_grid = grid.onedgrid.GaussLegendre(npoints=500)
    rgrid = grid.rtransform.BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned_grid)
    mgrid = grid.MolGrid.from_preset(
        atnums=np.array([atom1.Z, atom2.Z]),
        atcoords=np.array([pos_au_1, pos_au_2]),
        rgrid=rgrid,
        preset="fine",
        aim_weights=grid.BeckeWeights(),
        store=True,
    )
    cart_grid1 = mgrid.points - pos_au_1
    cart_grid2 = mgrid.points - pos_au_2
    hotcent_spline1 = R_spline(atom=atom1, nl=nl1)
    hotcent_spline2 = R_spline(atom=atom2, nl=nl2)
    psi1 = evaluate_psi(Ynl=Ynl1, rspline=hotcent_spline1, cart_grid=cart_grid1)
    psi2 = evaluate_psi(Ynl=Ynl2, rspline=hotcent_spline2, cart_grid=cart_grid2)
    return mgrid.integrate(psi1 * psi2 * mgrid.points[:,cart_component])


def twocenter_momentum(atom1, atom2, nl1, nl2, Ynl1, Ynl2, pos_au_1, pos_au_2, cart_component):
    """momentum in a.u. between two basis functions centered at different atoms"""
    oned_grid = grid.onedgrid.GaussLegendre(npoints=200)
    rgrid = grid.rtransform.BeckeRTransform(1e-5, R=1.5).transform_1d_grid(oned_grid)
    mgrid = grid.MolGrid.from_preset(
        atnums=np.array([atom1.Z, atom2.Z]),
        atcoords=np.array([pos_au_1, pos_au_2]),
        rgrid=rgrid,
        preset="fine",
        aim_weights=grid.BeckeWeights(),
        store=True,
    )
    cart_grid1 = mgrid.points - pos_au_1
    cart_grid2 = mgrid.points - pos_au_2
    hotcent_spline1 = R_spline(atom=atom1, nl=nl1)
    hotcent_spline2 = R_spline(atom=atom2, nl=nl2)
    psi1 = evaluate_psi(Ynl=Ynl1, rspline=hotcent_spline1, cart_grid=cart_grid1)
    psi2 = evaluate_psi(Ynl=Ynl2, rspline=hotcent_spline2, cart_grid=cart_grid2)
    psi2_interpolator = mgrid.interpolate(psi2)
    partial_psi2 = psi2_interpolator(mgrid.points, deriv=1)[:,cart_component]
    momentum = psi1 * partial_psi2
    return -1j*mgrid.integrate(momentum)

def grad_psi(Ynl, rspline, cart_grid, comp, h=1e-4):
    """finite difference gradient"""
    e = np.zeros(3)
    e[comp] = h
    return (evaluate_psi(Ynl, rspline, cart_grid+e) - evaluate_psi(Ynl, rspline, cart_grid-e)) / (2*h) 

def twocenter_momentum_finite_diff(atom1, atom2, nl1, nl2, Ynl1, Ynl2, pos_au_1, pos_au_2, cart_component):
    """momentum in a.u. between two basis functions centered at different atoms
        finite difference for gradient
    """
    oned_grid = grid.onedgrid.GaussLegendre(npoints=200)
    rgrid = grid.rtransform.BeckeRTransform(1e-5, R=1.5).transform_1d_grid(oned_grid)
    mgrid = grid.MolGrid.from_preset(
        atnums=np.array([atom1.Z, atom2.Z]),
        atcoords=np.array([pos_au_1, pos_au_2]),
        rgrid=rgrid,
        preset="fine",
        aim_weights=grid.BeckeWeights(),
        store=True,
    )
    cart_grid1 = mgrid.points - pos_au_1
    cart_grid2 = mgrid.points - pos_au_2
    hotcent_spline1 = R_spline(atom=atom1, nl=nl1)
    hotcent_spline2 = R_spline(atom=atom2, nl=nl2)
    psi1 = evaluate_psi(Ynl=Ynl1, rspline=hotcent_spline1, cart_grid=cart_grid1)
    partial_psi2 = grad_psi(Ynl=Ynl2, rspline=hotcent_spline2, cart_grid=cart_grid2, comp=cart_component)
    momentum = psi1 * partial_psi2
    return -1j*mgrid.integrate(momentum)

def momentum_atom_pair_block(pos1, pos2, atom1, atom2, maxl1, maxl2):
    """momentum in a.u. between two atoms (all basis states of first atom as bra and all basis states of second atom as ket)"""
    momentum = np.zeros((dim_atom_basis(maxl1), dim_atom_basis(maxl2), 3), dtype=complex)
    basis = list(first_center.keys())
    for i,a in enumerate(basis):
        for j,b in enumerate(basis):
            Y_nl1 = sp.lambdify((theta1, phi), first_center[a][0])
            Y_nl2 = sp.lambdify((theta1, phi), first_center[b][0])
            if a[0] in [s[1] for s in atom1.valence] and b[0] in [s[1] for s in atom2.valence]:
                nl1 = [s for s in atom1.valence if s[1] == a[0]][0]
                nl2 = [s for s in atom2.valence if s[1] == b[0]][0]
                for c in range(3):
                    if np.allclose(pos1, pos2):
                        momentum[i,j,c] = onsite_momentum_grid(atom=atom1, nl1=nl1, nl2=nl2, Ynl1=Y_nl1, Ynl2=Y_nl2, cart_component=c)
                    else:
                        momentum[i,j,c] = twocenter_momentum_finite_diff(atom1=atom1, atom2=atom2, nl1=nl1, nl2=nl2, Ynl1=Y_nl1, Ynl2=Y_nl2, pos_au_1=pos1, pos_au_2=pos2, cart_component=c)
    return momentum

def position_atom_pair_block(pos1, pos2, atom1, atom2, maxl1, maxl2):
    """position matrix elements in a.u. between two atoms (all basis states of first atom as bra and all basis states of second atom as ket)"""
    momentum = np.zeros((dim_atom_basis(maxl1), dim_atom_basis(maxl2), 3), dtype=complex)
    basis = list(first_center.keys())
    for i,a in enumerate(basis):
        for j,b in enumerate(basis):
            Y_nl1 = sp.lambdify((theta1, phi), first_center[a][0])
            Y_nl2 = sp.lambdify((theta1, phi), first_center[b][0])
            if a[0] in [s[1] for s in atom1.valence] and b[0] in [s[1] for s in atom2.valence]:
                nl1 = [s for s in atom1.valence if s[1] == a[0]][0]
                nl2 = [s for s in atom2.valence if s[1] == b[0]][0]
                for c in range(3):
                    if np.allclose(pos1, pos2):
                        shift = pos1[c] if i==j else 0
                        momentum[i,j,c] = shift + onsite_posop_grid(atom=atom1, nl1=nl1, nl2=nl2, Ynl1=Y_nl1, Ynl2=Y_nl2, cart_component=c)
                    else:
                        momentum[i,j,c] = twocenter_position(atom1=atom1, atom2=atom2, nl1=nl1, nl2=nl2, Ynl1=Y_nl1, Ynl2=Y_nl2, pos_au_1=pos1, pos_au_2=pos2, cart_component=c)
    return momentum





if __name__ == "__main__":
    from ase.build import graphene
    from hotcent.pos_op.utils import angstrom_to_bohr, bohr_to_angstrom
    # Get KS all-electron ground state of confined atom:
    element = "C"
    xc = "GGA_X_PBE+GGA_C_PBE"
    conf = PowerConfinement(r0=50.0, s=4)
    r0 = 3.2  # Bohr
    wf_conf = {
        "2s": PowerConfinement(r0=r0, s=8.2),
        "2p": PowerConfinement(r0=r0, s=8.2),
    }

    atom = AtomicDFT(
        element,
        xc=xc,
        confinement=conf,
        perturbative_confinement=False,
        configuration="[He] 2s2 2p2",
        valence=["2s", "2p"],
        scalarrel=True,
        maxiter=2500,
        timing=False,
        nodegpts=150,
        mix=0.2,
        txt="-",
    )
    # atom.rmin = 1e-6/atom.Z
    # atom.rmax = 150
    atom.run()

    atom.set_confinement(conf)
    atom.set_wf_confinement(wf_confinement=wf_conf)
    atom.run()
    onsite_momentum(atom, "C.txt")
    
    max_l = {"C": 1, "H": 0, "S": 2, "Mo": 2}
    graphene = graphene("CC", size=(1, 1, 1), vacuum=10)
    pos1 = angstrom_to_bohr(graphene.get_positions()[0])
    pos2 = angstrom_to_bohr(graphene.get_positions()[1])
    print("onsite position")
    print(bohr_to_angstrom(np.reshape(position_atom_pair_block(pos1=pos2, pos2=pos2, atom1=atom, atom2=atom, maxl1=1, maxl2=1), (16,3)))) #onsite
    print("onsite momentum")
    # print(np.reshape(momentum_atom_pair_block(pos1=pos1, pos2=pos1, atom1=atom, atom2=atom, maxl1=1, maxl2=1), (16,3))) #onsite
    print("offsite position")
    print(bohr_to_angstrom(np.reshape(position_atom_pair_block(pos1=pos2, pos2=pos1, atom1=atom, atom2=atom, maxl1=1, maxl2=1), (16,3)))) #offsite
    print("offsite momentum")
    print(np.reshape(momentum_atom_pair_block(pos1=pos1, pos2=pos2, atom1=atom, atom2=atom, maxl1=1, maxl2=1), (16,3))) #offsite
