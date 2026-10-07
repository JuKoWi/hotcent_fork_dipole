import matplotlib.pyplot as plt
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.pos_op.onsite_momentum import onsite_momentum, R_spline
from hotcent.pos_op.symbolic_integrals import first_center, theta1, phi
from hotcent.pos_op.rotation_transform import to_spherical
from hotcent.pos_op.utils import dim_atom_basis, angstrom_to_bohr
from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator
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

def check_basis_completeness(positions_au, atoms, maxl):
    dim_total = sum([dim_atom_basis(l) for l in maxl])
    S = np.zeros((dim_total, dim_total))
    r = np.zeros((3, dim_total, dim_total))
    atom_list = []
    for j, b in enumerate(atoms):
        atom_list += [b for j in range(dim_atom_basis(maxl[j]))]
    pos_list =[]
    for j,p in enumerate(positions_au):
        pos_list += [p for j in range(dim_atom_basis(maxl[j]))] 
    Ylm_list = []
    nl_list = []
    for i, b in enumerate(atoms):
        for j, lm in enumerate(list(first_center.keys())):
            if j == dim_atom_basis(maxl[i]):
                break
            if lm[0] in [s[1] for s in b.valence]:
                Ylm = sp.lambdify((theta1, phi), first_center[lm][0])
                Ylm_list.append(Ylm)
                nl_list.append([s for s in b.valence if s[1] == lm[0]][0])
    for i, a in enumerate(atom_list):
        for j, b in enumerate(atom_list):
            S[i,j] = twocenter_overlap(atom1=a, atom2=b, nl1=nl_list[i], nl2=nl_list[j], Ynl1=Ylm_list[i], Ynl2=Ylm_list[j], pos_au_1=pos_list[i], pos_au_2=pos_list[j])
            for c in range(3):
                r[c,i,j] = twocenter_position(atom1=a, atom2=b, nl1=nl_list[i], nl2=nl_list[j], Ynl1=Ylm_list[i], Ynl2=Ylm_list[j], pos_au_1=pos_list[i], pos_au_2=pos_list[j], cart_component=c)
    S_inv = np.linalg.inv(S)
    oned_grid = grid.onedgrid.GaussLegendre(npoints=200)
    rgrid = grid.rtransform.BeckeRTransform(1e-5, R=1.5).transform_1d_grid(oned_grid)
    mgrid = grid.MolGrid.from_preset(
        atnums=np.array([a.Z for a in atoms]),
        atcoords=np.array([p for p in positions_au]),
        rgrid=rgrid,
        preset="fine",
        aim_weights=grid.BeckeWeights(),
        store=True,
    )
    diff_grid = np.zeros((3, dim_total))
    diff = np.zeros((3, dim_total))
    coeffs = np.einsum('jk, ckl -> cjl', S_inv, r)
    RSR_diag = np.einsum('cab,bd,cda->ca', r, S_inv, r)                
    for c in range(3):
        for i, a in enumerate(atom_list):
            cart_grid = mgrid.points - pos_list[i]
            hotcent_spline = R_spline(atom=a, nl=nl_list[i])
            psi = evaluate_psi(Ynl=Ylm_list[i], rspline=hotcent_spline, cart_grid=cart_grid)   
            x = mgrid.points[:,c]
            psi_total = x * psi
            x2 = mgrid.integrate(psi_total**2)                          
            norm = mgrid.integrate((psi * (x-pos_list[i][c]))**2)
            for j,b in enumerate(atom_list):
                cart_grid = mgrid.points - pos_list[j]
                hotcent_spline = R_spline(atom=b, nl=nl_list[j])
                psi_total -= coeffs[c,j,i] * evaluate_psi(Ynl=Ylm_list[j], rspline=hotcent_spline, cart_grid=cart_grid)
            diff_grid[c, i] = mgrid.integrate(psi_total**2)/norm
            diff[c, i] = (x2 - RSR_diag[c, i])/norm                    
    return diff_grid, diff

def check_all_r_p_S(atom_dict, ase_atoms):
    """
        atom_dict: dict with hotcent atoms as values and element symbols as keys
        ase_atoms: ase.Atoms object
    """
    #perform slater koster calculation
    max_l_dict = {"C": 1, "H": 0, "S": 2, "Mo": 2}
    sk_integrals = SlaterKosterIntegrator(
        ase_atoms,
        skpath="skfiles_consistent/sk_unique",
        maxl_dict=max_l_dict,
        skpath_posop="skfiles_consistent/sk_posop_unique",
        format="unique",
        path_p_onsite="skfiles_consistent/onsite_momentum/"
    )
    r_hotcent = sk_integrals._calculate_lattice_dict("r")[(0,0,0)]
    S_hotcent = sk_integrals._calculate_lattice_dict("S")[(0,0,0)]
    p_hotcent = sk_integrals._calculate_lattice_dict("p")[(0,0,0)]

    #get list of atom wise information
    positions_au = angstrom_to_bohr(ase_atoms.get_positions())
    atoms = [atom_dict[symb] for symb in ase_atoms.get_chemical_symbols()]
    maxl = [max_l_dict[symb] for symb in ase_atoms.get_chemical_symbols()]

    # prepare iterations over all functions in the basis
    dim_total = sum([dim_atom_basis(l) for l in maxl])
    S = np.zeros((dim_total, dim_total))
    r = np.zeros((3, dim_total, dim_total))
    p = np.zeros((3, dim_total, dim_total))
    atom_list = []
    for j, b in enumerate(atoms):
        atom_list += [b for j in range(dim_atom_basis(maxl[j]))]
    pos_list =[]
    for j,p in enumerate(positions_au):
        pos_list += [p for j in range(dim_atom_basis(maxl[j]))] 
    Ylm_list = []
    nl_list = []
    for i, b in enumerate(atoms):
        for j, lm in enumerate(list(first_center.keys())):
            if j == dim_atom_basis(maxl[i]):
                break
            if lm[0] in [s[1] for s in b.valence]:
                Ylm = sp.lambdify((theta1, phi), first_center[lm][0])
                Ylm_list.append(Ylm)
                nl_list.append([s for s in b.valence if s[1] == lm[0]][0])
    for i, a in enumerate(atom_list):
        for j, b in enumerate(atom_list):
            S[i,j] = twocenter_overlap(atom1=a, atom2=b, nl1=nl_list[i], nl2=nl_list[j], Ynl1=Ylm_list[i], Ynl2=Ylm_list[j], pos_au_1=pos_list[i], pos_au_2=pos_list[j])
            for c in range(3):
                r[c,i,j] = twocenter_position(atom1=a, atom2=b, nl1=nl_list[i], nl2=nl_list[j], Ynl1=Ylm_list[i], Ynl2=Ylm_list[j], pos_au_1=pos_list[i], pos_au_2=pos_list[j], cart_component=c)
                p[c,i,j] = twocenter_momentum_finite_diff(atom1=a, atom2=b, nl1=nl_list[i], nl2=nl_list[j], Ynl1=Ylm_list[i], Ynl2=Ylm_list[j], pos_au_1=pos_list[i], pos_au_2=pos_list[j], cart_component=c)


if __name__ == "__main__":
    from ase.build import graphene, mx2 
    from hotcent.pos_op.utils import angstrom_to_bohr, bohr_to_angstrom
    import sys
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
    Ylm1 = sp.lambdify((theta1, phi), first_center['ss'][0])
    Ylm2 = sp.lambdify((theta1, phi), first_center['py'][0])
    # print(onsite_momentum_grid(atom=atom, nl1="2s", nl2="2p", Ynl1=Ylm1, Ynl2=Ylm2, cart_component=1))

    print(check_basis_completeness(positions_au=[pos1, pos2], atoms=[atom, atom], maxl=[1,1]))
    # print(check_basis_completeness(positions_au=[pos1, pos2], atoms=[atom, atom], maxl=[0,0]))

    pos1 += np.array([1,1,1])
    pos2 += np.array([1,1,1])

    print(check_basis_completeness(positions_au=[pos1, pos2], atoms=[atom, atom], maxl=[1,1]))
    # print(check_basis_completeness(positions_au=[pos1, pos2], atoms=[atom, atom], maxl=[0,0]))


    sys.exit()

    MoS2 = mx2(vacuum=20)
    mos2_positions = MoS2.get_positions()
    print(MoS2.get_chemical_symbols())

    atomS = AtomicDFT(
        "S",
        xc=xc,
        perturbative_confinement=False,
        confinement=PowerConfinement(r0=50, s=4),
        configuration="[Ne] 3s2 3p4 3d0",
        valence=["3s", "3p", "3d"],
        scalarrel=True,
        maxiter=2500,
        timing=False,
        # nodegpts=2500,
        mix=0.2,
        txt="-",
        # rmax=500,
    )
    atomS.run()
    

    atomMo = AtomicDFT(
        "Mo",
        xc=xc,
        perturbative_confinement=False,
        configuration="[Kr] 4d4 5s2 5p0",
        valence=["4d", "5s", "5p"],
        confinement=PowerConfinement(r0=50, s=4),
        scalarrel=True,
        maxiter=2500,
        timing=False,
        # nodegpts=150,
        mix=0.2,
        txt="-",
        # rmax=100,
    )
    atomMo.run()

    # Use parameters from 10.1021/ct4004959 (Heine 2013)
    rcovS = 3.9
    rcovMo = 4.3
    # confS = PowerConfinement(r0=50, s=4)
    # confMo = PowerConfinement(r0=50, s=4)

    wf_confS = {
        "3s": PowerConfinement(r0=rcovS, s=4.6),
        "3p": PowerConfinement(r0=rcovS, s=4.6),
        "3d": PowerConfinement(r0=rcovS, s=4.6),
    }

    wf_confMo = {
        "4d": PowerConfinement(r0=rcovMo, s=11.6),
        "5s": PowerConfinement(r0=rcovMo, s=11.6),
        "5p": PowerConfinement(r0=rcovMo, s=11.6),
    }

    # atomS.set_confinement(confS)
    atomS.set_wf_confinement(wf_confinement=wf_confS)
    atomS.run()

    # atomMo.set_confinement(confMo)
    atomMo.set_wf_confinement(wf_confinement=wf_confMo)
    atomMo.run()
    smallgrid, small = check_basis_completeness(positions_au=mos2_positions, atoms=[atomMo, atomS, atomS], maxl=[0,0,0])
    mediumgrid, medium = check_basis_completeness(positions_au=mos2_positions, atoms=[atomMo, atomS, atomS], maxl=[1,1,1])
    largegrid, large = check_basis_completeness(positions_au=mos2_positions, atoms=[atomMo, atomS, atomS], maxl=[2,2,2])

    print(smallgrid)
    print(small-smallgrid)
    print(mediumgrid)
    print(medium - mediumgrid)
    print(largegrid)
    print(large-largegrid)



    # print("onsite position")
    # print(bohr_to_angstrom(np.reshape(position_atom_pair_block(pos1=pos2, pos2=pos2, atom1=atom, atom2=atom, maxl1=1, maxl2=1), (16,3)))) #onsite
    # print("onsite momentum")
    # # print(np.reshape(momentum_atom_pair_block(pos1=pos1, pos2=pos1, atom1=atom, atom2=atom, maxl1=1, maxl2=1), (16,3))) #onsite
    # print("offsite position")
    # print(bohr_to_angstrom(np.reshape(position_atom_pair_block(pos1=pos2, pos2=pos1, atom1=atom, atom2=atom, maxl1=1, maxl2=1), (16,3)))) #offsite
    # print("offsite momentum")
    # print(np.reshape(momentum_atom_pair_block(pos1=pos1, pos2=pos2, atom1=atom, atom2=atom, maxl1=1, maxl2=1), (16,3))) #offsite
