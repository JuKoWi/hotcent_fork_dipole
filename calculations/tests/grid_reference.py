from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.pos_op.onsite_momentum import R_spline
from hotcent.pos_op.symbolic_integrals import first_center, theta1, phi
from hotcent.pos_op.rotation_transform import to_spherical
from hotcent.pos_op.utils import dim_atom_basis, angstrom_to_bohr
from hotcent.pos_op.mat_elem_evaluation import SlaterKosterIntegrator
import grid 
import numpy as np
import sympy as sp
import h5py
from ase.build import graphene, mx2 
from hotcent.pos_op.utils import angstrom_to_bohr 

MAX_L = {"C": 1, "H": 0, "S": 2, "Mo": 2}

def evaluate_psi(Ylm, rspline, cart_grid):
    """evaluate psi at a set of grid points"""
    spherical_coord = to_spherical(cart_grid)
    r_evaluated = rspline(spherical_coord[:,0])
    var_theta = spherical_coord[:,1]
    var_phi = spherical_coord[:,2]
    Y_evaluated = Ylm(var_theta, var_phi)
    return r_evaluated * Y_evaluated

def grad_psi(Ylm, rspline, cart_grid, comp, h=1e-4):
    """finite difference gradient"""
    e = np.zeros(3)
    e[comp] = h
    return (evaluate_psi(Ylm, rspline, cart_grid+e) - evaluate_psi(Ylm, rspline, cart_grid-e)) / (2*h) 

def atom_grid(atom, npoints):
    oned_grid = grid.onedgrid.GaussLegendre(npoints=npoints)
    rgrid = grid.rtransform.BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned_grid)
    return grid.atomgrid.AtomGrid(rgrid, degrees=[20])

def onsite_norm_grid(atom, nl, Ylm):
    """overlap between two basis states centered at the same atom"""
    return onsite_overlap_grid(atom=atom, nl1=nl, nl2=nl, Ylm1=Ylm, Ylm2=Ylm)

def onsite_overlap_grid(atom, nl1, nl2, Ylm1, Ylm2):
    hotcent_spline1 = R_spline(atom=atom, nl=nl1)
    hotcent_spline2 = R_spline(atom=atom, nl=nl2)
    at_grid = atom_grid(atom=atom, npoints=500)
    psi1 = evaluate_psi(Ylm=Ylm1, rspline=hotcent_spline1, cart_grid=at_grid.points)
    psi2 = evaluate_psi(Ylm=Ylm2, rspline=hotcent_spline2, cart_grid=at_grid.points)
    overlap = psi1 * psi2 
    return at_grid.integrate(overlap)

def onsite_posop_grid(atom, nl1, nl2, Ylm1, Ylm2, cart_component):
    """position operator matrix elements between basis states centered at the same atom"""
    hotcent_spline1 = R_spline(atom=atom, nl=nl1)
    hotcent_spline2 = R_spline(atom=atom, nl=nl2)
    at_grid = atom_grid(atom=atom, npoints=100)
    psi1 = evaluate_psi(Ylm=Ylm1, rspline=hotcent_spline1, cart_grid=at_grid.points)
    psi2 = evaluate_psi(Ylm=Ylm2, rspline=hotcent_spline2, cart_grid=at_grid.points)
    posop = psi1 * psi2 * at_grid.points[:,cart_component]
    return at_grid.integrate(posop)

def onsite_momentum_grid(atom, nl1, nl2, Ylm1, Ylm2, cart_component):
    """momentum in a.u. between two basis functions centered at the same atom"""
    hotcent_spline1 = R_spline(atom=atom, nl=nl1)
    hotcent_spline2 = R_spline(atom=atom, nl=nl2)
    at_grid = atom_grid(atom=atom, npoints=500)
    psi1 = evaluate_psi(Ylm=Ylm1, rspline=hotcent_spline1, cart_grid=at_grid.points)
    partial_psi2 = grad_psi(Ylm=Ylm2, rspline=hotcent_spline2, cart_grid=at_grid.points, comp=cart_component)
    return -1j*at_grid.integrate(psi1 * partial_psi2)

def make_twocenter_grid(atom1, atom2, pos_au_1, pos_au_2, npoints, rmin):
    oned_grid = grid.onedgrid.GaussLegendre(npoints=npoints)
    rgrid = grid.rtransform.BeckeRTransform(rmin, R=1.5).transform_1d_grid(oned_grid)
    if np.allclose(pos_au_1, pos_au_2):
        atnums, atcoords = np.array([atom1.Z]), np.array([pos_au_1])
    else: 
        atnums, atcoords = np.array([atom1.Z, atom2.Z]), np.array([pos_au_1, pos_au_2])
    return grid.MolGrid.from_preset(
        atnums=atnums,
        atcoords=atcoords,
        rgrid=rgrid,
        preset="fine",
        aim_weights=grid.BeckeWeights(),
        store=True,
    )

def pair_psi(mgrid, pos_au_1, pos_au_2, atom1, atom2, nl1, nl2, Ylm1, Ylm2):
    cart_grid1 = mgrid.points - pos_au_1
    cart_grid2 = mgrid.points - pos_au_2
    hotcent_spline1 = R_spline(atom=atom1, nl=nl1)
    hotcent_spline2 = R_spline(atom=atom2, nl=nl2)
    psi1 = evaluate_psi(Ylm=Ylm1, rspline=hotcent_spline1, cart_grid=cart_grid1)
    psi2 = evaluate_psi(Ylm=Ylm2, rspline=hotcent_spline2, cart_grid=cart_grid2)
    return psi1, psi2, hotcent_spline2

def twocenter_overlap(atom1, atom2, nl1, nl2, Ylm1, Ylm2, pos_au_1, pos_au_2):
    """overlap between two basis functions centered at different atoms"""
    mgrid = make_twocenter_grid(atom1=atom1, atom2=atom2, pos_au_1=pos_au_1, pos_au_2=pos_au_2, npoints=500, rmin=0.0)
    psi1, psi2, _ = pair_psi(mgrid=mgrid, pos_au_1=pos_au_1, pos_au_2=pos_au_2, atom1=atom1, atom2=atom2, nl1=nl1, nl2=nl2, Ylm1=Ylm1, Ylm2=Ylm2)
    return mgrid.integrate(psi1 * psi2)

def twocenter_position(atom1, atom2, nl1, nl2, Ylm1, Ylm2, pos_au_1, pos_au_2, cart_component):
    """position matrix elements between two basis functions centered at different atoms"""
    mgrid = make_twocenter_grid(atom1=atom1, atom2=atom2, pos_au_1=pos_au_1, pos_au_2=pos_au_2, npoints=500, rmin=0.0)
    psi1, psi2, _ = pair_psi(mgrid=mgrid, pos_au_1=pos_au_1, pos_au_2=pos_au_2, atom1=atom1, atom2=atom2, nl1=nl1, nl2=nl2, Ylm1=Ylm1, Ylm2=Ylm2)
    return mgrid.integrate(psi1 * psi2 * mgrid.points[:,cart_component])

def twocenter_momentum(atom1, atom2, nl1, nl2, Ylm1, Ylm2, pos_au_1, pos_au_2, cart_component):
    """momentum in a.u. between two basis functions centered at different atoms"""
    mgrid = make_twocenter_grid(atom1=atom1, atom2=atom2, pos_au_1=pos_au_1, pos_au_2=pos_au_2, npoints=500, rmin=0.0)
    psi1, psi2, _ = pair_psi(mgrid=mgrid, pos_au_1=pos_au_1, pos_au_2=pos_au_2, atom1=atom1, atom2=atom2, nl1=nl1, nl2=nl2, Ylm1=Ylm1, Ylm2=Ylm2)
    partial_psi2 = mgrid.interpolate(psi2)(mgrid.points, deriv=1)[:,cart_component]
    momentum = psi1 * partial_psi2
    return -1j*mgrid.integrate(momentum)

def twocenter_momentum_finite_diff(atom1, atom2, nl1, nl2, Ylm1, Ylm2, pos_au_1, pos_au_2, cart_component):
    """momentum in a.u. between two basis functions centered at different atoms
        finite difference for gradient
    """
    mgrid = make_twocenter_grid(atom1=atom1, atom2=atom2, pos_au_1=pos_au_1, pos_au_2=pos_au_2, npoints=500, rmin=0.0)
    psi1, _, spline2 = pair_psi(mgrid=mgrid, pos_au_1=pos_au_1, pos_au_2=pos_au_2, atom1=atom1, atom2=atom2, nl1=nl1, nl2=nl2, Ylm1=Ylm1, Ylm2=Ylm2)
    partial_psi2 = grad_psi(Ylm=Ylm2, rspline=spline2, cart_grid=mgrid.points - pos_au_2, comp=cart_component)
    momentum = psi1 * partial_psi2
    return -1j*mgrid.integrate(momentum)

def _lambdify_basis():
    return {key: sp.lambdify((theta1, phi), val[0]) for key, val in first_center.items()}

def sanity_report(S, r, p, label):
    print(f"[{label}] max|S - S^T|          = {np.abs(S - S.T).max():.2e}")
    print(f"[{label}] max|diag(S) - 1|      = {np.abs(np.diag(S) - 1).max():.2e}")
    print(f"[{label}] max|r_c - r_c^H|      = {max(np.abs(rc - rc.conj().T).max() for rc in r):.2e}")
    print(f"[{label}] max|p_c - p_c^H|      = {max(np.abs(pc - pc.conj().T).max() for pc in p):.2e}") 
    print(f"[{label}] max|Re p|, max|diag p| = {np.abs(p.real).max():.2e}, {max(np.abs(np.diag(pc)).max() for pc in p):.2e}")

def _basis_lists(atoms:list, positions_au, maxl):
    """Per-basis-function lists (atom, position, Ylm, nl), shared by both checks."""
    Y = _lambdify_basis()
    atom_list, pos_list, Ylm_list, nl_list = [], [], [], []
    for i, b in enumerate(atoms):
        for j, lm in enumerate(list(first_center.keys())):
            if j == dim_atom_basis(maxl[i]):
                break
            if lm[0] not in [s[1] for s in b.valence]:
                raise ValueError(f"l={lm[0]} is within maxl={maxl[i]} but not in the valence of {getattr(b, 'symbol', b)}")
            atom_list.append(b)
            pos_list.append(positions_au[i])
            Ylm_list.append(Y[lm])
            nl_list.append([s for s in b.valence if s[1] == lm[0]][0])
    return atom_list, np.array(pos_list), Ylm_list, nl_list

def check_basis_completeness(positions_au, atoms:list, maxl):
    dim_total = sum([dim_atom_basis(l) for l in maxl])
    S = np.zeros((dim_total, dim_total))
    r = np.zeros((3, dim_total, dim_total))

    atom_list, pos_list, Ylm_list, nl_list = _basis_lists(atoms=atoms, positions_au=positions_au, maxl=maxl)

    for i, a in enumerate(atom_list):
        for j, b in enumerate(atom_list):
            if np.allclose(pos_list[i], pos_list[j]):
                S[i,j] = 1 if i ==j else 0
            else:
                S[i,j] = twocenter_overlap(atom1=a, atom2=b, nl1=nl_list[i], nl2=nl_list[j], Ylm1=Ylm_list[i], Ylm2=Ylm_list[j], pos_au_1=pos_list[i], pos_au_2=pos_list[j])
            for c in range(3):
                if np.allclose(pos_list[i], pos_list[j]):
                    delta = 1 if i==j else 0
                    r[c,i,j] = delta*pos_list[i,c] *S[i,j] + onsite_posop_grid(atom=a, nl1=nl_list[i], nl2=nl_list[j], Ylm1=Ylm_list[i], Ylm2=Ylm_list[j], cart_component=c)
                else:
                    r[c,i,j] = twocenter_position(atom1=a, atom2=b, nl1=nl_list[i], nl2=nl_list[j], Ylm1=Ylm_list[i], Ylm2=Ylm_list[j], pos_au_1=pos_list[i], pos_au_2=pos_list[j], cart_component=c)

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
            psi = evaluate_psi(Ylm=Ylm_list[i], rspline=hotcent_spline, cart_grid=cart_grid)   
            x = mgrid.points[:,c]
            psi_total = x * psi
            x2 = mgrid.integrate(psi_total**2)                          
            norm = mgrid.integrate((psi * (x-pos_list[i][c]))**2)
            for j,b in enumerate(atom_list):
                cart_grid = mgrid.points - pos_list[j]
                hotcent_spline = R_spline(atom=b, nl=nl_list[j])
                psi_total -= coeffs[c,j,i] * evaluate_psi(Ylm=Ylm_list[j], rspline=hotcent_spline, cart_grid=cart_grid)
            diff_grid[c, i] = mgrid.integrate(psi_total**2)/norm
            diff[c, i] = (x2 - RSR_diag[c, i])/norm                    
    with h5py.File("bs_completeness.hdf5", 'w') as f:
        f.create_dataset('diff', data=diff)
        f.create_dataset('diff_grid', data=diff_grid)
        f.create_dataset('nl', data=nl_list)
        f.create_dataset('atoms', data=[a.symbol for a in atom_list])
        f.attrs['lmax'] = maxl

    return diff_grid, diff

def check_all_r_p_S(filename, atom_dict:dict, ase_atoms):
    """
        atom_dict: dict with hotcent atoms as values and element symbols as keys
        ase_atoms: ase.Atoms object
    """
    #perform slater koster calculation
    max_l_dict = {"C": 1, "H": 0, "S": 2, "Mo": 2}
    sk_integrals = SlaterKosterIntegrator(
        ase_atoms,
        skpath="sk_hdf5",
        maxl_dict=max_l_dict,
        format="hdf5",
    )
    r_hotcent = angstrom_to_bohr(sk_integrals._calculate_lattice_dict("r")[(0,0,0)])
    S_hotcent = sk_integrals._calculate_lattice_dict("S")[(0,0,0)]
    p_hotcent = sk_integrals._calculate_lattice_dict("p")[(0,0,0)]

    # #get list of atom wise information
    positions_au = angstrom_to_bohr(ase_atoms.get_positions())
    atoms = [atom_dict[symb] for symb in ase_atoms.get_chemical_symbols()]
    maxl = [max_l_dict[symb] for symb in ase_atoms.get_chemical_symbols()]

    # prepare iterations over all functions in the basis
    dim_total = sum([dim_atom_basis(l) for l in maxl])
    S = np.zeros((dim_total, dim_total), dtype=complex)
    r = np.zeros((3, dim_total, dim_total), dtype=complex)
    p = np.zeros((3, dim_total, dim_total), dtype=complex)
    atom_list = []
    for j, b in enumerate(atoms):
        atom_list += [b for j in range(dim_atom_basis(maxl[j]))]
    pos_list =[]
    for j,pos in enumerate(positions_au):
        pos_list += [pos for j in range(dim_atom_basis(maxl[j]))] 
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
            if np.allclose(pos_list[i],pos_list[j]):
                S[i,j] = onsite_overlap_grid(atom=a, nl1=nl_list[i], nl2=nl_list[j], Ylm1=Ylm_list[i], Ylm2=Ylm_list[j])
            else:
                S[i,j] = twocenter_overlap(atom1=a, atom2=b, nl1=nl_list[i], nl2=nl_list[j], Ylm1=Ylm_list[i], Ylm2=Ylm_list[j], pos_au_1=pos_list[i], pos_au_2=pos_list[j])
            for c in range(3):
                r[c,i,j] = twocenter_position(atom1=a, atom2=b, nl1=nl_list[i], nl2=nl_list[j], Ylm1=Ylm_list[i], Ylm2=Ylm_list[j], pos_au_1=pos_list[i], pos_au_2=pos_list[j], cart_component=c)
                p[c,i,j] = twocenter_momentum_finite_diff(atom1=a, atom2=b, nl1=nl_list[i], nl2=nl_list[j], Ylm1=Ylm_list[i], Ylm2=Ylm_list[j], pos_au_1=pos_list[i], pos_au_2=pos_list[j], cart_component=c)

    sanity_report(S, r, p, "grid")
    sanity_report(S_hotcent, r_hotcent, p_hotcent, "dftb")

    with h5py.File(filename, 'w') as f:
        dset_positions = f.create_dataset("positions", data=positions_au)
        dset_positions.attrs['unit'] = 'a.u.'
        dset_atomtypes = f.create_dataset("atom_types", data=ase_atoms.get_chemical_symbols())
        dset_maxl = f.create_dataset("maxl", data=maxl)

        dftb = f.create_group("dftb")
        dset_r_hotcent = dftb.create_dataset("r_dftb", data=r_hotcent)
        dset_r_hotcent.attrs['unit'] = 'a.u.'
        dset_p_hotcent = dftb.create_dataset("p_dftb", data=p_hotcent)
        dset_p_hotcent.attrs['unit'] = 'a.u.'
        dset_S_hotcent = dftb.create_dataset("S_dftb", data=S_hotcent)

        grid_group = f.create_group("grid")
        dset_r_grid = grid_group.create_dataset("r_grid", data=r)
        dset_r_grid.attrs['unit'] = 'a.u.'
        dset_p_grid = grid_group.create_dataset("p_grid", data=p)
        dset_r_grid.attrs['unit'] = 'a.u.'
        dset_S_grid = grid_group.create_dataset("S_grid", data=S)


if __name__ == "__main__":
    xc = "GGA_X_PBE+GGA_C_PBE"

    # # Get KS all-electron ground state of confined atom:
    # element = "C"
    # conf = PowerConfinement(r0=50.0, s=4)
    # r0 = 3.2  # Bohr
    # wf_conf = {
    #     "2s": PowerConfinement(r0=r0, s=8.2),
    #     "2p": PowerConfinement(r0=r0, s=8.2),
    # }
    # atomC = AtomicDFT(
    #     element,
    #     xc=xc,
    #     confinement=conf,
    #     perturbative_confinement=False,
    #     configuration="[He] 2s2 2p2",
    #     valence=["2s", "2p"],
    #     scalarrel=True,
    #     maxiter=2500,
    #     timing=False,
    #     nodegpts=1500,
    #     mix=0.2,
    #     txt="-",
    # )
    # # atom.rmin = 1e-6/atom.Z
    # # atom.rmax = 150
    # atomC.run()
    # atomC.set_confinement(conf)
    # atomC.set_wf_confinement(wf_confinement=wf_conf)
    # atomC.run()
    # graphene = graphene("CC", size=(1, 1, 1), vacuum=10)

    # check_basis_completeness(positions_au=[[0,0,0]], atoms=[atomC], maxl=[1])
    # check_all_r_p_S(filename="graphene_rpS.hdf5", atom_dict={'C': atomC}, ase_atoms=graphene)

    # atomS = AtomicDFT(
    #     "S",
    #     xc=xc,
    #     perturbative_confinement=False,
    #     confinement=PowerConfinement(r0=50, s=4),
    #     configuration="[Ne] 3s2 3p4 3d0",
    #     valence=["3s", "3p", "3d"],
    #     scalarrel=True,
    #     maxiter=2500,
    #     timing=False,
    #     # nodegpts=2500,
    #     mix=0.2,
    #     txt="-",
    #     # rmax=500,
    # )
    # atomS.run()
    # rcovS = 3.9
    # wf_confS = {
    #     "3s": PowerConfinement(r0=rcovS, s=4.6),
    #     "3p": PowerConfinement(r0=rcovS, s=4.6),
    #     "3d": PowerConfinement(r0=rcovS, s=4.6),
    # }
    # atomS.set_wf_confinement(wf_confinement=wf_confS)
    # atomS.run()

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
        # nodegpts=2500,
        mix=0.2,
        txt="-",
        # rmax=100,
    )
    atomMo.run()
    rcovMo = 4.3
    wf_confMo = {
        "4d": PowerConfinement(r0=rcovMo, s=11.6),
        "5s": PowerConfinement(r0=rcovMo, s=11.6),
        "5p": PowerConfinement(r0=rcovMo, s=11.6),
    }
    # Use parameters from 10.1021/ct4004959 (Heine 2013)
    atomMo.set_wf_confinement(wf_confinement=wf_confMo)
    atomMo.run()

    MoS2 = mx2(vacuum=20)
    # check_all_r_p_S(filename="mos2_rpS.hdf5", atom_dict={'Mo': atomMo, 'S': atomS}, ase_atoms=MoS2)
    check_basis_completeness(positions_au=[[0,0,0]], atoms=[atomMo], maxl=[1])

