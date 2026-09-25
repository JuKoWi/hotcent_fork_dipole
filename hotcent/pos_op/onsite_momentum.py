from hotcent.atomic_dft import AtomicBase
from hotcent.pos_op.slako_new import symbol_to_l
from hotcent.interpolation import build_interpolator
from hotcent.pos_op.utils import dim_atom_basis
from hotcent.pos_op.rotation_transform import transform_to_real
import numpy as np
import scipy as sc

def onsite_momentum(atom:AtomicBase, fname):
    """calculates the onsite components of momentum in atomic units. Basis functions with real spherical harmonics. 
    There is one block for each allowed combination of l:
    blocks: sp, pd, df 
    and a subblock for each component. Since momentum should be hermitian, only save the blocks above the 
    diagonal:
    013 # angular momenta covered by the valence set
    sp:
    #x  R I R I R I 
    #y  R I R I R I 
    #z  R I R I R I 
    pd:
    #x R I R I R I R I R I 
    #x R I R I R I R I R I 
    #x R I R I R I R I R I 
    
    #y R I R I R I R I R I 
    #y R I R I R I R I R I 
    #y R I R I R I R I R I 

    #z R I R I R I R I R I 
    #z R I R I R I R I R I 
    #z R I R I R I R I R I 
    df:
    ...
    
    """
    valence = atom.valence
    valence.sort(key=lambda v: symbol_to_l(v))
    angulars = [symbol_to_l(v) for v in valence]
    blocktitles = {0: 'sp', 1: 'pd', 2:'df'}
    components = (0,1,2) # corresponds to order x,y,z
    #TODO: no angular momentum twice
    #TODO: make also work for non-contiguous valence

    #build full matrix, then check hermiticity
    l_max = max(angulars)
    dim = dim_atom_basis(l_max)
    full_momentum = np.zeros((3, dim, dim), dtype=complex)
    for i, nl_symb in enumerate(valence):
        if symbol_to_l(nl_symb) + 1 in angulars:
            lbra = symbol_to_l(nl_symb)
            lket = lbra+1
            idx_bra1 = dim_atom_basis(lbra-1)
            idx_bra2 = dim_atom_basis(lbra)
            idx_ket1 = dim_atom_basis(lket-1)
            idx_ket2 = dim_atom_basis(lket)
            for c in components:
                block = calculate_block(lbra=lbra, lket=lket, atom=atom, component=c)
                full_momentum[c, idx_bra1:idx_bra2, idx_ket1:idx_ket2] = block
        if symbol_to_l(nl_symb) -1 in angulars:
            lbra = symbol_to_l(nl_symb)
            lket = lbra-1
            idx_bra1 = dim_atom_basis(lbra-1)
            idx_bra2 = dim_atom_basis(lbra)
            idx_ket1 = dim_atom_basis(lket-1)
            idx_ket2 = dim_atom_basis(lket)
            for c in components:
                block = calculate_block(lbra=lbra, lket=lket, atom=atom, component=c)
                full_momentum[c, idx_bra1:idx_bra2, idx_ket1:idx_ket2] = block
    X = transform_to_real()
    X = np.array(X, dtype=complex)[:dim, :dim]
    full_momentum_real_harmonics = np.einsum("ab, xbc, dc->xad", X, full_momentum, X.conjugate())
    hermitian_deviation = np.max(np.abs(full_momentum_real_harmonics - np.transpose(full_momentum_real_harmonics, axes=(0,2,1)).conjugate()))
    print(f"hermitian up  to {hermitian_deviation}")

    # select nonzero-blocks from above diagonal
    with open(fname, 'w') as file:
        l_string = ''
        for a in angulars:
            l_string += str(a)
        print(l_string, file=file)
        for i, nl_symb in enumerate(valence):
            if symbol_to_l(nl_symb) + 1 in angulars:
                print(blocktitles[symbol_to_l(nl_symb)], file=file)
                lbra = symbol_to_l(nl_symb)
                lket = lbra+1
                idx_bra1 = dim_atom_basis(lbra-1)
                idx_bra2 = dim_atom_basis(lbra)
                idx_ket1 = dim_atom_basis(lket-1)
                idx_ket2 = dim_atom_basis(lket)
                np.savetxt(X=full_momentum_real_harmonics[0,idx_bra1:idx_bra2, idx_ket1:idx_ket2].view(float), fname=file, delimiter='\t')
                file.write("\n")
                np.savetxt(X=full_momentum_real_harmonics[1,idx_bra1:idx_bra2, idx_ket1:idx_ket2].view(float), fname=file, delimiter='\t')
                file.write("\n")
                np.savetxt(X=full_momentum_real_harmonics[2,idx_bra1:idx_bra2, idx_ket1:idx_ket2].view(float), fname=file, delimiter='\t')
                file.write("\n")
    return full_momentum_real_harmonics

def load_onsite_momentum(file):
    lbra = {"sp": 0, "pd": 1, "df": 2}
    lmax = 3
    n = dim_atom_basis(lmax + 1)
    full_momentum = np.zeros((3, n, n), dtype=complex)

    with open(file, 'r') as f:
        f.readline()  # header
        while True:
            blockstring = f.readline().strip()
            if blockstring not in lbra:
                break  # end of file or unknown block label
            l = lbra[blockstring]

            dim_bra = 2 * l + 1
            dim_ket = 2 * (l + 1) + 1
            block = np.zeros((3, dim_bra, dim_ket), dtype=complex)
            for i in range(3):
                for j in range(dim_bra):
                    line = np.array(f.readline().split(), dtype=float)
                    block[i, j, :] = line[0::2] + 1j * line[1::2]
                f.readline()  # empty line

            idx_bra1 = dim_atom_basis(l - 1)
            idx_bra2 = dim_atom_basis(l)
            idx_ket1 = dim_atom_basis(l)
            idx_ket2 = dim_atom_basis(l + 1)

            full_momentum[:, idx_bra1:idx_bra2, idx_ket1:idx_ket2] = block
            full_momentum[:, idx_ket1:idx_ket2, idx_bra1:idx_bra2] = block.transpose(0, 2, 1).conj()

    return full_momentum
        
def calculate_block(lbra, lket, atom, component):
    block = np.zeros((2*lbra+1, 2*lket+1), dtype=complex)
    mbra = range(-lbra, lbra+1)
    mket = range(-lket, lket+1)
    for i, mb in enumerate(mbra):
        for j, mk in enumerate(mket):
            #x
            if (component == 0) and (abs(mb-mk) == 1):
                sign = 1 if (mb-mk) > 0 else -1
                if lbra == (lket+1):
                    block[i,j] = -sign* 0.5 * np.sqrt((lket + sign * mk +1) * (lket + sign * mk +2)/((2 * lket + 1) * (2*lket + 3))) * A_integral(atom, lbra, lket) 
                elif lbra == (lket -1):
                    block[i,j] = sign* 0.5 * np.sqrt((lket - sign* mk -1) * (lket - sign * mk)/((2*lket +1) * (2*lket + -1))) * B_integral(atom, lbra, lket)
            #y
            if (component == 1) and (abs(mb-mk) == 1):
                sign = 1 if (mb-mk) > 0 else -1
                if lbra == (lket+1):
                    block[i,j] = 1j * 0.5 * np.sqrt((lket + sign * mk +1) * (lket + sign * mk +2)/((2 * lket + 1) * (2*lket + 3))) * A_integral(atom, lbra, lket) 
                elif lbra == (lket -1):
                    block[i,j] = -1j* 0.5 * np.sqrt((lket - sign* mk -1) * (lket - sign * mk)/((2*lket +1) * (2*lket -1))) * B_integral(atom, lbra, lket)
            #z
            if (component == 2) and (abs(mb-mk) == 0):
                if lbra == (lket+1):
                    block[i,j] = np.sqrt((lket - mk +1) * (lket + mk +1)/((2*lket + 1) * (2*lket +3))) * A_integral(atom, lbra, lket)
                elif lbra == (lket-1):
                    block[i,j] = np.sqrt((lket - mk) * (lket +mk)/((2*lket +1) * (2*lket -1))) * B_integral(atom, lbra, lket) 
    return -1j * block

    
def R_spline(atom, nl):
    assert atom.solved, "Attribute is missing, please call run method first"
    if atom.Rnl_fct[nl] is None:
        rc = atom.rcutnl[nl] if nl in atom.rcutnl else None
        atom.Rnl_fct[nl] = build_interpolator(atom.rgrid, atom.Rnlg[nl], rc)
    return atom.Rnl_fct[nl]

def A_integral(atom, lbra, lket):
    nl_bra = [v for v in atom.valence if symbol_to_l(v) == lbra]
    nl_ket = [v for v in atom.valence if symbol_to_l(v) == lket]
    if (len(nl_bra) > 1) or (len(nl_ket) > 1):
        raise ValueError("Only one subshell per angular momentum implemented for valence sets")
    nl_bra = nl_bra[0]
    nl_ket = nl_ket[0]
    splinebra = R_spline(atom=atom, nl=nl_bra)
    splineket = R_spline(atom=atom, nl=nl_ket)
    r = atom.rgrid
    ket_dr = splineket.derivative()
    integral = sc.integrate.quad(lambda x: x**2 * splinebra(x) * (ket_dr(x) - lket * splineket(x)/x),0, r[-1])[0]
    return integral

def B_integral(atom, lbra, lket):
    nl_bra = [v for v in atom.valence if symbol_to_l(v) == lbra]
    nl_ket = [v for v in atom.valence if symbol_to_l(v) == lket]
    if (len(nl_bra) > 1) or (len(nl_ket) > 1):
        raise ValueError("Only one subshell per angular momentum implemented for valence sets")
    nl_bra = nl_bra[0]
    nl_ket = nl_ket[0]
    splinebra = R_spline(atom=atom, nl=nl_bra)
    splineket = R_spline(atom=atom, nl=nl_ket)
    r = atom.rgrid
    ket_dr = splineket.derivative()
    integral = sc.integrate.quad(lambda x: x**2 * splinebra(x) * (ket_dr(x) + (lket +1) * splineket(x)/x),0, r[-1])[0]
    # integral = sc.integrate.simpson(r**2 * splinebra(r) * (ket_dr(r) + (lket +1) * splineket(r)/r),x=r)
    return integral
