import sys

code = sys.argv[1].lower()

if 'hotcent' in code:
    from hotcent.atom_hotcent import HotcentAE as AE
elif 'gpaw' in code:
    from hotcent.atom_gpaw import GPAWAE as AE
 
atom = AE('Sn',
          xcname='LDA', 
          configuration='[Kr] 4d10 5s2 5p2',
          valence=['5s', '5p', '4d'],
          scalarrel=True,
          nodegpts=150,
          mix=0.2,
          txt='-',
          )

atom.run()
atom.plot_density(filename='Sn_densities.png')
atom.plot_Rnl(filename='Sn_orbitals.png')
atom.fit_sto('5s', 5, 4, filename='Sn_5s_STO.png')
atom.fit_sto('5p', 5, 4, filename='Sn_5p_STO.png')
atom.fit_sto('4d', 5, 4, filename='Sn_4d_STO.png')
atom.write_hsd(filename='Sn_wfc.hsd')
