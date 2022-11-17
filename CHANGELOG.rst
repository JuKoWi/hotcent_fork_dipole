=============
Release notes
=============


Development version
===================

* Extended the Slater-Koster tabulation procedure to *f* electrons.
  If one of the two atoms contains *f* electrons in the valence,
  the SKF files are now written in the 'extended' format instead
  of the 'simple' format.

* Maintenance related to changes in ASE, Matplotlib and LibXC.

* Bugfix: there was an off-by-one error in the number of zero-valued
  integrals printed in the Slater-Koster files for distances below
  'rmin'.

* The (optional) PyLibXC dependency now needs to be v5 or above.

* Bugfix: gradient corrections to the exchange-correlation potential
  for two-center integrals were wrong.

* The hotcent.slako.SlaterKosterTable class has been moved to
  hotcent.offsite_twocenter.Offsite2cTable. The default
  '<el1>-<el2>_no_repulsion.skf' template for the corresponding
  Slater-Koster file name has furthermore been changed to
  '<el1>-<el2>_offsite2c.skf' and the default superposition scheme
  is changed to density superposition.

* Dropped support for writing '.par' Slater-Koster files
  (only the SKF format remains).

* Added functionality for non-minimal basis sets (see
  AtomicBase.generate_nonminimal_basis()). Currently just a
  split-valence scheme for double-zeta basis sets is implemented.

* Hubbard parameters (U values) are now calculated as derivatives of
  the corresponding Hamiltonian matrix elements in the isolated atom
  (no longer as second derivatives of the total energy). The present
  version may hence produce U values that are slightly different from
  previous versions.

* Added the possibility to calculate spin constants (W values).
  This requires the otherwise optional PyLibXC dependency.

* Implemented analytical methods for Hubbard value and spin constant
  calculations based on the (Hartree-)XC kernel (instead of finite
  differentiation of atomic eigenvalues).

* Added the possibility to calculate higher-order corrections to
  Hamiltonian and (Hartree-) XC kernel matrix elements:

  - three-center expansion for off-site H integrals
  - two- and three-center expansions for on-site H integrals
  - two-center expansion for on- and off-site "gamma" and "W"
    integrals associated with the SCC and spin polarization energies,
    respectively.

* Added the possibility to calculate the needed (Hartree-)XC kernel
  and moment integrals when the difference density is expanded in
  spherical-harmonic multipoles (up to and including quadrupole moments
  and with as many radial functions as the 'zeta' count of the main
  basis set).

* Changed the tail smoothening procedure to a moving average for
  simplicity and robustness.


Version 1.0
===========

26 December 2019

* The GPAW-based atomic DFT calculator has been removed, as it was
  simply no longer needed (the native calculator now runs as
  fast and is easier to extend with more functionality).

* The native atomic DFT calculator has been renamed to 'AtomicDFT'
  and now resides in hotcent.atomic_dft.py. Its 'xcname' keyword
  argument (for the exchange-correlation functional) has furthermore
  been shortened to 'xc'.

* Calculation of Hubbard values (see examples/hubbard.py).

* Optional C-extensions allow for significantly faster atomic
  atomic DFT calculations (8-10x speedup) and construction of
  Slater-Koster tables (2-3x speedup).

* A new `stride` option for SlaterKosterTable.run() with default = 1.
  Setting higher integer values for `stride` means that the
  H and S integrals are only be explicitly calculated every
  `stride` points and will then be mapped on the final, denser grid
  using cubic spline interpolation. Use with care.

* No more backwards compatibility with Python2 (only Python3).

* The SlaterKosterTable.run() method now requires rmin, dr and N
  as arguments to specify the interatomic distances for which the
  Slater-Koster integrals are tabulated (rmin being the minimal
  distance, dr the grid spacing dr, and N the number of grid points).

* Major restructuring of hotcent.tools for fitting the confinement
  parameters, with also a new tutorial on this topic (`tutorial_2.ipynb`).


Version 0.9
===========

03 December 2019

* Start of versioning.
