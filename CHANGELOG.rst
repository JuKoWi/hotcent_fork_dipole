=============
Release notes
=============


Development version
===================

* The GPAW-based atomic DFT calculator has been removed, as it was
  simply no longer needed (the native calculator now runs as
  fast and is easier to extend with more functionality)

* The native atomic DFT calculator has been renamed to 'AtomicDFT'
  and now resides in hotcent.atomic_dft.py

* Calculation of Hubbard values (see examples/hubbard.py)

* Optional C-extensions allow for significantly faster (5-7x)
  atomic DFT calculations


Version 0.9
===========

03 December 2019

* Start of versioning.
