# hotcent

Calculating one- and two-center Slater-Koster integrals,
based on parts of the Hotbit code.


## Features:

* The main use for Hotcent is generating Slater-Koster tables,
typically in the [".skf" format](https://www.dftb.org/fileadmin/DFTB/public/misc/slakoformat.pdf)
to be used with DFTB codes such as DFTB+.

* The code allows to use different confinement potentials for
the different valence wave functions and for the electron density 
(which determines the effective potential used in the Hamiltonian 
integrals). 

* Both the potential- and density-superposition
schemes are available. 

* Currently only the PW92 (LDA) functional is implemented.


## Installation:

Just clone / download and update the $PYTHONPATH.
