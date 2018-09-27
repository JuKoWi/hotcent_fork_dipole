# Hotcent

Calculating one- and two-center Slater-Koster integrals,
based on parts of the [Hotbit](https://github.com/pekkosk/hotbit/) 
code.


## Features

* The main use for Hotcent is generating Slater-Koster tables,
typically in the [".skf"](https://www.dftb.org/fileadmin/DFTB/public/misc/slakoformat.pdf)
format to be used with DFTB codes such as DFTB+.

* The code allows to use different confinement potentials for
the different valence wave functions and for the electron density 
(which determines the effective potential used in the Hamiltonian 
integrals). 

* Both the potential superposition and density superposition
schemes are available. 

* With regards to exchange-correlation functionals, the PW92
(LDA) functional is natively available, and other functionals
can be applied through integration with the 
[GPAW](https://wiki.fysik.dtu.dk/gpaw/) code.


## Installation

* Set up the [ASE](https://wiki.fysik.dtu.dk/ase/) Python module. 

* Clone / download the Hotcent repository and update the `$PYTHONPATH` 
accordingly.

* If you want to use functionals other than LDA, Python must be able 
to import from the GPAW module. The GPAW C-code does not need to 
be compiled, and the PAW datasets are not needed either.
