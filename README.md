# Hotcent

Calculating one- and two-center Slater-Koster integrals,
based on parts of the [Hotbit](https://github.com/pekkosk/hotbit/) 
code. The development of Hotcent was started as part of the
following study: 

M. Van den Bossche, J. Chem. Phys. A. **2019**, 123 (13), 3038-3045
[(doi)](https://dx.doi.org/10.1021/acs.jpca.9b00927).


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
[GPAW](https://wiki.fysik.dtu.dk/gpaw/) code. In the potential
superposition scheme, it should be possible to use any of the 
functionals available through GPAW (and LibXC if GPAW has been
linked to it). When applying the density superposition scheme,
the choice is currently restricted to LDA- and GGA-type functionals.
LibXC functionals can now also be accessed via the PyLibXC module
(see hotcent.pylibxc_interface).


## Installation

* Set up the [ASE](https://wiki.fysik.dtu.dk/ase/) Python module. 

* Clone / download the Hotcent repository and update the `$PYTHONPATH` 
accordingly.

* If you want to use functionals other than LDA, the GPAW code must
be installed. 
