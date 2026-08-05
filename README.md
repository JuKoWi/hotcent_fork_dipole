# Hotcent with Position Elements

This project is an extension of *Hotcent*, a tool for generating tight-binding parameter files to be used in DFTB.
Compared to Hotcent it additionally generates parameters for the position operator elements (dipole elements) between the basis functions. 

The scripts in `hotcent/pos_op` are those that are required to do the position operator calculations.

Tutorials cover example calculations for a homoatomic system (graphene, tutorial_1) and a heteroatomic system (MoS2, tutorial_2)

## The original Hotcent

Hotcent was originally based on parts of the [Hotbit](
https://github.com/pekkosk/hotbit/) code but has much evolved
since then.

Hotcent was initially developed as part of the following study:

M. Van den Bossche, J. Chem. Phys. A. **2019**, 123 (13), 3038-3045
[(doi)](https://dx.doi.org/10.1021/acs.jpca.9b00927).

The code has then been considerably expanded and reworked for:

M. Van den Bossche, J. Chem. Theory Comput. **2024**
[(doi)](https://doi.org/10.1021/acs.jctc.4c00018).

Please consider citing the 2019 paper when using Hotcent version v1.0
(and earlier) and the 2024 paper for later Hotcent versions.

