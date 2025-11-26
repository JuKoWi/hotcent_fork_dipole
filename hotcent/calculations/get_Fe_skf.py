from optparse import OptionParser
from ase.units import Bohr
from ase.data import covalent_radii, atomic_numbers
from hotcent.offsite_twocenter import Offsite2cTable
from hotcent.confinement import PowerConfinement
from hotcent.atomic_dft import AtomicDFT
from hotcent.new_dipole.offsite_twocenter_new import Offsite2cTable


# Get KS all-electron ground state of confined atom:
element = 'Fe'
r0 = 1.85 * covalent_radii[atomic_numbers[element]] / Bohr
atom = AtomicDFT(element,
                 confinement=PowerConfinement(r0=r0, s=2),
                 perturbative_confinement=False,
                 configuration='[Ar] 3d7 4s1 4p0 ',
                 valence=['3d', '4s', '4p'],
                 timing=True,
                 )
atom.run()

# Compute Slater-Koster integrals:
rmin, dr, N = 0.5, 0.05, 250
off2c = Offsite2cTable(atom, atom, timing=True)
off2c.run(rmin, dr, N)
off2c.write(dftbplus_format=True, eigenvalues=atom.enl)  # writes to default Fe-Fe_offsite2c.skf filename


