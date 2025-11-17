import numpy as np
from ase import Atoms
from ase.build import graphene
from ase.visualize import view

class Matrix_R:
    """calculate twocenter integrals between localized basis functions"""
    def __init__(self, geometry):
        pass

    def run(self, cutoff, mat_type):
        """returns """
        pass

class Simulation:
    def __init__(self, unit_cell):
        self.supercell = unit_cell
    def create_supercell(self):
        pass
    def basis_indexing(self):
        """assign """
    def view_system(self):
        print(self.supercell.positions())
        view(self.supercell)

def create_system():
    structure = graphene()
    return structure

structure = create_system()
mysys = Simulation(unit_cell=structure)