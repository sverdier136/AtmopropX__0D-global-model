

class Specie:

    def __init__(self, name, mass, charge, nb_atoms):
        self.name = name
        self.mass = mass
        self.charge = charge
        self.nb_atoms = nb_atoms
        self.index = None

class Species:

    def __init__(self, species_list: list[Specie]):
        """Contains all species present in the problem. Should contain the electrons."""
        self.species = species_list
        self.names = [sp.name for sp in self.species]

        for i, sp in enumerate(self.species):
            sp.index = i

    @property
    def nb(self):
        """Number of species considered (electrons included)"""
        return len(self.species)

    def add(self, specie):
        self.species.append(specie)

    def get_index_by_name(self, name):
        for i, sp in enumerate(self.species):
            if sp.name == name:
                return i
        raise Exception(f"There is no {name} in this species group")
    
    def get_index_by_instance(self, inst):
        for i, sp in enumerate(self.species):
            if sp is inst:
                return i
        raise Exception(f"There is no {inst} in this species group")
    
    def get_specie_by_name(self, name):
        for sp in self.species:
            if sp.name == name:
                return sp
        raise Exception(f"There is no {name} in this species group")

# species_list = Species([Specie("I0", 10.57e-27, 0), Specie("I1", 10.57e-27, 0), Specie("I2", 10.57e-27, 0), Specie("I3", 10.57e-27, 0), Specie("I4", 10.57e-27, 0), Specie("I5", 10.57e-27, 0)])
