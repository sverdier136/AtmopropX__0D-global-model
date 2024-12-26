import numpy as np
from specie import Specie, Species


class Reaction:

    def __init__(self, species, reactives, products, rate_constant, energy_treshold, stoechio_coeffs=None):
        """
        Reaction class
            Inputs : 
                species : instance of class Species, lists all species present 
                reactives : list with all reactives names
                products : list with all products names
                rate_constant : function taking as arguments (T_g, T_e, )
                stoechio_coeffs : stoechiometric coefficients always positive"""
        self.species = species

        self.reactives = [self.species.get_specie_by_name(name) for name in reactives]
        self.reactives_indices = [self.species.get_index_by_instance(sp) for sp in self.reactives]

        self.products = [self.species.get_specie_by_name(name) for name in products]
        self.products_indices = [self.species.get_index_by_instance(sp) for sp in self.products]

        if stoechio_coeffs :
            self.stoechio_coeffs = np.array(stoechio_coeffs)
        else:
            # sets stoechio_coeffs at 1 for reactives and products if not defined
            self.stoechio_coeffs = np.zeros(self.species.nb)
            for i in self.reactives_indices:
                self.stoechio_coeffs[i] = 1
            for sp in self.products_indices:
                self.stoechio_coeffs[i] = 1

        self.energy_threshold = energy_treshold
        self.rate_constant = rate_constant     # func
        #self.

    def density_change_rate(self, T_g, T_e, densities):
        """Returns an np.array with the change rate for each species due to this reaction"""
        K = self.rate_constant(T_g, T_e)
        product = K * np.prod(densities[self.reactives_indices+self.products_indices]) # product of rate constant and densities of all the stuff
        rate = np.zeros(self.species.nb)
        for sp in self.reactives:
            i = self.species.get_index_by_instance(sp)
            rate[i] = - product * self.stoechio_coeffs[i]
        for sp in self.products:
            i = self.species.get_index_by_instance(sp)
            rate[i] = + product * self.stoechio_coeffs[i]
        
        return rate


    def energy_change_rate(self, T_g, T_e, densities):
        """Function meant to return the change in energy due to this specific equation.
            HOWEVER : seems like it is necessary to account for difference in temperature of atoms molecules and electrons...
            Thus 1 function per "Temperatur type" will be needed"""
        pass
        # K = self.rate_constant(T_g, T_e)
        # product = K * np.prod(densities[self.reactives_indices+self.products_indices]) # product of rate constant and densities of all the stuff
        # rate = np.zeros(self.species.nb)
        # for sp in self.reactives:
        #     rate[self.species.get_index_by_instance(sp)] = - product
        # for sp in self.products:
        #     rate[self.species.get_index_by_instance(sp)] = + product
        
        # return rate


if __name__ == "__main__":
    def K(T_g, T_e):
        return 2
    
    species_list = Species([Specie("I0", 10.57e-27, 0), Specie("I1", 10.57e-27, 0), Specie("I2", 10.57e-27, 0), Specie("I3", 10.57e-27, 0), Specie("I4", 10.57e-27, 0), Specie("I5", 10.57e-27, 0)])

    reac = Reaction(species_list, ["I2", "I4"], ["I5"], K, 10)
    densities = np.array([1,2,3,4,5,6])
    print(reac.change_rate(1,1,densities))
