import numpy as np
from numpy.typing import NDArray
from specie import Specie, Species


class Reaction:
    """
        Reaction class
            /!\ Electrons should be added to reactives and products only if they are not spectators (otherwise pbs with density_rate_change)
            Inputs : 
                species : instance of class Species, lists all species present 
                reactives : list with all reactives names
                products : list with all products names
                rate_constant : function taking as arguments (T_g, T_e, )
                stoechio_coeffs : stoechiometric coefficients always positive
    """

    def __init__(self, species, reactives, products, rate_constant, energy_treshold, stoechio_coeffs=None):
        """
        Reaction class
        /!\ Electrons should be added to reactives and products only if they are not spectators (otherwise pbs with density_rate_change)
            Inputs : 
                species : instance of class Species, lists all species present 
                reactives : list with all reactives names
                products : list with all products names
                rate_constant : function taking as arguments (T_g, T_e, )
                stoechio_coeffs : stoechiometric coefficients always positive"""
        self.species = species

        self.reactives = [self.species.get_specie_by_name(name) for name in reactives]
        self.reactives_indices = [self.species.get_index_by_instance(sp) for sp in self.reactives]
        assert max(self.reactives_indices) < self.species.nb , "Reactive index is greater than number of species"

        self.products = [self.species.get_specie_by_name(name) for name in products]
        self.products_indices = [self.species.get_index_by_instance(sp) for sp in self.products]
        assert max(self.products_indices) < self.species.nb , "Product index is greater than number of species"


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
        

    def density_change_rate(self, state: NDArray[float]): # type: ignore
        """Returns an np.array with the change rate for each species due to this reaction
        state has format : [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]"""
        K = self.rate_constant(state[self.species.nb:])
        product = K * np.prod(state[self.reactives_indices+self.products_indices]) # product of rate constant and densities of all the stuff
        rate = np.zeros(self.species.nb)
        for sp in self.reactives:
            i = self.species.get_index_by_instance(sp)
            rate[i] = - product * self.stoechio_coeffs[i]
        for sp in self.products:
            i = self.species.get_index_by_instance(sp)
            rate[i] = + product * self.stoechio_coeffs[i]
        
        return rate

    def energy_change_rate(self, state: NDArray[float]): # type: ignore
        K = self.rate_constant(T_g, T_e)
        product = K * np.prod(densities[self.reactives_indices+self.products_indices]) # product of rate constant and densities of all the stuff
        energ = self.energy_threshold * product
        return energ  
        # returns the contribution of this reaction to P_loss ; works for dissociation, ionization, and excitation ; does NOT include elastic collisions and wall losses

    


if __name__ == "__main__":
    def K(Ts):
        print("Temperatures : ",Ts)
        return 2
    
    species_list = Species([Specie("I0", 10.57e-27, 0), Specie("I1", 10.57e-27, 0), Specie("I2", 10.57e-27, 0), Specie("I3", 10.57e-27, 0), Specie("I4", 10.57e-27, 0), Specie("I5", 10.57e-27, 0)])

    reac = Reaction(species_list, ["I2", "I4"], ["I5"], K, 10)
    state = np.array([1,2,3,4,5,6, -181,-182]) # jusqu'à 6 = densité, apres T°
    print(reac.density_change_rate(state))
