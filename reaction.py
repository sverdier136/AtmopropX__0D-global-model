import numpy as np


species_list = ["I2", "I1/3", "I+"]

class Reaction:

    def __init__(self, species_list, reactives, products, rate_constant, energy_treshold, stoechio_coeffs=None):
        """Init of Reaction class.
            Inputs : 
            species_list : list of all species present (instances of class Specie)
            reactives : list with all reactives
            products : list with all products
            rate_constant : function taking as arguments (T_g, T_e, )
            stoechio_coeffs : stoechiometric coefficients positive if product, negative otherwise"""
        self.species = species_list
        if stoechio_coeffs :
            self.stoechio_coeffs = np.array(stoechio_coeffs)
        else:
            self.stoechio_coeffs = np.ones(len(self.species))
        self.energy_threshold = energy_treshold
        self.rate_constant = rate_constant     # func
        self.

    def change_rate(self, T_g, T_e, densities):
        """Returns an np.array with the change rate for each species due to this reaction"""
        return 