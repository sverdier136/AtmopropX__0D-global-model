from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from src.specie import Specie, Species
from src.reactions.reaction import Reaction
from src.config import *

class GasInjection(Reaction):
    """
    Represents the gas injection : adds quantity to ne, ...
    """

    def __init__(self, species: Species, injection_rates: NDArray[np.float], T_injection: float):
        """
        Dissociation class
            Inputs : 
                species : instance of class Species, lists all species present 
                injection_rate : np.array with number of particle injected per second for each specie
                    Must have same dimesnsions and be in the same order as species.names
                T_injection : temperature at which the gas is injected into the chamber
        """
        species_injected_names = [species.names[i] for i in range(species.nb) if injection_rates[i] != 0]
        super().__init__(species, [], species_injected_names)
        self.injection_rates = injection_rates
        self.T_injection = T_injection

    @override
    def density_change_rate(self, state):
        return self.injection_rates / V_chamber 

    
    @override
    def energy_change_rate(self, state):
        rate = np.zeros(3)

        for sp in self.products:
            rate[sp.nb_atoms] += 3/2 * self.injection_rates[sp.index] / V_chamber * self.T_injection

        return rate