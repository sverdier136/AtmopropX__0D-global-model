from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from src.specie import Specie, Species

from src.reactions.reaction import Reaction

class GasInjection(Reaction):
    """
    Represents the gas injection : adds quantity to ne, ...
    """

    def __init__(self, species: Species, injection_rates: NDArray[np.float]):
        """
        Dissociation class
            Inputs : 
                species : instance of class Species, lists all species present 
                injection_rate : np.array with number of particle injected per second for each specie
                    Must have same dimesnsions and be in the same order as species.names
        """
        species_injected_names = [species.names[i] for i in range(species.nb) if injection_rates[i] != 0]
        super().__init__(species, [], species_injected_names)
        self.injection_rates = injection_rates

    @override
    def density_change_rate(self, state):
        return self.injection_rates/

    
    @override
    def energy_change_rate(self, state):
        rate = np.zeros(3)

        K = self.rate_constant(state)
        reac_speed = K * np.prod(state[self.reactives_indices])
        mass_ratio = m_e / self.reactives[1].mass  # self.reactives[1].mass is mass of colliding_specie
        delta_temp = state[self.species.nb] - state[self.species.nb + self.reactives[1].nb_atoms]  # Te - Tspecie

        energy_change = 3 * mass_ratio * k_B * delta_temp * reac_speed 
        
        rate[0] = -energy_change
        rate[self.reactives[1].nb_atoms] = energy_change #mono / diatomic particles gain energy, electrons lose energy

        return rate