from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from src.specie import Specie, Species

from src.reactions.reaction import Reaction

class ElasticCollisionWithElectron(Reaction):

    def __init__(self, species: Species, colliding_specie: str, rate_constant, energy_treshold: float):
        super().__init__(species, [colliding_specie], [colliding_specie], rate_constant, energy_treshold)

    @override
    def density_change_rate(self, state):
        return np.zeros(self.species.nb)


    @override
    def electron_loss_power(self, state):
        K = self.rate_constant(state)
        mass_ratio = m_e / self.reactives[0].mass
        delta_temp = state[self.species.nb] - state[self.species.nb + self.reactives[0].nb_atoms ]

        energy_change = 3 * mass_ratio * k_B * delta_temp * state[0] * state[self.reactives_indices[0]] * K 
   
        return energy_change