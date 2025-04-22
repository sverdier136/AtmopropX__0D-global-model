from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from src.model_components.specie import Specie, Species
from src.model_components.reactions.reaction import Reaction
from src.model_components.chamber_caracteristics import Chamber
from src.model_components.reactions.general_elastic_collision import GeneralElasticCollision

# * A check + chamber
# ! omega a changer selon nom dans Chamber

class ElasticCollisionWithElectron(GeneralElasticCollision):
    """
    Elastic collision between a particle and an electron
    Works with 3 temperatures : Te, Tmono, Tdiat
    In reactives, electron must be in first position and colliding_specie next.
    """

    instance_counter = {}

    def __init__(self, species: Species, colliding_specie: str, rate_constant, energy_treshold: float, chamber: Chamber):
        """
        Elastic collision between molecule and electron
            Inputs : 
                species : instance of class Species, lists all species present 
                colliding_specie : name of specie that collides with an electron. Must be a string !
                rate_constant : function taking as argument state [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]
                energy_threshold : energy threshold of electron so that reaction occurs
        """
        super().__init__(species, [species.names[0], colliding_specie], [species.names[0], colliding_specie], rate_constant, energy_treshold, chamber)
        self.rate_constant = rate_constant

        name = f"{colliding_specie}"
        if name in self.instance_counter:
            self.name = f"ela_col_{name}_{self.instance_counter[name]}"
            self.instance_counter[name] += 1
        else:
            self.name = f"ela_col_{name}_0"
            self.instance_counter[name] = 1

    @override
    def density_change_rate(self, state):
        return np.zeros(self.species.nb)

    
    @override
    def energy_change_rate(self, state):
        rate = np.zeros(3)

        K = self.rate_constant(state)
        reac_speed = K * np.prod(state[self.reactives_indices])
        mass_ratio = m_e / self.reactives[1].mass  # self.reactives[1].mass is mass of colliding_specie
        delta_temp = state[self.species.nb] - state[self.species.nb + self.reactives[1].nb_atoms]  # Te - Tspecie

        energy_change = 3 * mass_ratio * e * delta_temp * reac_speed 
        
        rate[0] = -energy_change
        rate[self.reactives[1].nb_atoms] = energy_change #mono / diatomic particles gain energy, electrons lose energy
        self.var_tracker.add_value_to_variable("E_"+self.name, energy_change)

        return rate

    
    @override
    def colliding_specie_and_collision_frequency(self, state: NDArray[np.float64]):
        return self.reactives[1], self.rate_constant(state) * state[self.reactives[1].index]