import os
from pathlib import Path
from typing import override, Self
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant
import pandas as pd

from global_model_package.specie import Specie, Species
from global_model_package.chamber_caracteristics import Chamber
from global_model_package.constant_rate_calculation import ReactionRateConstant
from .reaction import Reaction

# * A check + chamber
# ! omega a changer selon nom dans Chamber

class RotationalExcitation(Reaction):
    """
    Elastic collision between a particle and an electron
    Works with 3 temperatures : Te, Tmono, Tdiat
    In reactives, electron must be in first position and colliding_specie next.
    """

    instance_counter = {}

    def __init__(self, species: Species, colliding_specie: str, rate_constant, chamber: Chamber, energy_threshold:float|None=None):
        """
        Elastic collision between molecule and electron
            Inputs : 
                species : instance of class Species, lists all species present 
                colliding_specie : name of specie that collides with an electron. Must be a string !
                rate_constant : function taking as argument state [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]
                chamber: Instance of Chamber used throughout the simulation
                energy_threshold : Optional. Energy threshold of electron so that reaction occurs
        """
        super().__init__(species, [species.names[0], colliding_specie], [species.names[0], colliding_specie], chamber)
        self.rate_constant = rate_constant
        self.energy_threshold = energy_threshold

        name = f"{colliding_specie}"
        if name in self.instance_counter:
            self.name = f"rot_exc_{name}_{self.instance_counter[name]}"
            self.instance_counter[name] += 1
        else:
            self.name = f"rot_exc_{name}_0"
            self.instance_counter[name] = 1

    @override
    def density_change_rate(self, state):
        return np.zeros(self.species.nb)

    
    @override
    def energy_change_rate(self, state):
        rate = np.zeros(3)

        K = self.rate_constant(state)
        self.var_tracker.add_value_to_variable("K_"+self.name, K)

        reac_speed = K * np.prod(state[self.reactives_indices])
        mass_ratio = m_e / self.reactives[1].mass  # self.reactives[1].mass is mass of colliding_specie
        delta_temp = state[self.species.nb] - state[self.species.nb + self.reactives[1].nb_atoms]  # Te - Tspecie

        energy_change = 3 * mass_ratio * e * delta_temp * reac_speed 
        
        rate[0] = -energy_change
        rate[self.reactives[1].nb_atoms] = energy_change #mono / diatomic particles gain energy, electrons lose energy
        self.var_tracker.add_value_to_variable("E_"+self.name, energy_change)

        return rate

    
    # @override
    # def colliding_specie_and_collision_frequency(self, state: NDArray[np.float64]):
    #     return self.reactives[1], self.rate_constant(state) * state[self.reactives[1].index]


    @classmethod
    def from_concatenated_txt_file(cls, species: Species, molecule_name, file_name: str, reaction_name, chamber) -> list[Self]:
        """
        Reads the cross-sections concatenated in a file downloaded from lxcat. It return a list of instances of ReactionRateConstant each with the right energy/cross sections lists.

        Parameters
        ----------
        file_name : str
            Name of the .txt file containing the concatenated cross-sections. Should not include the .txt extension.
        reaction_name: str
            Name of reaction as found on the first line of each reaction block, eg 'EXCITATION', 'IONISATION'...
        
        Returns
        ----------
        list[ReactionRateConstant]
        """
        assert ReactionRateConstant.CROSS_SECTIONS_PATH is not None, "ReactionRateConstant.CROSS_SECTIONS_PATH must be set before calling from_concatenated_txt_file."
        file_path = os.path.join(ReactionRateConstant.CROSS_SECTIONS_PATH, molecule_name, file_name+".txt")
        energy_cs_df_list: list[pd.DataFrame] = ReactionRateConstant.parse_concatenated_cross_sections_file(file_path, reaction_name) # type: ignore
        return [cls(
                    species, 
                    molecule_name, 
                    ReactionRateConstant(species, energy_cs_df["Energy"], energy_cs_df["Cross-section"]),  # type: ignore
                    chamber
                ) for energy_cs_df in energy_cs_df_list]