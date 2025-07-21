import os
from pathlib import Path
from typing import override, Self
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from global_model_package.specie import Specie, Species
from .reaction import Reaction
from global_model_package.chamber_caracteristics import Chamber
from global_model_package.constant_rate_calculation import ReactionRateConstant




class VibrationalExcitation(Reaction):
    """
        Represents vibrational excitation of a molecule by an electron
        where reaction speed is K * n_e * n_mol ...
        Works for 3 temperatures : that of electrons, monoatomic and diatomic particles

    """

    instance_counter = {}

    def __init__(self, 
                 species: Species, 
                 molecule_name: str, 
                 rate_constant, 
                 threshold_energy: float,
                 chamber: Chamber
                 ):
        """
        Represents vibrational excitation of a molecule by an electron
        where reaction speed is K * n_e * n_mol ...
        
        Inputs : 
            species : instance of class Species, lists all species present 
            molecule_name : name of molecule that will be excited. Must be a string !
            rate_constant : function taking as argument state [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]
            energy_threshold : energy threshold of electron so that reaction occurs
            chamber : instance of class Chamber, contains the chamber characteristics
        """
        # species.names[0] nom des électrons
        super().__init__(species, [species.names[0], molecule_name], [species.names[0], molecule_name], chamber)

        self.threshold_energy = threshold_energy
        self.rate_constant = rate_constant   # func

        name = f"{molecule_name}"
        if name in self.instance_counter:
            self.name = f"vib_exc_{name}_{self.instance_counter[name]}"
            self.instance_counter[name] += 1
        else:
            self.name = f"vib_exc_{name}_0"
            self.instance_counter[name] = 1
        
    @override
    def density_change_rate(self, state: NDArray[float]): # type: ignore
        """Returns an np.array with the change rate for each species due to this reaction
        state has format : [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]"""
        rate = np.zeros(self.species.nb)
        
        return rate

    
    @override
    def energy_change_rate(self, state):
        rate = np.zeros(3)

        K = self.rate_constant(state)
        energy_change = e*self.threshold_energy * K * np.prod(state[self.reactives_indices])
        rate[0] -= energy_change
        rate[self.reactives[1].nb_atoms] += energy_change

        self.var_tracker.add_value_to_variable("K_"+self.name, K)
        self.var_tracker.add_value_to_variable("E_e-_"+self.name, energy_change)
        return rate

    # @classmethod
    # def from_reaction_constant_rates(cls,species:Species, molecule_name:str, reaction_constant_rates: list[ReactionRateConstant], chamber:Chamber ) -> list[Self]:
    #     return [cls(species, molecule_name, rate_constant, rate_constant.energy_treshold, chamber) for rate_constant in reaction_constant_rates]

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
        energy_cs_df_list, energy_threshold_list = ReactionRateConstant.parse_concatenated_cross_sections_file(file_path, reaction_name, has_energy_threshold=True)
        return [cls(
                    species, 
                    molecule_name, 
                    ReactionRateConstant(species, energy_cs_df["Energy"], energy_cs_df["Cross-section"], energy_threshold),  # type: ignore
                    energy_threshold,  # type: ignore
                    chamber
                ) for energy_cs_df, energy_threshold in zip(energy_cs_df_list, energy_threshold_list)]
    























