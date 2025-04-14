from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from src.model_components.specie import Specie, Species
from src.model_components.reactions.reaction import Reaction
from src.model_components.chamber_caracteristics import Chamber


# * Checked by me

class InelasticCollision(Reaction):
    """
        Represents excitation of a molecule by an electron
        where reaction speed is K * n_e * n_mol ...
        Works for 3 temperatures : that of electrons, monoatomic and diatomic particles

    """

    def __init__(self, 
                 species: Species, 
                 molecule_name: str, 
                 chamber: Chamber
                 ):
        """
        Reaction class
        
        Inputs : 
            species : instance of class Species, lists all species present 
            reactives : list with all reactives names
            products : list with all products names
            rate_constant : function taking as argument state [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]
            energy_threshold : energy threshold of electron so that reaction occurs
            stoechio_coeffs : stoechiometric coefficients always positive
            spectators : list with spectators names (used to print reaction)
        """
        # species.names[0] nom des électrons
        super().__init__(species, [species.names[0], molecule_name], [species.names[0], molecule_name], chamber)
        self.charged_sp = [sp for sp in self.species.species if sp.charge != 0]
        self.non_charged_sp = [sp for sp in self.species.species if sp.charge == 0]
        print([sp.name for sp in self.species.species if sp.charge != 0])

    @override
    def density_change_rate(self, state: NDArray[float]): # type: ignore
        """Returns an np.array with the change rate for each species due to this reaction
        state has format : [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]"""
        rate = np.zeros(self.species.nb)
        
        return rate

    
    @override
    def energy_change_rate(self, state): ### PROBLEME !!!!!!
        rate = np.zeros(3)
        E_ion=0

        for sp in self.charged_sp:
            if sp.nb_atoms > 0:
                E_ion += sp.mass * state[sp.index] * self.chamber.u_B(state[self.species.nb], sp.mass)**2
        
        for sp in self.non_charged_sp:
            rate[sp.nb_atoms] += self.chamber.gamma_neutral(state[sp.index], state[self.species.nb + sp.nb_atoms], sp.mass) * E_ion * self.chamber.SIGMA_I

        self.var_tracker.add_value_to_variable_list('energy_change_inelastic_collision', rate)

        return rate

























