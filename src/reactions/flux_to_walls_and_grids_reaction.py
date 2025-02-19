from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from src.specie import Specie, Species
from src.auxiliary_funcs import *
from src.reactions.reaction import Reaction


# ! En cours
# ! Valable uniquement pour Chabert

class FluxToWallsAndThroughGrids(Reaction):
    """
    Elastic collision between a particle and an electron
    Works with 3 temperatures : Te, Tmono, Tdiat
    In reactives, electron must be in first position and colliding_specie next.
    """

    def __init__(self, species: Species, colliding_specie: str, rate_constant, energy_treshold: float):
        """
        Dissociation class
        /!\ Electrons should NOT be added to reactives and products
            Inputs : 
                species : instance of class Species, lists all species present 
                colliding_specie : name of specie that collides with an electron. Must be a string !
                rate_constant : function taking as argument state [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]
                energy_threshold : energy threshold of electron so that reaction occurs
        """
        super().__init__(species, [species.names[0], colliding_specie], [species.names[0], colliding_specie], rate_constant, energy_treshold)

    @override
    def density_change_rate(self, state):
        # ! gamma_e, ... a coder ds aux_funcs
        rate = np.zeros(self.species.nb)
        rate[0] = - gamma_e(state[0], state[self.species.nb]) * S_eff / V_chamber
        for sp in self.species.species[1:] :   # electron are skipped because handled before
            if sp.charge != 0:
                rate[sp.index] = - gamma_ion(state[sp.index], state[self.species.nb + sp.nb_atoms]) * S_ion / V_chamber
            else:
                rate[sp.index] = - gamma_neutral(state[sp.index], state[self.species.nb + sp.nb_atoms]) * S_neutral / V_chamber
        return rate

    
    @override
    def energy_change_rate(self, state):
        # ! latex a verifier pas homogène

        rate = np.zeros(3)

        E_kin = 7*e*state[self.species.nb]

        rate[0] = - E_kin * gamma_e(state[0], state[self.species.nb]) * S_eff / V_chamber

        for sp in self.species.species[1:] :   # electron are skipped because handled before
            if sp.charge != 0:
                rate[sp.index] = - gamma_ion(state[sp.index], state[self.species.nb + sp.nb_atoms]) * S_ion / V_chamber
            else:
                rate[sp.index]

        return rate