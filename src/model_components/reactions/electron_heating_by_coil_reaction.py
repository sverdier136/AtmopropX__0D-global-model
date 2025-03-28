from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from src.model_components.specie import Specie, Species
from src.model_components.reactions.reaction import Reaction
from src.model_components.chamber_caracteristics import Chamber


# * OK ! Vérif par Liam
# ! A recheck pour Chamber

class ElectronHeatingConstantAbsorbedPower(Reaction):
    """
    Represents the heating of electrons by the coil. It is supposed that the power transmitted to the electrons is constant 
    and therefore that the current in the coil changes to maintain this power.
    """

    def __init__(self, species: Species, power_absorbed: float, chamber: Chamber):
        """
        Electrons heating class
            Inputs : 
                species : instance of class Species, lists all species present 
                power_absorbed : power absorbed by the electrons
                chamber : instance of class Chamber, contains the chamber characteristics
        """
        super().__init__(species, [species[0]], [species[0]], chamber)
        self.power_absorbed = power_absorbed

    @override
    def density_change_rate(self, state):
        return np.zeros(self.species.nb) 

    
    @override
    def energy_change_rate(self, state):
        rate = np.zeros(3)
        rate[0] = self.power_absorbed
        return rate
    
class ElectronHeatingConstantCurrent(Reaction):
    """
    Represents the heating of electrons by the coil. It is supposed that the power transmitted to the electrons is constant 
    and therefore that the current in the coil changes to maintain this power.
    """

    def __init__(self, species: Species, chamber: Chamber):
        """
        Electrons heating class
            Inputs : 
                species : instance of class Species, lists all species present 
                power_absorbed : power absorbed by the electrons
                chamber : instance of class Chamber, contains the chamber characteristics
        """
        super().__init__(species, [species[0]], [species[0]], chamber)
        self.power_absorbed = power_absorbed

    @override
    def density_change_rate(self, state):
        return np.zeros(self.species.nb) 

    
    @override
    def energy_change_rate(self, state):
        rate = np.zeros(3)
        rate[0] = self.power_absorbed
        return rate