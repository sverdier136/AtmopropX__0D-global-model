from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from src.model_components.specie import Specie, Species
from src.model_components.reactions.reaction import Reaction
from src.model_components.chamber_caracteristics import Chamber



# ! A check

class ElectronHeating:
    """
    The general parent class representing the heating of electrons by the coil. MUST be inherited !
    """

    def __init__(self, species: Species, chamber: Chamber):
        """
        Electrons heating class
            Inputs : 
                species : instance of class Species, lists all species present 
                chamber : instance of class Chamber, contains the chamber characteristics
        """
        self.species = species
        self.chamber = chamber

    def absorbed_power(self, state, collision_frequencies) -> float:
        """
        Returns the absorbed power by the electrons
        """
        raise NotImplementedError("absorbed_power not implemented in subclass")
        

class ElectronHeatingConstantAbsorbedPower(ElectronHeating):
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
        super().__init__(species, chamber)
        self.power_absorbed = power_absorbed

    @override
    def absorbed_power(self, state) -> float:
        return self.absorbed_power
    

# ! Non codé, En cours
class ElectronHeatingConstantCurrent(ElectronHeating):
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
        super().__init__(species, chamber)
        self.power_absorbed = power_absorbed

    def normalised_concentrations(self , state: NDArray[np.float64]) :
        """Returns normalized concentrations (such that they sum up to 1)"""
        return state[:self.species.nb]/np.sum(state[:self.species.nb])

    def P_abs(self , R_ind):
        return R_ind* self.chamber.I_coil**2 / 2
    
    def R_ind(self, eps_p):
        '''plamsma resistance, used in calculating the power P_abs'''
        k_p = (self.chamber.omega / c_light) * np.sqrt(eps_p)
        a = 2 * pi * self.chamber.N**2 / (self.chamber.L * self.chamber.omega * eps_0)
        #jv are Besel functions
        b = 1j * k_p * self.chamber.R * jv(1, k_p * self.chamber.R) / (eps_p * jv(0, k_p * self.chamber.R))
        R_ind = a * np.real(b)
        self.add_value_to_variable("R_ind", R_ind)
        self.add_value_to_variable("R_ind_a", a)
        self.add_value_to_variable("R_ind_b", np.real(b))
        return R_ind
    
    def eps_p (self, collision_frequencies , state) :
        """Calcule la permittivité diélectrique relative due à toutes les réactions de collisions elastiques elctron-neutre. Ces réactions sont considérées séparément par chacun des eps_i
            collision_frequencies : np.array containing the collision frequencies for each specie in the order in which they appear in self.species"""
        #fonction utilisée dans f_dy
        normalized_c = self.normalised_concentrations(state)
        omega_pe_sq = (state[0] * e**2) / (m_e * eps_0)
        epsilons_i = 1 - omega_pe_sq / (self.chamber.omega * (self.chamber.omega -  1j*collision_frequencies))

        def equation(x):
            x = x[0] + x[1]*1j     
            value = np.sum(normalized_c*(epsilons_i-1)/(epsilons_i + 2*x)) + (1-x)/(3*x)   
            return [np.real(value), np.imag(value)]
        eps_p = fsolve(equation , [1, 0.01])
        return eps_p[0]+eps_p[1]*1j

    @override
    def absorbed_power(self, state) -> float:
        return self.P_abs(self.R_ind( self.eps_p(collision_frequencies, state)  ))