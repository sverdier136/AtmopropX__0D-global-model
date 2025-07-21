import os
from pathlib import Path
from typing import Self, Callable
import abc # abstract classes
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from global_model_package.specie import Specie, Species
from global_model_package.chamber_caracteristics import Chamber
from global_model_package.variable_tracker import VariableTracker
from global_model_package.constant_rate_calculation import ReactionRateConstant

# * Ok ! Vérifié par Liam
# ! Recheck pour chamber

class Reaction:
    """
        Instances of Reaction represents a reaction between two species, between a specie and the walls,...
        It uses variable state with format [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]

        Has 2 functions that must be reimplemented in classes inheriting 'Reaction' :
            - density_change_rate(state) : variation of species densities due to this specific reaction. Returns array of length species.nb
            - energy_change_rate(state) : variation of energies of electrons, monoatomic,... due to this specific reaction. Returns array of length number of temperature
    """

    def __init__(self, 
                 species: Species, 
                 reactives: list[str], 
                 products: list[str], 
                 chamber: Chamber,
                 stoechio_coeffs: list[float] | None=None, 
                 spectators: list[str] | None=None
                 ):
        """
        Reaction class
            Inputs : 
                species : instance of class Species, lists all species present 
                reactives : list with all reactives names
                products : list with all products names
                stoechio_coeffs : stoechiometric coefficients always positive, defaults to ones
                spectators : list with spectators names (used to print reaction)
        """

        self.species = species

        self.reactives: list[Specie] = [self.species.get_specie_by_name(name) for name in reactives]
        self.reactives_indices = [self.species.get_index_by_instance(sp) for sp in self.reactives]
        #assert max(self.reactives_indices) < self.species.nb , "Reactive index is greater than number of species"

        self.products: list[Specie] = [self.species.get_specie_by_name(name) for name in products]
        self.products_indices = [self.species.get_index_by_instance(sp) for sp in self.products]
        #assert max(self.products_indices) < self.species.nb , "Product index is greater than number of species"


        if stoechio_coeffs :
            self.stoechio_coeffs = np.array(stoechio_coeffs)
        else:
            # sets stoechio_coeffs at 1 for reactives and products if not defined
            self.stoechio_coeffs = np.zeros(self.species.nb)
            for i in self.reactives_indices:
                self.stoechio_coeffs[i] = 1
            for i in self.products_indices:
                self.stoechio_coeffs[i] = 1

        self.spectators = spectators
        self.chamber = chamber
        self.var_tracker: VariableTracker = None # type: ignore because changed at start of simulation
        

    def density_change_rate(self, state: NDArray[np.float64]): 
        """Returns an np.array with the change rate for each species due to this reaction
        state has format : [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]"""
        raise NotImplementedError("Classes inheriting Reaction should implement density_change_rate")

    def energy_change_rate(self, state: NDArray[np.float64]): 
        """Function meant to return the change in energy due to this specific equation.
            HOWEVER : seems like it is necessary to account for difference in temperature of atoms molecules and electrons...
            Thus 1 function per "Temperatur type" will be needed"""
        raise NotImplementedError("Classes inheriting Reaction should implement energy_change_rate")


    def __str__(self):
        """Returns string describing the reaction"""
        def format_species(species, species_indices):
            terms = []
            for idx, sp in zip(species_indices, species):
                coeff = self.stoechio_coeffs[idx]
                # Format coefficient: display as integer if it is a whole number, else as float with 2 decimals
                if coeff.is_integer():
                    coeff_str = f"{int(coeff)}" if coeff != 1 else ""
                else:
                    coeff_str = f"{coeff:.2f}" 
                term = f"{coeff_str} {sp.name}".strip()
                terms.append(term)
            return " + ".join(terms)

        reactives_str = format_species(self.reactives, self.reactives_indices)
        products_str = format_species(self.products, self.products_indices)
        return f"{reactives_str} -> {products_str} "
    
    def set_var_tracker(self, tracker: VariableTracker):
        """Sets the variable tracker for this reaction"""
        self.var_tracker = tracker


