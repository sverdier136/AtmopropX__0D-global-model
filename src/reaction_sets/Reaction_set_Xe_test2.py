from scipy.constants import pi, e, k, epsilon_0 as eps_0, c, m_e
from src.model_components.reactions.excitation_reaction import Excitation
from src.model_components.reactions.ionisation_reaction import Ionisation
from src.model_components.reactions.elastic_collision_with_electrons_reaction import ElasticCollisionWithElectron
from src.model_components.reactions.flux_to_walls_and_grids_reaction import FluxToWallsAndThroughGrids
from src.model_components.reactions.gas_injection_reaction import GasInjection
from src.model_components.specie import Species, Specie
#from src.constant_rate_constant import get_K_func

def get_species_and_reactions(chamber):

  species = Species([Specie("e", m_e, -e , 0 , 3/2), Specie("Xe", 2.18e-25, 0 , 1 , 3/2), Specie("Xe+", 2.18e-25, e , 1 , 3/2)])
  
  ### Ionisation
  ion_Xe = Ionisation(species, "Xe", "Xe+", get_K_func(species, "Xe", "ion_Xe"), 12.127, chamber) 
  
  ### Excitation
  exc_Xe = Excitation(species, "Xe", get_K_func(species, "Xe", "exc_Xe"), 11.6, chamber) 
  
  ### Terme source
  src_Xe = GasInjection(species, [0.0, 1.2e-19, 0.0], 500, chamber) 
  
  ### Sortie de Xe à travers les grilles
  out_Xe = FluxToWallsAndThroughGrids(species, "Xe", get_K_func(species, "Xe", "out_Xe"), 0, chamber) 
  
  ### Sortie de Xe+ à travers les grilles
  out_Xe+ = FluxToWallsAndThroughGrids(species, "Xe+", get_K_func(species, "Xe+", "out_Xe+"), 0, chamber) 
  
  # Reaction list
  reaction_list = [ion_Xe, exc_Xe, src_Xe, out_Xe, out_Xe+]

  return species, reaction_list
