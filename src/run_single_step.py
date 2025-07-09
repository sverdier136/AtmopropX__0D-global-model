import matplotlib.pyplot as plt
from pprint import pp
import numpy as np
from scipy.constants import k, e, pi
from src.global_model.model import GlobalModel
from src.config import config_dict
from src.global_model.chamber_caracteristics import Chamber
#from src.reaction_sets.Reaction_set_Xe_test1 import get_species_and_reactions
import src.reaction_sets.reaction_set_N_mono as reac_set_N_mono
import src.reaction_sets.Reaction_set_Xe_test1 as reac_set_Xe



chamber = Chamber(config_dict)
species_N, initial_state_N, reactions_list_N, electron_heating = reac_set_N_mono.get_species_and_reactions(chamber)
model_N = GlobalModel(species_N, reactions_list_N, chamber, electron_heating, simulation_name="Nmono_cstPabs")

species_X, initial_state_X, reactions_list_X, _ = reac_set_Xe.get_species_and_reactions(chamber)
model_X = GlobalModel(species_X, reactions_list_X, chamber, electron_heating, simulation_name="Xe_cstPabs")

initial_state = np.array([1.29e18, 2.24e23, 1.29e18, 3, 0.04, 0.043])

_ = model_N.f_dy(0.0, initial_state)
_ = model_X.f_dy(0.0, initial_state)

dic_N = model_N.var_tracker.tracked_variables
dic_X = model_X.var_tracker.tracked_variables
# pp(dic_N)
# pp(dic_X)
list_N = sorted(dic_N.items(), key=lambda x: x[0])
list_X = sorted(dic_X.items(), key=lambda x: x[0])
pp(list_N)
pp(list_X)
dic = {}
for i in range(len(list_N)):
    dic[list_N[i][0]] = list_N[i][1] + list_X[i][1]
print("N values         X values")
pp(dic)



