import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k, e, pi
from src.model import GlobalModel
from src.config import config_dict
from src.chamber_caracteristics import Chamber
from src.tests.Reaction_set_Xe_test1 import get_species_and_reactions



chamber = Chamber(config_dict)
species_list, reactions_list = get_species_and_reactions(chamber)
model = GlobalModel(species_list, reactions_list, chamber)

sol = model.solve(0, 5e-1)    # TODO Needs some testing

final_states = sol.y
print(final_states)

# Plot the results
time_points = sol.t
fig, ax1 = plt.subplots()

# Plot species concentrations on the first y-axis
for i, species in enumerate(species_list.species):
    ax1.plot(time_points, final_states[i], label=species)
ax1.set_ylabel('Concentration')
ax1.tick_params(axis='y')
ax1.legend(loc='upper left')

# Create a second y-axis for temperature
ax2 = ax1.twinx()
for i in range(2):
    ax2.plot(time_points, final_states[len(species_list.species) + i], label="Temp " + str(i), linestyle='--')
ax2.set_ylabel('Temperature')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right')

plt.xlabel('Time (s)')
plt.ylabel('Concentration')
plt.title('Species Concentrations Over Time')
plt.legend()
plt.grid()
plt.show()