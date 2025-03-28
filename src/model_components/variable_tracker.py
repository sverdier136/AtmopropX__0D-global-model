import os
import json

from src.model_components.specie import Species
class VariableTracker:

    def __init__(self, log_folder_path, log_file_name):
        self.log_file_path = os.path.join(log_folder_path, log_file_name)
        self.tracked_variables = {}

    def add_value_to_variable(self, key, value):
        if key in self.tracked_variables:
            self.tracked_variables[key].append(value)
        else:
            self.tracked_variables[key] = [value]

    def add_all_densities_and_temperatures(self, state, species: Species, prefix=""):
        for i, specie in enumerate(species.species):
            self.add_value_to_variable(prefix + specie.name+"_density", state[i])
        for i, specie in enumerate(species.species):
            self.add_value_to_variable(prefix + specie.name + '_temperature', state[species.nb + i])

    def save_tracked_variables(self):
        """Save in json file"""
        with open(self.log_file_path, 'w') as file:
            json.dump(self.tracked_variables, file, indent=4)