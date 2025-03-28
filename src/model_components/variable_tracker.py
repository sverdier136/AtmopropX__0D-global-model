
class VariableTracker:

    def __init__(self):
        pass

    def add_value_to_variable(self, key, value):
        if key in self.tracked_variables:
            self.tracked_variables[key].append(value)
        else:
            self.tracked_variables[key] = [value]

    def add_densities_and_temperature(self, state, prefix=""):
        for i, specie in enumerate(self.species.species):
            self.add_value_to_variable(prefix + specie.name+"_density", state[i])
        for i, specie in enumerate(self.species.species):
            self.add_value_to_variable(prefix + specie.name + '_temperature', state[self.species.nb + i])

    def save_tracked_variables(self, log_file='data.json'):
        """Save in json file"""
        with open(os.path.join(log_folder_path, log_file), 'w') as file:
            json.dump(self.tracked_variables, file, indent=4)