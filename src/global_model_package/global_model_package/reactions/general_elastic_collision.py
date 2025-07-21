from .reaction import Reaction

class GeneralElasticCollision(Reaction):
    """Should be inherited by all ellastic collisions considered in the calculation of the plasma dielectric permitivity
        
        Inherits from Reaction and adds the abstract function : get_eps_i that should be implemented by all children
    """

    def __init__(self, species, reactives, products, chamber, stoechio_coeffs = None, spectators = None):
        super().__init__(species, reactives, products, chamber, stoechio_coeffs, spectators)

    def get_eps_i(self, state):
        raise NotImplementedError("Classes inheriting ElasticCollision should implement get_eps_i")
    
    def colliding_specie_and_collision_frequency(self, state):
        raise NotImplementedError("Classes inheriting ElasticCollision should implement colliding_specie_and_collision_frequency")
