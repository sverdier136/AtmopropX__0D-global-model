from src.reactions.reaction import Reaction

class ElasticCollision(Reaction):
    """Should be inherited by all ellastic collisions considered in the calculation of the plasma dielectric permitivity
        Inherits from Reaction and adds the abstract function : get_eps_i that should be implemented by all children
    """

    def __init__(self, species, reactives, products, chamber, stoechio_coeffs = None, spectators = None):
        super().__init__(species, reactives, products, chamber, stoechio_coeffs, spectators)

    def get_eps_i(self, state):
        raise NotImplementedError("Classes inheriting EllasticCollision should implement get_eps_i")