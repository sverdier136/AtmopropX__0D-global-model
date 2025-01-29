from ..specie import Specie, Species


def test_Species_nb():
    species = Species([Specie("I0", 10.57e-27, 0), Specie("I1", 10.57e-27, 0), Specie("I2", 10.57e-27, 0), Specie("I3", 10.57e-27, 0), Specie("I4", 10.57e-27, 0), Specie("I5", 10.57e-27, 0)])
    assert species.nb == 6
