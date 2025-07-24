from scipy.constants import e, k, pi
import numpy as np



config_dict = {

        # Geometry
        'R': 1e-2,
        'L': 18e-2,
        
        # Neutral flow
        'target_pressure': 30.66, #in Pa

        # Electrical
        'omega': 40.66e6 * 2 * pi,
        'V_grid': 1000,  # potential difference
        'N': 1.5,
        'R_coil': 1.21e-4

}
