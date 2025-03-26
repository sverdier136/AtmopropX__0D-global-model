import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k, e, pi
from src.model import GlobalModel
from src.config import config_dict
from src.chamber_caracteristics import Chamber
from src.reaction_set_Xe import species_list, reactions_list


print("start")

if __name__ == "__main__":
    chamber = Chamber(config_dict)
    model = GlobalModel(species_list, reactions_list, chamber)

    # Solve for several values of I_coil

    I_coil = np.linspace(1, 40, 20)
    powers, final_states = model.solve_for_I_coil(I_coil)

    n_e = final_states[:, 0]
    n_Xe = final_states[:, 1]
    n_Xe_plus = final_states[:, 2]
    T_e = final_states[:, 3]
    T_mono = final_states[:, 4]
    T_dia = final_states[:, 5]

    print("plot start")

    thrust = model.eval_property(model.thrust_i, final_states)
    j_i = model.eval_property(model.j_i, final_states)
    plt.ylim(0, 200)
    plt.xlim(0, 1600)
    plt.plot(powers, j_i)
    plt.title("Current density as function of power in the coil")
    plt.show(block=False)
    print("thruster plot")

    #exit()
    # Temperature plot

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.plot(powers, T_e * k / e, 'g-')
    ax1.plot(powers, T_mono, 'b-')

    ax1.set_xlabel('$P_{RF}$ [W]')
    ax2.set_ylabel('$T_e$ [eV]', color='g')
    ax1.set_ylabel('$T_g$ [K]', color='b')

    ax1.set_xlim((0, 1600))
    ax2.set_ylim(((T_e * k / e).min(), 5.3))
    ax1.set_ylim((255, 540))

    plt.title("Gas and electron temperature as function of power")

    plt.show(block=False)
    print("2nd plot")

    # Density plot
    plt.xlim((0, 1600))
    plt.semilogy(powers, n_e, label='$n_e$')
    plt.semilogy(powers, n_Xe, label='$n_g$')
    plt.xlabel('n [$m^{-3}$]')
    plt.xlabel('$P_{RF}$ [W]')
    plt.legend()
    plt.title("Gas and electron (aka ion) density as function of power")
    plt.show()
    print("density plot")

