import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import k, e, pi
from src.global_model.model import GlobalModel
from src.config import config_dict
from src.global_model.chamber_caracteristics import Chamber
from src.reaction_sets.Reaction_set_Xe_test1 import get_species_and_reactions


print("start")

if __name__ == "__main__":
    chamber = Chamber(config_dict)
    species_list, reactions_list = get_species_and_reactions(chamber)
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

    # print the first and last values of results
    print("n_e (first 5) = ", n_e[:5], "n_e (last 5) = ", n_e[-5:], "length = ", len(n_e))
    print("n_Xe (first 5) = ", n_Xe[:5], "n_Xe (last 5) = ", n_Xe[-5:], "length = ", len(n_Xe))
    print("n_Xe_plus (first 5) = ", n_Xe_plus[:5], "n_Xe_plus (last 5) = ", n_Xe_plus[-5:], "length = ", len(n_Xe_plus))
    print("T_e (first 5) = ", T_e[:5], "T_e (last 5) = ", T_e[-5:], "length = ", len(T_e))
    print("T_mono (first 5) = ", T_mono[:5], "T_mono (last 5) = ", T_mono[-5:], "length = ", len(T_mono))
    print("T_dia (first 5) = ", T_dia[:5], "T_dia (last 5) = ", T_dia[-5:], "length = ", len(T_dia))

    print("plot start")

    # thrust = model.eval_property(model.thrust_i, final_states)
    # j_i = model.eval_property(model.j_i, final_states)
    # plt.ylim(0, 200)
    # plt.xlim(0, 1600)
    # plt.plot(powers, j_i)
    # plt.title("Current density as function of power in the coil")
    # plt.show(block=False)
    # print("thruster plot")

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
    #ax2.set_ylim(((T_e * k / e).min(), 5.3))
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

