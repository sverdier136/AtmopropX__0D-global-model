import numpy as np 
from scipy.constants import m_e, e, pi, k, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant
from scipy.integrate import trapezoid, solve_ivp, odeint
import matplotlib.pyplot as plt
from scipy.special import jv, j0, j1
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


#Local modules
from src.model_components.util import load_csv, load_cross_section
#from util import load_csv, load_cross_section
#from auxiliary_funcs import pressure, maxwellian_flux_speed, u_B, A_eff, A_eff_1, SIGMA_I, R_ind, h_L
from src.model_components.specie import Specie, Species
#from specie import Specie, Species

def rate_constant(T_e, E, cs, m):
    """Calculates a reaction rate constant """
    #T = T_e * k / e
    v = np.sqrt(2 * E * e / m)  # electrons speed
    a = (m / (2 * np.pi * e * T_e))**(3/2) * 4 * np.pi
    f = cs * v**3 * np.exp(- m * v**2 / (2 * e * T_e)) 
    k_rate = trapezoid(a*f, x=v)
    return k_rate


def get_K_func(species,specie:str,reaction:str):
    """ specie: string, the specie involved (N, N2 or O2, O)
        reaction: string, the type of reaction (ion_N,exc1_O,...)"""
    def get_K(state):
        T_e = state[species.nb]
        e_r,cs_r=load_cross_section(".\\cross_sections\\"+specie+'\\'+reaction+'.csv')
        k_rate=rate_constant(T_e,e_r,cs_r,m_e) #si on considère que T_e en première approx
        return k_rate 
    return get_K


if __name__ == "__main__":
    #species_list = Species([Specie("e", 9.1e-31, 0),Specie("Xe0", 10.57e-27, 0), Specie("Xe+", 10.57e-27, 0)])
    species_list = Species([Specie("e", 9.1e-31, -e, 0, 3/2), Specie("Xe", 2.18e-25, 0, 1, 3/2), Specie("Xe+", 2.18e-25, e, 1, 3/2)])
    state = np.array([1e18,3*1e19,3*1e19,3,0.03])
    T_e = state[3]
    #print(state[species_list.nb])

    get_K_ion=get_K_func(species_list, "Xe", "Ionization_Xe")
    get_K_exc=get_K_func(species_list, "Xe", "exc_Xe")
    get_K_el=get_K_func(species_list, "Xe", "ela_Xe")
    get_K_el_2 = get_K_func(species_list, "Xe", "Elastic_Xe")

    T=np.linspace(0.1, 30)
    k_rate_ion=[get_K_ion(np.array([1e18,3*1e19,3*1e19,t,0.03])) for t in T]
    k_rate_exc=[get_K_exc(np.array([1e18,3*1e19,3*1e19,t,0.03])) for t in T]
    k_rate_el=[get_K_el(np.array([1e18,3*1e19,3*1e19,t,0.03])) for t in T]
    k_rate_el_2=[get_K_el_2(np.array([1e18,3*1e19,3*1e19,t,0.03])) for t in T]
    # k_rate_exc=get_K_exc(state)
    # k_rate_el=get_K_el(state)

    # print("K_el_exp="+str(k_rate_el))
    # print("K_ion_exp="+str(k_rate_ion))
    # print("K_exc_exp="+str(k_rate_exc))

    #Threshold energies in V
    E_exc=11.6  #selon les valeurs des données du github: 8.32 eV
    E_iz=12.13 

    # #Expressions in Chabert
    K_el= 3e-13
    K_exc= 1.2921e-13 * np.exp(- E_exc / T_e)
    K_iz_1 = 6.73e-15 * (T_e)**0.5 * (3.97+0.643*T_e - 0.0368 * T_e**2) * np.exp(- E_iz/T_e)
    K_iz_2 = 6.73e-15 * (T_e)**0.5 * (-0.0001031*T_e**2 + 6.386 * np.exp(- E_iz/T_e))
    K_iz= 0.5 * (K_iz_1 + K_iz_2)

    def Kiz(T_e):
        E_iz=12.127
        K_iz_1 = 6.73e-15 * (T_e)**0.5 * (3.97+0.643*T_e - 0.0368 * T_e**2) * np.exp(- E_iz/T_e)
        K_iz_2 = 6.73e-15 * (T_e)**0.5 * (-0.0001031*T_e**2 + 6.386 * np.exp(- E_iz/T_e))
        return 0.5 * (K_iz_1 + K_iz_2)
    
    def Kexc(T_e):
        Eexc=11.6
        return 1.2921e-13 * np.exp(-Eexc/T_e)


    fig, ax = plt.subplots()
    ax.plot(T,[K_el for t in T], label="Formules de l'article de P. Chabert")
    ax.plot(T, k_rate_el_2, label = "Avec deuxièmes sections efficaces")
    ax.plot(T,k_rate_el, label="Avec les sections efficaces")
    ax.set_xlabel("Température (eV)")
    ax.set_ylabel(r"$K_{el}$ en $m^{-3}.s^{-1}$")
    plt.title(r"$K_{el}$ en fonction de la température")

    print(np.mean(k_rate_el_2))

    # axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
    # axins.plot(x, y)
    # axins.plot(x, k_rate_ion)

    # Définir les limites du zoom
    # x1, x2, y1, y2 = 5, 12, 0, 0.57e-13
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)

    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.legend()
    ax.grid()
    plt.show()

    # print("K_el="+str(K_el))
    # print("K_iz="+str(K_iz))
    # print("K_exc="+str(K_exc))

    # print(np.sqrt(1-2.959599013e-35j))
    # a=(1-1.4797995065e-35j)*6e-2
    # print(jv(0, a))
    # print(jv(1,a))
    #print((1.09868411346781+0.45508986056222733j)**2)