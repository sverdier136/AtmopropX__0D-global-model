import os
from pathlib import Path
import re
from typing import TypeVar, Type, Self

import numpy as np 
from numpy.typing import NDArray
import pandas as pd
from scipy.constants import m_e, e, pi, k, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant
from scipy.integrate import trapezoid, solve_ivp, odeint
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


#Local modules
from .util import load_csv, load_cross_section
from .specie import Specie, Species
#from specie import Specie, Species

#ReacRateType = TypeVar("ReacRateType", bound="ReactionRateConstant")


class ReactionRateConstant(object):

    def __init__(self, species: Species, energy_list: list[float]|NDArray[np.float64]|pd.Series, cross_section_list: list[float]|NDArray[np.float64]|pd.Series, energy_threshold: float|None = None) -> None:
        self.species: Species = species
        self.energy_treshold: float|None = energy_threshold
        self.energy_list: NDArray[np.float64] = np.array(energy_list)
        self.cross_section_list: NDArray[np.float64] = np.array(cross_section_list)
    
    def __call__(self, state: NDArray[np.float64]) -> float:
        """Calculates the reaction rate constant for a given state, which must be a NDArray[np.float]"""
        #T = T_e * k / e
        T_e = state[self.species.nb]
        # If the table of energy - cross sections doesn't go to high enough temperatures, it is extended by considering that for higher temperatures the cross section stay the same
        if 10 * e * T_e > np.max(self.energy_list):
            self.energy_list = np.append(self.energy_list, np.logspace(np.max(self.energy_list), 10 * e * T_e, 100))
            self.cross_section_list = np.append(self.cross_section_list, [self.cross_section_list[-1]]*100)
        v = np.sqrt(2 * self.energy_list * e / m_e)  # electrons speed
        a = (m_e / (2 * np.pi * e * T_e))**(3/2) * 4 * np.pi
        f = self.cross_section_list * v**3 * np.exp(- m_e * v**2 / (2 * e * T_e)) 
        k_rate: float = trapezoid(a*f, x=v)
        return k_rate

    @classmethod
    def from_concatenated_txt_file(cls, species: Species, file_path: str | Path, reaction_name) -> list[Self]:
        """
        Reads the cross-sections concatenated in a file downloaded from lxcat. It return a list of instances of ReactionRateConstant each with the right energy/cross sections lists.

        Parameters
        ----------
        file_path : str | Path
            Path to the .txt file containing the concatenated cross-sections
        reaction_name: str
            Name of reaction as found on the first line of each reaction block, eg 'EXCITATION', 'IONISATION'...
        
        Returns
        ----------
        list[ReactionRateConstant]
        """
        energy_cs_df_list, energy_threshold_list = cls.parse_concatenated_cross_sections_file(file_path, reaction_name)
        
        return [cls(species, energy_cs_df["Energy"], energy_cs_df["Cross-section"], energy_threshold) for energy_cs_df, energy_threshold in zip(energy_cs_df_list, energy_threshold_list)]
    
    
    @staticmethod
    def parse_concatenated_cross_sections_file(file_path: str | Path, reaction_name: str) -> tuple[list[pd.DataFrame], list[float]]:
        """
        Reads the cross-sections concatenated in a file downloaded from lxcat. 

        Parameters
        ----------
        file_path : str | Path
            Path to the .txt file containing the concatenated cross-sections
        reaction_name: str
            Name of reaction as found on the first line of each reaction block, eg 'EXCITATION', 'IONISATION'...
        
        Returns
        ----------
        (energy_cross_section_list, energy_threshold_list): tuple[list[pd.DataFrame], list[float]]
            First element of tuple is a list of DataFrame with first column "Energy" and second "Cross-section".
            Second element is the list of energy thresholds.
        """
        energy_cs_df = []
        energy_thresholds = []

        with open(file_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Detect reaction block start
            if reaction_name in line:
                reaction_type = line
                energy_line = lines[i + 2].strip()
                try:
                    energy_value = float(energy_line)
                except ValueError:
                    raise ValueError(f"The energy threshold can't be converted to a float. Here is the line : \"{repr(lines[i+2])}\".")

                # Find start of data table
                assert re.match(r"^\s*-{5,}", lines[i+8]), (f"Text file doesn't seem to have expected format, when looking for -----. You're line looks like : \"{repr(lines[i+8])}\". It should look like the following : \n"
                                                            "EXCITATION\n"
                                                            "N2(X,v=0) -> N2(X,v=0)\n"
                                                            "0.000000e+0\n"
                                                            "SPECIES: e / N2(X,v=0)\n"
                                                            "PROCESS: E + N2(X,v=0) -> E + N2(X,v=0), Excitation\n"
                                                            "PARAM.:  E = 0 eV\n"
                                                            "UPDATED: 2022-06-16 11:16:45\n"
                                                            "COLUMNS: Energy (eV) | Cross section (m2)\n"
                                                            "-----------------------------\n"
                                                            "1.000000e-2	3.689100e-21\n"
                                                            "2.000000e-2	3.721100e-21\n"
                                                            "3.000000e-2	3.753600e-21")
                i += 9  # skip past dashed line

                # Read the data table
                data = []
                while not re.match(r"^\s*-{5,}", lines[i]):
                    try:
                        parts = lines[i].strip().split()
                        assert len(parts)==2, "More than two parts on one line"
                        data.append([float(part) for part in parts])
                    except Exception as e:
                        raise IOError(f"When reading line \"{repr(lines[i])}\", the following error occured. Check the formatting of the file. \n {type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}") # type: ignore
                    i += 1

                energy_cs_df.append(pd.DataFrame(data, columns=["Energy", "Cross-section"]))
                energy_thresholds.append(energy_value)

                # # Prepare filename
                # clean_name = re.sub(r"[^\w\-_\. ]", "_", reaction_name.replace("->", "to"))
                # filename = f"{reaction_type}_{clean_name}_E={energy_value:.4f}eV.csv"
                # filepath = os.path.join(output_dir, filename)

                # # Write CSV
                # with open(filepath, "w", newline="") as csvfile:
                #     writer = csv.writer(csvfile)
                #     writer.writerow(["Energy (eV)", "Cross section (m²)"])
                #     for row in data:
                #         writer.writerow(row)

                # block_index += 1

            i += 1  # move to next line
        assert re.match(r"x+",lines[i-1]), f"File not ending with \"xxxx...x\". Ending with : \"{repr(lines[i-1])}\""
        return energy_cs_df, energy_thresholds





def rate_constant(T_e, E, cs, m):
    """Calculates a reaction rate constant """
    #T = T_e * k / e
    if 10 * e * T_e > np.max(E):
        E = np.append(E, np.logspace(np.max(E), 10 * e * T_e, 100))
        cs = np.append(cs, [cs[-1]]*100)
    v = np.sqrt(2 * E * e / m)  # electrons speed
    a = (m / (2 * np.pi * e * T_e))**(3/2) * 4 * np.pi
    f = cs * v**3 * np.exp(- m * v**2 / (2 * e * T_e)) 
    k_rate = trapezoid(a*f, x=v)
    return k_rate


def get_K_func(species,specie:str,reaction:str):
    """ specie: string, the specie involved (N, N2 or O2, O)
        reaction: string, the type of reaction (ion_N,exc1_O,...) """
    def get_K(state):
        T_e = state[species.nb]
        cross_sec_path =  Path(__file__).resolve().parent.parent.parent.parent.joinpath(f"cross_sections/{specie}/{reaction}.csv")
        e_r,cs_r=load_cross_section(cross_sec_path)
        k_rate=rate_constant(T_e,e_r,cs_r,m_e) #si on considère que T_e en première approx
        return k_rate 
    return get_K














if __name__ == "__main__":
    # #species_list = Species([Specie("e", 9.1e-31, 0),Specie("Xe0", 10.57e-27, 0), Specie("Xe+", 10.57e-27, 0)])
    species_list = Species([Specie("e", 9.1e-31, -e, 0, 3/2), Specie("Xe", 2.18e-25, 0, 1, 3/2), Specie("Xe+", 2.18e-25, e, 1, 3/2)])
    # state = np.array([10^18,3*10^19,3*10^19,3,0.03])
    # T_e = state[3]
    # #print(state[species_list.nb])

    # get_K_ion=get_K_func(species_list, "Xe", "Ionization_Xe")
    # get_K_exc=get_K_func(species_list, "Xe", "exc_Xe")
    # get_K_el=get_K_func(species_list, "Xe", "Elastic_Xe")

    # T=np.linspace(0.1, 30)
    # k_rate_ion=[get_K_ion(np.array([10^18,3*10^19,3*10^19,t,0.03])) for t in T]
    # k_rate_exc=[get_K_exc(np.array([10^18,3*10^19,3*10^19,t,0.03])) for t in T]
    # k_rate_el=[get_K_el(np.array([10^18,3*10^19,3*10^19,t,0.03])) for t in T]
    # # k_rate_exc=get_K_exc(state)
    # # k_rate_el=get_K_el(state)

    # # print("K_el_exp="+str(k_rate_el))
    # # print("K_ion_exp="+str(k_rate_ion))
    # # print("K_exc_exp="+str(k_rate_exc))

    # #Threshold energies in V
    # E_exc=11.6  #selon les valeurs des données du github: 8.32 eV
    # E_iz=12.13 

    # # #Expressions in Chabert
    # K_el= 1e-13
    # K_exc= 1.2921e-13 * np.exp(- E_exc / T_e)
    # K_iz_1 = 6.73e-15 * (T_e)**0.5 * (3.97+0.643*T_e - 0.0368 * T_e**2) * np.exp(- E_iz/T_e)
    # K_iz_2 = 6.73e-15 * (T_e)**0.5 * (-0.0001031*T_e**2 + 6.386 * np.exp(- E_iz/T_e))
    # K_iz= 0.5 * (K_iz_1 + K_iz_2)

    # def Kiz(T_e):
    #     E_iz=12.127
    #     K_iz_1 = 6.73e-15 * (T_e)**0.5 * (3.97+0.643*T_e - 0.0368 * T_e**2) * np.exp(- E_iz/T_e)
    #     K_iz_2 = 6.73e-15 * (T_e)**0.5 * (-0.0001031*T_e**2 + 6.386 * np.exp(- E_iz/T_e))
    #     return 0.5 * (K_iz_1 + K_iz_2)
    
    # def Kexc(T_e):
    #     Eexc=11.6
    #     return 1.2921e-13 * np.exp(-Eexc/T_e)


    # fig, ax = plt.subplots()
    # ax.plot(T,[1e-13 for t in T], label="Formules de l'article de P. Chabert")
    # ax.plot(T,k_rate_el, label="Avec les sections efficaces")
    # ax.set_xlabel("Température (eV)")
    # ax.set_ylabel(r"$K_{el}$ en $m^{-3}.s^{-1}$")
    # plt.title(r"$K_{el}$ en fonction de la température")

    # # axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
    # # axins.plot(x, y)
    # # axins.plot(x, k_rate_ion)

    # # Définir les limites du zoom
    # # x1, x2, y1, y2 = 5, 12, 0, 0.57e-13
    # # axins.set_xlim(x1, x2)
    # # axins.set_ylim(y1, y2)

    # # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # ax.legend()
    # ax.grid()
    # plt.show()

    # # print("K_el="+str(K_el))
    # # print("K_iz="+str(K_iz))
    # # print("K_exc="+str(K_exc))
    print("starting")
    constant_rates = ReactionRateConstant.from_concatenated_txt_file(species_list, "D:/Users/Charlelie/Downloads/Cross section.txt", "EXCITATION")
    print("hello")
    print(constant_rates[0]([10, 20, 30, 3]))
    # df = df_list[0]
    # print(df.head(10))
    # print(len(df_list))
    # print(len(e_list))
    # print(e_list)
    # print([len(d)+100*e for d, e in zip(df_list, e_list)])