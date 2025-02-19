import numpy as np
from scipy.constants import pi, e, k, epsilon_0 as eps_0, c, m_e
from scipy.special import jv

SIGMA_I = 1e-18 # Review this for iodine

def u_B(T_e, m_i):
    return np.sqrt(k * T_e / m_i)

def h_L(n_g, L):
    lambda_i = 1/(n_g * SIGMA_I)
    return 0.86 / np.sqrt(3 + (L / (2 * lambda_i)))

def h_R(n_g, R):
    lambda_i = 1/(n_g * SIGMA_I)
    return 0.8 / np.sqrt(4 + (R / lambda_i))

def maxwellian_flux_speed(T, m):
    # TODO : what is it, where does it come from, does it always hold true
    return np.sqrt((8 * k * T) / (pi * m))

def pressure(T, Q, v, A_out):
    """Calculates pressure in steady state without any plasma.
    T : Temperature in steady state
    Q : inflow rate in the chamber
    v : mean velocity of gas in steady state
    A_out : Effective area for which the gas can leave the chamber"""
    return (4 * k * T * Q) / (v * A_out)

def A_eff(n_g, R, L):
    return (2 * h_R(n_g, R) * pi * R * L) + (2 * h_L(n_g, L) * pi * R**2)

def A_eff_1(n_g, R, L, beta_i):
    return 2 * h_R(n_g, R) * pi * R * L + (2 - beta_i) * h_L(n_g, L) * pi * R**2

def eps_p(omega, n_e, n_g, K_el):
    omega_pe_sq = (n_e * e**2) / (m_e * eps_0)
    nu_m_i = 1j * K_el * n_g
    return 1 - (omega_pe_sq / (omega * (omega -  nu_m_i)))



