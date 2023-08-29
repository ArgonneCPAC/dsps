"""JAX implementation of a blackbody spectral energy density"""
import numpy as np
from jax import config
from jax import jit as jjit
from jax import lax

config.update("jax_enable_x64", True)

H_PLANCK = 6.62607015e-34  # J*s
K_BOLTZ = 1.380649e-23  # J/K
C_SPEED = 299792458.0  # m/s
WATTS_PER_LSUN = 3.828e26

LGH = np.log(H_PLANCK)
LGK = np.log(K_BOLTZ)
LGC = np.log(C_SPEED)
LG2 = np.log(2)

__all__ = ("blackbody_freq_density", "blackbody_wave_density")


@jjit
def blackbody_freq_density(freq_hz, temp_kelvin):
    """Blackbody frequency density, L_ν, in units [Lsun/Hz/sr/m/m]

    Parameters
    ----------
    freq_hz : ndarray of shape (n, )
        Photon frequency in units of Hz

    temp_kelvin : float
        Temperature in Kelvin

    Returns
    -------
    bb_lsun_per_hz : ndarray of shape (n, )
        Blackbody SED in units of Lsun/Hz/sr/m/m

    """
    bb_si = _blackbody_freq_density_si(freq_hz, temp_kelvin)  # W/Hz/sr/m/m
    bb_lsun_per_hz = bb_si / WATTS_PER_LSUN  # Lsun/Hz/sr/m/m
    return bb_lsun_per_hz


@jjit
def blackbody_wave_density(wave_aa, temp_kelvin):
    """Blackbody wavelength density, L_λ, in units [Lsun/AA/sr/m/m]

    Parameters
    ----------
    wave_aa : ndarray of shape (n, )
        Photon wavelength in units of angstroms

    temp_kelvin : float
        Temperature in Kelvin

    Returns
    -------
    bb_lsun_per_aa : ndarray of shape (n, )
        Blackbody SED in units of Lsun/AA/sr/m/m

    """
    wave_m = wave_aa / 1e10
    bb_si = _blackbody_wave_density_si(wave_m, temp_kelvin)
    bb_lsun_per_m = bb_si / WATTS_PER_LSUN  # Lsun/m/sr/m/m
    m_per_aa = 1e-10
    bb_lsun_per_aa = bb_lsun_per_m * m_per_aa
    return bb_lsun_per_aa


@jjit
def _freq_density_denom(freq_hz, temp_kelvin):
    hnu = H_PLANCK * freq_hz
    kt = K_BOLTZ * temp_kelvin
    x = hnu / kt
    denom = 1 / (lax.exp(x) - 1)
    return denom


@jjit
def _jax_bb_freq_density_exparg(freq_hz):
    lg_term1 = LG2 + LGH - 2 * LGC
    lg_term2 = 3 * lax.log(freq_hz)
    exparg = lg_term1 + lg_term2
    return exparg


@jjit
def _blackbody_freq_density_si(freq_hz, temp_kelvin):
    """Blackbody frequency density, L_ν, in SI units [W/Hz/sr/m/m]"""
    exparg = _jax_bb_freq_density_exparg(freq_hz)
    denom = _freq_density_denom(freq_hz, temp_kelvin)
    return denom * lax.exp(exparg)


@jjit
def _jax_bb_frequency_peak(temp_kelvin):
    alpha = 5.879e10
    peak_freq_hz = alpha * temp_kelvin
    return peak_freq_hz


@jjit
def _jax_bb_wavelength_peak(temp_kelvin):
    peak_freq_hz = _jax_bb_frequency_peak(temp_kelvin)
    peak_wave_m = C_SPEED / peak_freq_hz / 1.76
    return peak_wave_m


@jjit
def _wave_density_denom(wave_meters, temp_kelvin):
    hc = H_PLANCK * C_SPEED
    kt = K_BOLTZ * temp_kelvin
    x = hc / kt / wave_meters
    denom = 1 / (lax.exp(x) - 1)
    return denom


@jjit
def _blackbody_wave_density_si(wave_m, temp_kelvin):
    """Blackbody wavelength density, L_λ, in SI units [W/m/sr/m/m]"""
    denom_factor = _wave_density_denom(wave_m, temp_kelvin)
    lg_term1 = LG2 + LGH + 2 * LGC
    lg_term2 = 5 * lax.log(wave_m)
    exparg = lg_term1 - lg_term2
    wave_factor = lax.exp(exparg)
    return wave_factor * denom_factor
