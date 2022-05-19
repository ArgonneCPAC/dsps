"""
"""
from jax import jit as jjit
from jax import numpy as jnp


RV_C00 = 4.05
N09_X0_MIN = 0.0
N09_GAMMA_MIN = 0.0
N09_SLOPE_MIN = -3.0
N09_SLOPE_MAX = 3.0


@jjit
def _flux_ratio(k, rv, av):
    return 10 ** (-0.4 * _attenuation_curve(k, rv, av))


@jjit
def _get_optical_depth_V(logsm, logssfr, tau_mstar, tau_ssfr, tau_norm):
    return tau_mstar * (logsm - 10) + tau_ssfr * (logssfr + 10) + tau_norm


@jjit
def _get_attentuation_amplitude(logsm, logssfr, tau_mstar, tau_ssfr, tau_norm, cosi):
    tau_v = _get_optical_depth_V(logsm, logssfr, tau_mstar, tau_ssfr, tau_norm)
    x = tau_v / cosi
    logarg = (1 - jnp.exp(-x)) / x
    Av = -2.5 * jnp.log10(logarg)
    return Av


@jjit
def _get_eb_from_delta(delta):
    return -1.9 * delta + 0.85


@jjit
def _get_delta(logsm, logssfr, delta_mstar, delta_ssfr, delta_norm):
    return delta_mstar * (logsm - 10) + delta_ssfr * (logssfr + 10) + delta_norm


@jjit
def _attenuation_curve(k, rv, av):
    attenuation = av * k / rv
    return jnp.where(attenuation < 0, 0, attenuation)


@jjit
def calzetti00_k_lambda(x, rv):
    """Reddening curve k(λ) = A(λ) / E(B-V)

    Parameters
    ----------
    x : ndarray of shape (n, )
        Wavelength in microns

    rv : float

    Returns
    -------
    k_lambda : ndarray of shape (n, )
        Reddening curve

    """
    k1 = 2.659 * (-2.156 + 1.509 * 1 / x - 0.198 * 1 / x**2 + 0.011 * 1 / x**3) + rv
    k2 = 2.659 * (-1.857 + 1.040 * 1 / x) + rv
    return jnp.where(x < 0.63, k1, k2)


@jjit
def leitherer02_k_lambda(x, rv):
    """Reddening curve k(λ) = A(λ) / E(B-V)

    Parameters
    ----------
    x : ndarray of shape (n, )
        Wavelength in microns

    rv : float

    Returns
    -------
    k_lambda : ndarray of shape (n, )
        Reddening curve

    """
    k = 5.472 + (0.671 * 1 / x - 9.218 * 1e-3 / x**2 + 2.620 * 1e-3 / x**3)
    return k


@jjit
def drude_bump(x, x0, bump_width, Eb):
    num = (x * bump_width) ** 2
    denom = (x**2 - x0**2) ** 2 + num
    bump = num / denom
    return Eb * bump


@jjit
def power_law_vband_norm(x, delta):
    """Power law normalized at 0.55 microns (V band)."""
    return (x / 0.55) ** delta


@jjit
def noll09_k_lambda(x, x0, bump_width, Eb, delta):

    # Leitherer 2002 below 0.15 microns and Calzetti 2000 above
    k_c00 = calzetti00_k_lambda(x, RV_C00)
    k_l02 = leitherer02_k_lambda(x, RV_C00)
    k = jnp.where(x > 0.15, k_c00, k_l02)

    # Add the UV bump
    k = k + drude_bump(x, x0, bump_width, Eb)

    # Apply power-law correction
    k = k * power_law_vband_norm(x, delta)

    # Clip at zero
    k = jnp.where(k < 0, 0, k)

    return k


@jjit
def _Rvmod_sbl18(delta):
    num = RV_C00
    denom = (RV_C00 + 1) * (4400 / 5500) ** delta - RV_C00
    return num / denom


@jjit
def sbl18_k_lambda(x, x0, bump_width, Eb, delta):

    # Leitherer 2002 below 0.15 microns and Calzetti 2000 above
    k_c00 = calzetti00_k_lambda(x, RV_C00)
    k_l02 = leitherer02_k_lambda(x, RV_C00)
    k = jnp.where(x > 0.15, k_c00, k_l02)

    # Apply power-law correction
    k = k * power_law_vband_norm(x, delta)

    # Add the UV bump
    k = k + drude_bump(x, x0, bump_width, Eb)

    # Clip at zero
    k = jnp.where(k < 0, 0, k)

    return k
