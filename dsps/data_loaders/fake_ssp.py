"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from ..utils import _sig_slope


XTP = 7


@jjit
def _c0_at_lgzsun_vs_lgage(log_age_yr):
    ytp, x0, slope_k, lo, hi = -12.5, 8.5, 2, 0, -4.5
    c0_at_lgz0 = _sig_slope(log_age_yr, XTP, ytp, x0, slope_k, lo, hi)
    return c0_at_lgz0


@jjit
def _c1_at_lgzsun_vs_lgage(log_age_yr):
    ytp, x0, slope_k, lo, hi = -0.25, 8.5, 2, 0, 1.0
    c1_at_lgz0 = _sig_slope(log_age_yr, XTP, ytp, x0, slope_k, lo, hi)
    return c1_at_lgz0


@jjit
def _delta_c0_vs_lgmet(lgmet):
    delta_c0 = _sig_slope(lgmet, -2, 0, -2, 3, -2.1, 0)
    return delta_c0


@jjit
def _delta_c1_vs_lgmet(lgmet):
    delta_c1 = _sig_slope(lgmet, -2, 0, -3, 1, 0.85, 0)
    return delta_c1


@jjit
def _get_polynomial_coefficients(lgmet, log_age_yr):
    c0_at_lgz0 = _c0_at_lgzsun_vs_lgage(log_age_yr)
    c1_at_lgz0 = _c1_at_lgzsun_vs_lgage(log_age_yr)

    delta_c0 = _delta_c0_vs_lgmet(lgmet)
    delta_c1 = _delta_c1_vs_lgmet(lgmet)

    c0 = c0_at_lgz0 + delta_c0
    c1 = c1_at_lgz0 + delta_c1
    return c0, c1


@jjit
def _get_fake_ssp_spectrum(lgmet, log_age_yr, wave):
    c0, c1 = _get_polynomial_coefficients(lgmet, log_age_yr)
    lg_flux = c0 + c1 * jnp.log10(wave)
    return 10**lg_flux
