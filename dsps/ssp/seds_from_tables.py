"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap
from .weighted_ssps import _get_age_weights
from ..utils import _get_bin_edges, _get_triweights_singlepoint, _jax_get_dt_array
from ..metallicity.mzr import LGMET_LO, LGMET_HI


__all__ = ("compute_sed_galpop",)

SFR_MIN = 1e-13


@jjit
def _calc_age_met_weights_from_sfh_table(
    t_obs,
    lgZsun_bin_mids,
    lg_ages,
    lgt_table,
    logsm_table,
    lgmet,
    lgmet_scatter,
):
    lgt_birth_bin_mids, age_weights = _get_age_weights(
        t_obs, lg_ages, lgt_table, logsm_table
    )
    lgmet_bin_edges = _get_bin_edges(lgZsun_bin_mids, LGMET_LO, LGMET_HI)
    lgmet_weights = _get_triweights_singlepoint(lgmet, lgmet_scatter, lgmet_bin_edges)

    return age_weights, lgmet_weights


@jjit
def _calc_sed_kern(
    t_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    ssp_flux,
    t_table,
    logsm_table,
    lgmet,
    lgmet_scatter,
):
    n_met, n_ages, n_wave = ssp_flux.shape
    lgt_table = jnp.log10(t_table)
    _res = _calc_age_met_weights_from_sfh_table(
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        lgt_table,
        logsm_table,
        lgmet,
        lgmet_scatter,
    )
    age_weights, lgmet_weights = _res

    age_weights = age_weights.reshape((1, n_ages))
    lgmet_weights = lgmet_weights.reshape((n_met, 1))
    ssp_weights = age_weights * lgmet_weights
    ssp_weights = ssp_weights.reshape((n_met, n_ages, 1))
    sed = jnp.sum(ssp_flux * ssp_weights, axis=(0, 1))
    mstar_obs = 10 ** jnp.interp(jnp.log10(t_obs), lgt_table, logsm_table)
    return sed * mstar_obs


_a = (*[None] * 5, 0, 0, 0)
_calc_sed_vmap = jjit(vmap(_calc_sed_kern, in_axes=_a))


@jjit
def compute_sed_galpop(
    t_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    ssp_flux,
    t_table,
    sfh_table,
    mdf_params,
    sfr_min=SFR_MIN,
):
    """Calculate the SEDs of a galaxy population.

    Parameters
    ----------
    t_obs : float
        Time of observation in Gyr

    lgZsun_bin_mids : ndarray of shape (n_met, )
        SSP bins of log10(Z/Zsun)

    log_age_gyr : ndarray of shape (n_ages, )
        SSP bins of log10(age) in gyr

    ssp_flux : ndarray of shape (n_met, n_ages, n_wave)
        Array storing SSP luminosity in Lsun/Hz

    t_table : ndarray of shape (n_times, )
        Age of the universe in Gyr

    sfh_table : ndarray of shape (n_gals, n_times)
        SFR history for each galaxy in Msun/yr

    mdf_params : ndarray of shape (n_gals, 2)
        Median metallicity and log-normal scatter for each galaxy

    sfr_min : float, optional
        Lower bound clip on SFR. Used for log-safe purposes. Default is 0.0001 Msun/yr

    Returns
    -------
    sed_galpop : ndarray of shape (n_gals, n_wave)
        SED of each galaxy in in Lsun/Hz

    logsmh_galpop : ndarray of shape (n_gals, n_times)
        Base-10 log of cumulative history of stellar mass formed

    """
    dt_table = _jax_get_dt_array(t_table)
    sfh_table = jnp.where(sfh_table < sfr_min, sfr_min, sfh_table)
    smh = 1e9 * jnp.cumsum(sfh_table * dt_table, axis=1)
    logsmh_galpop = jnp.log10(smh)
    lgmet = mdf_params[:, 0]
    lgmet_scatter = mdf_params[:, 1]
    sed_galpop = _calc_sed_vmap(
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        ssp_flux,
        t_table,
        logsmh_galpop,
        lgmet,
        lgmet_scatter,
    )
    return sed_galpop, logsmh_galpop
