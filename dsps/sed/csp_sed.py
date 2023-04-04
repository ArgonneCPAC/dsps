"""Functions calculating the SED of a composite stellar population"""
from jax import jit as jjit
from jax import numpy as jnp
from .ssp_weights import _calc_ssp_weights_lognormal_mdf, _calc_ssp_weights_met_table
from .stellar_age_weights import _calc_logsm_table_from_sfh_table
from ..constants import SFR_MIN


@jjit
def calc_rest_sed_lognormal_mdf(
    gal_t_table,
    gal_sfr_table,
    gal_lgmet,
    gal_lgmet_scatter,
    ssp_lg_age,
    ssp_lgmet,
    ssp_flux,
    t_obs,
    sfr_min=SFR_MIN,
):
    """
    Calculate the SED of a galaxy defined by input tables of SFH and
    a lognormal metallicity distribution function

    Parameters
    ----------
    gal_t_table : ndarray of shape (n_t, )
        Age of the universe in Gyr at which the input galaxy SFH and metallicity
        have been tabulated

    gal_sfr_table : ndarray of shape (n_t, )
        Star formation history in Msun/yr evaluated at the input gal_t_table

    gal_lgmet : ndarray of shape (n_t, )
        Metallicity of the galaxy at the time of observation

    gal_lgmet_scatter : float
        Lognormal scatter in metallicity

    ssp_lg_age : ndarray of shape (n_ages, )
        Age of stellar populations of the input SSP table ssp_flux

    ssp_lgmet : ndarray of shape (n_met, )
        Metallicity of stellar populations of the input SSP table ssp_flux

    ssp_flux : ndarray of shape (n_met, n_ages, n_wave)
        SED of the SSP in units of Lsun/Hz/Msun

    t_obs : float
        Age of the universe in Gyr at the time the galaxy is observed

    Returns
    -------
    rest_sed : ndarray of shape (n_wave, )
        Restframe SED of the galaxy in units of Lsun/Hz

    """
    weights, age_weights, lgmet_weights = _calc_ssp_weights_lognormal_mdf(
        gal_t_table,
        gal_sfr_table,
        gal_lgmet,
        gal_lgmet_scatter,
        ssp_lg_age,
        ssp_lgmet,
        t_obs,
    )
    n_met, n_ages = weights.shape
    weights = weights.reshape((n_met, n_ages, 1))
    sed_unit_mstar = jnp.sum(ssp_flux * weights, axis=(0, 1))

    lgt_obs = jnp.log10(t_obs)
    lgt_table = jnp.log10(gal_t_table)
    logsm_table = _calc_logsm_table_from_sfh_table(gal_t_table, gal_sfr_table, sfr_min)
    logsm_obs = jnp.interp(lgt_obs, lgt_table, logsm_table)
    mstar_obs = 10**logsm_obs

    rest_sed = sed_unit_mstar * mstar_obs
    return rest_sed


@jjit
def calc_rest_sed_met_table(
    gal_t_table,
    gal_sfr_table,
    gal_lgmet_table,
    gal_lgmet_scatter,
    ssp_lg_age,
    ssp_lgmet,
    ssp_flux,
    t_obs,
    sfr_min=SFR_MIN,
):
    """
    Calculate the SED of a galaxy defined by input tables of SFH and metallicity

    Parameters
    ----------
    gal_t_table : ndarray of shape (n_t, )
        Age of the universe in Gyr at which the input galaxy SFH and metallicity
        have been tabulated

    gal_sfr_table : ndarray of shape (n_t, )
        Star formation history in Msun/yr evaluated at the input gal_t_table

    gal_lgmet_table : ndarray of shape (n_t, )
        Metallicity history evaluated at the input gal_t_table

    gal_lgmet_scatter : float
        Lognormal scatter in metallicity

    ssp_lg_age : ndarray of shape (n_ages, )
        Age of stellar populations of the input SSP table ssp_flux

    ssp_lgmet : ndarray of shape (n_met, )
        Metallicity of stellar populations of the input SSP table ssp_flux

    ssp_flux : ndarray of shape (n_met, n_ages, n_wave)
        SED of the SSP in units of Lsun/Hz/Msun

    t_obs : float
        Age of the universe in Gyr at the time the galaxy is observed

    Returns
    -------
    rest_sed : ndarray of shape (n_wave, )
        Restframe SED of the galaxy in units of Lsun/Hz

    """
    weights, age_weights, lgmet_weights = _calc_ssp_weights_met_table(
        gal_t_table,
        gal_sfr_table,
        gal_lgmet_table,
        gal_lgmet_scatter,
        ssp_lg_age,
        ssp_lgmet,
        t_obs,
    )
    n_met, n_ages = weights.shape
    weights = weights.reshape((n_met, n_ages, 1))
    sed_unit_mstar = jnp.sum(ssp_flux * weights, axis=(0, 1))

    lgt_obs = jnp.log10(t_obs)
    lgt_table = jnp.log10(gal_t_table)
    logsm_table = _calc_logsm_table_from_sfh_table(gal_t_table, gal_sfr_table, sfr_min)
    logsm_obs = jnp.interp(lgt_obs, lgt_table, logsm_table)
    mstar_obs = 10**logsm_obs

    rest_sed = sed_unit_mstar * mstar_obs
    return rest_sed
