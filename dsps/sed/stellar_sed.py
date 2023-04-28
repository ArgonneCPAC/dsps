"""Functions calculating the SED of a composite stellar population"""
import typing
from jax import jit as jjit
from jax import numpy as jnp
from .ssp_weights import calc_ssp_weights_sfh_table_lognormal_mdf
from .ssp_weights import calc_ssp_weights_sfh_table_met_table
from .stellar_age_weights import _calc_logsm_table_from_sfh_table
from ..constants import SFR_MIN

__all__ = ("calc_rest_sed_sfh_table_lognormal_mdf", "calc_rest_sed_sfh_table_met_table")


class RestSED(typing.NamedTuple):
    """namedtuple with 4 entries storing SED and information about the SSPs

    rest_sed : ndarray of shape (n_wave, )
        Restframe SED of the galaxy in units of Lsun/Hz

    weights : ndarray of shape (n_met, n_ages, 1)
        SSP weights of the joint distribution of stellar age and metallicity

    lgmet_weights : ndarray of shape (n_met, )
        SSP weights of the distribution of stellar metallicity

    age_weights : ndarray of shape (n_ages, )
        SSP weights of the distribution of stellar age

    """

    rest_sed: jnp.ndarray
    weights: jnp.ndarray
    lgmet_weights: jnp.ndarray
    age_weights: jnp.ndarray


@jjit
def calc_rest_sed_sfh_table_lognormal_mdf(
    gal_t_table,
    gal_sfr_table,
    gal_lgmet,
    gal_lgmet_scatter,
    ssp_lgmet,
    ssp_lg_age_gyr,
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
        log10(Z) of the galaxy at the time of observation

    gal_lgmet_scatter : float
        Lognormal scatter in metallicity

    ssp_lgmet : ndarray of shape (n_met, )
        Array of log10(Z) of the SSP templates

    ssp_lg_age_gyr : ndarray of shape (n_ages, )
        Array of log10(age/Gyr) of the SSP templates

    ssp_flux : ndarray of shape (n_met, n_ages, n_wave)
        SED of the SSP in units of Lsun/Hz/Msun

    t_obs : float
        Age of the universe in Gyr at the time the galaxy is observed

    Returns
    -------
    RestSED : namedtuple with the following entries:

        rest_sed : ndarray of shape (n_wave, )
            Restframe SED of the galaxy in units of Lsun/Hz

        weights : ndarray of shape (n_met, n_ages, 1)
            SSP weights of the joint distribution of stellar age and metallicity

        lgmet_weights : ndarray of shape (n_met, )
            SSP weights of the distribution of stellar metallicity

        age_weights : ndarray of shape (n_ages, )
            SSP weights of the distribution of stellar age

    """
    weights, lgmet_weights, age_weights = calc_ssp_weights_sfh_table_lognormal_mdf(
        gal_t_table,
        gal_sfr_table,
        gal_lgmet,
        gal_lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age_gyr,
        t_obs,
    )
    n_met, n_ages = weights.shape
    sed_unit_mstar = jnp.sum(
        ssp_flux * weights.reshape((n_met, n_ages, 1)), axis=(0, 1)
    )

    lgt_obs = jnp.log10(t_obs)
    lgt_table = jnp.log10(gal_t_table)
    logsm_table = _calc_logsm_table_from_sfh_table(gal_t_table, gal_sfr_table, sfr_min)
    logsm_obs = jnp.interp(lgt_obs, lgt_table, logsm_table)
    mstar_obs = 10**logsm_obs

    rest_sed = sed_unit_mstar * mstar_obs
    return RestSED(rest_sed, weights, lgmet_weights, age_weights)


@jjit
def calc_rest_sed_sfh_table_met_table(
    gal_t_table,
    gal_sfr_table,
    gal_lgmet_table,
    gal_lgmet_scatter,
    ssp_lgmet,
    ssp_lg_age_gyr,
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

    ssp_lgmet : ndarray of shape (n_met, )
        Metallicity of stellar populations of the input SSP table ssp_flux

    ssp_lg_age_gyr : ndarray of shape (n_ages, )
        Age of stellar populations of the input SSP table ssp_flux

    ssp_flux : ndarray of shape (n_met, n_ages, n_wave)
        SED of the SSP in units of Lsun/Hz/Msun

    t_obs : float
        Age of the universe in Gyr at the time the galaxy is observed

    Returns
    -------
    RestSED : namedtuple with the following entries:

        rest_sed : ndarray of shape (n_wave, )
            Restframe SED of the galaxy in units of Lsun/Hz

        weights : ndarray of shape (n_met, n_ages)
            SSP weights of the joint distribution of stellar age and metallicity

        lgmet_weights : ndarray of shape (n_met, )
            SSP weights of the distribution of stellar metallicity

        age_weights : ndarray of shape (n_ages, )
            SSP weights of the distribution of stellar age

    """
    weights, lgmet_weights, age_weights = calc_ssp_weights_sfh_table_met_table(
        gal_t_table,
        gal_sfr_table,
        gal_lgmet_table,
        gal_lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age_gyr,
        t_obs,
    )
    n_met, n_ages = weights.shape
    sed_unit_mstar = jnp.sum(
        ssp_flux * weights.reshape((n_met, n_ages, 1)), axis=(0, 1)
    )

    lgt_obs = jnp.log10(t_obs)
    lgt_table = jnp.log10(gal_t_table)
    logsm_table = _calc_logsm_table_from_sfh_table(gal_t_table, gal_sfr_table, sfr_min)
    logsm_obs = jnp.interp(lgt_obs, lgt_table, logsm_table)
    mstar_obs = 10**logsm_obs

    rest_sed = sed_unit_mstar * mstar_obs
    return RestSED(rest_sed, weights, lgmet_weights, age_weights)
