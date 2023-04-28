"""Kernels calculating SSP weights of a composite stellar population"""
import typing
from jax import jit as jjit
from jax import numpy as jnp
from .stellar_age_weights import calc_age_weights_from_sfh_table
from .metallicity_weights import calc_lgmet_weights_from_lognormal_mdf
from .metallicity_weights import calc_lgmet_weights_from_lgmet_table

__all__ = (
    "calc_ssp_weights_sfh_table_lognormal_mdf",
    "calc_ssp_weights_sfh_table_met_table",
)


class SSPWeights(typing.NamedTuple):
    weights: jnp.ndarray
    lgmet_weights: jnp.ndarray
    age_weights: jnp.ndarray


@jjit
def calc_ssp_weights_sfh_table_lognormal_mdf(
    gal_t_table,
    gal_sfr_table,
    gal_lgmet,
    gal_lgmet_scatter,
    ssp_lgmet,
    ssp_lg_age_gyr,
    t_obs,
):
    """Calculate SSP weights of a tabulated SFH and a lognormal MDF

    Parameters
    ----------
    gal_t_table : ndarray of shape (n_t, )
        Age of the universe in Gyr when the galaxy SFH is tabulated

    gal_sfr_table : ndarray of shape (n_t, )
        Tabulation of the galaxy SFH in Msun/yr at the times gal_t_table

    gal_lgmet : float
        log10(Z), center of the lognormal metallicity distribution function

    gal_lgmet_scatter : float
        lognormal scatter about gal_lgmet

    ssp_lgmet : ndarray of shape (n_ages, )
        Array of log10(Z) of the SSP templates

    ssp_lg_age_gyr : ndarray of shape (n_ages, )
        Array of log10(age/Gyr) of the SSP templates

    t_obs : float
        Age of the universe in Gyr at the time the galaxy is observed

    Returns
    -------
    SSPWeights : namedtuple with the following entries:

        weights : ndarray of shape (n_met, n_ages)
            SSP weights of the joint distribution of stellar age and metallicity

        lgmet_weights : ndarray of shape (n_met, )
            SSP weights of the distribution of stellar metallicity

        age_weights : ndarray of shape (n_ages, )
            SSP weights of the distribution of stellar age

    """
    age_weights = calc_age_weights_from_sfh_table(
        gal_t_table, gal_sfr_table, ssp_lg_age_gyr, t_obs
    )
    lgmet_weights = calc_lgmet_weights_from_lognormal_mdf(
        gal_lgmet, gal_lgmet_scatter, ssp_lgmet
    )

    weights = lgmet_weights.reshape((-1, 1)) * age_weights.reshape((1, -1))
    weights = weights / weights.sum()

    return SSPWeights(weights, lgmet_weights, age_weights)


@jjit
def calc_ssp_weights_sfh_table_met_table(
    gal_t_table,
    gal_sfr_table,
    gal_lgmet_table,
    gal_lgmet_scatter,
    ssp_lgmet,
    ssp_lg_age_gyr,
    t_obs,
):
    """Calculate SSP weights of a tabulated star-formation and metallicity history

    Parameters
    ----------
    gal_t_table : ndarray of shape (n_t, )
        Age of the universe in Gyr when the galaxy SFH is tabulated

    gal_sfr_table : ndarray of shape (n_t, )
        Tabulation of the galaxy SFH in Msun/yr at the times gal_t_table

    gal_lgmet_table : ndarray of shape (n_t, )
        log10(Z) tabulation of the galaxy stellar metallicity Z at the times gal_t_table

    gal_lgmet_scatter : float
        lognormal scatter about each stellar metallicity in gal_lgmet_table

    ssp_lgmet : ndarray of shape (n_ages, )
        Array of log10(Z) of the SSP templates

    ssp_lg_age_gyr : ndarray of shape (n_ages, )
        Array of log10(age/Gyr) of the stellar ages of the SSP templates

    t_obs : float
        Age of the universe in Gyr at the time the galaxy is observed

    Returns
    -------
    SSPWeights : namedtuple with the following entries:

        weights : ndarray of shape (n_met, n_ages)
            SSP weights of the joint distribution of stellar age and metallicity

        lgmet_weights : ndarray of shape (n_met, )
            SSP weights of the distribution of stellar metallicity

        age_weights : ndarray of shape (n_ages, )
            SSP weights of the distribution of stellar age

    """
    age_weights = calc_age_weights_from_sfh_table(
        gal_t_table, gal_sfr_table, ssp_lg_age_gyr, t_obs
    )
    lgmet_weights = calc_lgmet_weights_from_lgmet_table(
        gal_t_table,
        gal_lgmet_table,
        gal_lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age_gyr,
        t_obs,
    )

    weights = lgmet_weights * age_weights.reshape((1, -1))
    weights = weights / weights.sum()

    return SSPWeights(weights, lgmet_weights, age_weights)
