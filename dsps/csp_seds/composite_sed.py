"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from .ssp_weights import _calc_ssp_weights_lognormal_mdf, _calc_ssp_weights_met_table
from .stellar_age_weights import _calc_logsm_table_from_sfh_table
from ..constants import SFR_MIN


@jjit
def _calc_rest_sed_lognormal_mdf(
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
def _calc_rest_sed_met_table(
    t_obs,
    gal_t_table,
    gal_sfr_table,
    ssp_lg_age,
    ssp_lgmet,
    ssp_flux,
    gal_lgmet_table,
    gal_lgmet_scatter,
    sfr_min=SFR_MIN,
):
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
