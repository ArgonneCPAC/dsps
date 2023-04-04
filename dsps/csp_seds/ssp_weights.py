"""
"""
from jax import jit as jjit
from .stellar_age_weights import _calc_age_weights_from_sfh_table
from .metallicity_weights import _calc_lgmet_weights_from_lognormal_mdf
from .metallicity_weights import _calc_lgmet_weights_from_lgmet_table


@jjit
def _calc_ssp_weights_lognormal_mdf(
    gal_t_table, gal_sfr_table, lgmet, lgmet_scatter, ssp_lg_age, ssp_lgmet, t_obs
):
    age_weights = _calc_age_weights_from_sfh_table(
        gal_t_table, gal_sfr_table, ssp_lg_age, t_obs
    )
    lgmet_weights = _calc_lgmet_weights_from_lognormal_mdf(
        lgmet, lgmet_scatter, ssp_lgmet
    )

    weights = lgmet_weights.reshape((-1, 1)) * age_weights.reshape((1, -1))
    weights = weights / weights.sum()

    return weights, age_weights, lgmet_weights


@jjit
def _calc_ssp_weights_met_table(
    t_obs,
    gal_t_table,
    gal_sfr_table,
    ssp_lg_age,
    ssp_lgmet,
    gal_lgmet_table,
    lgmet_scatter,
):
    age_weights = _calc_age_weights_from_sfh_table(
        gal_t_table, gal_sfr_table, ssp_lg_age, t_obs
    )
    lgmet_weights = _calc_lgmet_weights_from_lgmet_table(
        gal_t_table, gal_lgmet_table, lgmet_scatter, ssp_lgmet, ssp_lg_age, t_obs
    )

    weights = lgmet_weights * age_weights.reshape((1, -1))
    weights = weights / weights.sum()

    return weights, age_weights, lgmet_weights
