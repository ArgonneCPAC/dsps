"""
"""
from jax import jit as jjit
from .stellar_age_weights import _calc_age_weights_from_sfh_table
from .metallicity_weights import _calc_lgmet_weights_from_lognormal_mdf


@jjit
def _calc_ssp_weights_lognormal_mdf(
    t_obs, gal_t_table, gal_sfr_table, ssp_lg_age, ssp_lgmet, lgmet, lgmet_scatter
):
    age_weights = _calc_age_weights_from_sfh_table(
        gal_t_table, gal_sfr_table, ssp_lg_age, t_obs
    )
    lgmet_weights = _calc_lgmet_weights_from_lognormal_mdf(
        lgmet, lgmet_scatter, ssp_lgmet
    )

    lgmet_weights = lgmet_weights.reshape((-1, 1))
    age_weights = age_weights.reshape((1, -1))
    weights = lgmet_weights * age_weights

    return weights, age_weights, lgmet_weights
