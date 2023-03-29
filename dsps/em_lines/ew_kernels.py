"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from .equivalent_width import _ew_kernel
from ..utils import _get_bin_edges, _get_triweights_singlepoint
from ..ssp.weighted_ssps import _get_age_weights
from ..metallicity.mzr import LGMET_LO, LGMET_HI

LGU_LO, LGU_HI = -10.0, 10.0


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
def _calc_age_met_lgu_weights_from_sfh_table(
    t_obs,
    lgZsun_bin_mids,
    lg_ages,
    lgU_bin_mids,
    lgt_table,
    logsm_table,
    lgmet,
    lgmet_scatter,
    lgu,
    lgu_scatter,
):
    lgt_birth_bin_mids, age_weights = _get_age_weights(
        t_obs, lg_ages, lgt_table, logsm_table
    )
    lgmet_bin_edges = _get_bin_edges(lgZsun_bin_mids, LGMET_LO, LGMET_HI)
    lgmet_weights = _get_triweights_singlepoint(lgmet, lgmet_scatter, lgmet_bin_edges)

    lgu_bin_edges = _get_bin_edges(lgU_bin_mids, LGU_LO, LGU_HI)
    lgu_weights = _get_triweights_singlepoint(lgu, lgu_scatter, lgu_bin_edges)
    return age_weights, lgmet_weights, lgu_weights


@jjit
def _calc_weighted_precomputed_ew_from_sfh_table_const_lgu_lgmet(
    t_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    lgU_bin_mids,
    ssp_ews,
    t_table,
    lgt_table,
    dt_table,
    sfh_table,
    logsm_table,
    lgmet,
    lgmet_scatter,
    lgu,
    lgu_scatter,
):
    n_lgu, n_met, n_ages = ssp_ews.shape

    _res = _calc_age_met_lgu_weights_from_sfh_table(
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        lgU_bin_mids,
        lgt_table,
        logsm_table,
        lgmet,
        lgmet_scatter,
        lgu,
        lgu_scatter,
    )
    age_weights, lgmet_weights, lgu_weights = _res

    age_weights = age_weights.reshape((1, 1, n_ages))
    lgmet_weights = lgmet_weights.reshape((1, n_met, 1))
    lgu_weights = lgu_weights.reshape((n_lgu, 1, 1))
    ssp_weights = age_weights * lgmet_weights * lgu_weights
    weighted_ew = jnp.sum(ssp_ews * ssp_weights, axis=(0, 1, 2))

    return weighted_ew


@jjit
def _calc_ew_from_sfh_table_const_lgu_lgmet(
    t_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    lgU_bin_mids,
    ssp_wave,
    ssp_flux,
    lgt_table,
    logsm_table,
    lgmet,
    lgmet_scatter,
    lgu,
    lgu_scatter,
    line_lo,
    line_mid,
    line_hi,
    cont_lo_lo,
    cont_lo_hi,
    cont_hi_lo,
    cont_hi_hi,
):
    n_lgu, n_met, n_ages, n_spec = ssp_flux.shape

    _res = _calc_age_met_lgu_weights_from_sfh_table(
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        lgU_bin_mids,
        lgt_table,
        logsm_table,
        lgmet,
        lgmet_scatter,
        lgu,
        lgu_scatter,
    )
    age_weights, lgmet_weights, lgu_weights = _res

    age_weights = age_weights.reshape((1, 1, n_ages, 1))
    lgmet_weights = lgmet_weights.reshape((1, n_met, 1, 1))
    lgu_weights = lgu_weights.reshape((n_lgu, 1, 1, 1))
    ssp_weights = age_weights * lgmet_weights * lgu_weights
    weighted_ssp = jnp.sum(ssp_flux * ssp_weights, axis=(0, 1, 2))

    ew, total_line_flux = _ew_kernel(
        ssp_wave,
        weighted_ssp,
        line_lo,
        line_mid,
        line_hi,
        cont_lo_lo,
        cont_lo_hi,
        cont_hi_lo,
        cont_hi_hi,
    )
    ew = jnp.where(ew < 0, 0, ew)
    total_line_flux = jnp.where(total_line_flux < 0, 0, total_line_flux)
    return ew, total_line_flux


@jjit
def _calc_ew_from_sfh_table_const_lgmet(
    t_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    ssp_wave,
    ssp_flux,
    lgt_table,
    logsm_table,
    lgmet,
    lgmet_scatter,
    line_lo,
    line_mid,
    line_hi,
    cont_lo_lo,
    cont_lo_hi,
    cont_hi_lo,
    cont_hi_hi,
):
    n_met, n_ages, n_spec = ssp_flux.shape

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

    age_weights = age_weights.reshape((1, n_ages, 1))
    lgmet_weights = lgmet_weights.reshape((n_met, 1, 1))
    ssp_weights = age_weights * lgmet_weights
    weighted_ssp = jnp.sum(ssp_flux * ssp_weights, axis=(0, 1))

    ew, total_line_flux = _ew_kernel(
        ssp_wave,
        weighted_ssp,
        line_lo,
        line_mid,
        line_hi,
        cont_lo_lo,
        cont_lo_hi,
        cont_hi_lo,
        cont_hi_hi,
    )
    ew = jnp.where(ew < 0, 0, ew)
    total_line_flux = jnp.where(total_line_flux < 0, 0, total_line_flux)
    return ew, total_line_flux
