"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from .stellar_ages import _get_sfh_tables, _get_age_weights_from_tables
from .stellar_ages import _get_lgt_birth, _get_lg_age_bin_edges
from .mzr import calc_lgmet_weights_from_logsm_table
from .mzr import _get_met_weights_singlegal, LGMET_LO, LGMET_HI
from .utils import _get_bin_edges


@jjit
def _get_age_weights(t_obs, lg_ages, lgt_table, logsm_table):
    lg_age_bin_edges = _get_lg_age_bin_edges(lg_ages)
    lgt_birth_bin_edges = _get_lgt_birth(t_obs, lg_age_bin_edges)
    lgt_birth_bin_mids = _get_lgt_birth(t_obs, lg_ages)
    age_weights = _get_age_weights_from_tables(
        lgt_birth_bin_edges, lgt_table, logsm_table
    )
    return lgt_birth_bin_mids, age_weights


@jjit
def _calc_weighted_ssp_from_sfh_table(
    t_obs,
    lgZsun_bin_mids,
    lg_ages,
    ssp_templates,
    t_table,
    lgt_table,
    dt_table,
    sfh_table,
    logsm_table,
    met_params,
):
    n_met, n_ages, n_filters = ssp_templates.shape

    lgt_birth_bin_mids, age_weights = _get_age_weights(
        t_obs, lg_ages, lgt_table, logsm_table
    )

    lgt_obs = jnp.log10(t_obs)
    lgmet_weights = calc_lgmet_weights_from_logsm_table(
        lgt_obs, lgZsun_bin_mids, lgt_table, logsm_table, met_params
    )
    lgmet_weights = lgmet_weights.reshape((n_met, 1, 1))
    age_weights = age_weights.reshape((1, n_ages, 1))
    w = lgmet_weights * age_weights
    wmags = w * ssp_templates
    return lgmet_weights, age_weights, jnp.sum(wmags, axis=(0, 1))


@jjit
def _calc_weighted_ssp_from_sfh_table_const_zmet(
    t_obs,
    lgZsun_bin_mids,
    lg_ages,
    ssp_templates,
    t_table,
    lgt_table,
    dt_table,
    sfh_table,
    logsm_table,
    lgmet,
    lgmet_scatter,
):
    n_met, n_ages, n_filters = ssp_templates.shape

    lgt_birth_bin_mids, age_weights = _get_age_weights(
        t_obs, lg_ages, lgt_table, logsm_table
    )

    lgmet_bin_edges = _get_bin_edges(lgZsun_bin_mids, LGMET_LO, LGMET_HI)
    lgmet_weights = _get_met_weights_singlegal(lgmet, lgmet_scatter, lgmet_bin_edges)
    lgmet_weights = lgmet_weights.reshape((n_met, 1, 1))
    age_weights = age_weights.reshape((1, n_ages, 1))
    w = lgmet_weights * age_weights
    wmags = w * ssp_templates
    return lgmet_weights, age_weights, jnp.sum(wmags, axis=(0, 1))


@jjit
def _calc_weighted_ssp_from_diffstar_params(
    t_obs,
    lgZsun_bin_mids,
    lg_ages,
    ssp_templates,
    mah_params,
    ms_params,
    q_params,
    met_params,
):
    _res = _get_sfh_tables(mah_params, ms_params, q_params)
    t_table, lgt_table, dt_table, sfh_table, logsm_table = _res

    return _calc_weighted_ssp_from_sfh_table(
        t_obs,
        lgZsun_bin_mids,
        lg_ages,
        ssp_templates,
        t_table,
        lgt_table,
        dt_table,
        sfh_table,
        logsm_table,
        met_params,
    )


@jjit
def _calc_weighted_ssp_from_diffstar_params_const_zmet(
    t_obs,
    lgZsun_bin_mids,
    lg_ages,
    ssp_templates,
    mah_params,
    ms_params,
    q_params,
    lgmet,
    lgmet_scatter,
):
    _res = _get_sfh_tables(mah_params, ms_params, q_params)
    t_table, lgt_table, dt_table, sfh_table, logsm_table = _res

    return _calc_weighted_ssp_from_sfh_table_const_zmet(
        t_obs,
        lgZsun_bin_mids,
        lg_ages,
        ssp_templates,
        t_table,
        lgt_table,
        dt_table,
        sfh_table,
        logsm_table,
        lgmet,
        lgmet_scatter,
    )
