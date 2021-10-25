"""
"""
from jax import vmap
from jax import jit as jjit
from jax import numpy as jnp
from .stellar_ages import _get_sfh_tables, _get_age_weights_from_tables
from .stellar_ages import _get_lgt_birth, _get_lg_age_bin_edges
from .metallicity import calc_lgmet_weights_from_logsm_table_single_t_birth


_a = (0, None, None, None, None)
calc_lgmet_weights_from_logsm_table = jjit(
    vmap(calc_lgmet_weights_from_logsm_table_single_t_birth, in_axes=_a)
)


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

    lg_age_bin_edges = _get_lg_age_bin_edges(lg_ages)
    lgt_birth_bin_edges = _get_lgt_birth(t_obs, lg_age_bin_edges)
    lgt_birth_bin_mids = _get_lgt_birth(t_obs, lg_ages)

    age_weights = _get_age_weights_from_tables(
        lgt_birth_bin_edges, lgt_table, logsm_table
    )

    lgmet_weights = calc_lgmet_weights_from_logsm_table(
        lgt_birth_bin_mids, lgZsun_bin_mids, lgt_table, logsm_table, met_params
    )

    w = lgmet_weights.T * age_weights.reshape((1, n_ages))
    wmags = w.reshape((n_met, n_ages, 1)) * ssp_templates
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
