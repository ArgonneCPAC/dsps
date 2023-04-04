"""
"""
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from .stellar_age_weights import _get_lgt_birth
from ..utils import triweighted_histogram, _get_bin_edges
from ..utils import _fill_empty_weights_singlepoint
from ..constants import LGMET_LO, LGMET_HI


@jjit
def _calc_lgmet_weights_from_lognormal_mdf(lgmet, lgmet_scatter, ssp_lgmet):
    lgmetbin_edges = _get_bin_edges(ssp_lgmet, LGMET_LO, LGMET_HI)
    lgmet_weights = _get_lgmet_weights_singlegal(lgmet, lgmet_scatter, lgmetbin_edges)
    return lgmet_weights


@jjit
def _calc_lgmet_weights_from_lgmet_table(
    gal_t_table,
    gal_lgmet_table,
    ssp_lgmet,
    ssp_lg_age,
    t_obs,
    lgmet_scatter,
):
    lgmet_at_ssp_lgages = _calc_lgmet_at_ssp_lgage_table(
        gal_t_table, gal_lgmet_table, ssp_lg_age, t_obs
    )

    lgmetbin_edges = _get_bin_edges(ssp_lgmet, LGMET_LO, LGMET_HI)
    lgmet_weight_matrix = _get_lgmet_weights_singlegal_zh(
        lgmet_at_ssp_lgages, lgmet_scatter, lgmetbin_edges
    )
    # Normalize so that sum of all matrix elements is unity
    lgmet_weight_matrix = lgmet_weight_matrix / ssp_lg_age.size

    lgmet_weight_matrix = jnp.swapaxes(lgmet_weight_matrix, 1, 0)

    return lgmet_weight_matrix


@jjit
def _get_lgmet_weights_singlegal(lgmet, lgmet_scatter, lgmetbin_edges):
    tw_hist_results = triweighted_histogram(lgmet, lgmet_scatter, lgmetbin_edges)

    tw_hist_results_sum = jnp.sum(tw_hist_results, axis=0)

    zmsk = tw_hist_results_sum == 0
    tw_hist_results_sum = jnp.where(zmsk, 1.0, tw_hist_results_sum)
    weights = tw_hist_results / tw_hist_results_sum

    return _fill_empty_weights_singlepoint(lgmet, lgmetbin_edges, weights)


_A = (0, None, None)
_get_lgmet_weights_singlegal_zh = jjit(vmap(_get_lgmet_weights_singlegal, in_axes=_A))


@jjit
def _calc_lgmet_at_ssp_lgage_table(gal_t_table, gal_lgmet_table, ssp_lg_age, t_obs):
    lgt_table = jnp.log10(gal_t_table)
    lgt_birth = _get_lgt_birth(t_obs, ssp_lg_age)
    lgmet_at_ssp_lgages = jnp.interp(lgt_birth, lgt_table, gal_lgmet_table)
    return lgmet_at_ssp_lgages
