"""Kernels calculating metallicity PDF-weighting of SSP tempates"""
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from .stellar_age_weights import _get_lgt_birth
from ..utils import triweighted_histogram, _get_bin_edges
from ..utils import _fill_empty_weights_singlepoint
from ..constants import LGMET_LO, LGMET_HI


__all__ = (
    "calc_lgmet_weights_from_lognormal_mdf",
    "calc_lgmet_weights_from_lgmet_table",
)


@jjit
def calc_lgmet_weights_from_lognormal_mdf(gal_lgmet, gal_lgmet_scatter, ssp_lgmet):
    """Calculate PDF-weights of lognormal metallicity distribution function

    Parameters
    ----------
    gal_lgmet : float
        log10(Z), center of the lognormal metallicity distribution function

    gal_lgmet_scatter : float
        lognormal scatter about gal_lgmet

    ssp_lgmet : ndarray of shape (n_ages, )
        Array of log10(Z) of the SSP templates

    Returns
    -------
    lgmet_weights : ndarray of shape (n_met, )
        SSP weights of the distribution of stellar metallicity

    """
    lgmetbin_edges = _get_bin_edges(ssp_lgmet, LGMET_LO, LGMET_HI)
    lgmet_weights = _get_lgmet_weights_singlegal(
        gal_lgmet, gal_lgmet_scatter, lgmetbin_edges
    )
    return lgmet_weights


@jjit
def calc_lgmet_weights_from_lgmet_table(
    gal_t_table,
    gal_lgmet_table,
    gal_lgmet_scatter,
    ssp_lgmet,
    ssp_lg_age_gyr,
    t_obs,
):
    """Calculate PDF-weights from a tabulated metallicity history

    Parameters
    ----------
    gal_t_table : ndarray of shape (n_t, )
        Age of the universe in Gyr when the galaxy metallicity is tabulated

    gal_lgmet_table : ndarray of shape (n_t, )
        Tabulation of log10(Z) at the times gal_t_table

    gal_lgmet_scatter : float
        lognormal scatter about gal_lgmet

    ssp_lgmet : ndarray of shape (n_ages, )
        Array of log10(Z) of the SSP templates

    ssp_lg_age_gyr : ndarray of shape (n_ages, )
        Array of log10(age/Gyr) of the stellar ages of the SSP templates

    Returns
    -------
    lgmet_weight_matrix : ndarray of shape (n_met, n_ages)
        SSP weights of the distribution of stellar metallicity

    """
    lgmet_at_ssp_lgages = _calc_lgmet_at_ssp_lgage_table(
        gal_t_table, gal_lgmet_table, ssp_lg_age_gyr, t_obs
    )

    lgmetbin_edges = _get_bin_edges(ssp_lgmet, LGMET_LO, LGMET_HI)
    lgmet_weight_matrix = _get_lgmet_weights_singlegal_zh(
        lgmet_at_ssp_lgages, gal_lgmet_scatter, lgmetbin_edges
    )
    # Normalize so that sum of all matrix elements is unity
    lgmet_weight_matrix = lgmet_weight_matrix / ssp_lg_age_gyr.size

    lgmet_weight_matrix = jnp.swapaxes(lgmet_weight_matrix, 1, 0)

    return lgmet_weight_matrix


@jjit
def _get_lgmet_weights_singlegal(gal_lgmet, gal_lgmet_scatter, lgmetbin_edges):
    tw_hist_results = triweighted_histogram(
        gal_lgmet, gal_lgmet_scatter, lgmetbin_edges
    )

    tw_hist_results_sum = jnp.sum(tw_hist_results, axis=0)

    zmsk = tw_hist_results_sum == 0
    tw_hist_results_sum = jnp.where(zmsk, 1.0, tw_hist_results_sum)
    weights = tw_hist_results / tw_hist_results_sum

    return _fill_empty_weights_singlepoint(gal_lgmet, lgmetbin_edges, weights)


_A = (0, None, None)
_get_lgmet_weights_singlegal_zh = jjit(vmap(_get_lgmet_weights_singlegal, in_axes=_A))


@jjit
def _calc_lgmet_at_ssp_lgage_table(gal_t_table, gal_lgmet_table, ssp_lg_age_gyr, t_obs):
    lgt_table = jnp.log10(gal_t_table)
    lgt_birth = _get_lgt_birth(t_obs, ssp_lg_age_gyr)
    lgmet_at_ssp_lgages = jnp.interp(lgt_birth, lgt_table, gal_lgmet_table)
    return lgmet_at_ssp_lgages
