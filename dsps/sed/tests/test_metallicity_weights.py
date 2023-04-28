"""
"""
from jax import random as jran
import numpy as np
from ...utils import _get_bin_edges
from ..metallicity_weights import _get_lgmet_weights_singlegal
from ..metallicity_weights import calc_lgmet_weights_from_lognormal_mdf
from ..metallicity_weights import calc_lgmet_weights_from_lgmet_table
from ...constants import T_BIRTH_MIN


SEED = 43
FSPS_LG_AGES = np.arange(5.5, 10.2, 0.05)  # log10 ages in years


def test_get_lgmet_weights_singlegal():
    n_bins = 22
    _lgzbin_mids = np.linspace(-3.7, -1.523, n_bins)
    lgzsunbin_mids = _lgzbin_mids - _lgzbin_mids[-3]
    lgzsunbins = _get_bin_edges(lgzsunbin_mids, -100.0, 100.0)

    lgz_scatter = 0.25
    ngals = 200
    lgzdata = np.linspace(-2.5, 0.5, ngals)

    lgzdata[:2] = -500
    lgzdata[-2:] = 500

    lgz = -0.5

    weights = _get_lgmet_weights_singlegal(lgz, lgz_scatter, lgzsunbins)
    assert weights.shape == (n_bins,)


def test_calc_lgmet_weights_from_lognormal_mdf():
    met_key = jran.PRNGKey(SEED)

    n_met = 15
    ssp_lgmet = np.linspace(-4, 0.5, n_met)
    lgmet = jran.uniform(
        met_key, minval=ssp_lgmet.min(), maxval=ssp_lgmet.max(), shape=()
    )

    lgmet_scatter = 0.1
    lgmet_weights = calc_lgmet_weights_from_lognormal_mdf(
        lgmet, lgmet_scatter, ssp_lgmet
    )
    assert lgmet_weights.shape == (n_met,)
    assert np.allclose(lgmet_weights.sum(), 1.0, rtol=1e-4)


def test_calc_lgmet_weights_from_lgmet_table():
    met_key = jran.PRNGKey(SEED)
    t_obs = 13.0
    n_t = 500
    gal_t_table = np.linspace(T_BIRTH_MIN, t_obs, n_t)

    n_met = 15
    ssp_lgmet = np.linspace(-4, 0.5, n_met)
    gal_lgmet_table = jran.uniform(
        met_key, minval=ssp_lgmet.min(), maxval=ssp_lgmet.max(), shape=(n_t,)
    )
    ssp_lg_age_gyr = FSPS_LG_AGES - 9.0
    n_ages = ssp_lg_age_gyr.size

    lgmet_scatter = 0.1
    lgmet_weights = calc_lgmet_weights_from_lgmet_table(
        gal_t_table,
        gal_lgmet_table,
        lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age_gyr,
        t_obs,
    )
    assert lgmet_weights.shape == (n_met, n_ages)
    assert np.allclose(lgmet_weights.sum(), 1.0, rtol=1e-4)
