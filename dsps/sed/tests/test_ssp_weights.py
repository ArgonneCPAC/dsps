"""
"""
import numpy as np
from jax import random as jran
from ..ssp_weights import calc_ssp_weights_sfh_table_lognormal_mdf
from ..ssp_weights import calc_ssp_weights_sfh_table_met_table
from ...constants import T_BIRTH_MIN


SEED = 43
FSPS_LG_AGES = np.arange(5.5, 10.2, 0.05)  # log10 ages in years


def test_calc_ssp_weights_lognormal_mdf():
    ran_key = jran.PRNGKey(SEED)
    t_obs = 13.0
    n_t = 500
    gal_t_table = np.linspace(T_BIRTH_MIN, t_obs, n_t)

    sfr_key, met_key = jran.split(ran_key, 2)
    gal_sfr_table = jran.uniform(sfr_key, minval=0, maxval=10, shape=(n_t,))

    n_ages = FSPS_LG_AGES.size
    ssp_lg_age_gyr = FSPS_LG_AGES - 9.0
    n_met = 15
    ssp_lgmet = np.linspace(-4, 0.5, n_met)

    lgmet = jran.uniform(
        met_key, minval=ssp_lgmet.min(), maxval=ssp_lgmet.max(), shape=()
    )
    lgmet_scatter = 0.1

    args = (
        gal_t_table,
        gal_sfr_table,
        lgmet,
        lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age_gyr,
        t_obs,
    )
    weight_info = calc_ssp_weights_sfh_table_lognormal_mdf(*args)
    weights, lgmet_weights, age_weights = weight_info
    assert weights.shape == (n_met, n_ages)
    assert lgmet_weights.shape == (n_met,)
    assert age_weights.shape == (n_ages,)

    assert np.allclose(age_weights.sum(), 1.0, rtol=1e-4)
    assert np.allclose(lgmet_weights.sum(), 1.0, rtol=1e-4)
    assert np.allclose(weights.sum(), 1.0, rtol=1e-4)

    # Test namedtuple fields
    assert weight_info.weights.shape == (n_met, n_ages)
    assert weight_info.lgmet_weights.shape == (n_met,)
    assert weight_info.age_weights.shape == (n_ages,)


def test_calc_ssp_weights_met_table():
    ran_key = jran.PRNGKey(SEED)
    t_obs = 13.0
    n_t = 500
    gal_t_table = np.linspace(T_BIRTH_MIN, t_obs, n_t)

    sfr_key, met_key = jran.split(ran_key, 2)
    gal_sfr_table = jran.uniform(sfr_key, minval=0, maxval=10, shape=(n_t,))

    n_ages = FSPS_LG_AGES.size
    ssp_lg_age_gyr = FSPS_LG_AGES - 9.0
    n_met = 15
    ssp_lgmet = np.linspace(-4, 0.5, n_met)

    gal_lgmet_table = jran.uniform(
        met_key, minval=ssp_lgmet.min(), maxval=ssp_lgmet.max(), shape=(n_t,)
    )
    gal_lgmet_scatter = 0.1

    args = (
        gal_t_table,
        gal_sfr_table,
        gal_lgmet_table,
        gal_lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age_gyr,
        t_obs,
    )
    weight_info = calc_ssp_weights_sfh_table_met_table(*args)
    weights, lgmet_weights, age_weights = weight_info
    assert weights.shape == (n_met, n_ages)
    assert lgmet_weights.shape == (n_met, n_ages)
    assert age_weights.shape == (n_ages,)

    assert np.allclose(age_weights.sum(), 1.0, rtol=1e-4)
    assert np.allclose(lgmet_weights.sum(), 1.0, rtol=1e-4)
    assert np.allclose(weights.sum(), 1.0, rtol=1e-4)

    # Test namedtuple fields
    assert weight_info.weights.shape == (n_met, n_ages)
    assert weight_info.lgmet_weights.shape == (n_met, n_ages)
    assert weight_info.age_weights.shape == (n_ages,)
