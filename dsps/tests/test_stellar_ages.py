"""
"""
import numpy as np
from ..stellar_ages import _get_lg_age_bin_edges, _get_lgt_birth, T_BIRTH_MIN
from ..stellar_ages import _get_sfh_tables, _get_age_weights_from_tables
from ..sfh_model import DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS
from ..utils import _jax_get_dt_array


FSPS_LG_AGES = np.arange(5.5, 10.2, 0.05)  # log10 ages in years


def linear_sfr(t_gyr):
    return t_gyr * 1e9


def linear_smh(t0, t_gyr):
    return 1e9 * 0.5 * (t_gyr ** 2 - t0 ** 2)


def test_age_bin_edges_have_correct_array_shape():
    lgt_ages = np.linspace(5.5, 10.5, 50)
    lgt_age_bins = _get_lg_age_bin_edges(lgt_ages)
    assert lgt_age_bins.size == lgt_ages.size + 1


def test_age_weights_are_mathematically_sensible():
    t_obs = 11.0
    mah_params = np.array(list(DEFAULT_MAH_PARAMS.values()))
    ms_params = np.array(list(DEFAULT_MS_PARAMS.values()))
    q_params = np.array(list(DEFAULT_Q_PARAMS.values()))
    res = _get_sfh_tables(mah_params, ms_params, q_params)
    t_table, lgt_table, dt_table, sfh_table, logsm_table = res

    lgt_ages = np.linspace(5.5, 10.5, 50) - 9.0
    lgt_age_bin_edges = _get_lg_age_bin_edges(lgt_ages)
    lgt_birth_bin_edges = _get_lgt_birth(t_obs, lgt_age_bin_edges)
    age_weights = _get_age_weights_from_tables(
        lgt_birth_bin_edges, lgt_table, logsm_table
    )
    assert age_weights.shape == lgt_ages.shape
    assert np.allclose(age_weights.sum(), 1.0)


def test_age_weights_agree_with_analytical_calculation_of_constant_sfr_weights():
    constant_sfr = 1.0 * 1e9  # Msun/Gyr

    # Analytically calculate age distributions for constant SFR (independent of t_obs)
    log_ages_gyr = FSPS_LG_AGES - 9
    ages_gyr = 10 ** log_ages_gyr
    dt_ages = _jax_get_dt_array(ages_gyr)
    mstar_age_bins = dt_ages * constant_sfr
    correct_weights = mstar_age_bins / mstar_age_bins.sum()

    # Calculate age distributions with DSPS
    t_obs = 16.0
    t_table = np.linspace(T_BIRTH_MIN, t_obs, 50_000)
    lgt_table = np.log10(t_table)
    mstar_table = constant_sfr * t_table
    logsm_table = np.log10(mstar_table)

    lgt_age_bin_edges = _get_lg_age_bin_edges(log_ages_gyr)
    lgt_birth_bin_edges = _get_lgt_birth(t_obs, lgt_age_bin_edges)

    dsps_age_weights = _get_age_weights_from_tables(
        lgt_birth_bin_edges, lgt_table, logsm_table
    )
    assert np.allclose(dsps_age_weights, correct_weights, atol=0.01)


def test_age_weights_agree_with_analytical_calculation_of_linear_sfr_weights():

    t_obs = 16.0

    # Analytically calculate age distributions for SFR(t) = t
    log_ages_gyr = FSPS_LG_AGES - 9
    lgt_age_bin_edges = _get_lg_age_bin_edges(log_ages_gyr)
    t_age_bin_edges_gyr = 10 ** lgt_age_bin_edges
    t_births_bin_edges = t_obs - t_age_bin_edges_gyr
    mstar_at_age_bins = linear_smh(T_BIRTH_MIN, t_births_bin_edges)
    dmstar_ages = -np.diff(mstar_at_age_bins)
    correct_weights = dmstar_ages / dmstar_ages.sum()

    # Calculate age distributions with DSPS
    t_table = np.linspace(T_BIRTH_MIN, t_obs, 50_000)
    lgt_table = np.log10(t_table)

    logsm_table = np.log10(linear_smh(T_BIRTH_MIN, t_table[1:]))
    lgt_birth_bin_edges = _get_lgt_birth(t_obs, lgt_age_bin_edges)
    dsps_age_weights = _get_age_weights_from_tables(
        lgt_birth_bin_edges, lgt_table[1:], logsm_table
    )
    assert np.allclose(dsps_age_weights, correct_weights, atol=0.001)
