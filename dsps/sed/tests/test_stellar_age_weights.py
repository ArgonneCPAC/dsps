"""
"""
import numpy as np
from ..stellar_age_weights import _get_lg_age_bin_edges
from ..stellar_age_weights import _calc_age_weights_from_logsm_table
from ...utils import _jax_get_dt_array
from ...constants import T_BIRTH_MIN


FSPS_LG_AGES = np.arange(5.5, 10.2, 0.05)  # log10 ages in years


def linear_sfr(t_gyr):
    return t_gyr * 1e9


def linear_smh(t0, t_gyr):
    return 1e9 * 0.5 * (t_gyr**2 - t0**2)


def test_age_bin_edges_have_correct_array_shape():
    lg_ages_gyr = FSPS_LG_AGES - 9
    lgt_age_bins = _get_lg_age_bin_edges(lg_ages_gyr)
    assert lgt_age_bins.size == lg_ages_gyr.size + 1


def test_age_weights_are_mathematically_sensible():
    t_obs = 11.0

    gal_t_table = np.linspace(0.05, 13.8, 75)
    gal_lgt_table = np.log10(gal_t_table)
    logsm_table = np.linspace(-1, 10, gal_t_table.size)

    ssp_lg_age_gyr = FSPS_LG_AGES - 9.0
    lgt_birth_bin_mids, age_weights = _calc_age_weights_from_logsm_table(
        gal_lgt_table, logsm_table, ssp_lg_age_gyr, t_obs
    )
    assert age_weights.shape == lgt_birth_bin_mids.shape
    assert age_weights.shape == ssp_lg_age_gyr.shape
    assert np.allclose(age_weights.sum(), 1.0)


def test_age_weights_agree_with_analytical_calculation_of_constant_sfr_weights():
    constant_sfr = 1.0 * 1e9  # Msun/Gyr

    # Analytically calculate age distributions for constant SFR (independent of t_obs)
    ssp_lg_age_gyr = FSPS_LG_AGES - 9
    dt_ages = _jax_get_dt_array(10**ssp_lg_age_gyr)
    mstar_age_bins = dt_ages * constant_sfr
    correct_weights = mstar_age_bins / mstar_age_bins.sum()

    # Calculate age distributions with DSPS
    t_obs = 16.0
    gal_t_table = np.linspace(T_BIRTH_MIN, t_obs, 50_000)
    gal_lgt_table = np.log10(gal_t_table)
    mstar_table = constant_sfr * gal_t_table
    logsm_table = np.log10(mstar_table)

    dsps_age_weights = _calc_age_weights_from_logsm_table(
        gal_lgt_table, logsm_table, ssp_lg_age_gyr, t_obs
    )[1]
    assert np.allclose(dsps_age_weights, correct_weights, atol=0.01)


def test_age_weights_agree_with_analytical_calculation_of_linear_sfr_weights():
    t_obs = 16.0

    # Analytically calculate age distributions for SFR(t) = t
    ssp_lg_age_gyr = FSPS_LG_AGES - 9
    lgt_age_bin_edges = _get_lg_age_bin_edges(ssp_lg_age_gyr)
    t_age_bin_edges_gyr = 10**lgt_age_bin_edges
    t_births_bin_edges = t_obs - t_age_bin_edges_gyr
    mstar_at_age_bins = linear_smh(T_BIRTH_MIN, t_births_bin_edges)
    dmstar_ages = -np.diff(mstar_at_age_bins)
    correct_weights = dmstar_ages / dmstar_ages.sum()

    # Calculate age distributions with DSPS
    gal_t_table = np.linspace(T_BIRTH_MIN, t_obs, 50_000)
    gal_lgt_table = np.log10(gal_t_table)

    logsm_table = np.log10(linear_smh(T_BIRTH_MIN, gal_t_table[1:]))
    dsps_age_weights = _calc_age_weights_from_logsm_table(
        gal_lgt_table[1:], logsm_table, ssp_lg_age_gyr, t_obs
    )[1]
    assert np.allclose(dsps_age_weights, correct_weights, atol=0.001)
