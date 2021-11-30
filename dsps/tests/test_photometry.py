"""
"""
import os
import numpy as np
from ..photometry import calc_obs_mag_history_singlegal, calc_obs_mags_galpop
from ..load_fsps_data import load_fsps_testing_data
from ..flat_wcdm import FSPS_COSMO


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DATA_DRN = os.path.join(os.path.dirname(_THIS_DRNAME), "data")


def test_calc_obs_mag_history_singlegal():
    res = load_fsps_testing_data(DATA_DRN)
    filter_data, ssp_data, lgZsun_bin_mids, log_age_gyr = res
    n_obs = 5
    z_obs_arr = np.linspace(0.1, 5, n_obs)
    obs_mags = calc_obs_mag_history_singlegal(
        ssp_data["wave"],
        ssp_data["flux"],
        filter_data["u_filter_wave"],
        filter_data["u_filter_trans"],
        z_obs_arr,
        *FSPS_COSMO
    )
    assert obs_mags.shape == (n_obs,)


def test_calc_obs_mags_galpop():
    res = load_fsps_testing_data(DATA_DRN)
    filter_data, ssp_data, lgZsun_bin_mids, log_age_gyr = res
    z_obs = 0.5
    n_gals = 10
    flux_galpop = np.tile(ssp_data["flux"], n_gals).reshape((n_gals, -1))
    assert np.allclose(flux_galpop[0, :], flux_galpop[1, :])
    obs_mags = calc_obs_mags_galpop(
        ssp_data["wave"],
        flux_galpop,
        filter_data["u_filter_wave"],
        filter_data["u_filter_trans"],
        z_obs,
        *FSPS_COSMO
    )
    assert obs_mags.shape == (n_gals,)
