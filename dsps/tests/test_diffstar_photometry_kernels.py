"""
"""
import numpy as np
from ..diffstar_photometry_kernels import (
    _calc_weighted_rest_mag_from_diffstar_params_const_zmet,
    _calc_weighted_rest_mag_from_diffstar_params_const_zmet_dust,
)
from ..sfh_model import DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS
from ..mzr import DEFAULT_MZR_PARAMS
from .retrieve_fake_fsps_data import load_fake_sps_data


def test_calc_weighted_rest_mag_from_diffstar_params_const_zmet():

    res = load_fake_sps_data()
    filter_waves, filter_trans, wave_ssp, spec_ssp, lgZsun_bin_mids, log_age_gyr = res
    t_obs = 11.0

    mah_params = np.array(list(DEFAULT_MAH_PARAMS.values()))
    ms_params = np.array(list(DEFAULT_MS_PARAMS.values()))
    q_params = np.array(list(DEFAULT_Q_PARAMS.values()))
    met_params = np.array(list(DEFAULT_MZR_PARAMS.values()))
    lgmet = -1.0
    lgmet_scatter = met_params[-1]

    args = (
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        wave_ssp,
        spec_ssp,
        filter_waves[0, :],
        filter_trans[0, :],
        *mah_params,
        *ms_params,
        *q_params,
        lgmet,
        lgmet_scatter,
    )
    mag = _calc_weighted_rest_mag_from_diffstar_params_const_zmet(*args)
    assert 0 < mag < 10


def test_calc_weighted_rest_mag_from_diffstar_params_const_zmet_dust():

    res = load_fake_sps_data()
    filter_waves, filter_trans, wave_ssp, spec_ssp, lgZsun_bin_mids, log_age_gyr = res
    t_obs = 11.0

    mah_params = np.array(list(DEFAULT_MAH_PARAMS.values()))
    ms_params = np.array(list(DEFAULT_MS_PARAMS.values()))
    q_params = np.array(list(DEFAULT_Q_PARAMS.values()))
    met_params = np.array(list(DEFAULT_MZR_PARAMS.values()))
    lgmet = -1.0
    lgmet_scatter = met_params[-1]

    args = (
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        wave_ssp,
        spec_ssp,
        filter_waves[0, :],
        filter_trans[0, :],
        *mah_params,
        *ms_params,
        *q_params,
        lgmet,
        lgmet_scatter,
    )
    mag_nodust = _calc_weighted_rest_mag_from_diffstar_params_const_zmet(*args)

    dust_x0, dust_gamma, dust_ampl, dust_slope, dust_Av = 0.25, 0.1, 0.5, 1.5, 1.0
    dust_params = dust_x0, dust_gamma, dust_ampl, dust_slope, dust_Av
    args2 = (*args, *dust_params)
    mag_dust = _calc_weighted_rest_mag_from_diffstar_params_const_zmet_dust(*args2)
    assert mag_nodust < mag_dust
