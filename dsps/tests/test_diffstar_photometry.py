"""
"""
import numpy as np
from ..diffstar_photometry import _calc_weighted_rest_mags_history
from ..sfh_model import DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS
from ..mzr import DEFAULT_MZR_PARAMS
from .retrieve_fake_fsps_data import load_fake_sps_data


def test_calc_weighted_rest_mag_history():

    res = load_fake_sps_data()
    filter_waves, filter_trans, wave_ssp, spec_ssp, lgZsun_bin_mids, log_age_gyr = res
    n_bands = filter_trans.shape[0]
    n_tobs = 20
    tarr = np.linspace(1, 13.7, n_tobs)

    mah_params = np.array(list(DEFAULT_MAH_PARAMS.values()))
    ms_params = np.array(list(DEFAULT_MS_PARAMS.values()))
    q_params = np.array(list(DEFAULT_Q_PARAMS.values()))
    met_params = np.array(list(DEFAULT_MZR_PARAMS.values()))
    lgmet = -1.0
    lgmet_scatter = met_params[-1]

    args = (
        tarr,
        lgZsun_bin_mids,
        log_age_gyr,
        wave_ssp,
        spec_ssp,
        filter_waves,
        filter_trans,
        *mah_params,
        *ms_params,
        *q_params,
        lgmet,
        lgmet_scatter,
    )
    mags = _calc_weighted_rest_mags_history(*args)
    assert mags.shape == (n_tobs, n_bands)
