"""
"""
import numpy as np
from ..diffstar_ew_kernels import _calc_ew_from_diffstar_params_const_lgu_lgmet
from ..sfh_model import DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS
from ..mzr import DEFAULT_MZR_PARAMS
from .retrieve_fake_fsps_data import load_fake_sps_data

OIIa, OIIb = 4996.0, 5000.0


def test_calc_weighted_rest_mag_from_diffstar_params_const_zmet():

    res = load_fake_sps_data()
    filter_waves, filter_trans, wave_ssp, _spec_ssp, lgZsun_bin_mids, log_age_gyr = res
    t_obs = 11.0

    lgU_bin_mids = np.array((-3.5, -2.5, -1.5))
    spec_ssp = np.array([_spec_ssp for __ in range(lgU_bin_mids.size)])

    mah_params = np.array(list(DEFAULT_MAH_PARAMS.values()))
    ms_params = np.array(list(DEFAULT_MS_PARAMS.values()))
    q_params = np.array(list(DEFAULT_Q_PARAMS.values()))
    met_params = np.array(list(DEFAULT_MZR_PARAMS.values()))
    lgmet = -1.0
    lgmet_scatter = met_params[-1]
    lgu = -2.0
    lgu_scatter = 0.2

    ewband1_lo, ewband1_hi = OIIa - 5, OIIa - 20
    ewband2_lo, ewband2_hi = OIIb + 5, OIIb + 20
    args = (
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        lgU_bin_mids,
        wave_ssp,
        spec_ssp,
        *mah_params,
        *ms_params,
        *q_params,
        lgmet,
        lgmet_scatter,
        lgu,
        lgu_scatter,
        ewband1_lo,
        ewband1_hi,
        ewband2_lo,
        ewband2_hi,
    )
    ew = _calc_ew_from_diffstar_params_const_lgu_lgmet(*args)
