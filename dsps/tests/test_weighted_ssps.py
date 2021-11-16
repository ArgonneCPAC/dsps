"""
"""
import numpy as np
from ..weighted_ssps import _calc_weighted_ssp_from_diffstar_params
from ..sfh_model import DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS
from ..mzr import DEFAULT_MZR_PARAMS


def test_calc_weighted_ssp_from_diffstar_params():

    t_obs = 11.0
    n_met, n_age, n_filters = 10, 20, 6
    lg_ages = np.linspace(6, 10, n_age) - 9
    ssp_templates = np.zeros((n_met, n_age, n_filters))
    lgZsun_bin_mids = np.linspace(-2, 0, n_met)

    mah_params = np.array(list(DEFAULT_MAH_PARAMS.values()))
    ms_params = np.array(list(DEFAULT_MS_PARAMS.values()))
    q_params = np.array(list(DEFAULT_Q_PARAMS.values()))
    met_params = np.array(list(DEFAULT_MZR_PARAMS.values()))

    args = (
        t_obs,
        lgZsun_bin_mids,
        lg_ages,
        ssp_templates,
        mah_params,
        ms_params,
        q_params,
        met_params,
    )
    lgmet_weights, age_weights, mags = _calc_weighted_ssp_from_diffstar_params(*args)
    assert lgmet_weights.shape == (n_met, 1, 1)
    assert age_weights.shape == (1, n_age, 1)
    assert mags.shape == (n_filters,)
