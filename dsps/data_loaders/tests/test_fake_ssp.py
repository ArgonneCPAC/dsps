"""
"""
import numpy as np
from ..fake_ssp import _get_fake_ssp_spectrum


def test_get_fake_ssp_spectrum():
    n_met, n_age, n_wave = 15, 105, 500
    lgmet_arr = np.linspace(-4, 0, n_met)
    log_age_yr_arr = np.linspace(5.5, 10.35, n_age)
    wave = np.linspace(100, 10_000, n_wave)
    for lgmet in lgmet_arr:
        for log_age_yr in log_age_yr_arr:
            ssp_sed = _get_fake_ssp_spectrum(lgmet, log_age_yr, wave)
            assert ssp_sed.shape == (n_wave,)
            assert np.all(np.isfinite(ssp_sed))
