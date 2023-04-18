"""
"""
import numpy as np
from ..retrieve_fake_fsps_data import load_fake_sps_data


def test_load_fake_sps_data():
    ssp_data = load_fake_sps_data()
    (n_met, n_age, n_wave) = ssp_data.ssp_flux.shape

    assert ssp_data.ssp_lgmet.shape == (n_met,)
    assert ssp_data.ssp_lg_age.shape == (n_age,)
    assert ssp_data.ssp_wave.shape == (n_wave,)

    assert np.all(np.isfinite(ssp_data.ssp_lgmet))
    assert np.all(np.isfinite(ssp_data.ssp_lg_age))
    assert np.all(np.isfinite(ssp_data.ssp_wave))
    assert np.all(np.isfinite(ssp_data.ssp_flux))
