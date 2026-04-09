""" """

import numpy as np

from ..retrieve_fake_fsps_data import (
    load_fake_filter_transmission_curves,
    load_fake_ssp_data,
)


def test_load_fake_ssp_data():
    ssp_data = load_fake_ssp_data()
    (n_met, n_age, n_wave) = ssp_data.ssp_flux.shape
    (_, _, n_line) = ssp_data.ssp_emline_luminosity.shape

    assert ssp_data.ssp_lgmet.shape == (n_met,)
    assert ssp_data.ssp_lg_age_gyr.shape == (n_age,)
    assert ssp_data.ssp_wave.shape == (n_wave,)
    assert ssp_data.ssp_emline_luminosity.shape == (n_met, n_age, n_line)

    assert np.all(np.isfinite(ssp_data.ssp_lgmet))
    assert np.all(np.isfinite(ssp_data.ssp_lg_age_gyr))
    assert np.all(np.isfinite(ssp_data.ssp_wave))
    assert np.all(np.isfinite(ssp_data.ssp_flux))
    assert isinstance(ssp_data.ssp_emline_wave.XXX, float)
    assert np.isfinite(ssp_data.ssp_emline_wave.XXX)
    assert np.all(np.isfinite(ssp_data.ssp_emline_luminosity))

    assert np.all(np.array(ssp_data.ssp_emline_luminosity) > 0)
    assert np.all(np.array(ssp_data.ssp_emline_luminosity) < 1e40)
    assert np.any(np.array(ssp_data.ssp_emline_luminosity) > 1e20)


def test_load_fake_ssp_data_nolines():
    ssp_data = load_fake_ssp_data(n_line=0)
    assert ssp_data.ssp_emline_wave is None
    assert ssp_data.ssp_emline_luminosity is None

    ssp_data = load_fake_ssp_data(n_line=1)
    assert ssp_data.ssp_emline_wave is not None
    assert ssp_data.ssp_emline_luminosity is not None


def test_load_fake_filter_transmission_curves():
    _res = load_fake_filter_transmission_curves()

    wave = _res[0]
    assert np.all(np.isfinite(wave))
    assert np.all(np.diff(wave) > 0)

    trans_curves = _res[1:]
    for tcurve in trans_curves:
        assert np.all(np.isfinite(tcurve))
        assert np.all(tcurve >= 0)
        assert np.all(tcurve <= 1)
        assert np.any(tcurve > 0)
