"""
"""
import os

import numpy as np
from jax.scipy.stats import norm

from .defaults import SSPData
from .retrieve_fsps_data import get_fsps_emline_info

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def load_fake_ssp_data():
    ssp_lgmet = _get_lgzlegend()
    ssp_lg_age_gyr = _get_log_age_gyr()
    ssp_wave = _get_ssp_wave()
    ssp_flux = _get_spec_ssp()
    _, ssp_emline_name = get_fsps_emline_info()
    n_lines = ssp_emline_name.size
    ssp_emline_wave = _get_emline_wave(n_lines)
    ssp_emline_luminosity = _get_ssp_emline_luminosity(n_lines)
    return SSPData(
        ssp_lgmet,
        ssp_lg_age_gyr,
        ssp_wave,
        ssp_flux,
        ssp_emline_name,
        ssp_emline_wave,
        ssp_emline_luminosity,
    )


def load_fake_filter_transmission_curves():
    wave = _get_ssp_wave()
    lgwave = np.log10(wave)
    u = _lsst_u_trans(lgwave)
    g = _lsst_g_trans(lgwave)
    r = _lsst_r_trans(lgwave)
    i = _lsst_i_trans(lgwave)
    z = _lsst_z_trans(lgwave)
    y = _lsst_y_trans(lgwave)
    return wave, u, g, r, i, z, y


def _get_log_age_gyr():
    log_age_gyr = np.arange(-3.5, 1.2, 0.05)
    return log_age_gyr


def _get_lgzlegend():
    lgzlegend = np.log10(zlegend)
    return lgzlegend


def _get_ssp_wave():
    n_wave_ssp = 1963
    ssp_wave = np.linspace(100, 20_000, n_wave_ssp)
    return ssp_wave


def _get_spec_ssp():
    drn = os.path.join(_THIS_DRNAME, "tests", "testing_data")
    ssp_wave = _get_ssp_wave()
    n_wave_ssp = ssp_wave.size
    ssp_plaw_data_c0 = np.loadtxt(os.path.join(drn, "ssp_plaw_data_c0.txt"))
    ssp_plaw_data_c1 = np.loadtxt(os.path.join(drn, "ssp_plaw_data_c1.txt"))
    n_met, n_age = ssp_plaw_data_c0.shape
    spec_ssp = np.zeros((n_met, n_age, n_wave_ssp))
    for iz in range(n_met):
        for iage in range(n_age):
            c0 = ssp_plaw_data_c0[iz, iage]
            c1 = ssp_plaw_data_c1[iz, iage]
            spec_ssp[iz, iage, :] = 10 ** (c0 + c1 * np.log10(ssp_wave))
    return spec_ssp


def _get_emline_wave(n_lines):
    ssp_emline_wave = np.logspace(3, 6, n_lines)
    return ssp_emline_wave


def _get_ssp_emline_luminosity(n_lines):
    drn = os.path.join(_THIS_DRNAME, "tests", "testing_data")
    ssp_plaw_data_c0 = np.loadtxt(os.path.join(drn, "ssp_plaw_data_c0.txt"))
    n_met, n_age = ssp_plaw_data_c0.shape

    ssp_emline_luminosity = np.zeros((n_met, n_age, n_lines))
    for iz in range(n_met):
        for iage in range(n_age):
            n_low = int(0.2 * n_lines)
            n_high = n_lines - n_low

            low = 10 ** np.random.uniform(-70, -2, size=n_low)
            high = 10 ** np.random.uniform(-2, 3, size=n_high)

            ssp_emline_luminosity[iz][iage] = np.concatenate([low, high])
            np.random.shuffle(ssp_emline_luminosity[iz][iage])

    return ssp_emline_luminosity


def _lsst_u_trans(x):
    return norm.pdf(x, loc=3.57, scale=0.022) / 80


def _lsst_g_trans(x):
    return norm.pdf(x, loc=3.68, scale=0.04) / 20


def _lsst_r_trans(x):
    return norm.pdf(x, loc=3.79, scale=0.03) / 25


def _lsst_i_trans(x):
    return norm.pdf(x, loc=3.875, scale=0.025) / 30


def _lsst_z_trans(x):
    return norm.pdf(x, loc=3.935, scale=0.017) / 47


def _lsst_y_trans(x):
    return norm.pdf(x, loc=3.985, scale=0.017) / 85


def _get_filter_waves():
    n_bands, n_filter_waves = 6, 1906
    wave_mins = np.array((3200.0, 3200.0, 3200.0, 4084.0, 4084.0, 4085.0))
    wave_maxs = np.array((9084.0, 9085.0, 9086.0, 10987.0, 10988.0, 10989.0))
    filter_waves = np.zeros((n_bands, n_filter_waves))
    for iband in range(n_bands):
        xmin, xmax = wave_mins[iband], wave_maxs[iband]
        filter_waves[iband, :] = np.linspace(xmin, xmax, n_filter_waves)
    return filter_waves


def _get_filter_trans():
    filter_waves = _get_filter_waves()
    n_bands, n_filter_waves = filter_waves.shape
    filter_trans = np.zeros((n_bands, n_filter_waves))

    func_list = (
        _lsst_u_trans,
        _lsst_g_trans,
        _lsst_r_trans,
        _lsst_i_trans,
        _lsst_z_trans,
        _lsst_y_trans,
    )
    for iband, func in enumerate(func_list):
        wave = filter_waves[iband, :]
        filter_trans[iband, :] = func(np.log10(wave))
    return filter_trans


zlegend = np.array(
    [
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0008,
        0.001,
        0.0012,
        0.0016,
        0.002,
        0.0025,
        0.0031,
        0.0039,
        0.0049,
        0.0061,
        0.0077,
        0.0096,
        0.012,
        0.015,
        0.019,
        0.024,
        0.03,
    ]
)
