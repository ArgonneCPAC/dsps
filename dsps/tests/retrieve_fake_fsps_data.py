"""
"""
import numpy as np
from scipy.stats import norm
import os


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def load_fake_sps_data():
    filter_waves = _get_filter_waves()
    filter_trans = _get_filter_trans()
    wave_ssp = _get_wave_ssp()
    spec_ssp = _get_spec_ssp()
    lgZsun_bin_mids = _get_lgZsun_bin_mids()
    log_age_gyr = _get_log_age_gyr()
    ret = filter_waves, filter_trans, wave_ssp, spec_ssp, lgZsun_bin_mids, log_age_gyr
    return ret


def _get_log_age_gyr():
    log_age_gyr = np.arange(-3.5, 1.2, 0.05)
    return log_age_gyr


def _get_lgZsun_bin_mids():
    lgZsun_bin_mids = np.log10(zlegend / zlegend[-3])
    return lgZsun_bin_mids


def _get_wave_ssp():
    n_wave_ssp = 1963
    x = np.arange(n_wave_ssp)
    c0, c1 = 2.358, 0.00257
    lg_wave_ssp = c0 + c1 * x
    wave_ssp = 10 ** lg_wave_ssp
    return wave_ssp


def _get_spec_ssp():
    drn = os.path.join(_THIS_DRNAME, "testing_data")
    wave_ssp = _get_wave_ssp()
    n_wave_ssp = wave_ssp.size
    ssp_plaw_data_c0 = np.loadtxt(os.path.join(drn, "ssp_plaw_data_c0.txt"))
    ssp_plaw_data_c1 = np.loadtxt(os.path.join(drn, "ssp_plaw_data_c1.txt"))
    n_met, n_age = ssp_plaw_data_c0.shape
    spec_ssp = np.zeros((n_met, n_age, n_wave_ssp))
    for iz in range(n_met):
        for iage in range(n_age):
            c0 = ssp_plaw_data_c0[iz, iage]
            c1 = ssp_plaw_data_c1[iz, iage]
            spec_ssp[iz, iage, :] = 10 ** (c0 + c1 * np.log10(wave_ssp))
    return spec_ssp


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
