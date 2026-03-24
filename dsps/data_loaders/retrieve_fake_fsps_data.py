"""
"""
import os

import numpy as np
from jax.scipy.stats import norm

from .defaults import SSPData

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def load_fake_ssp_data():
    ssp_lgmet = _get_lgzlegend()
    ssp_lg_age_gyr = _get_log_age_gyr()
    ssp_wave = _get_ssp_wave()
    ssp_flux = _get_spec_ssp()
    ssp_emline_wave = _get_emline_wave()
    ssp_emline_luminosity = _get_ssp_emline_luminosity()
    return SSPData(
        ssp_lgmet,
        ssp_lg_age_gyr,
        ssp_wave,
        ssp_flux,
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


def _get_emline_wave():
    return ssp_emline_wave


def _get_ssp_emline_luminosity():
    drn = os.path.join(_THIS_DRNAME, "tests", "testing_data")
    ssp_plaw_data_c0 = np.loadtxt(os.path.join(drn, "ssp_plaw_data_c0.txt"))
    n_met, n_age = ssp_plaw_data_c0.shape

    emline_wave = _get_emline_wave()
    n_lines = emline_wave.size

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

ssp_emline_wave = np.array(
    [
        9.231500e02,
        9.262260e02,
        9.307480e02,
        9.378040e02,
        9.497430e02,
        9.725370e02,
        1.025720e03,
        1.084940e03,
        1.215130e03,
        1.215670e03,
        1.486500e03,
        1.548190e03,
        1.550780e03,
        1.640430e03,
        1.660810e03,
        1.666000e03,
        1.814560e03,
        1.854720e03,
        1.862790e03,
        1.906680e03,
        1.908730e03,
        2.141706e03,
        2.321699e03,
        2.324249e03,
        2.325440e03,
        2.327680e03,
        2.328870e03,
        2.424774e03,
        2.471786e03,
        2.661184e03,
        2.669986e03,
        2.796399e03,
        2.803580e03,
        2.854544e03,
        3.110131e03,
        3.343194e03,
        3.427066e03,
        3.722747e03,
        3.727118e03,
        3.730119e03,
        3.799028e03,
        3.836528e03,
        3.868157e03,
        3.869917e03,
        3.889793e03,
        3.890213e03,
        3.968654e03,
        3.971255e03,
        4.069812e03,
        4.077564e03,
        4.102951e03,
        4.341748e03,
        4.364294e03,
        4.472814e03,
        4.622936e03,
        4.687024e03,
        4.712651e03,
        4.721393e03,
        4.741519e03,
        4.862763e03,
        4.932682e03,
        4.960370e03,
        5.008314e03,
        5.193346e03,
        5.201788e03,
        5.519327e03,
        5.539493e03,
        5.578974e03,
        5.756294e03,
        5.877358e03,
        6.302138e03,
        6.313902e03,
        6.365636e03,
        6.549959e03,
        6.564723e03,
        6.585369e03,
        6.680096e03,
        6.718396e03,
        6.732781e03,
        7.067276e03,
        7.137866e03,
        7.172785e03,
        7.239875e03,
        7.265442e03,
        7.325129e03,
        7.334131e03,
        7.334281e03,
        7.753361e03,
        8.581187e03,
        8.729659e03,
        9.017481e03,
        9.071246e03,
        9.126242e03,
        9.231642e03,
        9.533378e03,
        9.548693e03,
        9.852850e03,
        1.005221e04,
        1.011182e04,
        1.012623e04,
        1.032348e04,
        1.083343e04,
        1.094116e04,
        1.257043e04,
        1.282170e04,
        1.736700e04,
        1.817924e04,
        1.875640e04,
        1.945100e04,
        1.963035e04,
        2.058723e04,
        2.166134e04,
        2.625886e04,
        2.905336e04,
        3.039224e04,
        3.207023e04,
        3.297018e04,
        3.660373e04,
        3.740586e04,
        4.052296e04,
        4.488411e04,
        4.529303e04,
        4.653808e04,
        5.128695e04,
        5.608613e04,
        5.908249e04,
        5.982020e04,
        6.985379e04,
        7.319165e04,
        7.459915e04,
        7.502537e04,
        7.645379e04,
        7.812617e04,
        7.901982e04,
        8.991566e04,
        9.033578e04,
        9.510365e04,
        1.051062e05,
        1.231094e05,
        1.237196e05,
        1.281378e05,
        1.310227e05,
        1.432692e05,
        1.436803e05,
        1.477125e05,
        1.555537e05,
        1.871318e05,
        1.955832e05,
        2.183628e05,
        2.421346e05,
        2.589064e05,
        3.287145e05,
        3.348003e05,
        3.481461e05,
        3.601396e05,
        5.181530e05,
        5.734029e05,
        6.064374e05,
        6.318607e05,
        8.835771e05,
        1.218020e06,
        1.455368e06,
        1.576813e06,
        2.053030e06,
        3.703755e06,
        6.097653e06,
    ]
)
