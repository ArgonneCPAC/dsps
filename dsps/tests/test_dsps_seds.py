"""
"""
from collections import namedtuple
import numpy as np
from astropy.cosmology import Planck15
from ..seds_from_tables import compute_sed_galpop
from ..load_fsps_data import TASSO_DRN, load_fsps_testing_data
from ..seds_from_diffstar import compute_diffstarpop_restframe_seds
from ..restmag_from_diffstar import compute_diffstarpop_restframe_mags


_testing_data = namedtuple(
    "testing_data",
    [
        "filter_data",
        "lgZsun_bin_mids",
        "log_age_gyr",
        "ssp_wave",
        "ssp_flux",
        "tarr",
        "filter_waves",
        "filter_trans",
    ],
)


def _get_testing_data(
    drn=TASSO_DRN, n_ssp_wave=30, n_t=15, n_filter_wave=23, n_filters=3
):
    _res = load_fsps_testing_data(drn)
    filter_data, __, lgZsun_bin_mids, log_age_gyr = _res
    n_met = lgZsun_bin_mids.size
    n_age = log_age_gyr.size
    ssp_flux = np.ones(shape=(n_met, n_age, n_ssp_wave))
    ssp_wave = 10 ** np.linspace(3, 7, n_ssp_wave)
    filter_wave = np.linspace(4, 6, n_filter_wave)
    filter_waves = np.tile(filter_wave, n_filters).reshape((n_filters, n_filter_wave))
    filter_trans = np.ones_like(filter_waves)
    tarr = np.linspace(0.1, 13.8, n_t)
    return _testing_data(
        filter_data,
        lgZsun_bin_mids,
        log_age_gyr,
        ssp_wave,
        ssp_flux,
        tarr,
        filter_waves,
        filter_trans,
    )


def test_compute_sed_galpop_from_table():
    testing_data = _get_testing_data()
    lgZsun_bin_mids = testing_data.lgZsun_bin_mids
    log_age_gyr = testing_data.log_age_gyr
    ssp_flux = testing_data.ssp_flux
    tarr = testing_data.tarr
    n_t = tarr.size
    n_pop = 50
    sfh_tables = np.ones(shape=(n_pop, n_t))

    z_obs = 0.2
    t_obs = Planck15.age(z_obs).value
    lgmet_pop = np.zeros(n_pop)
    lgmet_scatter = 0.2 + np.zeros(n_pop)
    lgmet_params = np.array((lgmet_pop, lgmet_scatter)).T

    sed_args = (
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        ssp_flux,
        tarr,
        sfh_tables,
        lgmet_params,
    )
    seds, logsm_tables = compute_sed_galpop(*sed_args)
    assert seds.shape == (n_pop, ssp_flux.shape[-1])


def test_compute_sed_galpop_from_diffstar():
    testing_data = _get_testing_data()
    lgZsun_bin_mids = testing_data.lgZsun_bin_mids
    log_age_gyr = testing_data.log_age_gyr
    ssp_wave = testing_data.ssp_wave
    ssp_flux = testing_data.ssp_flux
    n_pop = 50

    diffmah_params = np.ones(shape=(n_pop, 6))
    u_ms_params = np.zeros(shape=(n_pop, 5))
    u_q_params = np.zeros(shape=(n_pop, 4))
    met_params = np.ones(shape=(n_pop, 2))

    t_obs = 13.0

    dstar_sed_args = (
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        ssp_wave,
        ssp_flux,
        diffmah_params,
        u_ms_params,
        u_q_params,
        met_params,
    )
    seds, sfh_tables, logsm_tables = compute_diffstarpop_restframe_seds(*dstar_sed_args)
    assert seds.shape == (n_pop, ssp_flux.shape[-1])

    # Test attenuation
    x0, gamma, ampl, slope1, Av = 0.0, 0.1, 0.5, -0.5, 1.0
    dust_params = x0, gamma, ampl, slope1, Av
    n_dust = len(dust_params)
    dust_params_pop = np.tile(dust_params, n_pop).reshape((n_pop, n_dust))
    seds, sfh_tables, logsm_tables = compute_diffstarpop_restframe_seds(
        *dstar_sed_args, dust_params=dust_params_pop
    )
    assert seds.shape == (n_pop, ssp_flux.shape[-1])


def test_compute_restframe_mags_galpop_from_diffstar():
    testing_data = _get_testing_data()
    lgZsun_bin_mids = testing_data.lgZsun_bin_mids
    log_age_gyr = testing_data.log_age_gyr
    ssp_wave = testing_data.ssp_wave
    ssp_flux = testing_data.ssp_flux
    filter_waves = testing_data.filter_waves
    filter_trans = testing_data.filter_trans
    n_pop = 50
    n_filters = filter_trans.shape[0]

    diffmah_params = np.ones(shape=(n_pop, 6))
    u_ms_params = np.zeros(shape=(n_pop, 5))
    u_q_params = np.zeros(shape=(n_pop, 4))
    met_params = np.ones(shape=(n_pop, 2))

    t_obs = 13.0
    rest_args = (
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        ssp_wave,
        ssp_flux,
        diffmah_params,
        u_ms_params,
        u_q_params,
        met_params,
        filter_waves,
        filter_trans,
    )

    mags_rest_subsample = compute_diffstarpop_restframe_mags(*rest_args)
    assert mags_rest_subsample.shape == (n_pop, n_filters)

    # Test attenuation
    x0, gamma, ampl, slope1, Av = 0.0, 0.1, 0.5, -0.5, 1.0
    dust_params = x0, gamma, ampl, slope1, Av
    n_dust = len(dust_params)
    dust_params_pop = np.tile(dust_params, n_pop).reshape((n_pop, n_dust))
    mags_rest_pop = compute_diffstarpop_restframe_mags(
        *rest_args, dust_params=dust_params_pop
    )
    assert mags_rest_pop.shape == (n_pop, n_filters)
