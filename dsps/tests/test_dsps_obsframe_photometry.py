"""
"""
import numpy as np
from astropy.cosmology import Planck15
from ..flat_wcdm import PLANCK15
from .test_dsps_seds import _get_testing_data
from ..obsmag_from_diffstar import compute_diffstarpop_obsframe_mags


def test_compute_obsframe_mags_galpop_from_diffstar():
    testing_data = _get_testing_data()
    lgZsun_bin_mids = testing_data.lgZsun_bin_mids
    log_age_gyr = testing_data.log_age_gyr
    ssp_wave = testing_data.ssp_wave
    ssp_flux = testing_data.ssp_flux
    filter_waves = testing_data.filter_waves
    filter_trans = testing_data.filter_trans
    n_pop = 50
    n_filters = filter_trans.shape[0]
    n_ssp_wave = ssp_wave.shape[0]
    assert n_ssp_wave > 0

    diffmah_params = np.ones(shape=(n_pop, 6))
    u_ms_params = np.zeros(shape=(n_pop, 5))
    u_q_params = np.zeros(shape=(n_pop, 4))
    met_params = np.ones(shape=(n_pop, 2))

    z_obs = 0.2
    t_obs = Planck15.age(z_obs).value
    obs_args = [
        t_obs,
        z_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        ssp_wave,
        ssp_flux,
        diffmah_params,
        u_ms_params,
        u_q_params,
        met_params,
        PLANCK15,
        filter_waves,
        filter_trans,
    ]
    mags_obs_pop = compute_diffstarpop_obsframe_mags(*obs_args)
    # mags_obs_pop, sed, sfh_table, logsm_table = _res
    assert mags_obs_pop.shape == (n_pop, n_filters)
    # assert sed.shape == (n_filters, n_pop, n_ssp_wave)

    # Test attenuation
    x0, gamma, ampl, slope1, Av = 0.0, 0.1, 0.5, -0.5, 1.0
    dust_params = x0, gamma, ampl, slope1, Av
    n_dust = len(dust_params)
    dust_params_pop = np.tile(dust_params, n_pop).reshape((n_pop, n_dust))
    mags_obs_pop = compute_diffstarpop_obsframe_mags(
        *obs_args, dust_params=dust_params_pop
    )
    # mags_obs_pop, sed, sfh_table, logsm_table = _res
    assert mags_obs_pop.shape == (n_pop, n_filters)
    # assert sed.shape == (n_filters, n_pop, n_ssp_wave)
