"""
"""
import pytest
import os
from collections import namedtuple
import numpy as np
from ..seds_from_tables import compute_sed_galpop
from ...load_fsps_data import TASSO_DRN, load_fsps_testing_data


if os.path.isdir(TASSO_DRN):
    HAS_FSPS_TEST_DATA = True
else:
    HAS_FSPS_TEST_DATA = False


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


@pytest.mark.skipif("not HAS_FSPS_TEST_DATA")
def test_compute_sed_galpop_from_table():
    testing_data = _get_testing_data()
    lgZsun_bin_mids = testing_data.lgZsun_bin_mids
    log_age_gyr = testing_data.log_age_gyr
    ssp_flux = testing_data.ssp_flux
    tarr = testing_data.tarr
    n_t = tarr.size
    n_pop = 50
    sfh_tables = np.ones(shape=(n_pop, n_t))

    t_obs = 11.3
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
    assert np.all(np.isfinite(logsm_tables))
    assert seds.shape == (n_pop, ssp_flux.shape[-1])
    assert np.all(np.isfinite(seds))
    assert np.all(seds >= 0)
