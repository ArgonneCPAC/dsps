"""
"""
import numpy as np
from ..ew_kernels import _calc_ew_from_sfh_table_const_lgu_lgmet
from ..ew_kernels import _calc_ew_from_sfh_table_const_lgmet
from ...metallicity.mzr import DEFAULT_MZR_PARAMS
from ...tests.retrieve_fake_fsps_data import load_fake_sps_data
from ...ssp.stellar_ages import _get_linspace_time_tables

OIIa, OIIb = 4996.0, 5000.0


def test_calc_ew_from_sfh_table_const_lgu_lgmet():

    res = load_fake_sps_data()
    filter_waves, filter_trans, wave_ssp, _spec_ssp, lgZsun_bin_mids, log_age_gyr = res
    t_obs = 11.0

    lgU_bin_mids = np.array((-3.5, -2.5, -1.5))
    spec_ssp = np.array([_spec_ssp for __ in range(lgU_bin_mids.size)])

    met_params = np.array(list(DEFAULT_MZR_PARAMS.values()))
    lgmet = -1.0
    lgmet_scatter = met_params[-1]
    lgu = -2.0
    lgu_scatter = 0.2

    line_mid = OIIb
    line_lo = line_mid - 15
    line_hi = line_mid + 15

    cont_lo_lo = line_mid - 100
    cont_lo_hi = line_mid - 50
    cont_hi_lo = line_mid + 50
    cont_hi_hi = line_mid + 100

    t_table, lgt_table, dt_table = _get_linspace_time_tables()
    logsm_table = np.linspace(-1, 10, lgt_table.size)

    args = (
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        lgU_bin_mids,
        wave_ssp,
        spec_ssp,
        lgt_table,
        logsm_table,
        lgmet,
        lgmet_scatter,
        lgu,
        lgu_scatter,
        line_lo,
        line_mid,
        line_hi,
        cont_lo_lo,
        cont_lo_hi,
        cont_hi_lo,
        cont_hi_hi,
    )
    ew, total_line_flux = _calc_ew_from_sfh_table_const_lgu_lgmet(*args)


def test_calc_ew_from_sfh_table_const_lgmet():

    res = load_fake_sps_data()
    filter_waves, filter_trans, wave_ssp, spec_ssp, lgZsun_bin_mids, log_age_gyr = res
    t_obs = 11.0

    met_params = np.array(list(DEFAULT_MZR_PARAMS.values()))
    lgmet = -1.0
    lgmet_scatter = met_params[-1]

    line_mid = OIIb
    line_lo = line_mid - 15
    line_hi = line_mid + 15

    cont_lo_lo = line_mid - 100
    cont_lo_hi = line_mid - 50
    cont_hi_lo = line_mid + 50
    cont_hi_hi = line_mid + 100

    t_table, lgt_table, dt_table = _get_linspace_time_tables()
    logsm_table = np.linspace(-1, 10, lgt_table.size)

    args = (
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        wave_ssp,
        spec_ssp,
        lgt_table,
        logsm_table,
        lgmet,
        lgmet_scatter,
        line_lo,
        line_mid,
        line_hi,
        cont_lo_lo,
        cont_lo_hi,
        cont_hi_lo,
        cont_hi_hi,
    )
    ew, total_line_flux = _calc_ew_from_sfh_table_const_lgmet(*args)
