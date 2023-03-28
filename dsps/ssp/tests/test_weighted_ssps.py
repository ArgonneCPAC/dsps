"""
"""
import numpy as np
import pytest
from ...metallicity.mzr import DEFAULT_MZR_PARAMS
from ..weighted_ssps import _calc_weighted_ssp_from_sfh_table
from ..weighted_ssps import _calc_weighted_ssp_from_sfh_table_const_zmet
from ..weighted_ssps import _calc_weighted_flux_from_sfh_table_age_correlated_zmet


def test_calc_weighted_ssp_from_sfh_table():

    t_obs = 11.0
    n_met, n_age, n_filters = 10, 20, 6
    lg_ages = np.linspace(6, 10, n_age) - 9
    ssp_templates = np.zeros((n_met, n_age, n_filters))
    lgZsun_bin_mids = np.linspace(-2, 0, n_met)

    met_params = np.array(list(DEFAULT_MZR_PARAMS.values()))
    t_table = np.linspace(0.1, 13.8, 50)
    lgt_table = np.log10(t_table)
    logsm_table = np.linspace(-1, 10, t_table.size)
    args = (
        t_obs,
        lgZsun_bin_mids,
        lg_ages,
        ssp_templates,
        lgt_table,
        logsm_table,
        met_params,
    )
    lgmet_weights, age_weights, mags = _calc_weighted_ssp_from_sfh_table(*args)
    assert lgmet_weights.shape == (n_met, 1, 1)
    assert age_weights.shape == (1, n_age, 1)
    assert mags.shape == (n_filters,)


def test_calc_weighted_ssp_from_sfh_table_const_zmet():

    t_obs = 11.0
    n_met, n_age, n_filters = 10, 20, 6
    lg_ages = np.linspace(6, 10, n_age) - 9
    ssp_templates = np.zeros((n_met, n_age, n_filters))
    lgZsun_bin_mids = np.linspace(-2, 0, n_met)

    t_table = np.linspace(0.1, 13.8, 50)
    lgt_table = np.log10(t_table)
    logsm_table = np.linspace(-1, 10, t_table.size)
    met_params = np.array(list(DEFAULT_MZR_PARAMS.values()))
    lgmet = -1.0
    lgmet_scatter = met_params[-1]

    args = (
        t_obs,
        lgZsun_bin_mids,
        lg_ages,
        ssp_templates,
        lgt_table,
        logsm_table,
        lgmet,
        lgmet_scatter,
    )
    res = _calc_weighted_ssp_from_sfh_table_const_zmet(*args)
    lgmet_weights, age_weights, mags = res
    assert lgmet_weights.shape == (n_met, 1, 1)
    assert age_weights.shape == (1, n_age, 1)
    assert mags.shape == (n_filters,)


def test_calc_weighted_flux_from_sfh_table_age_correlated_zmet():

    t_obs = 11.0
    n_met, n_age = 10, 20
    lg_ages = np.linspace(6, 10, n_age) - 9
    ssp_templates = np.zeros((n_met, n_age))
    lgZsun_bin_mids = np.linspace(-2, 0, n_met)

    t_table = np.linspace(0.1, 13.8, 50)
    lgt_table = np.log10(t_table)
    logsm_table = np.linspace(-1, 10, t_table.size)
    met_params = np.array(list(DEFAULT_MZR_PARAMS.values()))
    lgmet_young, lgmet_old = 0.0, -2.0
    lgmet_scatter = met_params[-1]

    args = (
        t_obs,
        lgZsun_bin_mids,
        lg_ages,
        ssp_templates,
        lgt_table,
        logsm_table,
        lgmet_young,
        lgmet_old,
        lgmet_scatter,
    )
    res = _calc_weighted_flux_from_sfh_table_age_correlated_zmet(*args)
    lgmet_weights, age_weights, flux = res
    assert lgmet_weights.shape == (n_met, n_age)
    assert age_weights.shape == (n_age,)
    assert flux.shape == ()


@pytest.mark.xfail
def test_tiny_scatter_edge_case():
    """Test ensures metallicity weights are correct even when scatter is very small"""
    raise NotImplementedError()
