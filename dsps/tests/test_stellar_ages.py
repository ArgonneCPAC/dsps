"""
"""
import numpy as np
from ..stellar_ages import _get_lg_age_bin_edges, _get_lgt_birth
from ..stellar_ages import _get_sfh_tables, _get_age_weights_from_tables
from ..sfh_model import DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS


def test_age_bin_edges():
    lgt_ages = np.linspace(5.5, 10.5, 50)
    lgt_age_bins = _get_lg_age_bin_edges(lgt_ages)
    assert lgt_age_bins.size == lgt_ages.size + 1


def test_age_weights():
    t_obs = 11.0
    mah_params = np.array(list(DEFAULT_MAH_PARAMS.values()))
    ms_params = np.array(list(DEFAULT_MS_PARAMS.values()))
    q_params = np.array(list(DEFAULT_Q_PARAMS.values()))
    res = _get_sfh_tables(mah_params, ms_params, q_params)
    t_table, lgt_table, dt_table, sfh_table, logsm_table = res

    lgt_ages = np.linspace(5.5, 10.5, 50) - 9.0
    lgt_age_bin_edges = _get_lg_age_bin_edges(lgt_ages)
    lgt_birth_bin_edges = _get_lgt_birth(t_obs, lgt_age_bin_edges)
    age_weights = _get_age_weights_from_tables(
        lgt_birth_bin_edges, lgt_table, logsm_table
    )
    assert age_weights.shape == lgt_ages.shape
    assert np.allclose(age_weights.sum(), 1.0)
