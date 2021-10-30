"""
"""
import numpy as np
from ..sfh_model import DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS
from ..mzr import DEFAULT_MZR_PARAMS
from ..zh_model import _calc_lgmet_history


def test_zh_model():
    mah_params = np.array(list(DEFAULT_MAH_PARAMS.values()))
    ms_params = np.array(list(DEFAULT_MS_PARAMS.values()))
    q_params = np.array(list(DEFAULT_Q_PARAMS.values()))
    met_params = np.array(list(DEFAULT_MZR_PARAMS.values()))

    tarr = np.linspace(0.1, 13.8, 100)
    lgzh = _calc_lgmet_history(tarr, mah_params, ms_params, q_params, met_params)
    assert np.all(np.isfinite(lgzh))
    assert lgzh.shape == tarr.shape
