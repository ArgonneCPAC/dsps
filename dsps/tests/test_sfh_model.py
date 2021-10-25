"""
"""
from ..sfh_model import MS_PARAM_BOUNDS, DEFAULT_MS_PARAMS
from ..sfh_model import DEFAULT_Q_PARAMS, Q_PARAM_BOUNDS


def test_sfh_parameter_bounds():
    for key, val in DEFAULT_MS_PARAMS.items():
        assert MS_PARAM_BOUNDS[key][0] < val < MS_PARAM_BOUNDS[key][1]

    for key, val in DEFAULT_Q_PARAMS.items():
        assert Q_PARAM_BOUNDS[key][0] < val < Q_PARAM_BOUNDS[key][1]
