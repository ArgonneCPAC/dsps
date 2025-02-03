"""
"""

import numpy as np
from jax import random as jran

from .. import mzr, umzr


def test_param_u_param_names_propagate_properly():

    gen = zip(umzr.DEFAULT_MZR_U_PARAMS._fields, umzr.DEFAULT_MZR_PARAMS._fields)
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = umzr.get_bounded_mzr_params(umzr.DEFAULT_MZR_U_PARAMS)
    assert set(inferred_default_params._fields) == set(umzr.DEFAULT_MZR_PARAMS._fields)

    inferred_default_u_params = umzr.get_unbounded_mzr_params(umzr.DEFAULT_MZR_PARAMS)
    assert set(inferred_default_u_params._fields) == set(
        umzr.DEFAULT_MZR_U_PARAMS._fields
    )


def test_get_bounded_mzr_params_fails_when_passing_params():
    try:
        umzr.get_bounded_mzr_params(umzr.DEFAULT_MZR_PARAMS)
        raise NameError("get_bounded_mzr_params should not accept u_params")
    except AttributeError:
        pass


def test_get_unbounded_deltapop_params_fails_when_passing_u_params():
    try:
        umzr.get_unbounded_mzr_params(umzr.DEFAULT_MZR_U_PARAMS)
        raise NameError("get_unbounded_mzr_params should not accept u_params")
    except AttributeError:
        pass


def test_param_inversion():
    ran_key = jran.PRNGKey(0)

    ntests = 100
    for __ in range(ntests):
        ran_key, test_key = jran.split(ran_key, 2)
        uran = jran.uniform(
            test_key, minval=-100, maxval=100, shape=(len(umzr.DEFAULT_MZR_U_PARAMS),)
        )
        params = umzr.get_bounded_mzr_params(umzr.MZRUParams(*uran))
        u_params = umzr.get_unbounded_mzr_params(umzr.MZRParams(*params))
        assert np.allclose(uran, u_params, rtol=0.01)
        assert np.all(np.isfinite(u_params))
        assert np.all(np.isfinite(params))


def test_monotonic_mzr():
    ran_key = jran.PRNGKey(0)

    logsm = np.linspace(1, 13, 500)

    ntests = 1000
    for __ in range(ntests):
        ran_key, u_key, time_key = jran.split(ran_key, 3)
        uran = jran.uniform(
            u_key, minval=-100, maxval=100, shape=(len(umzr.DEFAULT_MZR_U_PARAMS),)
        )
        params = umzr.get_bounded_mzr_params(umzr.MZRUParams(*uran))

        t = jran.uniform(time_key, minval=0.1, maxval=14, shape=())

        lgmet = umzr.mzr_model(logsm, t, *params)
        assert np.all(np.isfinite(lgmet))
        assert np.all(lgmet < 2)
        assert np.all(np.diff(lgmet) >= -0.01)

        assert params.mzr_t0_slope_lo >= params.mzr_t0_slope_hi


def test_default_mzr_umzr_agree():
    ran_key = jran.PRNGKey(0)

    logsm = np.linspace(1, 13, 500)

    ntests = 100
    for __ in range(ntests):
        ran_key, time_key = jran.split(ran_key, 2)
        tobs = jran.uniform(time_key, minval=0.1, maxval=14, shape=())
        lgmet_old = mzr.mzr_model(logsm, tobs, *mzr.DEFAULT_MET_PARAMS[:-1])
        lgmet_new = umzr.mzr_model(logsm, tobs, *umzr.DEFAULT_MZR_PARAMS)

        assert np.allclose(lgmet_old, lgmet_new, rtol=1e-4)


def test_default_umzr_params():
    gen = zip(umzr.DEFAULT_MZR_PARAMS, umzr.DEFAULT_MZR_PARAMS._fields)
    for param, key in gen:
        assert np.all(np.isfinite(param)), f"Parameter `{key}` is NaN"

    gen = zip(umzr.DEFAULT_MZR_U_PARAMS, umzr.DEFAULT_MZR_U_PARAMS._fields)
    for u_param, key in gen:
        assert np.all(np.isfinite(u_param)), f"Parameter `{key}` is NaN"
