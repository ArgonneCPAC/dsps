"""
"""

import numpy as np
from jax import random as jran

from .. import diffburst as db

TOL = 1e-4


def test_params_invert():
    ran_key = jran.PRNGKey(0)
    n_tests = 10
    for __ in range(n_tests):
        ran_key, u_key = jran.split(ran_key, 2)
        u = jran.uniform(
            u_key, minval=-10, maxval=10, shape=(len(db.DEFAULT_BURST_PARAMS),)
        )
        u_p = db.BurstUParams(*u)
        p = db._get_params_from_u_params(u_p)
        u_p2 = db._get_u_params_from_params(p)
        assert np.allclose(u_p, u_p2, rtol=TOL)


def test_age_weights_is_finite_and_zero_for_edge_cases():
    lgyr_peak, lgyr_max = 6.5, 8.0

    # all times in the lgyr_table are after lgyr_max
    lgyr_test = np.linspace(lgyr_max, lgyr_max + 2, 100)
    age_weights = db._pureburst_age_weights_from_params(lgyr_test, lgyr_peak, lgyr_max)
    assert np.all(np.isfinite(age_weights))
    assert np.allclose(age_weights, 0.0)

    # all times in the lgyr_table are before lgyr_min
    dlgyr_support = lgyr_max - lgyr_peak
    lgyr_min = lgyr_peak - dlgyr_support
    lgyr_test = np.linspace(lgyr_min - 2, lgyr_min, 100)
    age_weights = db._pureburst_age_weights_from_params(lgyr_test, lgyr_peak, lgyr_max)
    assert np.all(np.isfinite(age_weights))
    assert np.allclose(age_weights, 0.0)


def test_pureburst_age_weights_from_params_scales_with_lgyr_max_as_expected():
    n_tests = 10
    lgyr_peak = 5.5
    ran_key = jran.PRNGKey(0)
    for __ in range(n_tests):
        ran_key, lgyr_max_key = jran.split(ran_key, 2)
        lgyr_max = jran.uniform(lgyr_max_key, minval=7, maxval=8, shape=())
        params = db.BurstParams(db.DEFAULT_LGFBURST, lgyr_peak, lgyr_max)

        lgyr = np.linspace(4, 10.5, 100)
        age_weights = db._pureburst_age_weights_from_params(lgyr, lgyr_peak, lgyr_max)
        zmsk = lgyr > params.lgyr_max
        assert np.all(age_weights[zmsk] == 0)
        assert np.any(age_weights > 0)

        age_weight_at_lgyr_peak = db._pureburst_age_weights_from_params(
            lgyr_peak, lgyr_peak, lgyr_max
        )
        assert age_weight_at_lgyr_peak > 0

        lgyr_post_peak = np.linspace(lgyr_peak, lgyr_peak + 1, 10)
        age_weights_post_peak = db._pureburst_age_weights_from_params(
            lgyr_post_peak, lgyr_peak, lgyr_max
        )
        assert np.all(np.diff(age_weights_post_peak) < 0)

    lgyr_peak = 5.5

    lgyr_test = np.linspace(lgyr_peak, lgyr_peak + 1, 100)
    age_weight_younger = db._pureburst_age_weights_from_params(
        lgyr_test, lgyr_peak, 6.5
    )
    age_weights_older = db._pureburst_age_weights_from_params(lgyr_test, lgyr_peak, 8.0)
    assert age_weight_younger[0] > age_weights_older[0]


def test_age_weights_from_default_params_are_weights():
    lgyr = np.arange(5.5, 10.35, 0.05)
    lgyr_peak, lgyr_max = db.DEFAULT_BURST_PARAMS[1:]
    age_weights = db._pureburst_age_weights_from_params(lgyr, lgyr_peak, lgyr_max)
    assert np.all(np.isfinite(age_weights))
    assert np.all(age_weights >= 0)
    assert np.any(age_weights > 0)
    assert np.allclose(age_weights.sum(), 1.0, rtol=TOL)


def test_age_weights_from_random_params_are_weights():
    ran_key = jran.PRNGKey(0)
    lgyr = np.arange(5.5, 10.35, 0.05)
    n_tests = 10
    for __ in range(n_tests):
        ran_key, u_key = jran.split(ran_key, 2)
        u_p = jran.uniform(u_key, minval=-10, maxval=10, shape=(2,))
        age_weights = db._pureburst_age_weights_from_u_params(lgyr, *u_p)
        assert np.all(np.isfinite(age_weights))
        assert np.all(age_weights >= 0)
        assert np.any(age_weights > 0)
        assert np.allclose(age_weights.sum(), 1.0, rtol=TOL)


def test_age_weights_from_random_params_u_params_consistency():
    ran_key = jran.PRNGKey(0)
    lgyr = np.arange(5.5, 10.35, 0.05)
    n_tests = 10
    for __ in range(n_tests):
        ran_key, u_key = jran.split(ran_key, 2)
        u_p = jran.uniform(u_key, minval=-10, maxval=10, shape=(3,))
        p = db._get_params_from_u_params(u_p)
        age_weights = db._pureburst_age_weights_from_u_params(lgyr, *u_p[1:])
        age_weights2 = db._pureburst_age_weights_from_params(lgyr, *p[1:])
        assert np.allclose(age_weights, age_weights2, rtol=TOL)


def test_compute_bursty_age_weights_from_params():
    ran_key = jran.PRNGKey(0)
    lgyr_since_burst = np.arange(5.5, 10.35, 0.05)
    n_age = lgyr_since_burst.size
    age_weights = jran.uniform(ran_key, minval=0, maxval=1, shape=(n_age,))
    age_weights = age_weights / age_weights.sum()
    bursty_age_weights = db._compute_bursty_age_weights_from_params(
        lgyr_since_burst, age_weights, db.DEFAULT_BURST_PARAMS
    )
    assert np.all(np.isfinite(bursty_age_weights))
    assert np.all(bursty_age_weights >= 0)
    assert np.all(bursty_age_weights <= 1)
    assert np.allclose(bursty_age_weights.sum(), 1.0, rtol=TOL)


def test_compute_bursty_age_weights_from_u_params():
    ran_key = jran.PRNGKey(0)
    lgyr_since_burst = np.arange(5.5, 10.35, 0.05)
    n_tests = 10
    for __ in range(n_tests):
        ran_key, u_key, weights_key = jran.split(ran_key, 3)
        u = jran.uniform(
            u_key, minval=-10, maxval=10, shape=(len(db.DEFAULT_BURST_PARAMS),)
        )
        u_p = db.BurstUParams(*u)
        n_age = lgyr_since_burst.size
        age_weights = jran.uniform(weights_key, minval=0, maxval=1, shape=(n_age,))
        age_weights = age_weights / age_weights.sum()
        bursty_age_weights = db._compute_bursty_age_weights_from_u_params(
            lgyr_since_burst, age_weights, u_p
        )
        assert np.all(np.isfinite(bursty_age_weights))
        assert np.all(bursty_age_weights >= 0)
        assert np.all(bursty_age_weights <= 1)
        assert np.allclose(bursty_age_weights.sum(), 1.0, rtol=TOL)

        p = db._get_params_from_u_params(u_p)
        bursty_age_weights2 = db._compute_bursty_age_weights_from_params(
            lgyr_since_burst, age_weights, p
        )
        assert np.all(np.isfinite(bursty_age_weights2))
        assert np.all(bursty_age_weights2 >= 0)
        assert np.all(bursty_age_weights2 <= 1)
        assert np.allclose(bursty_age_weights2.sum(), 1.0, rtol=TOL)

        assert np.allclose(bursty_age_weights, bursty_age_weights2, rtol=TOL)


def test_calc_bursty_age_weights():
    ran_key = jran.PRNGKey(0)
    burst_params = db.DEFAULT_BURST_PARAMS
    n_age = 107
    ssp_lg_age_gyr = np.linspace(5.0 - 9, 10.5 - 9, n_age)
    smooth_age_weights = jran.uniform(ran_key, minval=0, maxval=1, shape=(n_age,))
    smooth_age_weights = smooth_age_weights / smooth_age_weights.sum()
    assert np.allclose(smooth_age_weights.sum(), 1.0, rtol=1e-4)

    ssp_lg_age_yr = ssp_lg_age_gyr + 9.0
    burst_weights = db._pureburst_age_weights_from_params(
        ssp_lg_age_yr, *burst_params[1:]
    )
    assert np.allclose(burst_weights.sum(), 1.0, rtol=1e-4)

    bursty_age_weights = db.calc_bursty_age_weights(
        burst_params, smooth_age_weights, ssp_lg_age_gyr
    )
    bursty_age_weights.shape == smooth_age_weights.shape
    assert np.all(np.isfinite(bursty_age_weights))
    assert np.allclose(bursty_age_weights.sum(), 1.0, rtol=1e-4)


def test_calc_bursty_age_weights_from_u_params():
    ran_key = jran.PRNGKey(0)
    n_age = 107
    ssp_lg_age_gyr = np.linspace(5.0 - 9, 10.5 - 9, n_age)
    smooth_key, ran_key = jran.split(ran_key, 2)
    smooth_age_weights = jran.uniform(smooth_key, minval=0, maxval=1, shape=(n_age,))
    smooth_age_weights = smooth_age_weights / smooth_age_weights.sum()
    n_tests = 10
    for i in range(n_tests):
        u_key, ran_key = jran.split(ran_key, 2)
        u = jran.uniform(
            u_key, minval=-20, maxval=20, shape=(len(db.DEFAULT_BURST_PARAMS),)
        )
        u_burst_params = db.BurstUParams(*u)
        bursty_age_weights = db.calc_bursty_age_weights_from_u_params(
            u_burst_params, smooth_age_weights, ssp_lg_age_gyr
        )
        assert np.all(np.isfinite(bursty_age_weights))
        assert np.allclose(bursty_age_weights.sum(), 1.0, rtol=1e-4)

        burst_params = db._get_params_from_u_params(u_burst_params)
        bursty_age_weights2 = db.calc_bursty_age_weights(
            burst_params, smooth_age_weights, ssp_lg_age_gyr
        )
        assert np.allclose(bursty_age_weights, bursty_age_weights2, rtol=1e-4)
