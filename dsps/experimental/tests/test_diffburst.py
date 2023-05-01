"""
"""
import numpy as np
from .. import diffburst as db


def test_params_invert():
    n_tests = 10
    for __ in range(n_tests):
        u_p = np.random.uniform(-10, 10, 2)
        p = db._get_params_from_u_params(u_p)
        u_p2 = db._get_u_params_from_params(p)
        assert np.allclose(u_p, u_p2, rtol=1e-3)


def test_age_weights_is_finite_and_zero_for_edge_cases():
    lgyr_peak, lgyr_max = 6.5, 8.0

    # all times in the lgyr_table are after lgyr_max
    lgyr_test = np.linspace(lgyr_max, lgyr_max + 2, 100)
    age_weights = db._age_weights_from_params(lgyr_test, (lgyr_peak, lgyr_max))
    assert np.all(np.isfinite(age_weights))
    assert np.allclose(age_weights, 0.0)

    # all times in the lgyr_table are before lgyr_min
    dlgyr_support = lgyr_max - lgyr_peak
    lgyr_min = lgyr_peak - dlgyr_support
    lgyr_test = np.linspace(lgyr_min - 2, lgyr_min, 100)
    age_weights = db._age_weights_from_params(lgyr_test, (lgyr_peak, lgyr_max))
    assert np.all(np.isfinite(age_weights))
    assert np.allclose(age_weights, 0.0)


def test_age_weights_from_params_scales_with_lgyr_max_as_expected():
    n_tests = 10
    lgyr_peak = 5.5
    for __ in range(n_tests):
        lgyr_max = np.random.uniform(7, 8)
        params = lgyr_peak, lgyr_max

        lgyr = np.linspace(4, 10.5, 100)
        age_weights = db._age_weights_from_params(lgyr, params)
        zmsk = lgyr > params[1]
        assert np.all(age_weights[zmsk] == 0)
        assert np.any(age_weights > 0)

        age_weight_at_lgyr_peak = db._age_weights_from_params(lgyr_peak, params)
        assert age_weight_at_lgyr_peak > 0

        lgyr_post_peak = np.linspace(lgyr_peak, lgyr_peak + 1, 10)
        age_weights_post_peak = db._age_weights_from_params(lgyr_post_peak, params)
        assert np.all(np.diff(age_weights_post_peak) < 0)

    lgyr_peak = 5.5

    lgyr_test = np.linspace(lgyr_peak, lgyr_peak + 1, 100)
    age_weight_younger = db._age_weights_from_params(lgyr_test, (lgyr_peak, 6.5))
    age_weights_older = db._age_weights_from_params(lgyr_test, (lgyr_peak, 8.0))
    assert age_weight_younger[0] > age_weights_older[0]


def test_age_weights_from_default_params_are_weights():
    lgyr = np.arange(5.5, 10.35, 0.05)
    age_weights = db._age_weights_from_params(lgyr, db.DEFAULT_PARAMS)
    assert np.all(np.isfinite(age_weights))
    assert np.all(age_weights >= 0)
    assert np.any(age_weights > 0)
    assert np.allclose(age_weights.sum(), 1.0, rtol=1e-3)


def test_age_weights_from_random_params_are_weights():
    lgyr = np.arange(5.5, 10.35, 0.05)
    n_tests = 10
    for __ in range(n_tests):
        u_p = np.random.uniform(-10, 10, 2)
        age_weights = db._age_weights_from_u_params(lgyr, u_p)
        assert np.all(np.isfinite(age_weights))
        assert np.all(age_weights >= 0)
        assert np.any(age_weights > 0)
        assert np.allclose(age_weights.sum(), 1.0, rtol=1e-3)


def test_age_weights_from_random_params_u_params_consistency():
    lgyr = np.arange(5.5, 10.35, 0.05)
    n_tests = 10
    for __ in range(n_tests):
        u_p = np.random.uniform(-10, 10, 2)
        age_weights = db._age_weights_from_u_params(lgyr, u_p)
        p = db._get_params_from_u_params(u_p)
        age_weights2 = db._age_weights_from_params(lgyr, p)
        assert np.allclose(age_weights, age_weights2, rtol=1e-3)


def test_compute_bursty_age_weights_from_params():
    lgyr_since_burst = np.arange(5.5, 10.35, 0.05)
    n_age = lgyr_since_burst.size
    age_weights = np.random.uniform(0, 1, n_age)
    age_weights = age_weights / age_weights.sum()
    fburst = np.random.uniform(0, 1)
    bursty_age_weights = db._compute_bursty_age_weights_from_params(
        lgyr_since_burst, age_weights, fburst, db.DEFAULT_PARAMS
    )
    assert np.all(np.isfinite(bursty_age_weights))
    assert np.all(bursty_age_weights >= 0)
    assert np.all(bursty_age_weights <= 1)
    assert np.allclose(bursty_age_weights.sum(), 1.0, rtol=1e-3)


def test_compute_bursty_age_weights_from_u_params():
    lgyr_since_burst = np.arange(5.5, 10.35, 0.05)
    n_tests = 10
    for __ in range(n_tests):
        u_p = np.random.uniform(-10, 10, 2)
        n_age = lgyr_since_burst.size
        age_weights = np.random.uniform(0, 1, n_age)
        age_weights = age_weights / age_weights.sum()
        fburst = np.random.uniform(0, 1)
        bursty_age_weights = db._compute_bursty_age_weights_from_u_params(
            lgyr_since_burst, age_weights, fburst, u_p
        )
        assert np.all(np.isfinite(bursty_age_weights))
        assert np.all(bursty_age_weights >= 0)
        assert np.all(bursty_age_weights <= 1)
        assert np.allclose(bursty_age_weights.sum(), 1.0, rtol=1e-3)

        p = db._get_params_from_u_params(u_p)
        bursty_age_weights2 = db._compute_bursty_age_weights_from_params(
            lgyr_since_burst, age_weights, fburst, p
        )
        assert np.all(np.isfinite(bursty_age_weights2))
        assert np.all(bursty_age_weights2 >= 0)
        assert np.all(bursty_age_weights2 <= 1)
        assert np.allclose(bursty_age_weights2.sum(), 1.0, rtol=1e-3)

        assert np.allclose(bursty_age_weights, bursty_age_weights2, rtol=1e-3)
