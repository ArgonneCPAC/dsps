"""
"""
import numpy as np
from .. import diffburst as db


def test_burst_age_weights_cdf_is_monotonic():
    log_age_yr = np.linspace(-500, 500, 1000)
    dburst_arr = np.linspace(-100, 100, 50)
    for dburst in dburst_arr:
        res = db._burst_age_weights_cdf(log_age_yr, dburst)
        assert np.all(np.diff(res) <= 0)


def test_burst_age_weights_cdf_is_correctly_bounded():
    dburst_arr = np.linspace(-100, 100, 50)
    for dburst in dburst_arr:
        log_age_yr = np.linspace(-500, 500, 1000)
        res = db._burst_age_weights_cdf(log_age_yr, dburst)
        assert np.all(res >= 0)
        assert np.all(res <= 1)

        log_age_yr = db.LGYR_STELLAR_AGE_MIN
        res = db._burst_age_weights(log_age_yr, dburst)
        assert np.allclose(res, 1.0, atol=1e-3)


def test_burst_age_weights_cdf_never_vanishes_everywhere():
    log_age_yr = np.linspace(-500, 500, 1000)
    dburst_arr = np.linspace(-100, 100, 50)
    for dburst in dburst_arr:
        log_age_yr = np.linspace(-500, 500, 1000)
        res = db._burst_age_weights_cdf(log_age_yr, dburst)
        assert np.any(res > 0)


def test_burst_age_weights_sum_to_unity():
    log_age_yr = np.linspace(-500, 500, 1000)
    dburst_arr = np.linspace(-100, 100, 50)
    for dburst in dburst_arr:
        res = db._burst_age_weights(log_age_yr, dburst)
        assert np.allclose(np.sum(res), 1.0)


def test_compute_bursty_age_weights():
    n_ages = 25
    lgyr_since_burst = np.linspace(5, 10, n_ages)
    age_weights = np.ones_like(lgyr_since_burst) / n_ages
    fburst = 0.01
    age_weights = db._compute_bursty_age_weights(
        lgyr_since_burst, age_weights, fburst, db.DEFAULT_DBURST
    )
    assert age_weights.shape == (n_ages,)
    assert np.allclose(age_weights.sum(), 1.0, rtol=1e-3)
    assert np.all(np.isfinite(age_weights))


def test_compute_bursty_age_weights_pop():
    n_ages = 25
    n_galpop = 50
    lgyr_since_burst = np.linspace(5, 10, n_ages)
    smooth_age_weights = np.ones((n_galpop, n_ages)) / n_ages
    fburst = 0.01 * np.ones(n_galpop)
    dburst = np.zeros(n_galpop) + db.DEFAULT_DBURST
    assert fburst.shape == (n_galpop,)
    assert dburst.shape == (n_galpop,)

    bursty_age_weights = db._compute_bursty_age_weights_pop(
        lgyr_since_burst, smooth_age_weights, fburst, dburst
    )
    assert bursty_age_weights.shape == (n_galpop, n_ages)
    assert np.all(np.isfinite(bursty_age_weights))
    assert np.allclose(np.sum(bursty_age_weights, axis=1), 1.0, rtol=1e-4)
