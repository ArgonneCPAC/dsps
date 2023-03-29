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
