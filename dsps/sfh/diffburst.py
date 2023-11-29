"""
"""
import typing

from jax import jit as jjit
from jax import numpy as jnp

from ..utils import _inverse_sigmoid, _sigmoid, triweight_gaussian

LGYR_PEAK_MIN = 5.0
LGAGE_MAX = 9.0
DLGAGE_MIN = 1.0
LGAGE_K = 0.1
LGFBURST_MIN = -8.0
LGFBURST_MAX = -1.0
LGFB_X0, LGFB_K = -2, 0.1
DEFAULT_LGFBURST = -4.0


class BurstParams(typing.NamedTuple):
    lgfburst: jnp.float32
    lgyr_peak: jnp.float32
    lgyr_max: jnp.float32


class BurstUParams(typing.NamedTuple):
    u_lgfburst: jnp.float32
    u_lgyr_peak: jnp.float32
    u_lgyr_max: jnp.float32


DEFAULT_PARAMS = BurstParams(-3.0, 5.5, 7.0)


@jjit
def calc_bursty_age_weights(burst_params, smooth_age_weights, ssp_lg_age_gyr):
    """Calculate the distribution of stellar ages of a smooth+bursty population

    Parameters
    ----------
    burst_params : namedtuple
        burst_params = (lgfburst, lgyr_peak, lgyr_max)

    smooth_age_weights : ndarray, shape (n_age, )
        Array storing the distribution of stellar ages, P(τ), from a smooth SFH

    ssp_lg_age_gyr : ndarray of shape (n_age, )
        Base-10 log of stellar age in Gyr
        The namedtuple dsps.SPSData.ssp_lg_age_gyr stores
        a grid of stellar ages in these units

    Returns
    -------
    age_weights : ndarray, shape (n_age, )
        P(τ) after adding a fractional contribution from the bursting population

    """
    burst_params = BurstParams(*burst_params)

    ssp_lg_age_yr = ssp_lg_age_gyr + 9
    burst_weights = _age_weights_from_params(ssp_lg_age_yr, burst_params)

    fb = 10**burst_params.lgfburst
    age_weights = fb * burst_weights + (1 - fb) * smooth_age_weights

    return age_weights


@jjit
def calc_bursty_age_weights_from_u_params(
    u_burst_params, smooth_age_weights, ssp_lg_age_gyr
):
    """Calculate the distribution of stellar ages of a smooth+bursty population
    when passed unbounded parameters

    Parameters
    ----------
    u_burst_params : namedtuple
        burst_params = (u_lgfburst, u_lgyr_peak, u_lgyr_max)

    smooth_age_weights : ndarray, shape (n_age, )
        Array storing the distribution of stellar ages, P(τ), from a smooth SFH

    ssp_lg_age_gyr : ndarray of shape (n_age, )
        Base-10 log of stellar age in Gyr
        The namedtuple dsps.SPSData.ssp_lg_age_gyr stores
        a grid of stellar ages in these units

    Returns
    -------
    age_weights : ndarray, shape (n_age, )
        P(τ) after adding a fractional contribution from the bursting population

    """
    burst_params = _get_params_from_u_params(u_burst_params)
    return calc_bursty_age_weights(burst_params, smooth_age_weights, ssp_lg_age_gyr)


@jjit
def _zero_safe_normalize(x):
    s = jnp.sum(x)
    s = jnp.where(s <= 0, 1, s)
    return x / s


@jjit
def _age_weights_from_params(lgyr, burst_params):
    burst_params = BurstParams(*burst_params)
    dlgyr_support = burst_params.lgyr_max - burst_params.lgyr_peak
    lgyr_min = burst_params.lgyr_peak - dlgyr_support
    twx0 = 0.5 * (lgyr_min + burst_params.lgyr_max)

    dlgyr = burst_params.lgyr_max - lgyr_min
    twh = dlgyr / 6

    tw_gauss = triweight_gaussian(lgyr, twx0, twh)

    age_weights = _zero_safe_normalize(tw_gauss)
    return age_weights


@jjit
def _age_weights_from_u_params(lgyr, u_burst_params):
    u_burst_params = BurstUParams(*u_burst_params)
    params = _get_params_from_u_params(u_burst_params)
    return _age_weights_from_params(lgyr, params)


@jjit
def _get_params_from_u_params(u_params):
    u_lgfburst, u_lgyr_peak, u_lgyr_max = u_params
    lgfburst = _get_lgfburst_from_u_lgfburst(u_lgfburst)
    lgyr_peak = _get_lgyr_peak_from_u_lgyr_peak(u_lgyr_peak)
    lgyr_max = _get_lgyr_max_from_lgyr_peak(lgyr_peak, u_lgyr_max)
    params = lgfburst, lgyr_peak, lgyr_max
    return params


@jjit
def _get_u_params_from_params(params):
    lgfburst, lgyr_peak, lgyr_max = params
    u_lgfburst = _get_u_lgfburst_from_lgfburst(lgfburst)
    u_lgyr_peak = _get_u_lgyr_peak_from_lgyr_peak(lgyr_peak)
    u_lgyr_max = _get_u_lgyr_max_from_lgyr_peak(lgyr_peak, lgyr_max)
    u_params = u_lgfburst, u_lgyr_peak, u_lgyr_max
    return u_params


@jjit
def _get_lgyr_peak_from_u_lgyr_peak(u_lgyr_peak):
    lo = LGYR_PEAK_MIN
    hi = LGAGE_MAX - DLGAGE_MIN
    x0 = 0.5 * (hi + lo)
    lgyr_peak = _sigmoid(u_lgyr_peak, x0, LGAGE_K, lo, hi)
    return lgyr_peak


@jjit
def _get_lgfburst_from_u_lgfburst(u_lgfburst):
    return _sigmoid(u_lgfburst, LGFB_X0, LGFB_K, LGFBURST_MIN, LGFBURST_MAX)


@jjit
def _get_u_lgfburst_from_lgfburst(lgfburst):
    return _inverse_sigmoid(lgfburst, LGFB_X0, LGFB_K, LGFBURST_MIN, LGFBURST_MAX)


@jjit
def _get_lgyr_max_from_lgyr_peak(lgyr_peak, u_lgyr_max):
    lo, hi = lgyr_peak + DLGAGE_MIN, LGAGE_MAX
    x0 = 0.5 * (hi + lo)
    lgyr_max = _sigmoid(u_lgyr_max, x0, LGAGE_K, lo, hi)
    return lgyr_max


@jjit
def _get_u_lgyr_peak_from_lgyr_peak(lgyr_peak):
    lo = LGYR_PEAK_MIN
    hi = LGAGE_MAX - DLGAGE_MIN
    x0 = 0.5 * (hi + lo)
    u_lgyr_peak = _inverse_sigmoid(lgyr_peak, x0, LGAGE_K, lo, hi)
    return u_lgyr_peak


@jjit
def _get_u_lgyr_max_from_lgyr_peak(lgyr_peak, lgyr_max):
    lo, hi = lgyr_peak + DLGAGE_MIN, LGAGE_MAX
    x0 = 0.5 * (hi + lo)
    u_lgyr_max = _inverse_sigmoid(lgyr_max, x0, LGAGE_K, lo, hi)
    return u_lgyr_max


DEFAULT_U_PARAMS = BurstUParams(
    *[float(u_p) for u_p in _get_u_params_from_params(DEFAULT_PARAMS)]
)


@jjit
def _compute_bursty_age_weights_from_params(
    lgyr_since_burst, age_weights, fburst, params
):
    burst_weights = _age_weights_from_params(lgyr_since_burst, params)
    age_weights = fburst * burst_weights + (1 - fburst) * age_weights
    return age_weights


@jjit
def _compute_bursty_age_weights_from_u_params(
    lgyr_since_burst, age_weights, fburst, u_params
):
    burst_weights = _age_weights_from_u_params(lgyr_since_burst, u_params)
    age_weights = fburst * burst_weights + (1 - fburst) * age_weights
    return age_weights
