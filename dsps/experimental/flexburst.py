"""
"""
import typing
from jax import numpy as jnp
from jax import jit as jjit
from ..utils import triweight_gaussian, _sigmoid, _inverse_sigmoid

LGYR_PEAK_MIN = 5.0
LGAGE_MAX = 9.0
DLGAGE_MIN = 1.0
LGAGE_K = 0.1


class BurstParams(typing.NamedTuple):
    lgyr_peak: jnp.float32
    lgyr_max: jnp.float32


class BurstUParams(typing.NamedTuple):
    u_lgyr_peak: jnp.float32
    u_lgyr_max: jnp.float32


DEFAULT_PARAMS = BurstParams(5.5, 7.0)


@jjit
def _zero_safe_normalize(x):
    s = jnp.sum(x)
    s = jnp.where(s <= 0, 1, s)
    return x / s


@jjit
def _age_weights_from_params(lgyr, params):
    lgyr_peak, lgyr_max = params

    dlgyr_support = lgyr_max - lgyr_peak
    lgyr_min = lgyr_peak - dlgyr_support
    twx0 = 0.5 * (lgyr_min + lgyr_max)

    dlgyr = lgyr_max - lgyr_min
    twh = dlgyr / 6

    tw_gauss = triweight_gaussian(lgyr, twx0, twh)

    age_weights = _zero_safe_normalize(tw_gauss)
    return age_weights


@jjit
def _age_weights_from_u_params(lgyr, u_params):
    params = _get_params_from_u_params(u_params)
    return _age_weights_from_params(lgyr, params)


@jjit
def _get_params_from_u_params(u_params):
    u_lgyr_peak, u_lgyr_max = u_params
    lgyr_peak = _get_lgyr_peak_from_u_lgyr_peak(u_lgyr_peak)
    lgyr_max = _get_lgyr_max_from_lgyr_peak(lgyr_peak, u_lgyr_max)
    params = lgyr_peak, lgyr_max
    return params


@jjit
def _get_u_params_from_params(params):
    lgyr_peak, lgyr_max = params
    u_lgyr_peak = _get_u_lgyr_peak_from_lgyr_peak(lgyr_peak)
    u_lgyr_max = _get_u_lgyr_max_from_lgyr_peak(lgyr_peak, lgyr_max)
    u_params = u_lgyr_peak, u_lgyr_max
    return u_params


@jjit
def _get_lgyr_peak_from_u_lgyr_peak(u_lgyr_peak):
    lo = LGYR_PEAK_MIN
    hi = LGAGE_MAX - DLGAGE_MIN
    x0 = 0.5 * (hi + lo)
    lgyr_peak = _sigmoid(u_lgyr_peak, x0, LGAGE_K, lo, hi)
    return lgyr_peak


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
    lgyr_peak = _inverse_sigmoid(lgyr_peak, x0, LGAGE_K, lo, hi)
    return lgyr_peak


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
