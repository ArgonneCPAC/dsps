"""
"""
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap

C0 = 1 / 2
C1 = 35 / 96
C3 = -35 / 864
C5 = 7 / 2592
C7 = -5 / 69984

LGYR_STELLAR_AGE_MIN = 5.0
DELTA_LGYR_STELLAR_AGES = 0.05
LGYR_BURST_DURATION_MAX = 9.0
DEFAULT_DBURST = 2.0


@jjit
def _tw_cuml_kern(x, m, h):
    """Triweight kernel version of an err function."""
    z = (x - m) / h
    zz = z * z
    zzz = zz * z
    val = C0 + C1 * z + C3 * zzz + C5 * zzz * zz + C7 * zzz * zzz * z
    val = jnp.where(z < -3, 0, val)
    val = jnp.where(z > 3, 1, val)
    return val


@jjit
def _tw_sigmoid(x, x0, tw_h, ymin, ymax):
    height_diff = ymax - ymin
    body = _tw_cuml_kern(x, x0, tw_h)
    return ymin + height_diff * body


@jjit
def _burst_age_weights_kern(lgyr_since_burst, lgyr_burst_duration, lgyr_age_min):
    delta = lgyr_burst_duration - lgyr_age_min
    x0 = lgyr_age_min + delta / 2.0
    tw_h = delta / 6.0
    return _tw_sigmoid(lgyr_since_burst, x0, tw_h, 1.0, 0.0)


@jjit
def _get_lgyr_burst_duration(dburst, lgyr_age_min, delta_lgyr_ages, lgyr_burst_max):
    lgyr_min = lgyr_age_min + delta_lgyr_ages
    lgyr_max = lgyr_burst_max
    dlgyr_tot = lgyr_max - lgyr_min
    tw_h = dlgyr_tot / 6.0
    x0 = (lgyr_max - lgyr_min) / 2.0
    lgyr_burst_duration = _tw_sigmoid(dburst, x0, tw_h, lgyr_min, lgyr_max)
    return lgyr_burst_duration


@jjit
def _burst_age_weights_cdf(
    lgyr_since_burst,
    dburst,
    lgyr_age_min=LGYR_STELLAR_AGE_MIN,
    delta_lgyr_ages=DELTA_LGYR_STELLAR_AGES,
    lgyr_burst_max=LGYR_BURST_DURATION_MAX,
):
    lgyr_burst_duration = _get_lgyr_burst_duration(
        dburst, lgyr_age_min, delta_lgyr_ages, lgyr_burst_max
    )
    return _burst_age_weights_kern(lgyr_since_burst, lgyr_burst_duration, lgyr_age_min)


@jjit
def _burst_age_weights(
    lgyr_since_burst,
    dburst,
    lgyr_age_min=LGYR_STELLAR_AGE_MIN,
    delta_lgyr_ages=DELTA_LGYR_STELLAR_AGES,
    lgyr_burst_max=LGYR_BURST_DURATION_MAX,
):
    cdf = _burst_age_weights_cdf(
        lgyr_since_burst, dburst, lgyr_age_min, delta_lgyr_ages, lgyr_burst_max
    )
    weights = cdf / jnp.sum(cdf)
    return weights


_A = (None, 0)
_burst_age_weights_pop = jjit(vmap(_burst_age_weights, in_axes=_A))
