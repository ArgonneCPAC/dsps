"""Mass-metallicity-redshift scaling relation with unbounding behavior
"""

from collections import OrderedDict, namedtuple

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..utils import _inverse_sigmoid, _sig_slope, _sigmoid

MZR_EVOL_T0 = 12.5
MZR_EVOL_K = 1.6

MZR_T0_XTP = 10.5
MZR_DT_XTP = 8.44
BOUNDING_K = 0.1
LGMET_SOLAR = float(np.log10(0.012))

MZR_SCATTER = 0.1

TLOOK_C = 8.44


MZR_T0_PDICT = OrderedDict(
    mzr_t0_ytp=0.05,
    mzr_t0_x0=10.5,
    mzr_t0_k=1,
    mzr_t0_slope_lo=0.4,
    mzr_t0_slope_hi=0.1,
)
MZR_T0_PBDICT = OrderedDict(
    mzr_t0_ytp=(-1.0, 0.25),
    mzr_t0_x0=(10.25, 10.75),
    mzr_t0_k=(0.5, 2),
    mzr_t0_slope_lo=(0.1, 1.0),
    mzr_t0_slope_hi=(0.0, 0.3),
)

MZR_TEVOL_PDICT = OrderedDict(
    c0_y_at_tlook_c=-1.455,
    c1_y_at_tlook_c=0.13,
    c0_early_time_slope=-0.959,
    c1_early_time_slope=0.067242,
)
MZR_TEVOL_PBDICT = OrderedDict(
    c0_y_at_tlook_c=(-2.0, -1.0),
    c1_y_at_tlook_c=(0.0, 0.5),
    c0_early_time_slope=(-1.5, -0.5),
    c1_early_time_slope=(0.0, 0.5),
)

DEFAULT_MZR_PDICT = OrderedDict()
DEFAULT_MZR_PDICT.update(MZR_T0_PDICT)
DEFAULT_MZR_PDICT.update(MZR_TEVOL_PDICT)

MZR_PBDICT = OrderedDict()
MZR_PBDICT.update(MZR_T0_PBDICT)
MZR_PBDICT.update(MZR_TEVOL_PBDICT)

_MZR_PNAMES = list(DEFAULT_MZR_PDICT.keys())
_MZR_UPNAMES = ["u_" + key for key in _MZR_PNAMES]

MZRParams = namedtuple("MZRParams", _MZR_PNAMES)
MZRUParams = namedtuple("MZRUParams", _MZR_UPNAMES)

DEFAULT_MZR_PARAMS = MZRParams(**DEFAULT_MZR_PDICT)


def get_ran_t0_params(ran_key, bounds_pdict=MZR_T0_PBDICT):
    nparams = len(bounds_pdict)
    bounds = list(bounds_pdict.values())
    params = []
    for ip in range(nparams):
        ran_key, p_key = jran.split(ran_key, 2)
        u = jran.uniform(p_key, minval=bounds[ip][0], maxval=bounds[ip][1], shape=())
        params.append(u)
    return params


@jjit
def mzr_model_t0(
    logsm, mzr_t0_ytp, mzr_t0_x0, mzr_t0_k, mzr_t0_slope_lo, mzr_t0_slope_hi
):
    mzr_t0 = _sig_slope(
        logsm,
        MZR_T0_XTP,
        mzr_t0_ytp,
        mzr_t0_x0,
        mzr_t0_k,
        mzr_t0_slope_lo,
        mzr_t0_slope_hi,
    )
    return mzr_t0 + LGMET_SOLAR


@jjit
def _delta_logz_vs_t_lookback(t_lookback, y_at_tc, early_time_slope, k):
    late_time_slope = y_at_tc / TLOOK_C
    args = MZR_DT_XTP, y_at_tc, TLOOK_C, k, late_time_slope, early_time_slope
    logZ_reduction = _sig_slope(t_lookback, *args)
    return logZ_reduction


@jjit
def _get_p_at_lgmstar(
    lgmstar,
    c0_y_at_tlook_c,
    c1_y_at_tlook_c,
    c0_early_time_slope,
    c1_early_time_slope,
):
    y_at_tlook_c = c0_y_at_tlook_c + c1_y_at_tlook_c * lgmstar
    early_time_slope = c0_early_time_slope + c1_early_time_slope * lgmstar
    return y_at_tlook_c, early_time_slope


@jjit
def _delta_logz_at_t_lookback(
    lgmstar,
    t_lookback,
    c0_y_at_tlook_c,
    c1_y_at_tlook_c,
    c0_early_time_slope,
    c1_early_time_slope,
):
    y_at_tlook_c, early_time_slope = _get_p_at_lgmstar(
        lgmstar,
        c0_y_at_tlook_c,
        c1_y_at_tlook_c,
        c0_early_time_slope,
        c1_early_time_slope,
    )
    logZ_reduction = _delta_logz_vs_t_lookback(
        t_lookback, y_at_tlook_c, early_time_slope, MZR_EVOL_K
    )
    return jnp.where(logZ_reduction > 0, 0, logZ_reduction)


@jjit
def mzr_evolution_model(
    logsm,
    cosmic_time,
    c0_y_at_tlook_c,
    c1_y_at_tlook_c,
    c0_early_time_slope,
    c1_early_time_slope,
):
    t_lookback = MZR_EVOL_T0 - cosmic_time
    t_lookback = jnp.where(t_lookback < 0, 0, t_lookback)
    logZ_reduction = _delta_logz_at_t_lookback(
        logsm,
        t_lookback,
        c0_y_at_tlook_c,
        c1_y_at_tlook_c,
        c0_early_time_slope,
        c1_early_time_slope,
    )
    return logZ_reduction


@jjit
def mzr_model(
    logsm,
    t,
    mzr_t0_ytp,
    mzr_t0_x0,
    mzr_t0_k,
    mzr_t0_slope_lo,
    mzr_t0_slope_hi,
    c0_y_at_tlook_c,
    c1_y_at_tlook_c,
    c0_early_time_slope,
    c1_early_time_slope,
):
    lgmet_at_t0 = mzr_model_t0(
        logsm, mzr_t0_ytp, mzr_t0_x0, mzr_t0_k, mzr_t0_slope_lo, mzr_t0_slope_hi
    )
    logZ_reduction = mzr_evolution_model(
        logsm,
        t,
        c0_y_at_tlook_c,
        c1_y_at_tlook_c,
        c0_early_time_slope,
        c1_early_time_slope,
    )
    return lgmet_at_t0 + logZ_reduction


@jjit
def _get_square_bounded_param(u_param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _sigmoid(u_param, mid, BOUNDING_K, lo, hi)


@jjit
def _get_square_unbounded_param(param, bound):
    lo, hi = bound
    mid = 0.5 * (lo + hi)
    return _inverse_sigmoid(param, mid, BOUNDING_K, lo, hi)


@jjit
def _get_mzr_t0_slope_hi_from_unbounded(mzr_t0_slope_lo, u_mzr_t0_slope_hi):
    x0 = 0.0
    ylo = MZR_PBDICT["mzr_t0_slope_hi"][0]
    yhi = mzr_t0_slope_lo
    mzr_t0_slope_hi = _sigmoid(u_mzr_t0_slope_hi, x0, BOUNDING_K, ylo, yhi)
    return mzr_t0_slope_hi


def _get_u_mzr_t0_slope_hi_from_bounded(mzr_t0_slope_lo, mzr_t0_slope_hi):
    x0 = 0.0
    ylo = MZR_PBDICT["mzr_t0_slope_hi"][0]
    yhi = mzr_t0_slope_lo
    u_mzr_t0_slope_hi = _inverse_sigmoid(mzr_t0_slope_hi, x0, BOUNDING_K, ylo, yhi)
    return u_mzr_t0_slope_hi


_C = (0, 0)
_get_params_kern = jjit(vmap(_get_square_bounded_param, in_axes=_C))
_get_u_params_kern = jjit(vmap(_get_square_unbounded_param, in_axes=_C))


@jjit
def get_bounded_mzr_params(u_params):
    u_parr = jnp.array([getattr(u_params, u_pname) for u_pname in _MZR_UPNAMES])

    # First ignore the non-square bounds
    params = MZRParams(
        *_get_params_kern(u_parr, jnp.array((list(MZR_PBDICT.values()))))
    )

    # Now compute the correct bounded value of mzr_t0_slope_hi
    mzr_t0_slope_hi = _get_mzr_t0_slope_hi_from_unbounded(
        params.mzr_t0_slope_lo, u_params.u_mzr_t0_slope_hi
    )
    # Overwrite mzr_t0_slope_hi with the correct value
    params = params._replace(mzr_t0_slope_hi=mzr_t0_slope_hi)

    return params


@jjit
def get_unbounded_mzr_params(params):
    parr = jnp.array([getattr(params, pname) for pname in _MZR_PNAMES])

    # First ignore the non-square bounds
    u_params = MZRUParams(
        *_get_u_params_kern(parr, jnp.array(list(MZR_PBDICT.values())))
    )

    # Now compute the correct bounded value of u_mzr_t0_slope_hi
    u_mzr_t0_slope_hi = _get_u_mzr_t0_slope_hi_from_bounded(
        params.mzr_t0_slope_lo, params.mzr_t0_slope_hi
    )

    # Overwrite u_mzr_t0_slope_hi with the correct value
    u_params = u_params._replace(u_mzr_t0_slope_hi=u_mzr_t0_slope_hi)

    return u_params


DEFAULT_MZR_U_PARAMS = MZRUParams(*get_unbounded_mzr_params(DEFAULT_MZR_PARAMS))
