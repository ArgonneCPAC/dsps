"""Model for the mass-metallicity relation"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from ..utils import _sig_slope


MAIOLINO08_PARAMS = OrderedDict()
MAIOLINO08_PARAMS[0.07] = (11.18, 9.04)
MAIOLINO08_PARAMS[0.7] = (11.57, 9.04)
MAIOLINO08_PARAMS[2.2] = (12.38, 8.99)
MAIOLINO08_PARAMS[3.5] = (12.76, 8.79)
MAIOLINO08_PARAMS[3.51] = (12.87, 8.90)

MZR_T0_PDICT = OrderedDict(
    mzr_t0_y0=0.05, mzr_t0_x0=10.5, mzr_t0_k=1, mzr_t0_slope_lo=0.4, mzr_t0_slope_hi=0.1
)
MZR_T0_PARAM_PBDICT = OrderedDict(
    mzr_t0_y0=(-1.0, 0.25),
    mzr_t0_x0=(10.25, 10.75),
    mzr_t0_k=(0.5, 2),
    mzr_t0_slope_lo=(0.1, 1.0),
    mzr_t0_slope_hi=(0.0, 0.3),
)

MZR_EVOL_T0 = 12.5
MZR_EVOL_K = 1.6
MZR_VS_T_PDICT = OrderedDict(
    c0_y_at_tlook_c=-1.455,
    c1_y_at_tlook_c=0.13,
    tlook_c=8.44,
    c0_early_time_slope=-0.959,
    c1_early_time_slope=0.067242,
)

MZR_SCATTER_PDICT = OrderedDict(mzr_scatter=0.1)
MZR_SCATTER_PBDICT = OrderedDict(mzr_scatter=(0.01, 0.5))

DEFAULT_MZR_PDICT = OrderedDict()
DEFAULT_MZR_PDICT.update(MZR_T0_PDICT)
DEFAULT_MZR_PDICT.update(MZR_VS_T_PDICT)
DEFAULT_MZR_PDICT.update(MZR_SCATTER_PDICT)


@jjit
def mzr_model_t0(
    logsm, mzr_t0_y0, mzr_t0_x0, mzr_t0_k, mzr_t0_slope_lo, mzr_t0_slope_hi
):
    mzr_t0_xtp = mzr_t0_x0
    return _sig_slope(
        logsm,
        mzr_t0_xtp,
        mzr_t0_y0,
        mzr_t0_x0,
        mzr_t0_k,
        mzr_t0_slope_lo,
        mzr_t0_slope_hi,
    )


@jjit
def maiolino08_metallicity_evolution(logsm, logm0, k0):
    x = logsm - logm0
    xsq = x * x
    return -12.0 - 0.0864 * xsq + k0


@jjit
def _delta_logz_vs_t_lookback(t_lookback, y_at_tc, tc, early_time_slope, k):
    late_time_slope = y_at_tc / tc
    xtp = tc
    args = xtp, y_at_tc, tc, k, late_time_slope, early_time_slope
    logZ_reduction = _sig_slope(t_lookback, *args)
    return logZ_reduction


@jjit
def _get_p_at_lgmstar(
    lgmstar,
    c0_y_at_tlook_c,
    c1_y_at_tlook_c,
    tlook_c,
    c0_early_time_slope,
    c1_early_time_slope,
):
    y_at_tlook_c = c0_y_at_tlook_c + c1_y_at_tlook_c * lgmstar
    tlook_c = tlook_c + jnp.zeros_like(lgmstar)
    early_time_slope = c0_early_time_slope + c1_early_time_slope * lgmstar
    return y_at_tlook_c, tlook_c, early_time_slope


@jjit
def _delta_logz_at_t_lookback(
    lgmstar,
    t_lookback,
    c0_y_at_tlook_c,
    c1_y_at_tlook_c,
    tlook_c,
    c0_early_time_slope,
    c1_early_time_slope,
):
    y_at_tlook_c, tlook_c, early_time_slope = _get_p_at_lgmstar(
        lgmstar,
        c0_y_at_tlook_c,
        c1_y_at_tlook_c,
        tlook_c,
        c0_early_time_slope,
        c1_early_time_slope,
    )
    logZ_reduction = _delta_logz_vs_t_lookback(
        t_lookback, y_at_tlook_c, tlook_c, early_time_slope, MZR_EVOL_K
    )
    return jnp.where(logZ_reduction > 0, 0, logZ_reduction)


@jjit
def mzr_evolution_model(
    logsm,
    cosmic_time,
    c0_y_at_tlook_c,
    c1_y_at_tlook_c,
    tlook_c,
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
        tlook_c,
        c0_early_time_slope,
        c1_early_time_slope,
    )
    return logZ_reduction


@jjit
def mzr_model(
    logsm,
    t,
    mzr_t0_y0,
    mzr_t0_x0,
    mzr_t0_k,
    mzr_t0_slope_lo,
    mzr_t0_slope_hi,
    c0_y_at_tlook_c,
    c1_y_at_tlook_c,
    tlook_c,
    c0_early_time_slope,
    c1_early_time_slope,
):
    lgmet_at_t0 = mzr_model_t0(
        logsm, mzr_t0_y0, mzr_t0_x0, mzr_t0_k, mzr_t0_slope_lo, mzr_t0_slope_hi
    )
    logZ_reduction = mzr_evolution_model(
        logsm,
        t,
        c0_y_at_tlook_c,
        c1_y_at_tlook_c,
        tlook_c,
        c0_early_time_slope,
        c1_early_time_slope,
    )
    return lgmet_at_t0 + logZ_reduction
