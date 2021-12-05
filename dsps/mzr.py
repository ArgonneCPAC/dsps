"""
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import ops as jops
from jax import vmap
from .utils import triweighted_histogram, _get_bin_edges, _tw_sigmoid


MAIOLINO08_PARAMS = OrderedDict()
MAIOLINO08_PARAMS[0.07] = (11.18, 9.04)
MAIOLINO08_PARAMS[0.7] = (11.57, 9.04)
MAIOLINO08_PARAMS[2.2] = (12.38, 8.99)
MAIOLINO08_PARAMS[3.5] = (12.76, 8.79)
MAIOLINO08_PARAMS[3.51] = (12.87, 8.90)

MZR_T0_PARAMS = OrderedDict(
    mzr_t0_y0=0.05, mzr_t0_x0=10.5, mzr_t0_k=1, mzr_t0_slope_lo=0.4, mzr_t0_slope_hi=0.1
)
MZR_T0_PARAM_BOUNDS = OrderedDict(
    mzr_t0_y0=(-1.0, 0.25),
    mzr_t0_x0=(10.25, 10.75),
    mzr_t0_k=(0.5, 2),
    mzr_t0_slope_lo=(0.1, 1.0),
    mzr_t0_slope_hi=(0.0, 0.3),
)

MZR_EVOL_T0 = 12.5
MZR_EVOL_K = 1.6
MZR_VS_T_PARAMS = OrderedDict(
    c0_y_at_tlook_c=-1.455,
    c1_y_at_tlook_c=0.13,
    tlook_c=8.44,
    c0_early_time_slope=-0.959,
    c1_early_time_slope=0.067242,
)

MZR_SCATTER_PARAMS = OrderedDict(mzr_scatter=0.1)
MZR_SCATTER_BOUNDS = OrderedDict(mzr_scatter=(0.01, 0.5))

DEFAULT_MZR_PARAMS = OrderedDict()
DEFAULT_MZR_PARAMS.update(MZR_T0_PARAMS)
DEFAULT_MZR_PARAMS.update(MZR_VS_T_PARAMS)
DEFAULT_MZR_PARAMS.update(MZR_SCATTER_PARAMS)

TODAY = 13.8
LGMET_LO, LGMET_HI = -10.0, 10.0
N_T_SMH_INTEGRATION = 100

LGAGE_CRIT_YR, LGAGE_CRIT_H = 8.0, 1.0


@jjit
def _sigmoid(x, logtc, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1.0 + jnp.exp(-k * (x - logtc)))


@jjit
def _sig_slope(x, y0, x0, slope_k, lo, hi):
    slope = _sigmoid(x, x0, slope_k, lo, hi)
    return y0 + slope * (x - x0)


@jjit
def mzr_model_t0(
    logsm, mzr_t0_y0, mzr_t0_x0, mzr_t0_k, mzr_t0_slope_lo, mzr_t0_slope_hi
):
    return _sig_slope(
        logsm, mzr_t0_y0, mzr_t0_x0, mzr_t0_k, mzr_t0_slope_lo, mzr_t0_slope_hi
    )


@jjit
def maiolino08_metallicity_evolution(logsm, logm0, k0):
    x = logsm - logm0
    xsq = x * x
    return -12.0 - 0.0864 * xsq + k0


@jjit
def _delta_logz_vs_t_lookback(t_lookback, y_at_tc, tc, early_time_slope, k):
    late_time_slope = y_at_tc / tc
    args = y_at_tc, tc, k, late_time_slope, early_time_slope
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


@jjit
def _fill_empty_weights_singlegal(lgmet, lgmetbin_edges, weights):
    zmsk = jnp.all(weights == 0, axis=0)
    lomsk = lgmet < lgmetbin_edges[0]
    himsk = lgmet > lgmetbin_edges[-1]

    lores = jnp.zeros(lgmetbin_edges.size - 1)
    hires = jnp.zeros(lgmetbin_edges.size - 1)

    lores = jops.index_update(lores, jops.index[0], 1.0)
    hires = jops.index_update(hires, jops.index[-1], 1.0)

    weights = jnp.where(zmsk & lomsk, lores, weights)
    weights = jnp.where(zmsk & himsk, hires, weights)
    return weights


@jjit
def _get_met_weights_singlegal(lgmet, lgmet_scatter, lgmetbin_edges):
    tw_hist_results = triweighted_histogram(lgmet, lgmet_scatter, lgmetbin_edges)

    tw_hist_results_sum = jnp.sum(tw_hist_results, axis=0)

    zmsk = tw_hist_results_sum == 0
    tw_hist_results_sum = jnp.where(zmsk, 1.0, tw_hist_results_sum)
    weights = tw_hist_results / tw_hist_results_sum

    return _fill_empty_weights_singlegal(lgmet, lgmetbin_edges, weights)


@jjit
def calc_lgmet_weights_from_logsm_table(
    lgt_obs,
    lgmet_bin_mids,
    lgt_table,
    logsm_table,
    met_params,
):
    lgmet_bin_edges = _get_bin_edges(lgmet_bin_mids, LGMET_LO, LGMET_HI)
    lgmet_scatter = met_params[-1]

    logsm_at_t_obs = jnp.interp(lgt_obs, lgt_table, logsm_table)
    lgmet = mzr_model(logsm_at_t_obs, 10 ** lgt_obs, *met_params[:-1])
    lgmet_weights = _get_met_weights_singlegal(lgmet, lgmet_scatter, lgmet_bin_edges)
    return lgmet_weights


@jjit
def calc_const_lgmet_weights(lgmet, lgmet_bin_mids, lgmet_scatter):
    lgmet_bin_edges = _get_bin_edges(lgmet_bin_mids, LGMET_LO, LGMET_HI)
    lgmet_weights = _get_met_weights_singlegal(lgmet, lgmet_scatter, lgmet_bin_edges)
    return lgmet_weights


_a = (0, None, None)
_get_met_weights_singlegal_vmap = jjit(vmap(_get_met_weights_singlegal, in_axes=_a))


@jjit
def _get_age_correlated_met_weights(
    lg_ages_gyr, lgmet_young, lgmet_old, lgmet_scatter, lgmet_bin_edges
):
    lg_ages_yr = lg_ages_gyr + 9
    lgmet = _tw_sigmoid(lg_ages_yr, LGAGE_CRIT_YR, LGAGE_CRIT_H, lgmet_young, lgmet_old)
    weights = _get_met_weights_singlegal_vmap(lgmet, lgmet_scatter, lgmet_bin_edges)
    return weights
