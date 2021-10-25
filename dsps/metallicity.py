"""
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import ops as jops
from .utils import triweighted_histogram, _get_bin_edges


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

MZR_VS_T_PARAMS = OrderedDict(
    mzr_y_at_tc_x0=11.04,
    mzr_y_at_tc_k=0.52,
    mzr_y_at_tc_ylo=-0.95,
    mzr_y_at_tc_yhi=0.07,
    mzr_tc_x0=10.62,
    mzr_tc_k=0.23,
    mzr_tc_ylo=8.32,
    mzr_tc_yhi=14.93,
    mzr_early_time_slope=-0.57,
    mzr_k=0.47,
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
def _y_at_tc_vs_logsm(logsm, y_at_tc_x0, y_at_tc_k, y_at_tc_ylo, y_at_tc_yhi):
    return _sigmoid(logsm, y_at_tc_x0, y_at_tc_k, y_at_tc_ylo, y_at_tc_yhi)


@jjit
def _tc_vs_logsm(logsm, tc_x0, tc_k, tc_ylo, tc_yhi):
    return _sigmoid(logsm, tc_x0, tc_k, tc_ylo, tc_yhi)


@jjit
def _delta_logz_vs_t(t, y_at_tc, tc, early_time_slope, k):
    late_time_slope = y_at_tc / tc
    args = y_at_tc, tc, k, late_time_slope, early_time_slope
    logZ_reduction = _sig_slope(t, *args)
    return logZ_reduction


@jjit
def mzr_evolution_model(
    logsm,
    t,
    mzr_y_at_tc_x0,
    mzr_y_at_tc_k,
    mzr_y_at_tc_ylo,
    mzr_y_at_tc_yhi,
    mzr_tc_x0,
    mzr_tc_k,
    mzr_tc_ylo,
    mzr_tc_yhi,
    mzr_early_time_slope,
    mzr_k,
):
    t_lookback = TODAY - t
    mzr_y_at_tc = _y_at_tc_vs_logsm(
        logsm, mzr_y_at_tc_x0, mzr_y_at_tc_k, mzr_y_at_tc_ylo, mzr_y_at_tc_yhi
    )
    mzr_tc = _tc_vs_logsm(logsm, mzr_tc_x0, mzr_tc_k, mzr_tc_ylo, mzr_tc_yhi)
    logZ_reduction = _delta_logz_vs_t(
        t_lookback, mzr_y_at_tc, mzr_tc, mzr_early_time_slope, mzr_k
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
    mzr_y_at_tc_x0,
    mzr_y_at_tc_k,
    mzr_y_at_tc_ylo,
    mzr_y_at_tc_yhi,
    mzr_tc_x0,
    mzr_tc_k,
    mzr_tc_ylo,
    mzr_tc_yhi,
    mzr_early_time_slope,
    mzr_k,
):
    lgmet_at_t0 = mzr_model_t0(
        logsm, mzr_t0_y0, mzr_t0_x0, mzr_t0_k, mzr_t0_slope_lo, mzr_t0_slope_hi
    )
    logZ_reduction = mzr_evolution_model(
        logsm,
        t,
        mzr_y_at_tc_x0,
        mzr_y_at_tc_k,
        mzr_y_at_tc_ylo,
        mzr_y_at_tc_yhi,
        mzr_tc_x0,
        mzr_tc_k,
        mzr_tc_ylo,
        mzr_tc_yhi,
        mzr_early_time_slope,
        mzr_k,
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
def calc_lgmet_weights_from_logsm_table_single_t_birth(
    lgt_birth,
    lgmet_bin_mids,
    lgt_table,
    logsm_table,
    met_params,
):
    lgmet_bin_edges = _get_bin_edges(lgmet_bin_mids, LGMET_LO, LGMET_HI)
    lgmet_scatter = met_params[-1]

    logsm_at_t_birth = jnp.interp(lgt_birth, lgt_table, logsm_table)
    lgmet = mzr_model(logsm_at_t_birth, 10 ** lgt_birth, *met_params[:-1])
    lgmet_weights = _get_met_weights_singlegal(lgmet, lgmet_scatter, lgmet_bin_edges)
    return lgmet_weights
