"""
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap
from jax import lax
from diffmah.individual_halo_assembly import _calc_halo_history
from .utils import _jax_get_dt_array

FB = 0.156
TODAY = 13.8
LGT0 = jnp.log10(TODAY)
T_SFH_MIN = 0.1

DEFAULT_MAH_PARAMS = OrderedDict(
    mah_logt0=LGT0,
    mah_logm0=12.0,
    mah_logtc=0.8,
    mah_k=3.5,
    mah_early=1.5,
    mah_late=0.5,
)

DEFAULT_MS_PARAMS = OrderedDict(
    ms_lgmcrit=12.0,
    ms_lgy_at_mcrit=-1.0,
    ms_indx_k=9.0,
    ms_indx_lo=2.0,
    ms_indx_hi=-1.0,
    ms_floor_low=1.1,
    ms_tau_dep=2.3,
)

DEFAULT_Q_PARAMS = OrderedDict(lg_qt=0.9, lg_qs=-0.5, lg_drop=-2.0, lg_rejuv=-2.0)


MS_PARAM_BOUNDS = OrderedDict(
    ms_lgmcrit=(9.0, 13.5),
    ms_lgy_at_mcrit=(-3.0, 0.5),
    ms_indx_k=(1.0, 15.0),
    ms_indx_lo=(0.0, 5.0),
    ms_indx_hi=(-2.0, 0.0),
    ms_floor_low=(0.5, 3.0),
    ms_tau_dep=(0.0, 10.0),
)
Q_PARAM_BOUNDS = OrderedDict(
    lg_qt=(0.1, 2.0), lg_qs=(-3.0, -0.01), lg_drop=(-3, 0.0), lg_rejuv=(-3, 0.0)
)


@jjit
def diffstar_sfh(t, mah_params, ms_params, q_params):
    lgt = jnp.log10(t)
    dt = _jax_get_dt_array(t)
    logt0, logmp, logtc, mah_k, early, late = mah_params
    dmhdt, log_mah = _calc_halo_history(lgt, logt0, logmp, logtc, mah_k, early, late)

    lgmcrit, lgy_at_mcrit, indx_k, indx_lo, indx_hi, floor_low, tau_dep = ms_params

    efficiency = _sfr_eff_plaw(
        lgt, log_mah, lgmcrit, lgy_at_mcrit, indx_k, indx_lo, indx_hi, floor_low
    )

    lag_matrix = _lag_kern(t, t, dt, tau_dep)
    lag_matrix_inst = jnp.identity(len(lgt)) / dt
    tau_w = jnp.where(tau_dep > 5.0 * jnp.mean(dt), 1.0, 0.0)
    lag_matrix = tau_w * lag_matrix + (1.0 - tau_w) * lag_matrix_inst

    main_sequence_sfr = FB * efficiency * dmhdt

    integrand = main_sequence_sfr * lag_matrix * dt
    lagged_sfr = jnp.sum(integrand, axis=1)

    lg_qt, lg_qs, lg_drop, lg_rejuv = q_params
    qfrac = _quenching_kern_rejuv(lgt, lg_qt, lg_qs, lg_drop, lg_rejuv)

    sfr = qfrac * lagged_sfr
    return sfr


@jjit
def _quenching_kern_rejuv(lgt, lg_qt, lg_qs, lg_drop, lg_rejuv):
    qs = 10 ** lg_qs
    return 10 ** rejuvenated_quenching_func(lgt, lg_qt, qs, lg_drop, lg_rejuv)


@jjit
def rejuvenated_quenching_func(t, t_q, q_dt, q_drop, q_rejuv):
    qs = q_dt / 12
    f2 = q_drop - q_rejuv
    return _jax_partial_u_tw_kern(t, t_q, qs, q_drop, f2)


@jjit
def _jax_partial_u_tw_kern(x, m, h, f1, f2):
    y = (x - m) / h
    z = f1 * _jax_tw(y + 3)
    w = f1 - f2 * _jax_tw(y - 3)
    return jnp.where(y < 0, z, w)


@jjit
def _jax_tw(y):
    v = -5 * y ** 7 / 69984 + 7 * y ** 5 / 2592 - 35 * y ** 3 / 864 + 35 * y / 96 + 0.5
    res = jnp.where(y < -3, 0, v)
    res = jnp.where(y > 3, 1, res)
    return res


@jjit
def _sfr_eff_plaw(
    lgt,
    lgm,
    lgmcrit,
    lgy_at_mcrit,
    indx_k,
    indx_lo,
    indx_hi,
    floor_low,
):
    slope = _sigmoid(lgm, lgmcrit, indx_k, indx_lo, indx_hi)
    eff = lgy_at_mcrit + slope * (lgm - lgmcrit)
    eff_floor = sigmoid_poly(lgm, lgmcrit, 10.0, lgy_at_mcrit - floor_low, -5)
    return 10 ** eff + 10 ** eff_floor


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    return ymin + (ymax - ymin) / (1 + jnp.exp(-k * (x - x0)))


@jjit
def sigmoid_poly(x, x0, k, ymin, ymax):
    arg = k * (x - x0)
    body = 0.5 * arg / jnp.sqrt(1 + arg ** 2) + 0.5
    return ymin + (ymax - ymin) * body


@jjit
def _gas_conversion_kern(t, t_acc, dt, tau_dep):

    w = tau_dep / 3.0
    m = t_acc

    tri_kern = lax.cond(
        t < t_acc,
        lambda x: 0.0,
        lambda x: lax.cond(
            x == t_acc,
            lambda x: tw_bin_jax_kern(m, w, x - dt / 2.0, x + dt / 2.0) / dt,
            lambda x: 2.0 * tw_bin_jax_kern(m, w, x - dt / 2.0, x + dt / 2.0) / dt,
            x,
        ),
        t,
    )

    return tri_kern


@jjit
def tw_bin_jax_kern(m, h, L, H):
    return tw_cuml_jax_kern(H, m, h) - tw_cuml_jax_kern(L, m, h)


@jjit
def tw_cuml_jax_kern(x, m, h):
    y = (x - m) / h
    return lax.cond(
        y < -3,
        lambda x: 0.0,
        lambda x: lax.cond(
            x > 3,
            lambda xx: 1.0,
            lambda xx: (
                -5 * xx ** 7 / 69984
                + 7 * xx ** 5 / 2592
                - 35 * xx ** 3 / 864
                + 35 * xx / 96
                + 1 / 2
            ),
            x,
        ),
        y,
    )


_a, _b = (0, None, 0, None), (None, 0, None, None)
_lag_kern = jjit(vmap(vmap(_gas_conversion_kern, in_axes=_b), in_axes=_a))
