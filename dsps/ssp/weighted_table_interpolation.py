"""
"""
from jax import numpy as jnp
from jax import jit as jjit


@jjit
def _calc_weights(x, x_table):
    n_table = x_table.size
    lgt_interp = jnp.interp(x, x_table, jnp.arange(0, n_table))
    it_lo = jnp.floor(lgt_interp).astype("i4")
    it_hi = it_lo + 1
    weight_hi = lgt_interp - it_lo
    weight_lo = 1 - weight_hi
    it_hi = jnp.where(it_hi > n_table - 1, n_table - 1, it_hi)
    return (it_lo, weight_lo), (it_hi, weight_hi)


@jjit
def _calc_weighted_table(x, x_table, y_table):
    (it_lo, weight_lo), (it_hi, weight_hi) = _calc_weights(x, x_table)
    return weight_lo * y_table[it_lo] + weight_hi * y_table[it_hi]


@jjit
def _calc_2d_weighted_table(x, y, x_table, y_table, z_table):
    (it_xlo, weight_xlo), (it_xhi, weight_xhi) = _calc_weights(x, x_table)
    (it_ylo, weight_ylo), (it_yhi, weight_yhi) = _calc_weights(y, y_table)

    z_xlo_ylo = z_table[it_xlo, it_ylo, :] * weight_xlo * weight_ylo
    z_xlo_yhi = z_table[it_xlo, it_yhi, :] * weight_xlo * weight_yhi
    z_xhi_ylo = z_table[it_xhi, it_ylo, :] * weight_xhi * weight_ylo
    z_xhi_yhi = z_table[it_xhi, it_yhi, :] * weight_xhi * weight_yhi

    return z_xlo_ylo + z_xlo_yhi + z_xhi_ylo * z_xhi_yhi
