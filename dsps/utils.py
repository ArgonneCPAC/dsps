"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from jax import ops as jops
from jax import vmap


@jjit
def _jax_get_dt_array(t):
    dt = jnp.zeros_like(t)
    tmids = 0.5 * (t[:-1] + t[1:])
    dtmids = jnp.diff(tmids)
    dt = jops.index_update(dt, jops.index[1:-1], dtmids)

    t_lo = t[0] - (t[1] - t[0]) / 2
    t_hi = t[-1] + dtmids[-1] / 2

    dt = jops.index_update(dt, jops.index[0], tmids[0] - t_lo)
    dt = jops.index_update(dt, jops.index[-1], t_hi - tmids[-1])
    return dt


@jjit
def _get_bin_edges(bin_mids, lowest_bin_edge, highest_bin_edge):
    """Calculate the lower and upper bounds on the array.

    Parameters
    ----------
    bin_mids : ndarray of shape (n, )

    Returns
    -------
    bin_edges : ndarray of shape (n+1, )
        Integration bounds on the bins

    """
    dbins = _jax_get_dt_array(bin_mids)

    bin_edges = jnp.zeros(dbins.size + 1)
    bin_edges = jops.index_update(bin_edges, jops.index[:-1], bin_mids - dbins / 2)

    bin_edges = jops.index_update(bin_edges, jops.index[0], lowest_bin_edge)
    bin_edges = jops.index_update(bin_edges, jops.index[-1], highest_bin_edge)

    return bin_edges


@jjit
def _tw_cuml_kern(x, m, h):
    """Triweight kernel version of an err function."""
    z = (x - m) / h
    val = (
        -5 * z ** 7 / 69984
        + 7 * z ** 5 / 2592
        - 35 * z ** 3 / 864
        + 35 * z / 96
        + 1 / 2
    )
    val = jnp.where(z < -3, 0, val)
    val = jnp.where(z > 3, 1, val)
    return val


@jjit
def _tw_sigmoid(x, x0, tw_h, ymin, ymax):
    height_diff = ymax - ymin
    body = _tw_cuml_kern(x, x0, tw_h)
    return ymin + height_diff * body


@jjit
def _get_tw_h_from_sigmoid_k(k):
    return 1 / (0.614 * k)


@jjit
def _triweighted_histogram_kernel(x, sig, lo, hi):
    """Triweight kernel integrated across the boundaries of a single bin."""
    a = _tw_cuml_kern(x, lo, sig)
    b = _tw_cuml_kern(x, hi, sig)
    return a - b


_a = [None, None, 0, 0]
_triweighted_histogram_vmap = jjit(vmap(_triweighted_histogram_kernel, in_axes=_a))


@jjit
def triweighted_histogram(x, sig, xbins):
    """Tri-weighted histogram.

    Parameters
    ----------
    x : ndarray of shape (n, )

    sig : float

    xbins : ndarray of shape (n_bins, )

    Returns
    -------
    result : ndarray of shape (n_bins, n)

    """
    return _triweighted_histogram_vmap(x, sig, xbins[:-1], xbins[1:])


@jjit
def triweight_gaussian(x, m, h):
    z = (x - m) / h
    val = 35 / 96 * (1 - (z / 3) ** 2) ** 3 / h
    msk = (z < -3) | (z > 3)
    return jnp.where(msk, 0, val)


@jjit
def interpolate_transmission_curve(wave, trans, n_out, pcut_lo=0, pcut_hi=1):
    """ """
    lowest_bin_edge = wave[0] - (wave[1] - wave[0]) / 2
    highest_bin_edge = wave[-1] + (wave[-1] - wave[-2]) / 2
    dwave = jnp.diff(_get_bin_edges(wave, lowest_bin_edge, highest_bin_edge))
    cuml = jnp.cumsum(dwave * trans)
    cuml = cuml / cuml[-1]

    msk = cuml >= pcut_lo
    msk &= cuml <= pcut_hi

    wave_lo, wave_hi = dwave[msk][0], dwave[msk][-1]
    wave_out = jnp.linspace(wave_lo, wave_hi, n_out)
    trans_out = jnp.jnp.interp(wave_out, wave, trans)

    return wave_out, trans_out


@jjit
def _fill_empty_weights_singlepoint(x, bin_edges, weights):
    zmsk = jnp.all(weights == 0, axis=0)
    lomsk = x < bin_edges[0]
    himsk = x > bin_edges[-1]

    lores = jnp.zeros(bin_edges.size - 1)
    hires = jnp.zeros(bin_edges.size - 1)

    lores = jops.index_update(lores, jops.index[0], 1.0)
    hires = jops.index_update(hires, jops.index[-1], 1.0)

    weights = jnp.where(zmsk & lomsk, lores, weights)
    weights = jnp.where(zmsk & himsk, hires, weights)
    return weights


@jjit
def _get_triweights_singlepoint(x, sig, bin_edges):
    tw_hist_results = triweighted_histogram(x, sig, bin_edges)

    tw_hist_results_sum = jnp.sum(tw_hist_results, axis=0)

    zmsk = tw_hist_results_sum == 0
    tw_hist_results_sum = jnp.where(zmsk, 1.0, tw_hist_results_sum)
    weights = tw_hist_results / tw_hist_results_sum

    return _fill_empty_weights_singlepoint(x, bin_edges, weights)


@jjit
def _sigmoid(x, x0, k, ylo, yhi):
    height_diff = yhi - ylo
    return ylo + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _sig_slope(x, y0, x0, slope_k, lo, hi):
    slope = _sigmoid(x, x0, slope_k, lo, hi)
    return y0 + slope * (x - x0)
