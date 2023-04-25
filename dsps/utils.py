"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap
from jax import random as jran


@jjit
def _jax_get_dt_array(t):
    dt = jnp.zeros_like(t)
    tmids = 0.5 * (t[:-1] + t[1:])
    dtmids = jnp.diff(tmids)

    dt = dt.at[1:-1].set(dtmids)

    t_lo = t[0] - (t[1] - t[0]) / 2
    t_hi = t[-1] + dtmids[-1] / 2

    dt = dt.at[0].set(tmids[0] - t_lo)
    dt = dt.at[-1].set(t_hi - tmids[-1])
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
    bin_edges = bin_edges.at[:-1].set(bin_mids - dbins / 2)

    bin_edges = bin_edges.at[0].set(lowest_bin_edge)
    bin_edges = bin_edges.at[-1].set(highest_bin_edge)

    return bin_edges


@jjit
def _tw_cuml_kern(x, m, h):
    """Triweight kernel version of an err function."""
    z = (x - m) / h
    val = (
        -5 * z**7 / 69984
        + 7 * z**5 / 2592
        - 35 * z**3 / 864
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
def _tw_sig_slope(x, xtp, ytp, x0, tw_h, lo, hi):
    slope = _tw_sigmoid(x, x0, tw_h, lo, hi)
    return ytp + slope * (x - xtp)


@jjit
def _fill_empty_weights_singlepoint(x, bin_edges, weights):
    zmsk = jnp.all(weights == 0, axis=0)
    lomsk = x < bin_edges[0]
    himsk = x > bin_edges[-1]

    lores = jnp.zeros(bin_edges.size - 1)
    hires = jnp.zeros(bin_edges.size - 1)

    lores = lores.at[0].set(1.0)
    hires = hires.at[-1].set(1.0)

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
def _inverse_sigmoid(y, x0, k, ylo, yhi):
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - jnp.log(lnarg) / k


@jjit
def _sig_slope(x, xtp, ytp, x0, slope_k, lo, hi):
    slope = _sigmoid(x, x0, slope_k, lo, hi)
    return ytp + slope * (x - xtp)


@jjit
def _mult_2d(w1, w2):
    return w1 * w2


@jjit
def _mult_3d(w1, w2, w3):
    return w1 * w2 * w3


_mult_2d_vmap = jjit(vmap(vmap(_mult_2d, in_axes=[None, 0]), in_axes=[0, None]))
_get_weight_matrices_2d = jjit(vmap(_mult_2d_vmap, in_axes=[0, 0]))


_mult_3d_vmap = jjit(
    vmap(
        vmap(vmap(_mult_3d, in_axes=[None, None, 0]), in_axes=[None, 0, None]),
        in_axes=[0, None, None],
    )
)
_get_weight_matrices_3d = jjit(vmap(_mult_3d_vmap, in_axes=[0, 0, 0]))


@jjit
def powerlaw_pdf(x, a, b, g):
    """pdf(x) propto x^{g-1}. Assumes a<b and g!=0

    Parameters
    ----------
    x : ndarray of shape (n, )
        Points at which to evaluate the powerlaw PDF

    a : ndarray of shape (n, )
        Lower bound on each powerlaw

    b : ndarray of shape (n, )
        Upper bound on each powerlaw

    g : ndarray of shape (n, )
        Index for each powerlaw

    Returns
    -------
    pdf : ndarray of shape (n, )
        Value of the PDF for each powerlaw

    """
    ag, bg = a**g, b**g
    return g * x ** (g - 1) / (bg - ag)


@jjit
def powerlaw_rvs(ran_key, a, b, g):
    """Power-law gen for pdf(x) propto x^{g-1} for a<=x<=b. Assumes a<b and g!=0

    Parameters
    ----------
    ran_key : jax.random.PRNGKey

    a : ndarray of shape (n, )
        Lower bound on each powerlaw

    b : ndarray of shape (n, )
        Upper bound on each powerlaw

    g : ndarray of shape (n, )
        Index for each powerlaw

    Returns
    -------
    y : ndarray of shape (n, )
        Monte Carlo realization of the input powerlaws

    """
    npts = a.shape[0]
    r = jran.uniform(ran_key, (npts,))
    ag, bg = a**g, b**g
    return (ag + (bg - ag) * r) ** (1.0 / g)
