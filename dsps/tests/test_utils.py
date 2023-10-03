"""
"""
import numpy as np
from jax import random as jran

from ..constants import T_TABLE_MIN
from ..utils import (
    _get_bin_edges,
    _get_triweights_singlepoint,
    _get_weight_matrices_2d,
    _get_weight_matrices_3d,
    _inverse_sigmoid,
    _mult_2d_vmap,
    _mult_3d_vmap,
    _sigmoid,
    cumtrapz,
    cumulative_mstar_formed,
    powerlaw_pdf,
    powerlaw_rvs,
    triweighted_histogram,
)


def test_sigmoid_inversion():
    xarr = np.linspace(-100, 100, 500)

    x0, k, ylo, yhi = 0, 0.1, -5, 5
    y = _sigmoid(xarr, x0, k, ylo, yhi)
    x2 = _inverse_sigmoid(y, x0, k, ylo, yhi)
    assert np.all(~np.isnan(x2))
    assert np.allclose(xarr, x2, rtol=1e-2)


def test_triweighted_histogram():
    n, n_bins = 5, 10
    xbins = np.arange(0, n_bins)
    x = np.linspace(0, 1, n)
    sig = 0.1
    hist = triweighted_histogram(x, sig, xbins)
    assert hist.shape == (n_bins - 1, n)


def test_get_triweights_singlepoint_x_below_lowest_bin_edge():
    x = -10
    sig = 0.1
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins)
    weights = _get_triweights_singlepoint(x, sig, bin_edges)
    assert weights.shape == (n_bins - 1,)
    assert weights[0] == 1.0
    assert np.allclose(weights[1:], 0)


def test_get_triweights_singlepoint_x_above_highest_bin_edge():
    x = 10
    sig = 0.1
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins)
    weights = _get_triweights_singlepoint(x, sig, bin_edges)
    assert weights.shape == (n_bins - 1,)
    assert weights[-1] == 1.0
    assert np.allclose(weights[:-1], 0)


def test_get_triweights_singlepoint_x_correctly_normalized():
    ran_key = jran.PRNGKey(0)
    n_bins = 4
    n_tests = 1000
    for itest in range(n_tests):
        ran_key, x_key, sig_key, edges_key = jran.split(ran_key, 4)
        x = jran.uniform(x_key, minval=-1, maxval=2)
        sig = jran.uniform(sig_key, minval=1, maxval=20)
        bin_edges = jran.uniform(edges_key, minval=-1, maxval=2, shape=(n_bins,))
        weights = _get_triweights_singlepoint(x, sig, bin_edges)
        assert weights.shape == (n_bins - 1,)
        assert np.allclose(weights.sum(), 1.0, atol=0.001)


def test_get_bin_edges():
    bin_mids = np.linspace(0, 9, 10)
    lowest_bin_edge = -5
    highest_bin_edge = 100
    bin_edges = _get_bin_edges(bin_mids, lowest_bin_edge, highest_bin_edge)
    assert bin_edges.shape == (bin_mids.size + 1,)
    assert bin_edges[0] == lowest_bin_edge
    assert bin_edges[-1] == highest_bin_edge


def test_weight_matrix_kernels_2d():
    ngals = 25
    n1, n2 = 10, 15
    w1arr = np.random.uniform(0, 1, n1)
    w2arr = np.random.uniform(0, 1, n2)
    w1arr = w1arr / w1arr.sum()
    w2arr = w2arr / w2arr.sum()

    res = _mult_2d_vmap(w1arr, w2arr)
    assert res.shape == (n1, n2)

    w1arr_pop = np.random.uniform(0, 1, size=(ngals, n1))
    w2arr_pop = np.random.uniform(0, 1, size=(ngals, n2))
    res_vmap = _get_weight_matrices_2d(w1arr_pop, w2arr_pop)
    assert res_vmap.shape == (ngals, n1, n2)


def test_weight_matrix_kernels_3d():
    ngals = 25
    n1, n2, n3 = 10, 15, 25
    w1arr = np.random.uniform(0, 1, n1)
    w2arr = np.random.uniform(0, 1, n2)
    w3arr = np.random.uniform(0, 1, n3)
    w1arr = w1arr / w1arr.sum()
    w2arr = w2arr / w2arr.sum()
    w3arr = w3arr / w3arr.sum()

    res = _mult_3d_vmap(w1arr, w2arr, w3arr)
    assert res.shape == (n1, n2, n3)

    w1arr_pop = np.random.uniform(0, 1, size=(ngals, n1))
    w2arr_pop = np.random.uniform(0, 1, size=(ngals, n2))
    w3arr_pop = np.random.uniform(0, 1, size=(ngals, n3))
    res_vmap = _get_weight_matrices_3d(w1arr_pop, w2arr_pop, w3arr_pop)
    assert res_vmap.shape == (ngals, n1, n2, n3)


def test_powerlaw_rvs():
    ran_key = jran.PRNGKey(0)
    npts = 2_000
    a = np.random.uniform(2, 3, npts)
    b = a + np.random.uniform(0, 1, npts)
    g = np.random.uniform(1, 4, npts)

    sample = powerlaw_rvs(ran_key, a, b, g)
    assert np.all(np.isfinite(sample))
    assert np.all(sample > a)
    assert np.all(sample < b)


def test_powerlaw_pdf():
    npts = 2_000
    a = np.random.uniform(2, 3, npts)
    b = a + np.random.uniform(0, 1, npts)
    g = np.random.uniform(1, 4, npts)
    x = np.random.uniform(a, b, npts)

    pdf = powerlaw_pdf(x, a, b, g)
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= 0)
    assert np.any(pdf > 0)
    a = np.random.uniform(2, 3, npts)
    b = a + np.random.uniform(0, 1, npts)
    g = np.random.uniform(1, 4, npts)
    x = np.random.uniform(a, b, npts)

    pdf = powerlaw_pdf(x, a, b, g)
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= 0)
    assert np.any(pdf > 0)


def test_cumtrapz():
    ran_key = jran.PRNGKey(0)
    n_x = 100
    n_tests = 10
    for __ in range(n_tests):
        x_key, y_key, ran_key = jran.split(ran_key, 3)
        xarr = np.sort(jran.uniform(x_key, minval=0, maxval=1, shape=(n_x,)))
        yarr = jran.uniform(y_key, minval=0, maxval=1, shape=(n_x,))
        jax_result = cumtrapz(xarr, yarr)
        np_result = [np.trapz(yarr[:-i], x=xarr[:-i]) for i in range(1, n_x)][::-1]
        assert np.allclose(jax_result[:-1], np_result, rtol=1e-4)
        assert np.allclose(jax_result[-1], np.trapz(yarr, x=xarr), rtol=1e-4)


def test_cumulative_mstar_formed():
    t_table = np.linspace(T_TABLE_MIN, 13.8, 200)
    sfh_table = np.random.uniform(0, 1, t_table.size)
    smh_table = cumulative_mstar_formed(t_table, sfh_table)
    assert smh_table.shape == t_table.shape
    assert np.all(smh_table > 0)
    assert np.all(np.diff(smh_table) > 0)
