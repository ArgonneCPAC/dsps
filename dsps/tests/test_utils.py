"""
"""
import numpy as np
from jax import random as jran
from ..utils import triweighted_histogram, _get_bin_edges, _get_triweights_singlepoint


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
    bin_mids = np.arange(0, 10)
    lowest_bin_edge = -5
    highest_bin_edge = 100
    bin_edges = _get_bin_edges(bin_mids, lowest_bin_edge, highest_bin_edge)
    assert bin_edges.shape == (bin_mids.size + 1,)
    assert bin_edges[0] == lowest_bin_edge
    assert bin_edges[-1] == highest_bin_edge
