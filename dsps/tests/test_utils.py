"""
"""
import numpy as np
from ..utils import triweighted_histogram, _get_bin_edges


def test_triweighted_histogram():
    n, n_bins = 5, 10
    xbins = np.arange(0, n_bins)
    x = np.linspace(0, 1, n)
    sig = 0.1
    hist = triweighted_histogram(x, sig, xbins)
    assert hist.shape == (n_bins - 1, n)


def test_get_bin_edges():
    bin_mids = np.arange(0, 10)
    lowest_bin_edge = -5
    highest_bin_edge = 100
    bin_edges = _get_bin_edges(bin_mids, lowest_bin_edge, highest_bin_edge)
    assert bin_edges.shape == (bin_mids.size + 1,)
    assert bin_edges[0] == lowest_bin_edge
    assert bin_edges[-1] == highest_bin_edge
