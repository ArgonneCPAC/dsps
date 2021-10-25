"""
"""
import numpy as np
from ..metallicity import _get_met_weights_singlegal
from ..utils import _get_bin_edges


def test_get_met_weights():
    n_bins = 22
    _lgzbin_mids = np.linspace(-3.7, -1.523, n_bins)
    lgzsunbin_mids = _lgzbin_mids - _lgzbin_mids[-3]
    lgzsunbins = _get_bin_edges(lgzsunbin_mids, -100.0, 100.0)

    lgz_scatter = 0.25
    ngals = 200
    lgzdata = np.linspace(-2.5, 0.5, ngals)

    lgzdata[:2] = -500
    lgzdata[-2:] = 500

    lgz = -0.5

    weights = _get_met_weights_singlegal(lgz, lgz_scatter, lgzsunbins)
    assert weights.shape == (n_bins,)
