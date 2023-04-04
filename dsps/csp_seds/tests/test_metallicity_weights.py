"""
"""
import numpy as np
from ...utils import _get_bin_edges
from ..metallicity_weights import _get_lgmet_weights_singlegal


def test_get_lgmet_weights_singlegal():
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

    weights = _get_lgmet_weights_singlegal(lgz, lgz_scatter, lgzsunbins)
    assert weights.shape == (n_bins,)
