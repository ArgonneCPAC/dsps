"""
"""
import numpy as np
from ..mzr import _get_met_weights_singlegal, MAIOLINO08_PARAMS
from ..mzr import MZR_VS_T_PARAMS, mzr_evolution_model
from ..mzr import maiolino08_metallicity_evolution as m08_zevol
from ..utils import _get_bin_edges
from ..flat_wcdm import PLANCK15, _lookback_time


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


def test_mzr_fit_agreement_with_maiolino08():
    ztest = np.array(list(MAIOLINO08_PARAMS.keys())[1:-1])
    cosmic_time = 13.8 - _lookback_time(ztest, *PLANCK15)
    lgsmarr_fit = np.linspace(9, 11, 50)
    m08_at_z0 = m08_zevol(lgsmarr_fit, *MAIOLINO08_PARAMS[0.07])

    for i, t in enumerate(cosmic_time):
        logZ_reduction = mzr_evolution_model(lgsmarr_fit, t, *MZR_VS_T_PARAMS.values())
        m08_at_z = m08_zevol(lgsmarr_fit, *MAIOLINO08_PARAMS[ztest[i]])
        logZ_reduction_correct = m08_at_z - m08_at_z0
        assert np.allclose(logZ_reduction, logZ_reduction_correct, atol=0.02)
