"""
"""
import numpy as np
from ..mzr import MAIOLINO08_PARAMS
from ..mzr import MZR_VS_T_PDICT, mzr_evolution_model
from ..mzr import maiolino08_metallicity_evolution as m08_zevol
from ...cosmology.flat_wcdm import PLANCK15, lookback_time


def test_mzr_fit_agreement_with_maiolino08():
    ztest = np.array(list(MAIOLINO08_PARAMS.keys())[1:-1])
    cosmic_time = 13.8 - lookback_time(ztest, *PLANCK15)
    lgsmarr_fit = np.linspace(9, 11, 50)
    m08_at_z0 = m08_zevol(lgsmarr_fit, *MAIOLINO08_PARAMS[0.07])

    for i, t in enumerate(cosmic_time):
        logZ_reduction = mzr_evolution_model(lgsmarr_fit, t, *MZR_VS_T_PDICT.values())
        m08_at_z = m08_zevol(lgsmarr_fit, *MAIOLINO08_PARAMS[ztest[i]])
        logZ_reduction_correct = m08_at_z - m08_at_z0
        assert np.allclose(logZ_reduction, logZ_reduction_correct, atol=0.02)
