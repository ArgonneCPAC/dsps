"""
"""
import numpy as np

from ...cosmology.flat_wcdm import PLANCK15, lookback_time
from ..mzr import MAIOLINO08_PARAMS, MZR_VS_T_PDICT
from ..mzr import maiolino08_metallicity_evolution as m08_zevol
from ..mzr import mzr_evolution_model


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


def test_default_mzr_params_are_importable_from_metallicity_defaults_module():
    from ...metallicity.defaults import DEFAULT_MET_PARAMS
    from ...metallicity.mzr import DEFAULT_MET_PDICT

    assert np.allclose(list(DEFAULT_MET_PDICT.values()), DEFAULT_MET_PARAMS)
