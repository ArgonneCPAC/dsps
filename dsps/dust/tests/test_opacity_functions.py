"""
"""
import numpy as np
from ..opacity_functions import rolling_plaw_opacity, _dust_opacity_cowley


def test_rolling_plaw_opacity_approximates_cowley():
    wave_micron = np.logspace(1, 4, 5000)
    kappa_cowley = _dust_opacity_cowley(wave_micron)
    kappa_aph = rolling_plaw_opacity(wave_micron)
    assert np.allclose(kappa_cowley, kappa_aph, rtol=0.1)
