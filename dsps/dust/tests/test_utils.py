"""
"""
import numpy as np
from jax.scipy.stats import norm
from ..utils import get_filter_effective_wavelength


def test_get_filter_effective_wavelength_tophat_transmission_curve():
    wave = np.linspace(0, 10, 5_000)
    trans = np.zeros_like(wave)
    msk = (wave > 4) & (wave < 6)
    trans[msk] = 1.0
    redshift = 0.0
    lambda_eff = get_filter_effective_wavelength(wave, trans, redshift)
    assert np.allclose(lambda_eff, 5.0, rtol=0.001)


def test_get_filter_effective_wavelength_gaussian_transmission_curve():
    wave = np.linspace(100, 1_000, 5_000)

    mu_rest, std = 500, 50
    trans = norm.pdf(wave, mu_rest, std)

    redshift = 0.0
    lambda_eff = get_filter_effective_wavelength(wave, trans, redshift)
    assert np.allclose(mu_rest, lambda_eff, atol=0.01)

    redshift = 1.0
    lambda_eff = get_filter_effective_wavelength(wave, trans, redshift)
    mu_eff_correct = mu_rest / (1 + redshift)
    assert np.allclose(mu_eff_correct, lambda_eff, atol=0.01)
