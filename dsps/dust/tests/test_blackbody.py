"""
"""
import numpy as np
from .. import blackbody as jbb
import warnings
import pytest

try:
    from astropy.modeling.models import BlackBody
    from astropy import units as u
    from astropy import constants as const

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

NO_ASTROPY_MSG = "Must have astropy installed to run this unit test"


@pytest.mark.skipif(not HAS_ASTROPY, reason=NO_ASTROPY_MSG)
def test_jax_blackbody_freq_density_unit_conversion():
    wave_m = np.logspace(-5, -2.3, 50) * u.m
    freq_hz = (const.c / wave_m).to(u.Hz)

    u_freq_density_lsun = u.Lsun / u.Hz / u.sr / u.m**2
    temp_test = [1, 5, 500, 5000, 50_000, 500_000]
    for temp in temp_test:
        with warnings.catch_warnings(record=True):
            bb_freq_density_astropy_si = BlackBody(temperature=temp * u.K)(wave_m)

        bb_freq_density_astropy_lsun = bb_freq_density_astropy_si.to(
            u_freq_density_lsun
        )
        bb_freq_density_jax_lsun = jbb.blackbody_freq_density(freq_hz, temp)
        assert np.allclose(bb_freq_density_jax_lsun, bb_freq_density_astropy_lsun.value)


@pytest.mark.skipif(not HAS_ASTROPY, reason=NO_ASTROPY_MSG)
def test_jax_blackbody_freq_density_agrees_with_astropy_in_si_units():
    wave_m = np.logspace(-5, -2.3, 50) * u.m
    freq_hz = (const.c / wave_m).to(u.Hz)

    u_freq_density_si = u.Watt / u.Hz / u.sr / u.m**2
    temp_test = [1, 5, 500, 5000, 50_000, 500_000]
    for temp in temp_test:
        with warnings.catch_warnings(record=True):
            bb_freq_density_astropy_si = BlackBody(temperature=temp * u.K)(wave_m)
        bb_freq_density_astropy_si = bb_freq_density_astropy_si.to(u_freq_density_si)
        bb_freq_density_jax_si = jbb._blackbody_freq_density_si(freq_hz, temp)
        assert np.allclose(bb_freq_density_jax_si, bb_freq_density_astropy_si.value)
        assert np.all(np.isfinite(bb_freq_density_jax_si))


@pytest.mark.skipif(not HAS_ASTROPY, reason=NO_ASTROPY_MSG)
def test_jax_blackbody_freq_density_vs_wave_density_are_consistent_in_si_units():
    """Enforce ν*L_ν = λ*L_λ"""
    wave_m = np.logspace(-5, -2.3, 50) * u.m
    freq_hz = (const.c / wave_m).to(u.Hz)

    factor = wave_m * wave_m / const.c.value

    temp_test = [1, 5, 500, 5000, 50_000, 500_000]
    for temp in temp_test:
        bb_freq_density_jax_si = jbb._blackbody_freq_density_si(freq_hz, temp)
        bb_wave_density_jax_si = jbb._blackbody_wave_density_si(wave_m, temp)

        assert np.allclose(bb_freq_density_jax_si, bb_wave_density_jax_si * factor)


@pytest.mark.skipif(not HAS_ASTROPY, reason=NO_ASTROPY_MSG)
def test_jax_blackbody_wave_density_units():
    """Enforce agreement between SI and code units for wave density"""
    wave_m = np.logspace(-5, -2.3, 50) * u.m
    wave_aa = wave_m.to(u.angstrom)

    u_wave_density_si = u.Watt / u.m / u.sr / u.m**2
    code_units = u.Lsun / u.angstrom / u.sr / u.m**2

    temp_test = [1, 5, 500, 5000, 50_000, 500_000]
    for temp in temp_test:
        bb_wave_density_jax_si = jbb._blackbody_wave_density_si(wave_m, temp)

        bb_wave_density_si = bb_wave_density_jax_si * u_wave_density_si
        bb_wave_density_astropy_converted = bb_wave_density_si.to(code_units)

        bb_wave_density_jax = jbb.blackbody_wave_density(wave_aa, temp)
        assert np.allclose(bb_wave_density_jax, bb_wave_density_astropy_converted.value)
