"""
"""
import numpy as np
from ..flat_wcdm import _distance_modulus, _angular_diameter_distance
from ..flat_wcdm import _lookback_time, _Om, _delta_vir, _virial_dynamical_time

try:
    from astropy.cosmology import Planck15, WMAP5, Flatw0waCDM

    LCDM_COSMO_LIST = Planck15, WMAP5
except ImportError:
    LCDM_COSMO_LIST = []


def test_flat_lcdm():
    zray = np.linspace(0.001, 10, 500)
    for cosmo in LCDM_COSMO_LIST:
        params = cosmo.Om0, cosmo.Ode0, -1.0, 0.0, cosmo.h

        dmod_astropy = cosmo.distmod(zray).value
        dmod_jax = _distance_modulus(zray, *params)
        assert np.allclose(dmod_astropy, dmod_jax, rtol=0.001)

        angdist_astropy = cosmo.angular_diameter_distance(zray).value
        angdist_jax = _angular_diameter_distance(zray, *params)
        assert np.allclose(angdist_astropy, angdist_jax, rtol=0.005)

        lookback_astropy = cosmo.lookback_time(zray).value
        lookback_jax = _lookback_time(zray, *params)
        assert np.allclose(lookback_astropy, lookback_jax, rtol=0.005)

        om_astropy = cosmo.Om(zray)
        om_jax = _Om(zray, *params[:-1])
        assert np.allclose(om_astropy, om_jax, rtol=0.01)


def test_flat_wcdm_distances():
    n_test = 10
    zray = np.linspace(0.001, 10, 50)
    for lcdm_cosmo in LCDM_COSMO_LIST:
        for itest in range(n_test):
            w0 = np.random.uniform(-1.5, -0.5)
            wa = np.random.uniform(-0.5, 0.5)
            w_cosmo = Flatw0waCDM(lcdm_cosmo.H0, lcdm_cosmo.Om0, w0=w0, wa=wa)
            params = w_cosmo.Om0, w_cosmo.Ode0, w0, wa, w_cosmo.h

            dmod_astropy = w_cosmo.distmod(zray).value
            dmod_jax = _distance_modulus(zray, *params)
            assert np.allclose(dmod_astropy, dmod_jax, rtol=0.001)

            angdist_astropy = w_cosmo.angular_diameter_distance(zray).value
            angdist_jax = _angular_diameter_distance(zray, *params)
            assert np.allclose(angdist_astropy, angdist_jax, rtol=0.001)

            lookback_astropy = w_cosmo.lookback_time(zray).value
            lookback_jax = _lookback_time(zray, *params)
            assert np.allclose(lookback_astropy, lookback_jax, rtol=0.001)

            om_astropy = w_cosmo.Om(zray)
            om_jax = _Om(zray, *params[:-1])
            assert np.allclose(om_astropy, om_jax, rtol=0.001)


def test_delta_vir():
    """Enforce delta~178 at high-z and monotonically decreases thereafter"""
    zray = np.linspace(0.001, 10, 50)
    for cosmo in LCDM_COSMO_LIST:
        params = cosmo.Om0, cosmo.Ode0, -1.0, 0.0, cosmo.h
        z_high = np.atleast_1d(500.0)
        delta = float(_delta_vir(z_high, *params[:-1]))
        assert np.allclose(delta, 178, atol=1)
        delta_ray = _delta_vir(zray, *params[:-1])
        assert np.all(np.diff(delta_ray) > 0)


def test_dynamical_time():
    """Enforce dynamical times increase as dark energy becomes operative"""
    zray = np.linspace(0.001, 10, 50)
    for cosmo in LCDM_COSMO_LIST:
        params = cosmo.Om0, cosmo.Ode0, -1.0, 0.0, cosmo.h
        tcross = _virial_dynamical_time(zray, *params)
        assert np.all(np.diff(tcross) < 0)

        z0 = np.atleast_1d(0.0)
        tcross_z0 = float(_virial_dynamical_time(z0, *params))
        assert np.allclose(tcross_z0, 4, atol=1)
