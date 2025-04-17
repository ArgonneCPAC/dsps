""" """

import numpy as np
import pytest

from .. import flat_wcdm

try:
    from astropy.cosmology import WMAP5 as WMAP5_astropy
    from astropy.cosmology import Flatw0waCDM
    from astropy.cosmology import Planck15 as Planck15_astropy

    HAS_ASTROPY = True
    ASTROPY_COSMO_LIST = [Planck15_astropy, WMAP5_astropy]
    DSPS_COSMO_LIST = [flat_wcdm.PLANCK15, flat_wcdm.WMAP5]
except ImportError:
    HAS_ASTROPY = False

NO_ASTROPY_MSG = "Must have astropy installed to run this unit test"


@pytest.mark.skipif(not HAS_ASTROPY, reason=NO_ASTROPY_MSG)
def test_predefined_cosmologies():
    zray = np.linspace(0.001, 10, 500)
    gen = zip(ASTROPY_COSMO_LIST, DSPS_COSMO_LIST)
    for cosmo_pair in gen:
        cosmo_astropy, cosmo_dsps = cosmo_pair
        assert np.allclose(cosmo_astropy.Om0, cosmo_dsps.Om0, rtol=1e-3)
        assert np.allclose(cosmo_astropy.h, cosmo_dsps.h, rtol=1e-3)

        dmod_astropy = cosmo_astropy.distmod(zray).value
        dmod_jax = flat_wcdm.distance_modulus(zray, *cosmo_dsps)
        assert np.allclose(dmod_astropy, dmod_jax, rtol=0.001)

        angdist_astropy = cosmo_astropy.angular_diameter_distance(zray).value
        angdist_jax = flat_wcdm.angular_diameter_distance(zray, *cosmo_dsps)
        assert np.allclose(angdist_astropy, angdist_jax, rtol=0.005)

        lookback_astropy = cosmo_astropy.lookback_time(zray).value
        lookback_jax = flat_wcdm.lookback_time(zray, *cosmo_dsps)
        assert np.allclose(lookback_astropy, lookback_jax, rtol=0.005)

        om_astropy = cosmo_astropy.Om(zray)
        om_jax = flat_wcdm._Om(zray, *cosmo_dsps[:-1])
        assert np.allclose(om_astropy, om_jax, rtol=0.01)


@pytest.mark.skipif(not HAS_ASTROPY, reason=NO_ASTROPY_MSG)
def test_flat_lcdm():
    zray = np.linspace(0.001, 10, 500)
    for cosmo in ASTROPY_COSMO_LIST:
        cosmo_dsps = flat_wcdm.CosmoParams(cosmo.Om0, -1, 0, cosmo.h)
        dmod_astropy = cosmo.distmod(zray).value
        dmod_jax = flat_wcdm.distance_modulus(zray, *cosmo_dsps)
        assert np.allclose(dmod_astropy, dmod_jax, rtol=0.001)

        angdist_astropy = cosmo.angular_diameter_distance(zray).value
        angdist_jax = flat_wcdm.angular_diameter_distance(zray, *cosmo_dsps)
        assert np.allclose(angdist_astropy, angdist_jax, rtol=0.005)

        lookback_astropy = cosmo.lookback_time(zray).value
        lookback_jax = flat_wcdm.lookback_time(zray, *cosmo_dsps)
        assert np.allclose(lookback_astropy, lookback_jax, rtol=0.005)

        om_astropy = cosmo.Om(zray)
        om_jax = flat_wcdm._Om(zray, *cosmo_dsps[:-1])
        assert np.allclose(om_astropy, om_jax, rtol=0.01)


@pytest.mark.skipif(not HAS_ASTROPY, reason=NO_ASTROPY_MSG)
def test_flat_wcdm_distances():
    n_test = 10
    zray = np.linspace(0.001, 10, 50)
    for cosmo in ASTROPY_COSMO_LIST:
        for itest in range(n_test):
            w0 = np.random.uniform(-1.5, -0.5)
            wa = np.random.uniform(-0.5, 0.5)
            w_cosmo = Flatw0waCDM(cosmo.H0, cosmo.Om0, w0=w0, wa=wa)
            cosmo_dsps = flat_wcdm.CosmoParams(cosmo.Om0, w0, wa, cosmo.h)

            dmod_astropy = w_cosmo.distmod(zray).value

            dmod_jax = flat_wcdm.distance_modulus(zray, *cosmo_dsps)
            assert np.allclose(dmod_astropy, dmod_jax, rtol=0.001)

            angdist_astropy = w_cosmo.angular_diameter_distance(zray).value
            angdist_jax = flat_wcdm.angular_diameter_distance(zray, *cosmo_dsps)
            assert np.allclose(angdist_astropy, angdist_jax, rtol=0.001)

            lookback_astropy = w_cosmo.lookback_time(zray).value
            lookback_jax = flat_wcdm.lookback_time(zray, *cosmo_dsps)
            assert np.allclose(lookback_astropy, lookback_jax, rtol=0.001)

            om_astropy = w_cosmo.Om(zray)
            om_jax = flat_wcdm._Om(zray, *cosmo_dsps[:-1])
            assert np.allclose(om_astropy, om_jax, rtol=0.001)


@pytest.mark.skipif(not HAS_ASTROPY, reason=NO_ASTROPY_MSG)
def test_delta_vir():
    """Enforce delta~178 at high-z and monotonically decreases thereafter"""
    zray = np.linspace(0.001, 10, 50)
    for cosmo in ASTROPY_COSMO_LIST:
        cosmo_dsps = flat_wcdm.CosmoParams(cosmo.Om0, -1, 0, cosmo.h)
        z_high = np.atleast_1d(500.0)
        delta = flat_wcdm._delta_vir(z_high, *cosmo_dsps[:-1])
        assert np.allclose(delta, 178, atol=1)
        delta_ray = flat_wcdm._delta_vir(zray, *cosmo_dsps[:-1])
        assert np.all(np.diff(delta_ray) > 0)


@pytest.mark.skipif(not HAS_ASTROPY, reason=NO_ASTROPY_MSG)
def test_dynamical_time():
    """Enforce dynamical times increase as dark energy becomes operative"""
    zray = np.linspace(0.001, 10, 50)
    for cosmo in ASTROPY_COSMO_LIST:
        cosmo_dsps = flat_wcdm.CosmoParams(cosmo.Om0, -1, 0, cosmo.h)

        tcross = flat_wcdm.virial_dynamical_time(zray, *cosmo_dsps)
        assert np.all(np.diff(tcross) < 0)

        z0 = np.atleast_1d(0.0)
        tcross_z0 = flat_wcdm.virial_dynamical_time(z0, *cosmo_dsps)
        assert np.allclose(tcross_z0, 4, atol=1)


def test_cosmology_defaults():
    from ...cosmology import DEFAULT_COSMOLOGY, PLANCK15

    assert np.allclose(DEFAULT_COSMOLOGY, PLANCK15)


def test_top_level_imports():
    from ...cosmology import DEFAULT_COSMOLOGY, CosmoParams

    CosmoParams(*DEFAULT_COSMOLOGY)


@pytest.mark.skipif(not HAS_ASTROPY, reason=NO_ASTROPY_MSG)
def test_age_at_z():
    zray = np.linspace(0.001, 10, 500)
    for cosmo in ASTROPY_COSMO_LIST:
        cosmo_dsps = flat_wcdm.CosmoParams(cosmo.Om0, -1, 0, cosmo.h)
        age_astropy = cosmo.age(zray).value
        age_jax = flat_wcdm.age_at_z(zray, *cosmo_dsps)
        assert np.allclose(age_astropy, age_jax, rtol=0.01)


@pytest.mark.skipif(not HAS_ASTROPY, reason=NO_ASTROPY_MSG)
def test_age_at_z0():
    for cosmo in ASTROPY_COSMO_LIST:
        cosmo_dsps = flat_wcdm.CosmoParams(cosmo.Om0, -1, 0, cosmo.h)
        age_astropy = cosmo.age(0.0).value
        age_jax = flat_wcdm.age_at_z0(*cosmo_dsps)
        assert np.allclose(age_astropy, age_jax, rtol=0.01)


@pytest.mark.skipif(not HAS_ASTROPY, reason=NO_ASTROPY_MSG)
def test_differential_comoving_volume():
    zarr = np.linspace(0.01, 2.0, 50)
    cosmo_params = Planck15_astropy.Om0, -1.0, 0.0, Planck15_astropy.h
    diff_com_vol = flat_wcdm.differential_comoving_volume(zarr, *cosmo_params)
    diff_com_vol_astropy = Planck15_astropy.differential_comoving_volume(zarr).value
    assert np.allclose(diff_com_vol, diff_com_vol_astropy, rtol=0.01)
