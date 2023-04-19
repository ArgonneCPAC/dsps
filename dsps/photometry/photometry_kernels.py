"""Kernels of common photometry integrals"""
from jax import numpy as jnp
from jax import jit as jjit
from ..cosmology.flat_wcdm import distance_modulus_to_z

AB0 = 1.13492e-13  # 3631 Jansky placed at 10 pc in units of Lsun/Hz


__all__ = ("calc_obs_mag", "calc_rest_mag")


@jjit
def calc_obs_mag(
    wave_spec_rest, lum_spec, wave_filter, trans_filter, redshift, Om0, w0, wa, h
):
    """Calculate the apparent magnitude of an SED observed through a filter

    Parameters
    ----------
    wave_spec_rest : ndarray of shape (n_wave, )

    lum_spec : ndarray of shape (n_wave, )

    wave_filter : ndarray of shape (n_filter_wave, )

    trans_filter : ndarray of shape (n_filter_wave, )

    redshift : float

    Om0 : float

    w0 : float

    wa : float

    h : float

    Returns
    -------
    obs_mag : float

    """
    flux_source = _obs_flux_ssp(
        wave_spec_rest, lum_spec, wave_filter, trans_filter, redshift
    )
    flux_ab0 = _flux_ab0_at_10pc(wave_filter, trans_filter)
    mag_no_dimming = -2.5 * jnp.log10(flux_source / flux_ab0)
    dimming = _cosmological_dimming(redshift, Om0, w0, wa, h)
    obs_mag = mag_no_dimming + dimming
    return obs_mag


@jjit
def _cosmological_dimming_from_table(z, z_table, distance_modulus_table):
    distance_modulus = jnp.interp(z, z_table, distance_modulus_table)
    return distance_modulus - 2.5 * jnp.log10(1 + z)


@jjit
def _cosmological_dimming(z, Om0, w0, wa, h):
    dmod = distance_modulus_to_z(z, Om0, w0, wa, h)
    return dmod - 2.5 * jnp.log10(1 + z)


@jjit
def _calc_obs_mag_no_dimming(wave_spec_rest, lum_spec, wave_filter, trans_filter, z):
    flux_source = _obs_flux_ssp(wave_spec_rest, lum_spec, wave_filter, trans_filter, z)
    flux_ab0 = _flux_ab0_at_10pc(wave_filter, trans_filter)
    return -2.5 * jnp.log10(flux_source / flux_ab0)


@jjit
def calc_rest_mag(wave_spec_rest, lum_spec, wave_filter, trans_filter):
    """Calculate the restframe magnitude of an SED observed through a filter

    Parameters
    ----------
    wave_spec_rest : ndarray of shape (n_wave, )

    lum_spec : ndarray of shape (n_wave, )

    wave_filter : ndarray of shape (n_filter_wave, )

    trans_filter : ndarray of shape (n_filter_wave, )

    Returns
    -------
    rest_mag : float

    """
    flux_source = _rest_flux_ssp(wave_spec_rest, lum_spec, wave_filter, trans_filter)
    flux_ab0 = _flux_ab0_at_10pc(wave_filter, trans_filter)
    rest_mag = -2.5 * jnp.log10(flux_source / flux_ab0)
    return rest_mag


@jjit
def _obs_flux_ssp(wave_spec_rest, lum_spec, wave_filter, trans_filter, z):
    lum_zshift_phot = jnp.interp(
        wave_filter, wave_spec_rest * (1 + z), lum_spec, left=0, right=0
    )
    integrand = trans_filter * lum_zshift_phot / wave_filter
    lum_filter = jnp.trapz(integrand, x=wave_filter)
    return lum_filter


@jjit
def _rest_flux_ssp(wave_spec_rest, lum_spec, wave_filter, trans_filter):
    lum_phot = jnp.interp(wave_filter, wave_spec_rest, lum_spec, left=0, right=0)
    integrand = trans_filter * lum_phot / wave_filter
    lum_filter = jnp.trapz(integrand, x=wave_filter)
    return lum_filter


@jjit
def _flux_ab0_at_10pc(wave_filter, trans_filter):
    integrand = trans_filter * AB0 / wave_filter
    lum_ab0_filter = jnp.trapz(integrand, x=wave_filter)
    return lum_ab0_filter
