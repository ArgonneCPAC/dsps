"""
"""
import numpy as np
from jax import vmap as jvmap, jit as jjit
from .photometry_kernels import _obs_flux_ssp, _calc_obs_mag_no_dimming, _calc_obs_mag

_a = [None, 0, None, None, None]
_b = [None, None, None, None, 0]
_obs_flux_ssp_vmap = jjit(
    jvmap(jvmap(jvmap(_obs_flux_ssp, in_axes=_b), in_axes=_a), in_axes=_a)
)
_calc_obs_mag_no_dimming_vmap = jjit(
    jvmap(jvmap(jvmap(_calc_obs_mag_no_dimming, in_axes=_b), in_axes=_a), in_axes=_a)
)

_calc_obs_mag_no_dimming_vmap_singlemet = jjit(
    jvmap(jvmap(_calc_obs_mag_no_dimming, in_axes=_b), in_axes=_a)
)

_c = [None, 0, None, None, None, None, None]
_d = [None, None, None, None, 0, None, None]
_e = [None, None, 0, 0, None, None, None]

_calc_obs_mag_vmap = jjit(
    jvmap(jvmap(jvmap(_calc_obs_mag, in_axes=_d), in_axes=_c), in_axes=_c)
)

_a = (*[None] * 4, 0, *[None] * 5)
_calc_obs_mag_vmap_z = jjit(jvmap(_calc_obs_mag, in_axes=_a))

_b = (None, 0, *[None] * 8)
_calc_obs_mag_vmap_spec = jjit(jvmap(_calc_obs_mag, in_axes=_b))


@jjit
def calc_obs_mag_history_singlegal(
    wave_spec, lum_spec, wave_filter, trans_filter, z_obs, Om0, Ode0, w0, wa, h
):
    """Calculate the history of the observed flux of a single galaxy
    through a particular filter.

    Parameters
    ----------
    wave_spec_rest : ndarray of shape (n_wave_spec, )
        Rest-frame wavelengths of the spectrum

    lum_spec : ndarray of shape (n_wave_spec, )
        Spectrum of each galaxy in Lsun/Hz

    wave_filter : ndarray of shape (n_wave_filter, )
        Wave length of the filter transmission curve

    trans_filter : ndarray of shape (n_wave_filter, )
        Fraction of the incident flux transmitted through the filter

    z_obs : ndarray of shape (n_obs, )
        Array of redshifts of the observed galaxies

    Om0: float
        Omega matter at z=0

    Ode0: float
        Omega DE at z=0

    w0 : float
        DE eqn of state today

    wa : float
        DE eqn of state deriv

    h : float
        Little h

    Returns
    -------
    obs_mags : ndarray of shape (n_obs, )

    """
    obs_mags = _calc_obs_mag_vmap_z(
        wave_spec, lum_spec, wave_filter, trans_filter, z_obs, Om0, Ode0, w0, wa, h
    )
    return obs_mags


@jjit
def calc_obs_mags_galpop(
    wave_spec, lum_spec, wave_filter, trans_filter, z_obs, Om0, Ode0, w0, wa, h
):
    """Calculate the history of the observed flux of a galaxy population
    at a single redshift observed through a particular filter.

    Parameters
    ----------
    wave_spec_rest : ndarray of shape (n_wave_spec, )
        Rest-frame wavelengths of the spectrum

    lum_spec : ndarray of shape (n_gals, n_wave_spec)
        Spectrum of each galaxy in Lsun/Hz

    wave_filter : ndarray of shape (n_wave_filter, )
        Wave length of the filter transmission curve

    trans_filter : ndarray of shape (n_wave_filter, )
        Fraction of the incident flux transmitted through the filter

    z_obs : float
        Redshift of the observed galaxies

    Om0: float
        Omega matter at z=0

    Ode0: float
        Omega DE at z=0

    w0 : float
        DE eqn of state today

    wa : float
        DE eqn of state deriv

    h : float
        Little h

    Returns
    -------
    obs_mags : ndarray of shape (n_gals, )

    """
    obs_mags = _calc_obs_mag_vmap_spec(
        wave_spec, lum_spec, wave_filter, trans_filter, z_obs, Om0, Ode0, w0, wa, h
    )
    return obs_mags


def interpolate_filter_trans_curves(wave_filters, trans_filters, n=None):
    """Interpolate a collection of filter transmission curves to a common length.
    Convenience function for analyses vmapping over broadband colors.

    Parameters
    ----------
    wave_filters : sequence of n_filters ndarrays

    trans_filters : sequence of n_filters ndarrays

    n : int, optional
        Desired length of the output transmission curves.
        Default is equal to the smallest length transmission curve

    Returns
    -------
    wave_filters : ndarray of shape (n_filters, n)

    trans_filters : ndarray of shape (n_filters, n)

    """
    wave0 = wave_filters[0]
    wave_min, wave_max = wave0.min(), wave0.max()

    if n is None:
        n = np.min([x.size for x in wave_filters])

    for wave, trans in zip(wave_filters, trans_filters):
        wave_min = min(wave_min, wave.min())
        wave_max = max(wave_max, wave.max())

    wave_collector = []
    trans_collector = []
    for wave, trans in zip(wave_filters, trans_filters):
        wave_min, wave_max = wave.min(), wave.max()
        new_wave = np.linspace(wave_min, wave_max, n)
        new_trans = np.interp(new_wave, wave, trans)
        wave_collector.append(new_wave)
        trans_collector.append(new_trans)
    return np.array(wave_collector), np.array(trans_collector)
