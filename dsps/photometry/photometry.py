"""
"""
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
