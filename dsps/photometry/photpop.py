"""Functions used to compute photometry for collections of SEDs"""
from jax import jit as jjit
from jax import vmap
from .photometry_kernels import calc_obs_mag, calc_rest_mag


_z = [*[None] * 4, 0, *[None] * 4]
_f = [None, None, 0, 0, None, *[None] * 4]
_ssp = [None, 0, *[None] * 7]
_calc_obs_mag_vmap_f = jjit(vmap(calc_obs_mag, in_axes=_f))
_calc_obs_mag_vmap_f_ssp = jjit(
    vmap(vmap(_calc_obs_mag_vmap_f, in_axes=_ssp), in_axes=_ssp)
)
_calc_obs_mag_vmap_f_ssp_z = jjit(vmap(_calc_obs_mag_vmap_f_ssp, in_axes=_z))


@jjit
def precompute_ssp_obsmags_on_z_table(
    ssp_wave,
    ssp_fluxes,
    filter_waves,
    filter_trans,
    z_table,
    Om0,
    w0,
    wa,
    h,
):
    """Precompute observed magnitudes of a collection of SEDs on a redshift grid

    Parameters
    ----------
    ssp_wave : array of shape (n_spec, )
        Wavelength of the SEDs in Angstroms

    ssp_fluxes : array of shape (n_met, n_age, n_spec)
        Flux of the SEDs in Lsun/Hz, normalized to unit stellar mass

    filter_waves : array of shape (n_filters, n_trans_curve)
        Wavelength of the filter transmission curves in Angstroms

    filter_trans : array of shape (n_filters, n_trans_curve)
        Transmission curves defining fractional transmission of the filters

    z_table : array of shape (n_redshift, )
        Array of redshifts at which the magnitudes will be computed

    Om0 : float
        Cosmological matter density at z=0

    w0 : float
        Dark energy equation of state parameter

    wa : float
        Dark energy equation of state parameter

    h : float
        Hubble parameter

    Returns
    -------
    ssp_photmag_table : array of shape (n_redshift, n_met, n_age, n_filters)

    """
    ssp_obsmag_table = _calc_obs_mag_vmap_f_ssp_z(
        ssp_wave, ssp_fluxes, filter_waves, filter_trans, z_table, Om0, w0, wa, h
    )
    return ssp_obsmag_table


_calc_rest_mag_vmap_f = jjit(vmap(calc_rest_mag, in_axes=[None, None, 0, 0]))
_calc_rest_mag_vmap_f_ssp = jjit(
    vmap(
        vmap(_calc_rest_mag_vmap_f, in_axes=[None, 0, None, None]),
        in_axes=[None, 0, None, None],
    )
)


@jjit
def precompute_ssp_restmags(ssp_wave, ssp_fluxes, filter_waves, filter_trans):
    """Precompute restframe magnitudes of a collection of SEDs

    Parameters
    ----------
    ssp_wave : array of shape (n_spec, )
        Wavelength of the SEDs in Angstroms

    ssp_fluxes : array of shape (n_met, n_age, n_spec)
        Flux of the SEDs in Lsun/Hz, normalized to unit stellar mass

    filter_waves : array of shape (n_filters, n_trans_curve)
        Wavelength of the filter transmission curves in Angstroms

    filter_trans : array of shape (n_filters, n_trans_curve)
        Transmission curves defining fractional transmission of the filters

    Returns
    -------
    ssp_photmag_table : array of shape (n_met, n_age, n_filters)

    """
    ssp_restmag_table = _calc_rest_mag_vmap_f_ssp(
        ssp_wave, ssp_fluxes, filter_waves, filter_trans
    )
    return ssp_restmag_table
