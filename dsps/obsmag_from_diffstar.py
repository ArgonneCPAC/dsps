"""
"""
from jax import jit as jjit
from jax import vmap
from dsps.photometry_kernels import _calc_obs_mag
from .seds_from_diffstar import _calc_diffstar_sed_kern
from .seds_from_diffstar import _calc_diffstar_attenuated_sed_kern


@jjit
def _calc_diffstar_obs_mag_kern(
    t_obs,
    z_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    ssp_wave,
    ssp_flux,
    mah_params,
    u_ms_params,
    u_q_params,
    met_params,
    cosmo_params,
    wave_filter,
    trans_filter,
):
    sed, sfh_table, logsm_table = _calc_diffstar_sed_kern(
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        ssp_flux,
        mah_params,
        u_ms_params,
        u_q_params,
        met_params,
    )
    mag = _calc_obs_mag(ssp_wave, sed, wave_filter, trans_filter, z_obs, *cosmo_params)
    return mag


@jjit
def _calc_diffstar_obs_mag_attenuation_kern(
    t_obs,
    z_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    ssp_wave,
    ssp_flux,
    mah_params,
    u_ms_params,
    u_q_params,
    met_params,
    dust_params,
    cosmo_params,
    wave_filter,
    trans_filter,
):
    sed, sfh_table, logsm_table = _calc_diffstar_attenuated_sed_kern(
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        ssp_wave,
        ssp_flux,
        mah_params,
        u_ms_params,
        u_q_params,
        met_params,
        dust_params,
    )
    mag = _calc_obs_mag(ssp_wave, sed, wave_filter, trans_filter, z_obs, *cosmo_params)
    return mag


_c = [*[None] * 6, 0, 0, 0, 0, None, None, None]
_d = [*[None] * 11, 0, 0]
_calc_diffstar_obs_mags_vmap = jjit(
    vmap(vmap(_calc_diffstar_obs_mag_kern, in_axes=_d), in_axes=_c)
)

_c = [*[None] * 6, 0, 0, 0, 0, 0, None, None, None]
_d = [*[None] * 12, 0, 0]
_calc_diffstar_obs_mags_attenuation_vmap = jjit(
    vmap(vmap(_calc_diffstar_obs_mag_attenuation_kern, in_axes=_d), in_axes=_c)
)


def compute_diffstarpop_obsframe_mags(
    t_obs,
    z_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    ssp_wave,
    ssp_flux,
    mah_params,
    u_ms_params,
    u_q_params,
    met_params,
    cosmo_params,
    wave_filters,
    trans_filters,
    dust_params=None,
):
    """Calculate the observed magnitudes of a population of Diffstar galaxies that are
    all observed at the same time, t_obs.

    Parameters
    ----------
    t_obs : float
        Age of the universe at the time of observation in units of Gyr

    z_obs : float
        Redshift of the universe at the time of observation

    lgZsun_bin_mids : ndarray of shape (n_met, )
        SSP bins of log10(Z/Zsun)

    log_age_gyr : ndarray of shape (n_ages, )
        SSP bins of log10(age) in gyr

    ssp_wave : ndarray of shape (n_wave, )
        Array storing the wavelength in Angstroms of the SSP luminosities

    ssp_flux : ndarray of shape (n_met, n_ages, n_wave)
        Array storing SSP luminosity in Lsun/Hz

    mah_params : ndarray of shape (n_gals, 6)
        Diffmah parameters of each galaxy:
        (logt0, logmp, logtc, k, early, late)

    u_ms_params : ndarray of shape (n_gals, 5)
        Unbounded versions of the Diffstar main sequence parameters of each galaxy:
        (u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep)

    u_q_params : ndarray of shape (n_gals, 4)
        Unbounded versions of the Diffstar quenching parameters of each galaxy:
        (u_lg_qt, u_lg_qs, u_lg_drop, u_lg_rejuv)

    met_params : ndarray of shape (n_gals, 2)
        Metallicity parameters of each galaxy:
        (lgmet, lgmet_scatter)

    cosmo_params : ndarray of shape (5, )
        Flat LCDM parameters:
        (Om0, Ode0, w0, wa, h)

    wave_filters : ndarray of shape (n_filters, n_wave_filters)
        Wavelengths in nm of the filter transmission curve

    trans_filters : ndarray of shape (n_filters, n_wave_filters)
        Fraction of light that passes through each filter

    dust_params : ndarray of shape (n_gals, 5), optional
        Dust parameters controlling attenuation within each galaxy:
        (dust_x0, dust_gamma, dust_ampl, dust_slope, dust_Av)
        Default behavior is no attenuation

    Returns
    -------
    obs_mags : ndarray of shape (n_gals, n_filters)
        Restframe magnitude of each galaxy through each filter

    """
    if dust_params is None:
        obs_mags = _calc_diffstar_obs_mags_vmap(
            t_obs,
            z_obs,
            lgZsun_bin_mids,
            log_age_gyr,
            ssp_wave,
            ssp_flux,
            mah_params,
            u_ms_params,
            u_q_params,
            met_params,
            cosmo_params,
            wave_filters,
            trans_filters,
        )
    else:
        obs_mags = _calc_diffstar_obs_mags_attenuation_vmap(
            t_obs,
            z_obs,
            lgZsun_bin_mids,
            log_age_gyr,
            ssp_wave,
            ssp_flux,
            mah_params,
            u_ms_params,
            u_q_params,
            met_params,
            dust_params,
            cosmo_params,
            wave_filters,
            trans_filters,
        )
    return obs_mags
