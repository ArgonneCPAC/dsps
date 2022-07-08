"""
"""
from jax import jit as jjit
from jax import vmap
from .photometry_kernels import _calc_rest_mag
from .seds_from_diffstar import _calc_diffstar_sed_kern
from .seds_from_diffstar import _calc_diffstar_attenuated_sed_kern


__all__ = ("compute_diffstarpop_restframe_mags",)


@jjit
def _calc_diffstar_rest_mag_kern(
    t_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    ssp_wave,
    ssp_flux,
    mah_params,
    u_ms_params,
    u_q_params,
    met_params,
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
    rest_mag = _calc_rest_mag(ssp_wave, sed, wave_filter, trans_filter)
    return rest_mag


@jjit
def _calc_diffstar_rest_mag_attenuation_kern(
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
    rest_mag = _calc_rest_mag(ssp_wave, sed, wave_filter, trans_filter)
    return rest_mag


_a = [*[None] * 5, 0, 0, 0, 0, None, None]
_b = [*[None] * 9, 0, 0]
_calc_diffstar_rest_mags_vmap = jjit(
    vmap(vmap(_calc_diffstar_rest_mag_kern, in_axes=_b), in_axes=_a)
)


_a = [*[None] * 5, 0, 0, 0, 0, 0, None, None]
_b = [*[None] * 10, 0, 0]
_calc_diffstar_rest_mags_attenuation_vmap = jjit(
    vmap(vmap(_calc_diffstar_rest_mag_attenuation_kern, in_axes=_b), in_axes=_a)
)


def compute_diffstarpop_restframe_mags(
    t_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    ssp_wave,
    ssp_flux,
    mah_params,
    u_ms_params,
    u_q_params,
    met_params,
    wave_filters,
    trans_filters,
    dust_params=None,
):
    """Calculate the restframe magnitudes of a population of Diffstar galaxies that are
    all observed at the same time, t_obs.

    Parameters
    ----------
    t_obs : float
        Age of the universe at the time of observation in units of Gyr

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
    rest_mags : ndarray of shape (n_gals, n_filters)
        Restframe magnitude of each galaxy through each filter

    """
    if dust_params is None:
        rest_mags = _calc_diffstar_rest_mags_vmap(
            t_obs,
            lgZsun_bin_mids,
            log_age_gyr,
            ssp_wave,
            ssp_flux,
            mah_params,
            u_ms_params,
            u_q_params,
            met_params,
            wave_filters,
            trans_filters,
        )
    else:
        rest_mags = _calc_diffstar_rest_mags_attenuation_vmap(
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
            wave_filters,
            trans_filters,
        )
    return rest_mags
