"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap
from diffmah.individual_halo_assembly import _calc_halo_history
from diffstar.stars import calculate_sm_sfr_history_from_mah
from .stellar_ages import _get_linspace_time_tables
from .attenuation_kernels import sbl18_k_lambda, _flux_ratio
from .attenuation_kernels import RV_C00
from .seds_from_tables import _calc_sed_kern

__all__ = ("compute_diffstarpop_restframe_seds",)


@jjit
def _get_diffstar_sfh_tables(mah_params, ms_params, q_params):
    t_table, lgt_table, dt_table = _get_linspace_time_tables()
    dmhdt, log_mah = _calc_halo_history(lgt_table, *mah_params)
    mstar_table, sfh_table = calculate_sm_sfr_history_from_mah(
        lgt_table, dt_table, dmhdt, log_mah, ms_params, q_params
    )
    logsm_table = jnp.log10(mstar_table)
    return t_table, lgt_table, dt_table, sfh_table, logsm_table


@jjit
def _calc_diffstar_sed_kern(
    t_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    ssp_flux,
    mah_params,
    ms_params,
    q_params,
    met_params,
):
    lgmet, lgmet_scatter = met_params
    _res = _get_diffstar_sfh_tables(mah_params, ms_params, q_params)
    t_table = _res[0]
    sfh_table = _res[3]
    logsm_table = _res[4]
    sed = _calc_sed_kern(
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        ssp_flux,
        t_table,
        logsm_table,
        lgmet,
        lgmet_scatter,
    )
    return (sed, sfh_table, logsm_table)


_e = [None, None, None, None, 0, 0, 0, 0]
_calc_diffstarpop_seds_vmap = jjit(vmap(_calc_diffstar_sed_kern, in_axes=_e))


@jjit
def _calc_diffstar_attenuated_sed_kern(
    t_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    ssp_wave,
    ssp_flux,
    mah_params,
    ms_params,
    q_params,
    met_params,
    dust_params,
):

    sed, sfh_table, logsm_table = _calc_diffstar_sed_kern(
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        ssp_flux,
        mah_params,
        ms_params,
        q_params,
        met_params,
    )
    dust_x0, bump_width, dust_Eb, dust_delta, dust_Av = dust_params
    wave_micron = ssp_wave / 10_000
    att_k = sbl18_k_lambda(wave_micron, dust_x0, bump_width, dust_Eb, dust_delta)
    attenuation = _flux_ratio(att_k, RV_C00, dust_Av)
    attenuated_sed = attenuation * sed
    return attenuated_sed, sfh_table, logsm_table


_f = [None, None, None, None, None, 0, 0, 0, 0, 0]
_calc_diffstar_attenuated_sed_vmap = jjit(
    vmap(_calc_diffstar_attenuated_sed_kern, in_axes=_f)
)


def compute_diffstarpop_restframe_seds(
    t_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    ssp_wave,
    ssp_flux,
    mah_params,
    u_ms_params,
    u_q_params,
    met_params,
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

    dust_params : ndarray of shape (n_gals, 5), optional
        Dust parameters controlling attenuation within each galaxy:
        (dust_x0, dust_gamma, dust_ampl, dust_slope, dust_Av)
        Default behavior is no attenuation

    Returns
    -------
    rest_seds : ndarray of shape (n_gals, n_wave)
        Restframe magnitude of each galaxy through each filter

    """
    if dust_params is None:
        rest_seds, sfh_tables, logsm_tables = _calc_diffstarpop_seds_vmap(
            t_obs,
            lgZsun_bin_mids,
            log_age_gyr,
            ssp_flux,
            mah_params,
            u_ms_params,
            u_q_params,
            met_params,
        )
    else:
        rest_seds, sfh_tables, logsm_tables = _calc_diffstar_attenuated_sed_vmap(
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
    return rest_seds, sfh_tables, logsm_tables
