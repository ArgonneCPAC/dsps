"""
"""
from jax import jit as jjit
from jax import vmap
from jax import numpy as jnp
from .photometry_kernels import _calc_rest_mag, _calc_obs_mag
from .weighted_ssps import _calc_weighted_ssp_from_diffstar_params_const_zmet
from .weighted_ssps import _calc_weighted_flux_from_diffstar_age_correlated_zmet
from .attenuation_kernels import _flux_ratio, sbl18_k_lambda, RV_C00
from .utils import _tw_sigmoid


@jjit
def _calc_weighted_rest_mag_from_diffstar_params_const_zmet(
    t_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    spec_wave,
    spec_flux,
    filter_wave,
    filter_flux,
    mah_logt0,
    mah_logmp,
    mah_logtc,
    mah_k,
    mah_early,
    mah_late,
    lgmcrit,
    lgy_at_mcrit,
    indx_k,
    indx_lo,
    indx_hi,
    floor_low,
    tau_dep,
    lg_qt,
    lg_qs,
    lg_drop,
    lg_rejuv,
    lgmet,
    lgmet_scatter,
):
    mah_params = mah_logt0, mah_logmp, mah_logtc, mah_k, mah_early, mah_late
    ms_params = lgmcrit, lgy_at_mcrit, indx_k, indx_lo, indx_hi, floor_low, tau_dep
    q_params = lg_qt, lg_qs, lg_drop, lg_rejuv
    _res = _calc_weighted_ssp_from_diffstar_params_const_zmet(
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        spec_flux,
        mah_params,
        ms_params,
        q_params,
        lgmet,
        lgmet_scatter,
    )
    lgmet_weights, age_weights, weighted_ssp = _res

    rest_mag = _calc_rest_mag(spec_wave, weighted_ssp, filter_wave, filter_flux)
    return rest_mag


_a = (None, None, None, 0, None, None, None, None, None, None)
_calc_weighted_flux_from_diffstar_age_correlated_zmet_vmap = jjit(
    vmap(_calc_weighted_flux_from_diffstar_age_correlated_zmet, in_axes=_a)
)


@jjit
def _calc_weighted_rest_mag_from_diffstar_params_age_correlated_zmet(
    t_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    spec_wave,
    spec_flux,
    filter_wave,
    filter_flux,
    mah_logt0,
    mah_logmp,
    mah_logtc,
    mah_k,
    mah_early,
    mah_late,
    lgmcrit,
    lgy_at_mcrit,
    indx_k,
    indx_lo,
    indx_hi,
    floor_low,
    tau_dep,
    lg_qt,
    lg_qs,
    lg_drop,
    lg_rejuv,
    lgmet_young,
    lgmet_old,
    lgmet_scatter,
):
    mah_params = mah_logt0, mah_logmp, mah_logtc, mah_k, mah_early, mah_late
    ms_params = lgmcrit, lgy_at_mcrit, indx_k, indx_lo, indx_hi, floor_low, tau_dep
    q_params = lg_qt, lg_qs, lg_drop, lg_rejuv

    spec_flux = jnp.moveaxis(spec_flux, 2, 0)
    _res = _calc_weighted_flux_from_diffstar_age_correlated_zmet_vmap(
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        spec_flux,
        mah_params,
        ms_params,
        q_params,
        lgmet_young,
        lgmet_old,
        lgmet_scatter,
    )
    lgmet_weights, age_weights, weighted_ssp = _res

    rest_mag = _calc_rest_mag(spec_wave, weighted_ssp, filter_wave, filter_flux)
    return rest_mag


@jjit
def _calc_weighted_rest_mag_from_diffstar_params_const_zmet_dust(
    t_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    spec_wave,
    spec_flux,
    filter_wave,
    filter_flux,
    mah_logt0,
    mah_logmp,
    mah_logtc,
    mah_k,
    mah_early,
    mah_late,
    lgmcrit,
    lgy_at_mcrit,
    indx_k,
    indx_lo,
    indx_hi,
    floor_low,
    tau_dep,
    lg_qt,
    lg_qs,
    lg_drop,
    lg_rejuv,
    lgmet,
    lgmet_scatter,
    dust_x0,
    dust_gamma,
    dust_ampl,
    dust_slope,
    dust_Av,
):
    mah_params = mah_logt0, mah_logmp, mah_logtc, mah_k, mah_early, mah_late
    ms_params = lgmcrit, lgy_at_mcrit, indx_k, indx_lo, indx_hi, floor_low, tau_dep
    q_params = lg_qt, lg_qs, lg_drop, lg_rejuv
    _res = _calc_weighted_ssp_from_diffstar_params_const_zmet(
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        spec_flux,
        mah_params,
        ms_params,
        q_params,
        lgmet,
        lgmet_scatter,
    )
    lgmet_weights, age_weights, weighted_ssp = _res
    wave_micron = spec_wave / 10_000
    att_k = sbl18_k_lambda(wave_micron, dust_x0, dust_gamma, dust_ampl, dust_slope)
    attenuation = _flux_ratio(att_k, RV_C00, dust_Av)
    attenuated_ssp = attenuation * weighted_ssp

    rest_mag = _calc_rest_mag(spec_wave, attenuated_ssp, filter_wave, filter_flux)
    return rest_mag


@jjit
def _calc_weighted_obs_mag_from_diffstar_params_const_zmet_dust(
    t_obs,
    z_obs,
    Om0,
    Ode0,
    w0,
    wa,
    h,
    lgZsun_bin_mids,
    log_age_gyr,
    spec_wave,
    spec_flux,
    filter_wave,
    filter_flux,
    mah_logt0,
    mah_logmp,
    mah_logtc,
    mah_k,
    mah_early,
    mah_late,
    lgmcrit,
    lgy_at_mcrit,
    indx_k,
    indx_lo,
    indx_hi,
    floor_low,
    tau_dep,
    lg_qt,
    lg_qs,
    lg_drop,
    lg_rejuv,
    lgmet,
    lgmet_scatter,
    dust_x0,
    dust_gamma,
    dust_ampl,
    dust_slope,
    dust_Av,
):
    mah_params = mah_logt0, mah_logmp, mah_logtc, mah_k, mah_early, mah_late
    ms_params = lgmcrit, lgy_at_mcrit, indx_k, indx_lo, indx_hi, floor_low, tau_dep
    q_params = lg_qt, lg_qs, lg_drop, lg_rejuv
    _res = _calc_weighted_ssp_from_diffstar_params_const_zmet(
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        spec_flux,
        mah_params,
        ms_params,
        q_params,
        lgmet,
        lgmet_scatter,
    )
    lgmet_weights, age_weights, weighted_ssp = _res
    wave_micron = spec_wave / 10_000
    att_k = sbl18_k_lambda(wave_micron, dust_x0, dust_gamma, dust_ampl, dust_slope)
    attenuation = _flux_ratio(att_k, RV_C00, dust_Av)
    attenuated_ssp = attenuation * weighted_ssp

    obs_mag = _calc_obs_mag(
        spec_wave, attenuated_ssp, filter_wave, filter_flux, z_obs, Om0, Ode0, w0, wa, h
    )
    return obs_mag, weighted_ssp, attenuation


@jjit
def _calc_weighted_obs_mag_from_diffstar_params_const_zmet_agedep_dust(
    t_obs,
    z_obs,
    Om0,
    Ode0,
    w0,
    wa,
    h,
    lgZsun_bin_mids,
    log_age_gyr,
    spec_wave,
    spec_flux,
    filter_wave,
    filter_flux,
    mah_logt0,
    mah_logmp,
    mah_logtc,
    mah_k,
    mah_early,
    mah_late,
    lgmcrit,
    lgy_at_mcrit,
    indx_k,
    indx_lo,
    indx_hi,
    floor_low,
    tau_dep,
    lg_qt,
    lg_qs,
    lg_drop,
    lg_rejuv,
    lgmet,
    lgmet_scatter,
    dust_x0,
    dust_gamma,
    dust_ampl,
    dust_slope_young,
    dust_slope_old,
    dust_Av_young,
    dust_Av_old,
):
    mah_params = mah_logt0, mah_logmp, mah_logtc, mah_k, mah_early, mah_late
    ms_params = lgmcrit, lgy_at_mcrit, indx_k, indx_lo, indx_hi, floor_low, tau_dep
    q_params = lg_qt, lg_qs, lg_drop, lg_rejuv

    wave_micron = spec_wave / 10_000

    dust_slope_arr = _tw_sigmoid(
        log_age_gyr + 9, 8, 1, dust_slope_young, dust_slope_old
    )
    dust_Av_arr = _tw_sigmoid(log_age_gyr + 9, 8, 1, dust_Av_young, dust_Av_old)

    att_curves = _calc_sbl18_attenuation_vmap(
        wave_micron, dust_x0, dust_gamma, dust_ampl, dust_slope_arr, dust_Av_arr
    )

    n_met, n_ages, n_wave = spec_flux.shape
    spec_flux = att_curves.reshape((1, n_ages, n_wave)) * spec_flux

    _res = _calc_weighted_ssp_from_diffstar_params_const_zmet(
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        spec_flux,
        mah_params,
        ms_params,
        q_params,
        lgmet,
        lgmet_scatter,
    )
    lgmet_weights, age_weights, weighted_ssp = _res

    obs_mag = _calc_obs_mag(
        spec_wave, weighted_ssp, filter_wave, filter_flux, z_obs, Om0, Ode0, w0, wa, h
    )
    return obs_mag, weighted_ssp, att_curves


@jjit
def _calc_sbl18_attenuation(
    wave_micron, dust_x0, dust_gamma, dust_ampl, dust_slope, dust_Av
):
    att_k = sbl18_k_lambda(wave_micron, dust_x0, dust_gamma, dust_ampl, dust_slope)
    attenuation = _flux_ratio(att_k, RV_C00, dust_Av)
    return attenuation


_a = (None, None, None, None, 0, 0)
_calc_sbl18_attenuation_vmap = jjit(vmap(_calc_sbl18_attenuation, in_axes=_a))
