"""Kernels calculating stellar age PDF-weighting of SSP tempates"""
from jax import numpy as jnp
from jax import jit as jjit
from ..utils import _jax_get_dt_array
from ..constants import SFR_MIN, T_BIRTH_MIN, N_T_LGSM_INTEGRATION
from ..cosmology import TODAY

__all__ = ("calc_age_weights_from_sfh_table",)


@jjit
def calc_age_weights_from_sfh_table(
    gal_t_table, gal_sfr_table, ssp_lg_age_gyr, t_obs, sfr_min=SFR_MIN
):
    """Calculate PDF-weights of stellar ages from a tabulated SFH

    Parameters
    ----------
    gal_t_table : ndarray of shape (n_t, )
        Age of the universe in Gyr when the galaxy SFH is tabulated

    gal_sfr_table : ndarray of shape (n_t, )
        Tabulation of the galaxy SFH in Msun/yr at the times gal_t_table

    ssp_lg_age_gyr : ndarray of shape (n_ages, )
        log10(stellar age) in Gyr of the SSP templates

    t_obs : float
        Age of the universe in Gyr at the time the galaxy is observed

    Returns
    -------
    age_weights : ndarray of shape (n_ages, )
        PDF weights of the SSP spectra

    """
    logsm_table = _calc_logsm_table_from_sfh_table(gal_t_table, gal_sfr_table, sfr_min)
    gal_lgt_table = jnp.log10(gal_t_table)
    age_weights = _calc_age_weights_from_logsm_table(
        gal_lgt_table, logsm_table, ssp_lg_age_gyr, t_obs
    )[1]
    return age_weights


@jjit
def _calc_age_weights_from_logsm_table(lgt_table, logsm_table, ssp_lg_age_gyr, t_obs):
    """Calculate PDF weights of the SSP spectra of a composite SED

    Parameters
    ----------
    lgt_table : ndarray of shape (n_t, )
        log10 of the age of the universe in Gyr when the galaxy SFH is tabulated

    logsm_table : ndarray of shape (n_t, )
        log10 of the cumulative stellar mass formed in-situ at each time in lgt_table

    ssp_lg_age_gyr : ndarray of shape (n_ages, )
        Array of stellar ages of the SSP templates

    t_obs : float
        Age of the universe in Gyr at the time the galaxy is observed

    Returns
    -------
    lgt_birth_bin_mids : ndarray of shape (n_ages, )
        Age of the universe in Gyr at the birth time
        of the stellar populations observed at t_obs

    age_weights : ndarray of shape (n_ages, )
        PDF weights of the SSP spectra

    """
    lg_age_bin_edges = _get_lg_age_bin_edges(ssp_lg_age_gyr)
    lgt_birth_bin_edges = _get_lgt_birth(t_obs, lg_age_bin_edges)
    lgt_birth_bin_mids = _get_lgt_birth(t_obs, ssp_lg_age_gyr)

    logsm_at_t_birth_bin_edges = jnp.interp(lgt_birth_bin_edges, lgt_table, logsm_table)
    delta_mstar_at_t_birth = -jnp.diff(10**logsm_at_t_birth_bin_edges)
    age_weights = delta_mstar_at_t_birth / delta_mstar_at_t_birth.sum()

    return lgt_birth_bin_mids, age_weights


@jjit
def _get_linspace_time_tables(t0=TODAY):
    """Convenience function returning time arrays used in SFH integrations

    Parameters
    ----------
    t0 : float, optional
        Age of the universe in Gyr at z=0

    Returns
    -------
    t_table : ndarray of shape (n, )
        Age of the universe in Gyr

    lgt_table : ndarray of shape (n, )
        log10(t_table)

    dt_table : ndarray of shape (n, )
        Duration in Gyr between the midpoints bracketing each element of t_table

    """
    t_table = jnp.linspace(T_BIRTH_MIN, t0, N_T_LGSM_INTEGRATION)
    lgt_table = jnp.log10(t_table)
    dt_table = _jax_get_dt_array(t_table)
    return t_table, lgt_table, dt_table


@jjit
def _get_lgt_birth(t_obs, lg_ages):
    """Age of the universe in Gyr at the birth time
    of the stellar populations observed at t_obs"""
    t_birth = t_obs - 10**lg_ages
    t_birth = jnp.where(t_birth < T_BIRTH_MIN, T_BIRTH_MIN, t_birth)
    lgt_birth = jnp.log10(t_birth)
    return lgt_birth


@jjit
def _get_lg_age_bin_edges(lg_ages):
    """Calculate the lower and upper bounds on the array of ages of the SSP templates.

    Parameters
    ----------
    lgt_ages : ndarray of shape (n, )
        Base-10 logarithm of the age of the SSP template

    Returns
    -------
    lgt_age_bounds : ndarray of shape (n+1, )
        Integration bounds on the SSP templates

    Notes
    -----
    For a galaxy with star formation history SFH(t),
    this function can be used to help calculate M_age[i](t_obs),
    the total mass of stars with age 10**lgt_ages[i] at time t_obs.

    To calculate M_age[i](t_obs), SFH(t) should be integrated across the time interval
    (t_obs - 10**lgt_age_bounds[i+1], t_obs - 10**lgt_age_bounds[i+1]).

    """
    dlgtau = _jax_get_dt_array(lg_ages)

    lg_age_bin_edges = jnp.zeros(dlgtau.size + 1)
    lg_age_bin_edges = lg_age_bin_edges.at[:-1].set(lg_ages - dlgtau / 2)

    dlowest = (lg_ages[1] - lg_ages[0]) / 2
    lowest = lg_ages[0] - dlowest
    highest = lg_ages[-1] + dlgtau[-1] / 2
    lg_age_bin_edges = lg_age_bin_edges.at[0].set(lowest)
    lg_age_bin_edges = lg_age_bin_edges.at[-1].set(highest)

    return lg_age_bin_edges


@jjit
def _calc_logsm_table_from_sfh_table(gal_t_table, gal_sfr_table, sfr_min):
    """Integrate the input SFH to calculate the stellar mass formed in-situ

    gal_t_table : ndarray of shape (n_t, )
        Age of the universe in Gyr when the galaxy SFH is tabulated

    gal_sfr_table : ndarray of shape (n_t, )
        Tabulation of the galaxy SFH in Msun/yr at the times gal_t_table

    sfr_min : float
        Minimum star formation rate in Msun/yr

    """
    dt_table = _jax_get_dt_array(gal_t_table)

    gal_sfr_table = jnp.where(gal_sfr_table < sfr_min, sfr_min, gal_sfr_table)
    gal_mstar_table = jnp.cumsum(gal_sfr_table * dt_table) * 1e9
    logsm_table = jnp.log10(gal_mstar_table)
    return logsm_table
