"""
"""
import numpy as np
from jax import random as jran
from ..composite_sed import calc_rest_sed_lognormal_mdf, calc_rest_sed_met_table
from ...constants import T_BIRTH_MIN

SEED = 43
FSPS_LG_AGES = np.arange(5.5, 10.2, 0.05)  # log10 ages in years


def test_calc_rest_sed_lognormal_mdf():
    ran_key = jran.PRNGKey(SEED)
    t_obs = 13.0
    n_t = 500
    gal_t_table = np.linspace(T_BIRTH_MIN, t_obs, n_t)

    sfr_key, met_key, sed_key = jran.split(ran_key, 3)
    gal_sfr_table = jran.uniform(sfr_key, minval=0, maxval=10, shape=(n_t,))

    n_ages = FSPS_LG_AGES.size
    ssp_lg_age = FSPS_LG_AGES - 9.0
    n_met = 15
    ssp_lgmet = np.linspace(-4, 0.5, n_met)

    gal_lgmet = jran.uniform(
        met_key, minval=ssp_lgmet.min(), maxval=ssp_lgmet.max(), shape=()
    )
    gal_lgmet_scatter = 0.1

    n_wave = 20
    ssp_flux = jran.uniform(sed_key, minval=0, maxval=1, shape=(n_met, n_ages, n_wave))

    args = (
        gal_t_table,
        gal_sfr_table,
        gal_lgmet,
        gal_lgmet_scatter,
        ssp_lg_age,
        ssp_lgmet,
        ssp_flux,
        t_obs,
    )
    sed = calc_rest_sed_lognormal_mdf(*args)
    assert sed.shape == (n_wave,)
    assert np.any(sed > 0)


def test_calc_rest_sed_lgmet_table():
    ran_key = jran.PRNGKey(SEED)
    t_obs = 13.0
    n_t = 500
    gal_t_table = np.linspace(T_BIRTH_MIN, t_obs, n_t)

    sfr_key, met_key, sed_key = jran.split(ran_key, 3)
    gal_sfr_table = jran.uniform(sfr_key, minval=0, maxval=10, shape=(n_t,))

    n_ages = FSPS_LG_AGES.size
    ssp_lg_age = FSPS_LG_AGES - 9.0
    n_met = 15
    ssp_lgmet = np.linspace(-4, 0.5, n_met)

    gal_lgmet_table = jran.uniform(
        met_key, minval=ssp_lgmet.min(), maxval=ssp_lgmet.max(), shape=(n_t,)
    )
    gal_lgmet_scatter = 0.1

    n_wave = 20
    ssp_flux = jran.uniform(sed_key, minval=0, maxval=1, shape=(n_met, n_ages, n_wave))

    args = (
        gal_t_table,
        gal_sfr_table,
        gal_lgmet_table,
        gal_lgmet_scatter,
        ssp_lg_age,
        ssp_lgmet,
        ssp_flux,
        t_obs,
    )
    sed = calc_rest_sed_met_table(*args)
    assert sed.shape == (n_wave,)
    assert np.any(sed > 0)
