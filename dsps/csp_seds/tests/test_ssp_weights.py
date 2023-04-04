"""
"""
import numpy as np
from jax import random as jran
from ..ssp_weights import _calc_ssp_weights_lognormal_mdf
from ...constants import T_BIRTH_MIN


FSPS_LG_AGES = np.arange(5.5, 10.2, 0.05)  # log10 ages in years
SEED = 43


def test_calc_ssp_weights_lognormal_mdf():
    ran_key = jran.PRNGKey(SEED)
    t_obs = 13.0
    n_t = 500
    gal_t_table = np.linspace(T_BIRTH_MIN, t_obs, n_t)

    sfr_key, met_key = jran.split(ran_key, 2)
    gal_sfr_table = jran.uniform(sfr_key, minval=0, maxval=10, shape=())

    n_ages = FSPS_LG_AGES.size
    ssp_lg_age = FSPS_LG_AGES - 9.0
    n_met = 15
    ssp_lgmet = np.linspace(-4, 0.5, n_met)

    lgmet = jran.uniform(
        met_key, minval=ssp_lgmet.min(), maxval=ssp_lgmet.max(), shape=()
    )
    lgmet_scatter = 0.1

    args = (
        t_obs,
        gal_t_table,
        gal_sfr_table,
        ssp_lg_age,
        ssp_lgmet,
        lgmet,
        lgmet_scatter,
    )
    weights, age_weights, lgmet_weights = _calc_ssp_weights_lognormal_mdf(*args)
    assert weights.shape == (n_met, n_ages)
    assert np.allclose(age_weights.sum(), 1.0, rtol=1e-4)
    assert np.allclose(lgmet_weights.sum(), 1.0, rtol=1e-4)
    assert np.allclose(weights.sum(), 1.0, rtol=1e-4)
