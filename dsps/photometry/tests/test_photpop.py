"""
"""
import numpy as np
from ..photpop import precompute_ssp_restmags
from ..photpop import precompute_ssp_obsmags_on_z_table


def test_precompute_ssp_restmags():
    n_met, n_age, n_filters, n_wave = 12, 50, 3, 1_000
    ssp_wave = np.linspace(500, 10_000, n_wave)
    ssp_fluxes = np.random.uniform(0, 1, size=(n_met, n_age, n_wave))

    filter_wave = np.linspace(ssp_wave.min(), ssp_wave.max(), 200)
    filter_waves = np.array([filter_wave for i in range(n_filters)])
    filter_trans = np.array([np.ones(filter_wave.size) for i in range(n_filters)])

    ssp_restmags = precompute_ssp_restmags(
        ssp_wave, ssp_fluxes, filter_waves, filter_trans
    )
    assert ssp_restmags.shape == (n_met, n_age, n_filters)
    assert np.all(np.isfinite(ssp_restmags))
    assert not np.all(ssp_restmags == 0)


def test_precompute_ssp_obsmags_on_z_table():
    cosmology = 0.3, -1.0, 0.0, 0.67
    n_met, n_age, n_filters, n_wave = 12, 50, 3, 1_000
    ssp_wave = np.linspace(500, 10_000, n_wave)
    ssp_fluxes = np.random.uniform(0, 1, size=(n_met, n_age, n_wave))
    filter_wave = np.linspace(ssp_wave.min(), ssp_wave.max(), 200)
    filter_waves = np.array([filter_wave for i in range(n_filters)])
    filter_trans = np.array([np.ones(filter_wave.size) for i in range(n_filters)])
    z_table = np.array((0.1, 1.0))
    n_redshift = z_table.size
    ssp_obsmags = precompute_ssp_obsmags_on_z_table(
        ssp_wave, ssp_fluxes, filter_waves, filter_trans, z_table, *cosmology
    )
    assert ssp_obsmags.shape == (n_redshift, n_met, n_age, n_filters)
    assert np.all(np.isfinite(ssp_obsmags))
    assert not np.all(ssp_obsmags == 0)
