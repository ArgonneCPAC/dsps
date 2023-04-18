"""
"""
import numpy as np
import os
from jax import jit as jjit, vmap
from ..photometry_kernels import _calc_obs_mag_no_dimming, calc_rest_mag, calc_obs_mag
from ...cosmology.flat_wcdm import FSPS_COSMO


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DATA_DRN = os.path.join(
    os.path.dirname(os.path.dirname(_THIS_DRNAME)), "tests", "testing_data"
)
LSST_BAND_FNPAT = "lsst_{}_transmission.npy"

_a = (*[None] * 4, 0, *[None] * 4)
_calc_obs_mag_vmap_z = jjit(vmap(calc_obs_mag, in_axes=_a))


def test_obs_mag_no_dimming_agrees_with_rest_mag_at_z0():
    z_obs = 0.0
    ssp_bnames = ("fsps_ssp_imet_20_iage_5.npy", "fsps_ssp_imet_2_iage_90.npy")
    ssp_fnames = [os.path.join(DATA_DRN, bn) for bn in ssp_bnames]

    for ssp_fn in ssp_fnames:
        ssp_data = np.load(ssp_fn)
        wave_spec = ssp_data["wave"]
        lum_spec = ssp_data["flux"]

        for band in ("u", "g", "r", "i", "z", "y"):
            fn = os.path.join(DATA_DRN, LSST_BAND_FNPAT.format(band))
            filter_data = np.load(fn)
            band_wave = filter_data["wave"]
            band_trans = filter_data["transmission"]
            mags_obs = _calc_obs_mag_no_dimming(
                wave_spec, lum_spec, band_wave, band_trans, z_obs
            )
            mags_rest = calc_rest_mag(wave_spec, lum_spec, band_wave, band_trans)
            assert np.allclose(mags_obs, mags_rest)


def test_obs_mag_agrees_with_fsps_across_redshift():
    zobs_ray = np.load(os.path.join(DATA_DRN, "fsps_lsst_mags_zobs.npy"))
    ssp_bnames = ("fsps_ssp_imet_20_iage_5.npy", "fsps_ssp_imet_2_iage_90.npy")
    ssp_fnames = [os.path.join(DATA_DRN, bn) for bn in ssp_bnames]

    ssp_mag_bnames = (
        "fsps_lsst_mags_zobs_imet_20_iage_5.npy",
        "fsps_lsst_mags_zobs_imet_2_iage_90.npy",
    )
    ssp_mag_fnames = [os.path.join(DATA_DRN, bn) for bn in ssp_mag_bnames]

    gen = zip(ssp_fnames, ssp_mag_fnames)
    for ssp_fn, ssp_mag_fn in gen:
        ssp_data = np.load(ssp_fn)
        wave_spec = ssp_data["wave"]
        lum_spec = ssp_data["flux"]

        mag_data = np.load(ssp_mag_fn)
        for iband, band in enumerate(("u", "g", "r", "i", "z", "y")):
            filter_data = np.load(os.path.join(DATA_DRN, LSST_BAND_FNPAT.format(band)))
            band_wave = filter_data["wave"]
            band_trans = filter_data["transmission"]

            pred_mags = _calc_obs_mag_vmap_z(
                wave_spec, lum_spec, band_wave, band_trans, zobs_ray, *FSPS_COSMO
            )

            assert np.allclose(mag_data[band], pred_mags, atol=0.05)
