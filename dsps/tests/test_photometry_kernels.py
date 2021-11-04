"""
"""
import numpy as np
import os
from ..photometry_kernels import _calc_obs_mag_no_dimming, _calc_rest_mag
from ..flat_wcdm import FSPS_COSMO


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DATA_DRN = os.path.join(os.path.dirname(_THIS_DRNAME), "data")
LSST_BAND_FNPAT = "lsst_{}_transmission.npy"


def test_obs_mag_agrees_with_rest_mag_at_z0():
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
            mags_rest = _calc_rest_mag(wave_spec, lum_spec, band_wave, band_trans)
            assert np.allclose(mags_obs, mags_rest)
