"""
"""
import os
import numpy as np

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TASSO_DRN = "/Users/aphearin/work/DATA/SPS_validation/FSPS_ssp_data"


def load_fsps_data(drn):
    zlegend = np.load(os.path.join(drn, "zlegend.npy"))
    lgZsun_bin_mids = np.log10(zlegend / zlegend[-3])

    log_age_gyr = np.load(os.path.join(drn, "log_age.npy")) - 9

    spec_ssp = np.load(os.path.join(drn, "ssp_spec_flux_lines.npy"))
    wave_ssp = np.load(os.path.join(drn, "ssp_spec_wave.npy"))

    # Interpolate the filters so that they all have the same length
    filter_names = "u", "g", "r", "i", "z", "y"
    fwpat = "lsst_{}_transmission.npy"
    filters = [np.load(os.path.join(drn, fwpat.format(band))) for band in filter_names]
    filter_size = np.max([f.shape[0] for f in filters])
    filter_waves_out = []
    filter_trans_out = []
    for f in filters:
        xout = np.linspace(f[:, 0].min(), f[:, 0].max(), filter_size)
        yout = np.interp(xout, f[:, 0], f[:, 1])
        filter_waves_out.append(xout)
        filter_trans_out.append(yout)
    filter_waves = np.array(filter_waves_out)
    filter_trans = np.array(filter_trans_out)

    return filter_waves, filter_trans, wave_ssp, spec_ssp, lgZsun_bin_mids, log_age_gyr
