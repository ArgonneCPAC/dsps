"""
"""
import os
import numpy as np

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TASSO_DRN = "/Users/aphearin/work/DATA/SPS_validation/FSPS_ssp_data"
BEBOP_DRN = "/lcrc/project/halotools/FSPS_ssp_data"


def load_fsps_testing_data(drn, imet=20, iage=5):
    zlegend = np.load(os.path.join(drn, "zlegend.npy"))
    lgZsun_bin_mids = np.log10(zlegend / zlegend[-3])

    log_age_gyr = np.load(os.path.join(drn, "log_age.npy")) - 9

    ssp_bname = "fsps_ssp_imet_{0}_iage_{1}.npy".format(imet, iage)
    ssp_data = np.load(os.path.join(drn, ssp_bname))

    # Interpolate the filters so that they all have the same length
    filter_names = "u", "g", "r", "i", "z", "y"
    fwpat = "lsst_{}_transmission.npy"
    filters = [np.load(os.path.join(drn, fwpat.format(band))) for band in filter_names]
    filter_size = np.max([f.shape[0] for f in filters])
    filter_waves_out = []
    filter_trans_out = []
    for f in filters:
        wave, trans = f["wave"], f["transmission"]
        xout = np.linspace(wave.min(), wave.max(), filter_size)
        yout = np.interp(xout, wave, trans)
        filter_waves_out.append(xout)
        filter_trans_out.append(yout)
    dt_list = []
    for filter_name in filter_names:
        colname = "{0}_filter_wave".format(filter_name)
        dt_list.append((colname, "f4"))
        colname = "{0}_filter_trans".format(filter_name)
        dt_list.append((colname, "f4"))
    dt = np.dtype(dt_list)
    filter_data = np.zeros(filter_size, dtype=dt)
    for ifilter, filter_name in enumerate(filter_names):
        colname = "{0}_filter_wave".format(filter_name)
        filter_data[colname] = filter_waves_out[ifilter]
        colname = "{0}_filter_trans".format(filter_name)
        filter_data[colname] = filter_trans_out[ifilter]

    return filter_data, ssp_data, lgZsun_bin_mids, log_age_gyr
