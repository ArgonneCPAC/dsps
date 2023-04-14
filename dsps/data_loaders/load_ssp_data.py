"""
"""
from collections import OrderedDict
import os
import h5py


def load_default_ssp_templates(drn=None, bn=None):
    if drn is None:
        drn = os.environ["DSPS_DRN"]
    if bn is None:
        raise NotImplementedError("WIP")
    fn = os.path.join(drn, bn)
    assert os.path.isfile(fn), "{0} does not exist".format(fn)

    ssp_data = OrderedDict()
    with h5py.File(fn, "r") as hdf:
        for key in hdf:
            ssp_data[key] = hdf[key][...]

    return ssp_data
