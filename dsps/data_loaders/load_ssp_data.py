"""
"""
from collections import OrderedDict
import os
import h5py

from .defaults import DEFAULT_SSP_BNAME, DEFAULT_SSP_KEYS, SSPData


def load_default_ssp_templates(
    fn=None,
    drn=None,
    bn=DEFAULT_SSP_BNAME,
    ssp_keys=DEFAULT_SSP_KEYS,
):
    """Load SSP templates from disk, defaulting to DSPS package data location

    Parameters
    ----------
    fn : string, optional
        Absolute path to hdf5 file storing the SSP data.
        This argument supersedes drn and bn.
        Default is None, in which case DSPS will look
        in the directory stored in the DSPS_DRN environment variable

    drn : string, optional
        Directory to hdf5 file storing the SSP data
        This argument is only used if fn is not supplied
        Default behavior is the DSPS_DRN environment variable

    bn : string, optional
        Basename of hdf5 file storing the SSP data
        This argument is only used if fn is not supplied

    """
    if fn is None:
        if drn is None:
            try:
                drn = os.environ["DSPS_DRN"]
            except KeyError:
                msg = (
                    "Since you did not pass the fn or drn argument\n"
                    "then you must have the DSPS_DRN environment variable set"
                )
                raise ValueError(msg)
        fn = os.path.join(drn, bn)
    assert os.path.isfile(fn), "{0} does not exist".format(fn)

    ssp_data = OrderedDict()
    with h5py.File(fn, "r") as hdf:
        for key in hdf:
            ssp_data[key] = hdf[key][...]

    return SSPData(*[ssp_data[key] for key in ssp_keys])
