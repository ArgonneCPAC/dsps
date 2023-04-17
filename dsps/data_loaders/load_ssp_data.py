"""
"""
import typing
from collections import OrderedDict
import os
import h5py
import numpy as np

from .defaults import DEFAULT_SSP_BNAME, DEFAULT_SSP_KEYS


class SSPData(typing.NamedTuple):
    """NamedTuple with info about SSP templates

    ssp_lgmet : ndarray of shape (n_met, )
        Array of log10(Z) of the SSP templates
        where dimensionless Z is the mass fraction of elements heavier than He

    ssp_lg_age : ndarray of shape (n_ages, )
        Array of log10(age/Gyr) of the SSP templates

    ssp_wave : ndarray of shape (n_wave, )

    ssp_flux : ndarray of shape (n_met, n_ages, n_wave)
        SED of the SSP in units of Lsun/Hz/Msun

    """

    ssp_lgmet: np.ndarray
    ssp_lg_age: np.ndarray
    ssp_wave: np.ndarray
    ssp_flux: np.ndarray


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
            drn = os.environ["DSPS_DRN"]
        fn = os.path.join(drn, bn)
    assert os.path.isfile(fn), "{0} does not exist".format(fn)

    ssp_data = OrderedDict()
    with h5py.File(fn, "r") as hdf:
        for key in hdf:
            ssp_data[key] = hdf[key][...]

    return SSPData(*[ssp_data[key] for key in ssp_keys])
