"""
"""
import os
from collections import OrderedDict

import h5py

from .defaults import DEFAULT_SSP_BNAME, DEFAULT_SSP_KEYS, SSPData
from .retrieve_fake_fsps_data import load_fake_ssp_data
from .retrieve_fsps_data import _get_emlines_nested_namedtuple


def load_ssp_templates(
    fn=None,
    drn=None,
    bn=DEFAULT_SSP_BNAME,
    default_ssp_keys=DEFAULT_SSP_KEYS,
    dummy=False,
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

    Returns
    -------
    NamedTuple with 4(+3 optional) entries storing info about SSP templates

        ssp_lgmet : ndarray of shape (n_met, )
            Array of log10(Z) of the SSP templates
            where dimensionless Z is the mass fraction of elements heavier than He

        ssp_lg_age_gyr : ndarray of shape (n_ages, )
            Array of log10(age/Gyr) of the SSP templates

        ssp_wave : ndarray of shape (n_wave, )

        ssp_flux : ndarray of shape (n_met, n_ages, n_wave)
            SED of the SSP in units of Lsun/Hz/Msun

        ssp_emlines (optional):
            namedtuple with n_lines fields, one for each emission line

            a further nested namedtuple which stores the following fields
            on each emission line:
                emline_wave: float
                    line wavelength in Angstroms

                emline_luminosity: ndarray of shape (n_met, n_age)
                    Array of emission line luminosities in units of Lsun/Msun

    """
    if dummy:
        return load_fake_ssp_data()

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

    ssp_data_dict = OrderedDict()

    with h5py.File(fn, "r") as hdf:
        for key in default_ssp_keys:
            ssp_data_dict[key] = hdf[key][...]

        if "ssp_emlines" in hdf.keys():
            grp = hdf["ssp_emlines"]
            ssp_emline_name = grp["ssp_emline_name"][...]
            ssp_emline_name = [name.decode("utf-8") for name in ssp_emline_name]
            ssp_emline_wave = grp["ssp_emline_wave"][...]
            ssp_emline_luminosity = grp["ssp_emline_luminosity"][...]
            ssp_data_dict["ssp_emlines"] = _get_emlines_nested_namedtuple(
                ssp_emline_name, ssp_emline_wave, ssp_emline_luminosity
            )

        return SSPData(**ssp_data_dict)
