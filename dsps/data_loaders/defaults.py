"""Globals storing defaults for various DSPS options"""

import typing

import numpy as np

DEFAULT_SSP_BNAME = "ssp_data_fsps_v3.2_lgmet_age.h5"


class SSPData(typing.NamedTuple):
    """NamedTuple with 4 (+1 optional) entries storing info about SSP templates

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

    ssp_lgmet: np.ndarray
    ssp_lg_age_gyr: np.ndarray
    ssp_wave: np.ndarray
    ssp_flux: np.ndarray
    ssp_emlines: np.ndarray = None


DEFAULT_SSP_KEYS = ("ssp_lgmet", "ssp_lg_age_gyr", "ssp_wave", "ssp_flux")


class TransmissionCurve(typing.NamedTuple):
    """NamedTuple with 2 entries storing info about the transmission curve

    wave : ndarray of shape (n, )
        Array of λ/AA

    transmission : ndarray of shape (n, )
        Fraction of the flux transmitted through the filter

    """

    wave: np.ndarray
    transmission: np.ndarray
