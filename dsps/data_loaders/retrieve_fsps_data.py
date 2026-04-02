"""Use python-fsps to retrieve a block of Simple Stellar Population (SSP) data"""
from collections import namedtuple

import numpy as np

try:
    import fsps

    HAS_FSPS = True
except (ImportError, RuntimeError):
    HAS_FSPS = False

from pathlib import Path

from .defaults import SSPData

BASE_PATH = Path(__file__).resolve().parent.parent
EMLINES_INFO_PATH = BASE_PATH / "data/emlines_info.dat"


def get_fsps_emline_info(fn=EMLINES_INFO_PATH):
    with open(fn, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        ref_emline_wave = np.array(
            [line.split(",")[0] for line in lines], dtype="float64"
        )
        ref_emline_name = np.array([line.split(",")[1] for line in lines])
    return ref_emline_wave, ref_emline_name


def get_matched_emline_name(ref_emline_wave, ref_emline_name, find_emline_wave):
    """
    ref_emline_wave: ndarray of shape (n_lines, )
        Array of emission line wavelength in Angstroms from "emline_info.dat"
    ref_emline_name: ndarray of shape (n_lines, )
        Array of emission line names from "emline_info.dat"
    find_emline_wave: float64
        emission line wavelength to be found a match for in "emline_info.dat"

    """
    isclose = np.isclose(ref_emline_wave, find_emline_wave, atol=0, rtol=1e-6)
    if isclose.sum() == 1:
        return ref_emline_name[isclose].item()
    else:
        return ""


def retrieve_ssp_data_from_fsps(add_neb_emission=True, **kwargs):
    """Use python-fsps to populate arrays and matrices of data
    for the default simple stellar populations (SSPs) in the shapes expected by DSPS

    Parameters
    ----------
    add_neb_emission : bool, optional
        Argument passed to fsps.StellarPopulation. Default is True.

    kwargs : optional
        Any keyword arguments passed to the retrieve_ssp_data_from_fsps function will be
        passed on to fsps.StellarPopulation.

    Returns
    -------
    ssp_lgmet : ndarray of shape (n_met, )
        Array of log10(Z) of the SSP templates
        where dimensionless Z is the mass fraction of elements heavier than He

    ssp_lg_age_gyr : ndarray of shape (n_ages, )
        Array of log10(age/Gyr) of the SSP templates

    ssp_wave : ndarray of shape (n_wave, )

    ssp_flux : ndarray of shape (n_met, n_ages, n_wave)
        SED of the SSP in units of Lsun/Hz/Msun

    ssp_emline_name (optional): ndarray of shape (n_lines, )
        string Array of line names

    ssp_emline_wave (optional): ndarray of shape (n_lines, )
        Array of line wavelengths in Angstroms

    ssp_emline_luminosity (optional): ndarray of shape (n_met, n_age, n_lines)
        Array of emission line luminosities in units of Lsun/Msun

    Notes
    -----
    The retrieve_ssp_data_from_fsps function is just a wrapper around
    python-fsps without any other dependencies. This standalone function
    should be straightforward to modify to use python-fsps to build
    alternate SSP data blocks.

    All DSPS functions operate on plain ndarrays, so user-supplied data
    storing alternate SSP models is supported. You will just need to
    pack your SSP data into arrays with shapes matching the shapes of
    the arrays returned by this function.

    """
    assert HAS_FSPS, "Must have python-fsps installed to use this function"

    sp = fsps.StellarPopulation(
        zcontinuous=0,
        **kwargs,
    )
    ssp_lgmet = np.log10(sp.zlegend)
    nzmet = ssp_lgmet.size
    ssp_lg_age_gyr = sp.log_age - 9.0
    spectrum_collector = []
    if hasattr(sp, "emline_luminosity") is True:
        emline_luminosity_collector = []
    for zmet_indx in range(1, ssp_lgmet.size + 1):
        print("...retrieving zmet = {0} of {1}".format(zmet_indx, nzmet))
        sp = fsps.StellarPopulation(
            zcontinuous=0,
            zmet=zmet_indx,
            add_neb_emission=add_neb_emission,
            **kwargs,
        )
        _wave, _fluxes = sp.get_spectrum()
        spectrum_collector.append(_fluxes)

        if hasattr(sp, "emline_luminosity") is True:
            emline_luminosity_collector.append(sp.emline_luminosity)

    ssp_wave = np.array(_wave)
    ssp_flux = np.array(spectrum_collector)

    if hasattr(sp, "emline_luminosity") is True:
        ssp_emline_wave = np.array(sp.emline_wavelengths)
        ssp_emline_luminosity = np.array(emline_luminosity_collector)
        ref_emline_wave, ref_emline_name = get_fsps_emline_info()
        ssp_emline_name = [
            get_matched_emline_name(ref_emline_wave, ref_emline_name, find_emline)
            for find_emline in ssp_emline_wave
        ]
        emline_fields = [
            name.decode("utf-8")
            .replace(".", "p")
            .replace("-", "_")
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            for name in ssp_emline_name
        ]
        ssp_emlines = _get_emlines_nested_namedtuple(
            emline_fields, ssp_emline_wave, ssp_emline_luminosity
        )
        return SSPData(
            ssp_lgmet,
            ssp_lg_age_gyr,
            ssp_wave,
            ssp_flux,
            ssp_emlines,
        )
    else:
        return SSPData(ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux)


def _get_emlines_nested_namedtuple(fields, emline_wave, emline_luminosity):
    EmissionLine = namedtuple("EmissionLine", ["emline_wave", "emline_luminosity"])
    values = [
        EmissionLine(emline_wave[i].item(), emline_luminosity[:, :, i])
        for i in range(len(fields))
    ]

    EmissionLines = namedtuple("EmissionLines", fields)
    emlines = EmissionLines(*values)
    return emlines
