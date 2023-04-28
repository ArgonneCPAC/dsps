"""Use python-fsps to retrieve a block of Simple Stellar Population (SSP) data"""
import numpy as np

try:
    import fsps

    HAS_FSPS = True
except ImportError:
    HAS_FSPS = False

from .load_ssp_data import SSPData


def retrieve_ssp_data_from_fsps():
    """Use python-fsps to populate arrays and matrices of data
    for the default simple stellar populations (SSPs) in the shapes expected by DSPS

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

    sp = fsps.StellarPopulation(zcontinuous=0)
    ssp_lgmet = np.log10(sp.zlegend)
    nzmet = ssp_lgmet.size
    ssp_lg_age_gyr = sp.log_age - 9.0
    spectrum_collector = []
    for zmet_indx in range(1, ssp_lgmet.size + 1):
        print("...retrieving zmet = {0} of {1}".format(zmet_indx, nzmet))
        sp = fsps.StellarPopulation(
            zcontinuous=0, zmet=zmet_indx, add_neb_emission=True
        )
        _wave, _fluxes = sp.get_spectrum()
        spectrum_collector.append(_fluxes)
    ssp_wave = np.array(_wave)
    ssp_flux = np.array(spectrum_collector)

    return SSPData(ssp_lgmet, ssp_lg_age_gyr, ssp_wave, ssp_flux)
