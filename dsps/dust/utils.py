"""
"""
from jax import jit as jjit

from ..utils import trapz


@jjit
def get_filter_effective_wavelength(filter_wave, filter_trans, redshift):
    """Calculate the effective wavelength of a filter transmission curve.

    Used to approximate the attenuation curve as a constant evaluated at lambda_eff

    Parameters
    ----------
    filter_wave : ndarray of shape (n, )

    filter_trans : ndarray of shape (n, )

    redshift : float

    Returns
    -------
    lambda_eff : float

    """
    norm = trapz(filter_wave, filter_trans)
    lambda_eff_rest = trapz(filter_wave, filter_trans * filter_wave) / norm
    lambda_eff = lambda_eff_rest / (1 + redshift)
    return lambda_eff
