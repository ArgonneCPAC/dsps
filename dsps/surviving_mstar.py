"""
"""
from collections import OrderedDict
from jax import jit as jjit
from .utils import _tw_sigmoid


PARAMS = OrderedDict(x0=2, yhi=0.58)


@jjit
def _surviving_mstar(lg_age_myr):
    """Calculate the fraction of stellar mass that survives as a population ages.

    Parameters
    ----------
    lg_age_myr : ndarray of shape (n, )
        Base-10 log of the age of the stellar population in Myr

    Returns
    -------
    frac_surviving : ndarray of shape (n, )
        Surviving fraction

    """
    frac_surviving = _tw_sigmoid(lg_age_myr, PARAMS["x0"], 1, 1, PARAMS["yhi"])
    return frac_surviving
