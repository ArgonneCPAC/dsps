"""Functions calculating the stellar mass locked in remnants as a function of time"""
from collections import OrderedDict
from copy import deepcopy
from jax import jit as jjit
from ..utils import _sig_slope


DEFAULT_PARAMS = OrderedDict(a=-1.42, b=7, lgk1=1, d=1.3, e=0.2)
SALPETER_PARAMS = deepcopy(DEFAULT_PARAMS)
CHABRIER_PARAMS = deepcopy(DEFAULT_PARAMS)
KROUPA_PARAMS = deepcopy(DEFAULT_PARAMS)
VAN_DOKKUM_PARAMS = deepcopy(DEFAULT_PARAMS)

SALPETER_PARAMS.update(a=-1.67)
CHABRIER_PARAMS.update(a=-1.42)
KROUPA_PARAMS.update(a=-1.45)
VAN_DOKKUM_PARAMS.update(a=-1.4)


@jjit
def remnant_mass(
    lg_age_yr,
    a=DEFAULT_PARAMS["a"],
    b=DEFAULT_PARAMS["b"],
    lgk1=DEFAULT_PARAMS["lgk1"],
    d=DEFAULT_PARAMS["d"],
    e=DEFAULT_PARAMS["e"],
):
    """Calculate the fraction of stellar mass in remnants as a population ages.

    Default behavior assumes Chabrier IMF.
    Calibrations for the following alternative parameters are also available:
        SALPETER_PARAMS, KROUPA_PARAMS, VAN_DOKKUM_PARAMS

    Parameters
    ----------
    lg_age_yr : ndarray of shape (n, )
        Base-10 log of the age of the stellar population in years

    Returns
    -------
    frac_remnant: ndarray of shape (n, )
        Remnant fraction

    """
    return _log_remnant_mass(lg_age_yr, a, b, lgk1, d, e)


@jjit
def _log_remnant_mass(lg_age_yr, a, b, lgk1, d, e):
    k1 = 10**lgk1
    xtp = b
    return 10 ** _sig_slope(lg_age_yr, xtp, a, b, k1, d, e)
