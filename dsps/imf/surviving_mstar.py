"""Functions calculating the surviving stellar mass as a function of time"""
from collections import OrderedDict
from copy import deepcopy
from jax import jit as jjit
from ..utils import _sigmoid, _sig_slope


PARAMS = OrderedDict(x0=2, yhi=0.58)
DEFAULT_PARAMS = OrderedDict(
    a=0.225,
    b=8.0,
    lgk1=-0.5,
    d=0.145,
    e=0.08,
    f=6.5,
    lgk2=0.7,
    h=0.005,
)
SALPETER_PARAMS = deepcopy(DEFAULT_PARAMS)
CHABRIER_PARAMS = deepcopy(DEFAULT_PARAMS)
KROUPA_PARAMS = deepcopy(DEFAULT_PARAMS)
VAN_DOKKUM_PARAMS = deepcopy(DEFAULT_PARAMS)

SALPETER_PARAMS.update(a=0.13, d=0.055)
CHABRIER_PARAMS.update(a=0.225, d=0.145)
KROUPA_PARAMS.update(a=0.211, d=0.13)
VAN_DOKKUM_PARAMS.update(a=0.238, d=0.156)


@jjit
def surviving_mstar(
    lg_age_yr,
    a=DEFAULT_PARAMS["a"],
    b=DEFAULT_PARAMS["b"],
    lgk1=DEFAULT_PARAMS["lgk1"],
    d=DEFAULT_PARAMS["d"],
    e=DEFAULT_PARAMS["e"],
    f=DEFAULT_PARAMS["f"],
    lgk2=DEFAULT_PARAMS["lgk2"],
    h=DEFAULT_PARAMS["h"],
):
    """Calculate the fraction of stellar mass that survives as a population ages.

    Default behavior assumes Chabrier IMF.
    Calibrations for the following alternative parameters are also available:
        SALPETER_PARAMS, KROUPA_PARAMS, VAN_DOKKUM_PARAMS

    Parameters
    ----------
    lg_age_yr : ndarray of shape (n, )
        Base-10 log of the age of the stellar population in years

    Returns
    -------
    frac_surviving : ndarray of shape (n, )
        Surviving fraction

    """
    frac_returned = _returned_mass(lg_age_yr, a, b, lgk1, d, e, f, lgk2, h)
    return 1 - frac_returned


@jjit
def _returned_mass(lg_age_yr, a, b, lgk1, d, e, f, lgk2, h):
    k1 = 10**lgk1
    xtp = b
    z = _sig_slope(lg_age_yr, xtp, a, b, k1, d, e)

    k2 = 10**lgk2
    h = _sigmoid(lg_age_yr, f, k2, h, z)
    return h
