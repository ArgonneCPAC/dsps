""" """

import os
import random
from glob import glob

import h5py
import numpy as np
from jax import random as jran
from jax.scipy.stats import norm

from .defaults import TransmissionCurve

TOL = 0.01


def load_transmission_curve(fn=None, bn_pat=None, drn=None):
    """Load filter transmission curves from disk,
    defaulting to DSPS package data location

    Parameters
    ----------
    fn : string, optional
        Absolute path to file storing the transmission curve.
        This argument supersedes drn and bn_pat.
        Default is None, in which case DSPS will look in the filters subdirectory
        of the drn stored in the DSPS_DRN environment variable

    bn_pat : string, optional
        Basename pattern of file storing the transmission curve.
        For example, "lsst_r*" or "suprimecam_b*"
        There must exist a uniquely matching file in the input drn.

    drn : string, optional
        Directory to files storing filter transmission curves
        This argument is only used if fn is not supplied
        Default behavior is the filters subdirectory
        of the drn stored in the DSPS_DRN environment variable

    bn : string, optional
        Basename of file storing the filter transmission curve.
        This argument is only used if fn is not supplied

    Returns
    -------
    NamedTuple with 2 entries storing the transmission curve

        wave : ndarray of shape (n, )
            Array of λ/AA

        transmission : ndarray of shape (n, )
            Fraction of the flux transmitted through the filter

    """
    if fn is None:
        if drn is None:
            try:
                drn = os.environ["DSPS_DRN"]
                drn = os.path.join(drn, "filters")
                assert os.path.isdir(drn), "Directory does not exist:\n{0}".format(drn)
            except KeyError:
                msg = (
                    "Since you did not pass the fn or drn argument\n"
                    "then you must have the DSPS_DRN environment variable set"
                )
                raise ValueError(msg)

        fn_pat = os.path.join(drn, bn_pat)
        fn_list = glob(fn_pat)
        msg = "There is no filename with pattern {0} located in {1}"
        assert len(fn_list) != 0, msg.format(bn_pat, drn)
        msg = "There is more than one matching filename with pattern {0} located in {1}"
        assert len(fn_list) < 2, msg.format(bn_pat, drn)
        fn = fn_list[0]

    assert os.path.isfile(fn), "{0} does not exist".format(fn)
    with h5py.File(fn, "r") as hdf:
        wave = hdf["wave"][...]
        transmission = hdf["transmission"][...]

        if transmission.min() < -TOL:
            raise ValueError(f"Negatively-valued transmission curve for {fn}")
        elif transmission.min() > 1 + TOL:
            raise ValueError(f"transmission curve exceeds unity for {fn}")
        else:
            transmission = np.clip(transmission, 0.0, 1.0)

    return TransmissionCurve(wave, transmission)


def load_random_transmission_curve(
    ran_key=None, tcurve_center=None, wave_range=(1_000, 10_000), scale=300
):
    if ran_key is None:
        seed = random.randint(0, 2**32 - 1)
        ran_key = jran.key(seed)

    if tcurve_center is None:
        xmin = wave_range[0] + scale * 2
        xmax = wave_range[1] - scale * 2
        tcurve_center = jran.uniform(ran_key, minval=xmin, maxval=xmax)
    else:
        assert wave_range[0] < tcurve_center < wave_range[1]

    wave = np.linspace(*wave_range, 200)

    _transmission = norm.pdf(wave, loc=tcurve_center, scale=scale)
    transmission = _transmission / _transmission.max()
    tcurve = TransmissionCurve(wave, transmission)

    return tcurve
