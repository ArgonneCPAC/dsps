"""
"""
import os
import numpy as np
from ..surviving_mstar import _surviving_mstar


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def test_surviving_mstar():
    TEST_DRN = os.path.join(_THIS_DRNAME, "testing_data")
    lgageray_myr = np.loadtxt(os.path.join(TEST_DRN, "lgageray_myr.txt"))
    mstar_surviving_fsps = np.loadtxt(os.path.join(TEST_DRN, "mstar_surviving.txt"))
    mstar_surviving_dsps = _surviving_mstar(lgageray_myr)
    assert np.allclose(mstar_surviving_fsps, mstar_surviving_dsps, atol=0.025)
