""""""

import numpy as np
import pytest

try:
    from astropy import units as u

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = True

from .. import constants

NO_ASTROPY_MSG = "Must have astropy installed to run this unit test"


@pytest.mark.skipif(not HAS_ASTROPY, reason=NO_ASTROPY_MSG)
def test_lsun_cgs_agrees_with_astropy():
    lsun_cgs = u.Lsun.to(u.erg / u.s)
    assert np.allclose(lsun_cgs, constants.L_SUN_CGS, rtol=1e-3)
