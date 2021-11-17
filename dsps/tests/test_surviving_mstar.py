"""
"""
import os
import numpy as np
from ..surviving_mstar import surviving_mstar
from ..surviving_mstar import _returned_mass
from ..surviving_mstar import SALPETER_PARAMS, CHABRIER_PARAMS
from ..surviving_mstar import KROUPA_PARAMS, VAN_DOKKUM_PARAMS


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TEST_DRN = os.path.join(_THIS_DRNAME, "testing_data")


def test_surviving_mstar_default():
    log_ages_yr = np.loadtxt(os.path.join(TEST_DRN, "lg_ages_mstar_surviving.txt"))
    mstar_surviving_fsps = np.loadtxt(
        os.path.join(TEST_DRN, "chabrier_mstar_surviving.txt")
    )
    mstar_surviving_dsps = surviving_mstar(log_ages_yr)
    assert np.allclose(mstar_surviving_fsps, mstar_surviving_dsps, atol=0.02)


def test_surviving_mstar_alt_imfs():
    log_ages_yr = np.loadtxt(os.path.join(TEST_DRN, "lg_ages_mstar_surviving.txt"))
    salpeter_msurv_fsps = np.loadtxt(
        os.path.join(TEST_DRN, "salpeter_mstar_surviving.txt")
    )
    chabrier_msurv_fsps = np.loadtxt(
        os.path.join(TEST_DRN, "chabrier_mstar_surviving.txt")
    )
    kroupa_msurv_fsps = np.loadtxt(os.path.join(TEST_DRN, "kroupa_mstar_surviving.txt"))
    van_dokkum_msurv_fsps = np.loadtxt(
        os.path.join(TEST_DRN, "van_dokkum_mstar_surviving.txt")
    )

    salpeter_msurv_dsps = 1 - _returned_mass(log_ages_yr, *SALPETER_PARAMS.values())
    assert np.allclose(salpeter_msurv_dsps, salpeter_msurv_fsps, atol=0.02)

    chabrier_msurv_dsps = 1 - _returned_mass(log_ages_yr, *CHABRIER_PARAMS.values())
    assert np.allclose(chabrier_msurv_dsps, chabrier_msurv_fsps, atol=0.02)

    kroupa_msurv_dsps = 1 - _returned_mass(log_ages_yr, *KROUPA_PARAMS.values())
    assert np.allclose(kroupa_msurv_dsps, kroupa_msurv_fsps, atol=0.02)

    van_dokkum_msurv_dsps = 1 - _returned_mass(log_ages_yr, *VAN_DOKKUM_PARAMS.values())
    assert np.allclose(van_dokkum_msurv_dsps, van_dokkum_msurv_fsps, atol=0.02)
