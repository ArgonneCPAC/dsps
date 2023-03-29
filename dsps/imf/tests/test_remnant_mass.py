"""
"""
import os
import numpy as np
from ..remnant_mass import remnant_mass
from ..remnant_mass import SALPETER_PARAMS, CHABRIER_PARAMS
from ..remnant_mass import KROUPA_PARAMS, VAN_DOKKUM_PARAMS


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TEST_DRN = os.path.join(_THIS_DRNAME, "testing_data")


def test_surviving_mstar_default():
    log_ages_yr = np.loadtxt(os.path.join(TEST_DRN, "lg_ages_mstar_surviving.txt"))
    chabrier_msurv = np.loadtxt(os.path.join(TEST_DRN, "chabrier_mstar_surviving.txt"))
    chabrier_msurv_norem = np.loadtxt(
        os.path.join(TEST_DRN, "chabrier_mstar_surviving_norem.txt")
    )
    mstar_remnant_fsps = chabrier_msurv - chabrier_msurv_norem
    mstar_remnant_dsps = remnant_mass(log_ages_yr)
    assert np.allclose(mstar_remnant_fsps, mstar_remnant_dsps, atol=0.05)


def test_remnant_mstar_alt_imfs():
    log_ages_yr = np.loadtxt(os.path.join(TEST_DRN, "lg_ages_mstar_surviving.txt"))
    salpeter_msurv = np.loadtxt(os.path.join(TEST_DRN, "salpeter_mstar_surviving.txt"))
    chabrier_msurv = np.loadtxt(os.path.join(TEST_DRN, "chabrier_mstar_surviving.txt"))
    kroupa_msurv = np.loadtxt(os.path.join(TEST_DRN, "kroupa_mstar_surviving.txt"))
    van_dokkum_msurv = np.loadtxt(
        os.path.join(TEST_DRN, "van_dokkum_mstar_surviving.txt")
    )

    salpeter_msurv_norem = np.loadtxt(
        os.path.join(TEST_DRN, "salpeter_mstar_surviving_norem.txt")
    )
    chabrier_msurv_norem = np.loadtxt(
        os.path.join(TEST_DRN, "chabrier_mstar_surviving_norem.txt")
    )
    kroupa_msurv_norem = np.loadtxt(
        os.path.join(TEST_DRN, "kroupa_mstar_surviving_norem.txt")
    )
    van_dokkum_msurv_norem = np.loadtxt(
        os.path.join(TEST_DRN, "van_dokkum_mstar_surviving_norem.txt")
    )

    mstar_remnant_chabrier_fsps = chabrier_msurv - chabrier_msurv_norem
    mstar_remnant_salpeter_fsps = salpeter_msurv - salpeter_msurv_norem
    mstar_remnant_kroupa_fsps = kroupa_msurv - kroupa_msurv_norem
    mstar_remnant_van_dokkum_fsps = van_dokkum_msurv - van_dokkum_msurv_norem

    mstar_remnant_chabrier_dsps = remnant_mass(log_ages_yr, *CHABRIER_PARAMS.values())
    mstar_remnant_salpeter_dsps = remnant_mass(log_ages_yr, *SALPETER_PARAMS.values())
    mstar_remnant_kroupa_dsps = remnant_mass(log_ages_yr, *KROUPA_PARAMS.values())
    mstar_remnant_van_dokkum_dsps = remnant_mass(
        log_ages_yr, *VAN_DOKKUM_PARAMS.values()
    )

    assert np.allclose(
        mstar_remnant_chabrier_fsps, mstar_remnant_chabrier_dsps, atol=0.05
    )
    assert np.allclose(
        mstar_remnant_salpeter_fsps, mstar_remnant_salpeter_dsps, atol=0.05
    )
    assert np.allclose(mstar_remnant_kroupa_fsps, mstar_remnant_kroupa_dsps, atol=0.05)
    assert np.allclose(
        mstar_remnant_van_dokkum_fsps, mstar_remnant_van_dokkum_dsps, atol=0.05
    )
