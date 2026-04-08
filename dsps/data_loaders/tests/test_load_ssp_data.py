""" """

import os

import numpy as np
import pytest

from ..defaults import DEFAULT_SSP_BNAME_EMLINES, SSPData
from ..load_ssp_data import load_ssp_templates

DSPS_DATA_DRN = os.environ.get("DSPS_DRN", None)
ENV_VAR_MSG = "load_ssp_templates can only be tested if DSPS_DRN is in the env"


HAS_EMLINE_DSPS_DATA = False
if os.path.isdir(DSPS_DATA_DRN):
    if os.path.isfile(os.path.join(DSPS_DATA_DRN, DEFAULT_SSP_BNAME_EMLINES)):
        HAS_EMLINE_DSPS_DATA = True
NO_EMLINE_DSPS_DATA = f"Must have {DEFAULT_SSP_BNAME_EMLINES} to run this test"


@pytest.mark.skipif(DSPS_DATA_DRN is None, reason=ENV_VAR_MSG)
def test_load_ssp_templates():
    ssp_data = load_ssp_templates(drn=DSPS_DATA_DRN)
    assert len(ssp_data) == len(SSPData._fields)


def test_load_dummy_ssp_templates():
    ssp_data = load_ssp_templates(dummy=True)
    assert len(ssp_data) == len(SSPData._fields)


def test_and_freeze_sspdata_field_names():
    ssp_data = SSPData(None, None, None, None, None, None)
    expected_names = (
        "ssp_lgmet",
        "ssp_lg_age_gyr",
        "ssp_wave",
        "ssp_flux",
        "ssp_emline_wave",
        "ssp_emline_luminosity",
    )
    assert set(expected_names) == set(ssp_data._fields)


@pytest.mark.skipif(HAS_EMLINE_DSPS_DATA is False, reason=NO_EMLINE_DSPS_DATA)
def test_load_ssp_templates_emlines():
    ssp_data = load_ssp_templates(
        bn=DEFAULT_SSP_BNAME_EMLINES, ssp_keys=SSPData._fields
    )
    n_met = len(ssp_data.ssp_lgmet)
    n_age = len(ssp_data.ssp_lg_age_gyr)
    n_lines = len(ssp_data.ssp_emline_wave)

    assert ssp_data.ssp_emline_luminosity.shape == (n_met, n_age, n_lines)
    n_wave = ssp_data.ssp_flux.shape[-1]
    assert ssp_data.ssp_flux.shape == (n_met, n_age, n_wave)
    for x in ssp_data:
        assert np.all(np.isfinite(x))
