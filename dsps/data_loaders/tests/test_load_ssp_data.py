"""
"""
import os
import pytest
from ..load_ssp_data import load_ssp_templates
from ..defaults import SSPData

DSPS_DATA_DRN = os.environ.get("DSPS_DRN", None)
ENV_VAR_MSG = "load_ssp_templates can only be tested if DSPS_DRN is in the env"


@pytest.mark.skipif(DSPS_DATA_DRN is None, reason=ENV_VAR_MSG)
def test_load_ssp_templates():
    ssp_data = load_ssp_templates(drn=DSPS_DATA_DRN)
    assert len(ssp_data) == len(SSPData._fields)
