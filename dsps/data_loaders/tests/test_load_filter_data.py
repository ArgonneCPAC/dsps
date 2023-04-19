"""
"""
import os
import pytest
import numpy as np
from glob import glob
from ..load_filter_data import load_transmission_curve
from ..defaults import TransmissionCurve

DSPS_DATA_DRN = os.environ.get("DSPS_DRN", None)
ENV_VAR_MSG = "load_transmission_curve can only be tested if DSPS_DRN is in the env"

APH_DSPS_DRN = "/Users/aphearin/work/DATA/DSPS_data"


def _enforce_filter_is_sensible(filter_data):
    assert len(filter_data) == len(TransmissionCurve._fields)
    assert np.all(np.isfinite(filter_data.wave))
    assert np.all(np.isfinite(filter_data.transmission))
    assert np.all(filter_data.transmission >= 0)
    assert np.all(filter_data.transmission <= 1)
    assert np.any(filter_data.transmission > 0)


@pytest.mark.skipif(DSPS_DATA_DRN is None, reason=ENV_VAR_MSG)
def test_load_any_existent_filter_data_from_fnames():
    drn = os.path.join(DSPS_DATA_DRN, "filters")
    fn_list = glob(os.path.join(drn, "*transmission.h5"))
    for fn in fn_list:
        filter_data = load_transmission_curve(fn=fn)
        _enforce_filter_is_sensible(filter_data)


@pytest.mark.skipif(DSPS_DATA_DRN is None, reason=ENV_VAR_MSG)
def test_load_any_existent_filter_data_from_bnpat():
    drn = os.path.join(DSPS_DATA_DRN, "filters")
    fn_list = glob(os.path.join(drn, "*transmission.h5"))
    bn_list = [os.path.basename(fn) for fn in fn_list]

    if APH_DSPS_DRN == DSPS_DATA_DRN:
        assert len(bn_list) > 0

    for bn_pat in bn_list:
        filter_data = load_transmission_curve(bn_pat=bn_pat)
        _enforce_filter_is_sensible(filter_data)

    for bn_pat in bn_list:
        filter_data = load_transmission_curve(drn=drn, bn_pat=bn_pat)
        _enforce_filter_is_sensible(filter_data)
