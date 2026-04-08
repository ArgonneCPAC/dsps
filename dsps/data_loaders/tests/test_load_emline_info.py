""""""

from importlib.resources import files

import numpy as np

from .. import load_emline_info as lemi
from ..load_ssp_data import load_ssp_templates

EMLINES_INFO_BNAME = "emlines_info.dat"


def test_read_emlines_info_fsps_returns_something():
    fn = files("dsps").joinpath("data", EMLINES_INFO_BNAME)
    emline_dict, line_wave_arr = lemi.read_emlines_info_fsps(fn, testmode=True)
    assert np.allclose(np.array(list(emline_dict.values())), line_wave_arr, rtol=1e-3)


def test_read_emlines_info_fsps_field_names():
    ssp_data = load_ssp_templates(
        bn="fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5"
    )
    fn = files("dsps").joinpath("data", EMLINES_INFO_BNAME)
    emline_dict, line_wave_arr = lemi.read_emlines_info_fsps(fn, testmode=True)

    seq = [x != y for x, y in zip(emline_dict.keys(), ssp_data.ssp_emline_wave._fields)]
    n_disagree = sum(seq)
    assert n_disagree == 0


def test_read_emlines_info_fsps_line_wave_values():
    ssp_data = load_ssp_templates(
        bn="fsps_v0.4.7_mist_c3k_a_kroupa_wNE_logGasU-2.0_logGasZ0.0.h5"
    )
    fn = files("dsps").joinpath("data", EMLINES_INFO_BNAME)
    emline_dict, line_wave_arr = lemi.read_emlines_info_fsps(fn, testmode=True)

    assert np.allclose(ssp_data.ssp_emline_wave, line_wave_arr, rtol=1e-3)
