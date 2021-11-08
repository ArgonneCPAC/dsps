"""
"""
from .retrieve_fake_fsps_data import load_fake_sps_data


def test_load_fake_sps_data():
    ret = load_fake_sps_data()
    filter_waves, filter_trans, wave_ssp, spec_ssp, lgZsun_bin_mids, log_age_gyr = ret
