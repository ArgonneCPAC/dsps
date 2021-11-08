"""
"""
import numpy as np
from ..equivalent_width import _ew_kernel
from ..utils import triweight_gaussian


def get_fake_exp_continuum(exp_peak):
    lgpeak = np.log10(exp_peak)
    x = np.logspace(lgpeak - 2, lgpeak + 1, 500)
    y = 3.2 * np.exp(-0.5 * (x - exp_peak) ** 2 / 10 ** 2)
    return x, y


def get_fake_line(x, line, width, area):
    return triweight_gaussian(x, line, width) * area


def test_ew_kernel():
    wave, continuum = get_fake_exp_continuum(2)
    line_mid, line_width, line_area = 4, 0.05, 0.04
    line = get_fake_line(wave, line_mid, line_width, line_area)
    w1, w2 = line_mid - 15 * line_width, line_mid - 6 * line_width
    w3, w4 = line_mid + 6 * line_width, line_mid + 15 * line_width
    ew, inferred_line_area = _ew_kernel(wave, continuum + line, w1, w2, w3, w4)
    assert np.allclose(line_area, inferred_line_area, rtol=0.02)
