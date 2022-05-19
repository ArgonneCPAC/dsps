"""
"""
import os
import numpy as np
from ..attenuation_kernels import RV_C00, calzetti00_k_lambda, leitherer02_k_lambda
from ..attenuation_kernels import noll09_k_lambda, _attenuation_curve, sbl18_k_lambda


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TEST_DRN = os.path.join(_THIS_DRNAME, "testing_data")


def test_calzetii00():
    x = np.loadtxt(os.path.join(TEST_DRN, "calzetti_2000_wavelength_microns.txt"))
    k = np.loadtxt(os.path.join(TEST_DRN, "calzetti_2000_k_lambda.txt"))
    k2 = calzetti00_k_lambda(x, RV_C00)
    assert np.allclose(k, k2)

    Av = 1.0
    attenuation = np.loadtxt(os.path.join(TEST_DRN, "calzetti_2000_attenuation.txt"))
    attenuation2 = _attenuation_curve(k2, RV_C00, Av)
    assert np.allclose(attenuation, attenuation2)


def test_leitherer02():
    x = np.loadtxt(os.path.join(TEST_DRN, "leitherer_2002_wavelength_microns.txt"))
    k = np.loadtxt(os.path.join(TEST_DRN, "leitherer_2002_k_lambda.txt"))
    k2 = leitherer02_k_lambda(x, RV_C00)
    assert np.allclose(k, k2)

    Av = 1.0
    attenuation = np.loadtxt(os.path.join(TEST_DRN, "leitherer_2002_attenuation.txt"))
    attenuation2 = _attenuation_curve(k2, RV_C00, Av)
    assert np.allclose(attenuation, attenuation2)


def test_noll09():
    x = np.loadtxt(os.path.join(TEST_DRN, "noll_2009_wavelength_microns.txt"))
    kpat = "noll_2009_k_lambda{}.txt"
    apat = "noll_2009_attenuation{}.txt"
    kbnames = kpat.format(1), kpat.format(2), kpat.format(3)
    abnames = apat.format(1), apat.format(2), apat.format(3)
    for kbn, abn in zip(kbnames, abnames):
        kfn = os.path.join(TEST_DRN, kbn)
        k = np.loadtxt(kfn)
        x0, gamma, ampl, delta, av = _read_noll09_header(kfn)
        k2 = noll09_k_lambda(x, x0, gamma, ampl, delta)
        assert np.allclose(k, k2)

        afn = os.path.join(TEST_DRN, abn)
        attenuation = np.loadtxt(afn)
        attenuation2 = _attenuation_curve(k2, RV_C00, av)
        assert np.allclose(attenuation, attenuation2)


def test_salim18():
    x = np.loadtxt(os.path.join(TEST_DRN, "salim_2018_wavelength_microns.txt"))
    kpat = "salim_2018_k_lambda{}.txt"
    apat = "salim_2018_attenuation{}.txt"
    kbnames = kpat.format(1), kpat.format(2)
    abnames = apat.format(1), apat.format(2)
    for kbn, abn in zip(kbnames, abnames):
        kfn = os.path.join(TEST_DRN, kbn)
        k = np.loadtxt(kfn)
        x0, gamma, ampl, delta, av = _read_noll09_header(kfn)
        k2 = sbl18_k_lambda(x, x0, gamma, ampl, delta)
        assert np.allclose(k, k2)

        afn = os.path.join(TEST_DRN, abn)
        attenuation = np.loadtxt(afn)
        attenuation2 = _attenuation_curve(k2, RV_C00, av)
        assert np.allclose(attenuation, attenuation2)


def _read_noll09_header(fn):
    with open(fn, "r") as f:
        x0 = float(next(f).strip()[2:].split()[1])
        gamma = float(next(f).strip()[2:].split()[1])
        ampl = float(next(f).strip()[2:].split()[1])
        delta = float(next(f).strip()[2:].split()[1])
        av = float(next(f).strip()[2:].split()[1])
        return x0, gamma, ampl, delta, av


def test_salim18_fig2_dashed_black():
    fn = os.path.join(TEST_DRN, "salim18_fig2_dashed_black.txt")
    with open(fn, "r") as f:
        raw_header_line1 = next(f).strip()
        raw_header_line2 = next(f).strip()
        data_collector = []
        for data_line in f:
            data_collector.append(next(f).strip())
    delta = float(raw_header_line1.split()[-1])
    Eb = float(raw_header_line2.split()[-1])
    waves = np.array([float(x.split(",")[0]) for x in data_collector])
    att_salim18 = np.array([float(x.split(",")[1]) for x in data_collector])

    bump_width = 350
    bump_center = 2175
    att_dsps = sbl18_k_lambda(waves / 1e4, bump_center, bump_width, Eb, delta) / RV_C00
    assert np.allclose(att_salim18, att_dsps, atol=1), "delta={0}".format(delta)


def test_salim18_fig2_dashed_red():
    fn = os.path.join(TEST_DRN, "salim18_fig2_dashed_red.txt")
    with open(fn, "r") as f:
        raw_header_line1 = next(f).strip()
        raw_header_line2 = next(f).strip()
        data_collector = []
        for data_line in f:
            data_collector.append(next(f).strip())
    delta = float(raw_header_line1.split()[-1])
    Eb = float(raw_header_line2.split()[-1])
    waves = np.array([float(x.split(",")[0]) for x in data_collector])
    att_salim18 = np.array([float(x.split(",")[1]) for x in data_collector])

    bump_width = 350
    bump_center = 2175
    att_dsps = sbl18_k_lambda(waves / 1e4, bump_center, bump_width, Eb, delta) / RV_C00
    assert np.allclose(att_salim18, att_dsps, atol=1), "delta={0}".format(delta)
