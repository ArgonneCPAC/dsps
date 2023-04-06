"""
"""
import os
import numpy as np
from ..attenuation_kernels import RV_C00, calzetti00_k_lambda, leitherer02_k_lambda
from ..attenuation_kernels import noll09_k_lambda, _attenuation_curve, sbl18_k_lambda
from ..attenuation_kernels import triweight_k_lambda, _l02_below_c00_above
from ..attenuation_kernels import _get_filter_effective_wavelength
from ..attenuation_kernels import _get_effective_attenuation, _get_eb_from_delta

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
        x0, gamma, ampl, slope, av = _read_noll09_header(kfn)
        k2 = noll09_k_lambda(x, x0, gamma, ampl, slope)
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
        x0, gamma, ampl, slope, av = _read_noll09_header(kfn)
        k2 = sbl18_k_lambda(x, x0, gamma, ampl, slope)
        assert np.allclose(k, k2)

        afn = os.path.join(TEST_DRN, abn)
        attenuation = np.loadtxt(afn)
        attenuation2 = _attenuation_curve(k2, RV_C00, av)
        assert np.allclose(attenuation, attenuation2)


def test_triweight_k_lambda():
    x_microns_target = np.linspace(0.097, 2.2, 100)
    noll09_target = _l02_below_c00_above(x_microns_target)
    tw_approx = triweight_k_lambda(x_microns_target)
    assert np.allclose(noll09_target, tw_approx, rtol=0.1)


def _read_noll09_header(fn):
    with open(fn, "r") as f:
        x0 = float(next(f).strip()[2:].split()[1])
        gamma = float(next(f).strip()[2:].split()[1])
        ampl = float(next(f).strip()[2:].split()[1])
        slope = float(next(f).strip()[2:].split()[1])
        av = float(next(f).strip()[2:].split()[1])
        return x0, gamma, ampl, slope, av


def test_get_filter_effective_wavelength():
    wave = np.linspace(0, 10, 5_000)
    trans = np.zeros_like(wave)
    msk = (wave > 4) & (wave < 6)
    trans[msk] = 1.0
    redshift = 0.0
    lambda_eff = _get_filter_effective_wavelength(wave, trans, redshift)
    assert np.allclose(lambda_eff, 5.0, rtol=0.001)


def test_get_effective_attenuation():
    filter_wave = np.logspace(2, 5, 5_000)
    filter_trans = np.ones_like(filter_wave)
    redshift = 0.0

    dust_delta = -0.3
    dust_eb = _get_eb_from_delta(dust_delta)
    dust_Av = 0.5
    att_curve_params = np.array((dust_eb, dust_delta, dust_Av))

    # No unobscured sightlines
    args = filter_wave, filter_trans, redshift, att_curve_params
    res = _get_effective_attenuation(*args)
    assert np.all(np.isfinite(res))
    assert res.shape == ()

    # With obscured sightlines
    frac_unobscured = 0.5
    args = filter_wave, filter_trans, redshift, att_curve_params, frac_unobscured
    res = _get_effective_attenuation(*args)
    assert np.all(np.isfinite(res))
    assert res.shape == ()
