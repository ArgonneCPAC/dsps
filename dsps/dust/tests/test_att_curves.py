"""
"""
import os
import numpy as np
from ..att_curves import calzetti00_k_lambda, calzetti00_att_curve
from ..att_curves import leitherer02_k_lambda, leitherer02_att_curve
from ..att_curves import noll09_k_lambda, sbl18_k_lambda
from ..att_curves import _l02_below_c00_above, triweight_k_lambda
from ..att_curves import noll09_k_att_curve, sbl18_k_att_curve
from ..att_curves import _frac_transmission_from_k_lambda


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TEST_DRN = os.path.join(_THIS_DRNAME, "testing_data")


def test_calzetii00_k_lambda():
    x = np.loadtxt(os.path.join(TEST_DRN, "calzetti_2000_wavelength_microns.txt"))
    k = np.loadtxt(os.path.join(TEST_DRN, "calzetti_2000_k_lambda.txt"))
    k2 = calzetti00_k_lambda(x)
    assert np.allclose(k, k2)


def test_leitherer02_k_lambda():
    x = np.loadtxt(os.path.join(TEST_DRN, "leitherer_2002_wavelength_microns.txt"))
    k = np.loadtxt(os.path.join(TEST_DRN, "leitherer_2002_k_lambda.txt"))
    k2 = leitherer02_k_lambda(x)
    assert np.allclose(k, k2)


def test_calzetti00_att_curve():
    wave_microns = np.loadtxt(
        os.path.join(TEST_DRN, "calzetti_2000_wavelength_microns.txt")
    )
    correct_attenuation = np.loadtxt(
        os.path.join(TEST_DRN, "calzetti_2000_attenuation.txt")
    )
    Av = 1.0
    dsps_attenuation = calzetti00_att_curve(wave_microns, Av)
    assert np.allclose(dsps_attenuation, correct_attenuation)


def test_leitherer02_att_curve():
    wave_microns = np.loadtxt(
        os.path.join(TEST_DRN, "leitherer_2002_wavelength_microns.txt")
    )
    correct_attenuation = np.loadtxt(
        os.path.join(TEST_DRN, "leitherer_2002_attenuation.txt")
    )
    Av = 1.0
    dsps_attenuation = leitherer02_att_curve(wave_microns, Av)
    assert np.allclose(dsps_attenuation, correct_attenuation)


def _read_noll09_header(fn):
    with open(fn, "r") as f:
        x0 = float(next(f).strip()[2:].split()[1])
        gamma = float(next(f).strip()[2:].split()[1])
        ampl = float(next(f).strip()[2:].split()[1])
        slope = float(next(f).strip()[2:].split()[1])
        av = float(next(f).strip()[2:].split()[1])
        return x0, gamma, ampl, slope, av


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
        k2 = noll09_k_lambda(x, ampl, slope, uv_bump=x0, uv_bump_width=gamma)
        assert np.allclose(k, k2)

        afn = os.path.join(TEST_DRN, abn)
        correct_attenuation = np.loadtxt(afn)
        dsps_attenuation = noll09_k_att_curve(
            x, av, ampl, slope, uv_bump=x0, uv_bump_width=gamma
        )
        assert np.allclose(correct_attenuation, dsps_attenuation)


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
        k2 = sbl18_k_lambda(x, ampl, slope, uv_bump=x0, uv_bump_width=gamma)
        assert np.allclose(k, k2)

        afn = os.path.join(TEST_DRN, abn)
        correct_attenuation = np.loadtxt(afn)
        dsps_attenuation = sbl18_k_att_curve(
            x, av, ampl, slope, uv_bump=x0, uv_bump_width=gamma
        )
        assert np.allclose(correct_attenuation, dsps_attenuation)


def test_triweight_k_lambda():
    x_microns_target = np.linspace(0.097, 2.2, 100)
    noll09_target = _l02_below_c00_above(x_microns_target)
    tw_approx = triweight_k_lambda(x_microns_target)
    assert np.allclose(noll09_target, tw_approx, rtol=0.1)


def test_transmission_fraction_ftrans_floor_effect():
    x = np.logspace(-2, 4, 5_000)
    k_lambda = calzetti00_k_lambda(x)
    av = 1.0
    ftrans_floor = 0.0
    for ftrans_floor in (0.0, 0.1, 0.5, 1.0):
        ftrans = _frac_transmission_from_k_lambda(k_lambda, av, ftrans_floor)
        assert np.all(np.isfinite(ftrans))
        assert np.all(ftrans <= 1)
        assert np.all(ftrans >= ftrans_floor)


def test_transmission_fraction_av_effect():
    x = np.logspace(-2, 4, 5_000)
    k_lambda = calzetti00_k_lambda(x)
    ftrans_floor = 0.0

    av = 0.0
    ftrans = _frac_transmission_from_k_lambda(k_lambda, av, ftrans_floor)
    assert np.all(np.isfinite(ftrans))
    assert np.allclose(ftrans, 1.0)

    # Transmission fractions should decrease with increasing av
    for av in (0.1, 0.5, 1.0, 5.0):
        ftrans_new = _frac_transmission_from_k_lambda(k_lambda, av, ftrans_floor)
        assert np.all(ftrans_new <= ftrans)
        assert np.any(ftrans_new < ftrans)
        ftrans = ftrans_new
