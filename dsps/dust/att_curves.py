"""Models for dust attenuation curves.

Each model is typically defined by a pair of functions:
one function providing a model for a "reddening curve" k(λ),
and a second function implementing the "attenuation curve" A(λ),
where the relationship between these two quantities is given by:

    A(λ) = k(λ) * (Av/Rv)

The quantity Av is a normalization parameter.
Throughout this module Rv=4.05 is held constant.

The attenuation curve A(λ) defines the quantity F_trans(λ),
the fraction of flux transmitted through dust, according the following equation:

    F_trans(λ) = 10^(-0.4*A(λ))

The function _frac_transmission_from_k_lambda is used to compute F_trans(λ).

"""
from jax import numpy as jnp
from jax import jit as jjit
from ..utils import _tw_sig_slope


RV_C00 = 4.05
UV_BUMP_W0 = 0.2175  # Center of UV bump in micron
UV_BUMP_DW = 0.0350  # Width of UV bump in micron


@jjit
def _frac_transmission_from_k_lambda(k_lambda, av, ftrans_floor=0.0):
    """Fraction of flux transmitted through dust for an input reddening curve.

    See Lower+22 for details.

    Parameters
    ----------
    k_lambda : ndarray of shape (n, )
        Reddening curve k(λ) computed by a model defined elsewhere

        The reddening curve k(λ) is related to the attenuation curve A(λ) as follows:

        k(λ) = A(λ) * (4.05/av)

    av : float
        V-band normalization of the attenuation curve.
        av=0 corresponds to zero dust attenuation

    ftrans_floor : float, optional
        Minimum value of the transmission fraction.
        Default is 0.0, in which case dust can obscure up to 100% of emitted flux.

    Returns
    -------
    ftrans : ndarray of shape (n, )
        F_trans(λ) is related to the attenuation curve A(λ) as follows:

        F_trans(λ) = 10^(-0.4*A(λ))

        When input variable ftrans_floor is nonzero, this equation becomes:

        F_trans(λ) = ftrans_floor + (1-ftrans_floor)*10^(-0.4*A(λ))

    """
    A_lambda = _att_curve_from_k_lambda(k_lambda, av)
    ftrans = 10 ** (-0.4 * A_lambda)

    ftrans = ftrans_floor + (1 - ftrans_floor) * ftrans

    return ftrans


@jjit
def _l02_below_c00_above(wave_micron, xc=0.15):
    """Piecewise reddening curve k(λ)

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in micron

    Returns
    -------
    k_lambda : ndarray of shape (n, )
        The reddening curve k(λ) is related to the attenuation curve A(λ) as follows:

        k(λ) = A(λ) * (4.05/av)

    Notes
    -----
    Used by noll09_k_lambda and sbl18_k_lambda

    """
    k_lambda_c00 = calzetti00_k_lambda(wave_micron)
    k_lambda_l02 = leitherer02_k_lambda(wave_micron)
    k_lambda = jnp.where(wave_micron > xc, k_lambda_c00, k_lambda_l02)
    return k_lambda


@jjit
def power_law_vband_norm(wave_micron, plaw_slope, vband_micron=0.55):
    """Power law normalized at V-band wavelength λ_V=0.55 micron.

    Used to modify a baseline reddening curve model k_0(λ) as follows:

    k(λ) = k_0(λ) * (λ/λ_V)**δ

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in micron

    plaw_slope : float
        Slope δ of the power-law modification (λ/λ_V)**δ

    Returns
    -------
    res : ndarray of shape (n, )

    """
    x = wave_micron / vband_micron
    return x**plaw_slope


@jjit
def noll09_k_att_curve(
    wave_micron,
    av,
    uv_bump_ampl,
    plaw_slope,
    uv_bump=UV_BUMP_W0,
    uv_bump_width=UV_BUMP_DW,
):
    """Attenuation curve A(λ) = k(λ) * (av/4.05) from Noll+2009

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in micron

    av : float
        V-band normalization of the attenuation curve.
        av=0 corresponds to zero dust attenuation

    uv_bump_ampl : float
        Amplitude of the UV bump feature at 0.2175 micron

    plaw_slope : float
        Slope of the power-law modification to k(λ)

    uv_bump : float, optional
        Centroid of the UV bump feature in micron. Default is UV_BUMP_W0=0.2175

    uv_bump_width : float, optional
        Width of the UV bump feature in micron. Default is UV_BUMP_DW=0.0350

    Returns
    -------
    A_lambda : ndarray of shape (n, )
        Attenuation curve A(λ) defining the fraction of flux transmitted by dust:

        F_trans(λ) = 10^(-0.4*A(λ))

    """
    k_lambda = noll09_k_lambda(
        wave_micron, uv_bump_ampl, plaw_slope, uv_bump, uv_bump_width
    )
    A_lambda = _att_curve_from_k_lambda(k_lambda, av)
    return A_lambda


@jjit
def noll09_k_lambda(
    wave_micron, uv_bump_ampl, plaw_slope, uv_bump=UV_BUMP_W0, uv_bump_width=UV_BUMP_DW
):
    """Reddening curve k(λ) = A(λ) * (4.05/av) from Noll+2009

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in micron

    uv_bump_ampl : float
        Amplitude of the UV bump feature at 0.2175 micron

    plaw_slope : float
        Slope of the power-law modification to k(λ)

    uv_bump : float, optional
        Centroid of the UV bump feature in micron. Default is UV_BUMP_W0=0.2175

    uv_bump_width : float, optional
        Width of the UV bump feature in micron. Default is UV_BUMP_DW=0.0350

    Returns
    -------
    k_lambda : ndarray of shape (n, )
        The reddening curve k(λ) is related to the attenuation curve A(λ) as follows:

        k(λ) = A(λ) * (4.05/av)

    """
    # Leitherer 2002 below 0.15 micron and Calzetti 2000 above
    k_lambda = _l02_below_c00_above(wave_micron, xc=0.15)

    # Add the UV bump
    k_lambda = k_lambda + _drude_bump(wave_micron, uv_bump, uv_bump_width, uv_bump_ampl)

    # Apply power-law correction
    k_lambda = k_lambda * power_law_vband_norm(wave_micron, plaw_slope)

    # Clip at zero
    k_lambda = jnp.where(k_lambda < 0, 0, k_lambda)

    return k_lambda


@jjit
def sbl18_k_att_curve(
    wave_micron,
    av,
    uv_bump_ampl,
    plaw_slope,
    uv_bump=UV_BUMP_W0,
    uv_bump_width=UV_BUMP_DW,
):
    """Attenuation curve A(λ) = k(λ) * (av/4.05) from Salim+2018

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in micron

    av : float
        V-band normalization of the attenuation curve.
        av=0 corresponds to zero dust attenuation

    uv_bump_ampl : float
        Amplitude of the UV bump feature at 0.2175 micron

    plaw_slope : float
        Slope of the power-law modification to k(λ)

    uv_bump : float, optional
        Centroid of the UV bump feature in micron. Default is UV_BUMP_W0=0.2175

    uv_bump_width : float, optional
        Width of the UV bump feature in micron. Default is UV_BUMP_DW=0.0350

    Returns
    -------
    A_lambda : ndarray of shape (n, )
        Attenuation curve A(λ) defining the fraction of flux transmitted by dust:

        F_trans(λ) = 10^(-0.4*A(λ))

    """
    k_lambda = sbl18_k_lambda(
        wave_micron, uv_bump_ampl, plaw_slope, uv_bump, uv_bump_width
    )
    A_lambda = _att_curve_from_k_lambda(k_lambda, av)
    return A_lambda


@jjit
def sbl18_k_lambda(
    wave_micron, uv_bump_ampl, plaw_slope, uv_bump=UV_BUMP_W0, uv_bump_width=UV_BUMP_DW
):
    """Reddening curve k(λ) = A(λ) * (4.05/av) from Salim+2018

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in micron

    uv_bump_ampl : float
        Amplitude of the UV bump feature at 0.2175 micron

    plaw_slope : float
        Slope of the power-law modification to k(λ)

    uv_bump : float, optional
        Centroid of the UV bump feature in micron. Default is UV_BUMP_W0=0.2175

    uv_bump_width : float, optional
        Width of the UV bump feature in micron. Default is UV_BUMP_DW=0.0350

    Returns
    -------
    k_lambda : ndarray of shape (n, )
        The reddening curve k(λ) is related to the attenuation curve A(λ) as follows:

        k(λ) = A(λ) * (4.05/av)

    """
    # Leitherer 2002 below 0.15 micron and Calzetti 2000 above
    k_lambda = _l02_below_c00_above(wave_micron, xc=0.15)

    # Apply power-law correction
    k_lambda = k_lambda * power_law_vband_norm(wave_micron, plaw_slope)

    # Add the UV bump
    k_lambda = k_lambda + _drude_bump(wave_micron, uv_bump, uv_bump_width, uv_bump_ampl)

    # Clip at zero
    k_lambda = jnp.where(k_lambda < 0, 0, k_lambda)

    return k_lambda


@jjit
def calzetti00_att_curve(wave_micron, av):
    """Attenuation curve A(λ) = k(λ) * (av/4.05) from Calzetti (2000)

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in micron

    av : float
        V-band normalization of the attenuation curve.
        av=0 corresponds to zero dust attenuation

    Returns
    -------
    A_lambda : ndarray of shape (n, )
        Attenuation curve A(λ) defining the fraction of flux transmitted by dust:

        F_trans(λ) = 10^(-0.4*A(λ))

    """
    k_lambda = calzetti00_k_lambda(wave_micron)
    A_lambda = _att_curve_from_k_lambda(k_lambda, av)
    return A_lambda


@jjit
def leitherer02_att_curve(wave_micron, av):
    """Attenuation curve A(λ) = k(λ) * (av/4.05)

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in micron

    av : float
        V-band normalization of the attenuation curve.
        av=0 corresponds to zero dust attenuation

    Returns
    -------
    A_lambda : ndarray of shape (n, )
        Attenuation curve A(λ) defining the fraction of flux transmitted by dust:

        F_trans(λ) = 10^(-0.4*A(λ))

    """
    k_lambda = leitherer02_k_lambda(wave_micron)
    A_lambda = _att_curve_from_k_lambda(k_lambda, av)
    return A_lambda


@jjit
def calzetti00_k_lambda(wave_micron):
    """Reddening curve k(λ) = A(λ) * (4.05/av) from Calzetti (2000)

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in micron

    Returns
    -------
    k_lambda : ndarray of shape (n, )
        The reddening curve k(λ) is related to the attenuation curve A(λ) as follows:

        k(λ) = A(λ) * (4.05/av)

    """
    y = 1 / wave_micron
    k1 = 2.659 * (-2.156 + 1.509 * y - 0.198 * y**2 + 0.011 * y**3) + RV_C00
    k2 = 2.659 * (-1.857 + 1.040 * y) + RV_C00
    k_lambda = jnp.where(wave_micron < 0.63, k1, k2)
    return k_lambda


@jjit
def leitherer02_k_lambda(wave_micron):
    """Reddening curve k(λ) = A(λ) * (4.05/av)

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in micron

    Returns
    -------
    k_lambda : ndarray of shape (n, )
        The reddening curve k(λ) is related to the attenuation curve A(λ) as follows:

        k(λ) = A(λ) * (4.05/av)

    """
    y = 1 / wave_micron
    k_lambda = 5.472 + (0.671 * y - 9.218 * 1e-3 * y**2 + 2.620 * 1e-3 * y**3)
    return k_lambda


@jjit
def triweight_k_lambda(
    wave_micron, xtp=-1.0, ytp=1.15, x0=0.5, tw_h=0.5, lo=-0.65, hi=-1.95
):
    """Smooth approximation to Noll+09 k(λ)

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in micron

    Returns
    -------
    k_lambda : ndarray of shape (n, )
        The reddening curve k(λ) is related to the attenuation curve A(λ) as follows:

        k(λ) = A(λ) * (4.05/av)

    """
    lgx = jnp.log10(wave_micron)
    lgk_lambda = _tw_sig_slope(lgx, xtp, ytp, x0, tw_h, lo, hi)
    k_lambda = 10**lgk_lambda
    return k_lambda


@jjit
def _att_curve_from_k_lambda(k_lambda, av):
    """Normalize k(λ) according to (av/4.05) and clip at zero

    A(λ) = k(λ) * (av/4.05)

    """
    att_curve = av * k_lambda / RV_C00
    att_curve = jnp.where(att_curve < 0, 0, att_curve)
    return att_curve


@jjit
def _drude_bump(x, x0, gamma, ampl):
    """Drude profile of a bump feature seen in reddening curves

    The UV bump is typically located at λ=0.2175 micron,
    but _drude_bump can be used to introduce a generic bump in a power-law type model"""
    bump = x**2 * gamma**2 / ((x**2 - x0**2) ** 2 + x**2 * gamma**2)
    return ampl * bump


@jjit
def _get_eb_from_delta(delta):
    """Eb--delta relationship calibrated in Kriek & Conroy 2013"""
    return -1.9 * delta + 0.85
