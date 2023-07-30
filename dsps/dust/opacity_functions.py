"""Models for dust opacity"""
from jax import jit as jjit
from jax import numpy as jnp
import numpy as np
from collections import OrderedDict
from dsps.utils import _tw_sig_slope


DUST_OPACITY_PARAMS = OrderedDict(
    dust_opacity_ytp=1.1,
    dust_opacity_x0=2.0,
    dust_opacity_index_uv=-2.0,
    dust_opacity_index_ir=-1.5,
)
DUST_OPACITY_XTP = 2.0
DUST_OPACITY_H = 0.25


@jjit
def rolling_plaw_opacity(
    wave_micron,
    dust_opacity_ytp=DUST_OPACITY_PARAMS["dust_opacity_ytp"],
    dust_opacity_x0=DUST_OPACITY_PARAMS["dust_opacity_x0"],
    dust_opacity_index_uv=DUST_OPACITY_PARAMS["dust_opacity_index_uv"],
    dust_opacity_index_ir=DUST_OPACITY_PARAMS["dust_opacity_index_ir"],
):
    """Dust opacity model based on a rolling power law

    Parameters
    ----------
    wave_micron : ndarray of shape (n, )
        Wavelength in microns

    dust_opacity_ytp : float, optional

    dust_opacity_x0 : float, optional

    dust_opacity_index_uv : float, optional

    dust_opacity_index_ir : float, optional

    Returns
    -------
    kappa : ndarray of shape (n, )
        Dust opacity in [cm^2/g]

    """
    lgwave_micron = jnp.log10(wave_micron)
    lgkappa = _rolling_plaw_opacity_kern(
        lgwave_micron,
        dust_opacity_ytp,
        dust_opacity_x0,
        dust_opacity_index_uv,
        dust_opacity_index_ir,
    )
    return 10**lgkappa


@jjit
def _rolling_plaw_opacity_kern(
    lgwave_micron,
    dust_opacity_ytp,
    dust_opacity_x0,
    dust_opacity_index_uv,
    dust_opacity_index_ir,
):
    lgkappa = _tw_sig_slope(
        lgwave_micron,
        DUST_OPACITY_XTP,
        dust_opacity_ytp,
        dust_opacity_x0,
        DUST_OPACITY_H,
        dust_opacity_index_uv,
        dust_opacity_index_ir,
    )
    return lgkappa


def _dust_opacity_cowley(wave, kappa1=140, wave1=30, waveb=100, beta=-1.5):
    """Dust opacity model based on Cowley+16, https://arxiv.org/abs/1607.05717"""
    alpha = -2.0
    opacity_uv = kappa1 * (wave / wave1) ** alpha
    opacity_ir = kappa1 * ((waveb / wave1) ** alpha) * ((wave / waveb) ** beta)
    msk = wave < waveb
    return np.where(msk, opacity_uv, opacity_ir)
