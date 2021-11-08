"""
"""
from jax import jit as jjit
from jax import numpy as jnp


@jjit
def _get_quadfit_weights(x, x1, x2, x3, x4):
    msk_lo = (x >= x1) & (x <= x2)
    msk_hi = (x >= x3) & (x <= x4)
    msk = msk_lo | msk_hi
    return jnp.where(msk, 1, 0)


@jjit
def _get_integration_weights(x, x2, x3):
    msk = (x >= x2) & (x <= x3)
    return jnp.where(msk, 1, 0)


@jjit
def _weighted_quadratic_fit(x, y, w):
    deg = 2
    lhs = jnp.vander(x, deg + 1)
    rhs = y

    lhs *= w[:, jnp.newaxis]
    rhs *= w

    # scale lhs to improve condition number and solve
    scale = jnp.sqrt((lhs * lhs).sum(axis=0))
    lhs /= scale[jnp.newaxis, :]

    c, resids, rank, s = jnp.linalg.lstsq(lhs, rhs)
    c = (c.T / scale).T  # broadcast scale coefficients

    return c


@jjit
def _ew_kernel(wave, flux, x1, x2, x3, x4):
    quadfit_w = _get_quadfit_weights(wave, x1, x2, x3, x4)
    c = _weighted_quadratic_fit(wave, flux, quadfit_w)
    c2, c1, c0 = c

    int_w = _get_integration_weights(wave, x2, x3)
    continuum_integrand = int_w * (c0 + c1 * wave + c2 * wave * wave)
    spec_integrand = int_w * flux

    continuum_flux = jnp.trapz(continuum_integrand, x=wave)
    spec_flux = jnp.trapz(spec_integrand, x=wave)
    line_area = spec_flux - continuum_flux
    ew = spec_flux / continuum_flux - 1
    return ew, line_area
