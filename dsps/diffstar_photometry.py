"""
"""
from jax import jit as jjit
from jax import vmap
from .diffstar_photometry_kernels import (
    _calc_weighted_rest_mag_from_diffstar_params_const_zmet,
)


_a = [*[None] * 5, 0, 0, *[None] * 19]
_calc_weighted_rest_mags_from_diffstar_params = jjit(
    vmap(_calc_weighted_rest_mag_from_diffstar_params_const_zmet, in_axes=_a)
)

_b = [0, *[None] * 25]
_calc_weighted_rest_mags_history = jjit(
    vmap(_calc_weighted_rest_mags_from_diffstar_params, in_axes=_b)
)
