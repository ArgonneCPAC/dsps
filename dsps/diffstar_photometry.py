"""
"""
from jax import jit as jjit
from jax import vmap
from .diffstar_photometry_kernels import (
    _calc_weighted_rest_mag_from_diffstar_params_const_zmet,
    _calc_weighted_obs_mag_from_diffstar_params_const_zmet_dust,
    _calc_weighted_obs_mag_from_diffstar_params_const_zmet_agedep_dust,
)


_a = [*[None] * 5, 0, 0, *[None] * 19]
_calc_weighted_rest_mags_from_diffstar_params = jjit(
    vmap(_calc_weighted_rest_mag_from_diffstar_params_const_zmet, in_axes=_a)
)

_b = [0, *[None] * 25]
_calc_weighted_rest_mags_history = jjit(
    vmap(_calc_weighted_rest_mags_from_diffstar_params, in_axes=_b)
)

_c = [*[None] * 11, 0, 0, *[None] * 24]
_calc_weighted_obs_mags_from_diffstar_params_dust = jjit(
    vmap(
        _calc_weighted_obs_mag_from_diffstar_params_const_zmet_dust,
        in_axes=_c,
        out_axes=(0, None, None),
    )
)

_c = [*[None] * 11, 0, 0, *[None] * 26]
_calc_weighted_obs_mags_from_diffstar_params_agedep_dust = jjit(
    vmap(
        _calc_weighted_obs_mag_from_diffstar_params_const_zmet_agedep_dust,
        in_axes=_c,
        out_axes=(0, None, None),
    )
)

_d = [0, 0, *[None] * 35]
_calc_weighted_obs_mags_history_dust = jjit(
    vmap(
        _calc_weighted_obs_mags_from_diffstar_params_dust,
        in_axes=_d,
        out_axes=(0, 0, None),
    )
)

_d = [0, 0, *[None] * 37]
_calc_weighted_obs_mags_history_agedep_dust = jjit(
    vmap(
        _calc_weighted_obs_mags_from_diffstar_params_agedep_dust,
        in_axes=_d,
        out_axes=(0, 0, None),
    )
)
