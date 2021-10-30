"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from .sfh_model import diffstar_sfh
from .mzr import mzr_model
from .utils import _jax_get_dt_array


@jjit
def _calc_lgmet_history(t, mah_params, ms_params, q_params, met_params):
    dt = _jax_get_dt_array(t)
    sfh = diffstar_sfh(t, mah_params, ms_params, q_params)
    smh = jnp.cumsum(sfh * dt) * 1e9
    lgzh = mzr_model(jnp.log10(smh), t, *met_params[:-1])
    return lgzh
