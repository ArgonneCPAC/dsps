"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from .blackbody import blackbody_wave_density
from .opacity_functions import rolling_plaw_opacity


FOURPI = 4.0 * jnp.pi


@jjit
def _graybody_emission(wave_aa, temp_k, dust_mass, opacity_params):
    bb = blackbody_wave_density(wave_aa, temp_k)  # Lsun/AA/sr/m/m
    wave_micron = wave_aa / 1e4
    kappa = rolling_plaw_opacity(wave_micron, *opacity_params)  # cm^2/g
