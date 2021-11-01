"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

c_speed = 2.99792458e8  # m/s

# rho_crit(z=0) in [Msun * h^2 * kpc^-3]
RHO_CRIT0_KPC3_UNITY_H = 277.536627  # multiply by h**2 in cosmology conversion

MPC = 3.08567758149e24  # Mpc in cm
YEAR = 31556925.2  # year in seconds


@jjit
def _rho_de_z(z, w0, wa):
    a = 1.0 / (1.0 + z)
    de_z = a ** (-3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * (1.0 - a))
    return de_z


@jjit
def _Ez(z, Om0, Ode0, w0, wa):
    zp1 = 1.0 + z
    t = Om0 * zp1 ** 3 + Ode0 * _rho_de_z(z, w0, wa)
    E = jnp.sqrt(t)
    return E


@jjit
def _integrand_oneOverEz(z, Om0, Ode0, w0, wa):
    return 1 / _Ez(z, Om0, Ode0, w0, wa)


@jjit
def _integrand_oneOverEz1pz(z, Om0, Ode0, w0, wa):
    return 1.0 / _Ez(z, Om0, Ode0, w0, wa) / (1.0 + z)


@jjit
def _comoving_distance_to_z(z, Om0, Ode0, w0, wa, h):
    z_table = jnp.linspace(0, z, 256)
    integrand = _integrand_oneOverEz(z_table, Om0, Ode0, w0, wa)
    # The 1E-5 factor comes from the conversion between the
    # speed of light in m/s to km/s and H0 = 100 * h.
    return jnp.trapz(integrand, x=z_table) * c_speed * 1e-5 / h


@jjit
def _luminosity_distance_to_z(z, Om0, Ode0, w0, wa, h):
    return _comoving_distance_to_z(z, Om0, Ode0, w0, wa, h) * (1 + z)


@jjit
def _angular_diameter_distance_to_z(z, Om0, Ode0, w0, wa, h):
    return _comoving_distance_to_z(z, Om0, Ode0, w0, wa, h) / (1 + z)


@jjit
def _distance_modulus_to_z(z, Om0, Ode0, w0, wa, h):
    d_lum = _luminosity_distance_to_z(z, Om0, Ode0, w0, wa, h)
    mu = 5.0 * jnp.log10(d_lum * 1e5)
    return mu


@jjit
def _hubble_time(z, Om0, Ode0, w0, wa, h):
    E0 = _Ez(z, Om0, Ode0, w0, wa)
    htime = 1e-16 * MPC / YEAR / h / E0
    return htime


@jjit
def _lookback_to_z(z, Om0, Ode0, w0, wa, h):
    z_table = jnp.linspace(0, z, 256)
    integrand = 1 / _Ez(z_table, Om0, Ode0, w0, wa) / (1 + z_table)
    res = jnp.trapz(integrand, x=z_table)
    th = _hubble_time(0.0, Om0, Ode0, w0, wa, h)
    return th * res


@jjit
def _rho_crit(z, Om0, Ode0, w0, wa, h):
    """Critical density in units of physical Msun/kpc**3"""
    rho_crit0 = RHO_CRIT0_KPC3_UNITY_H * h * h
    return rho_crit0 * _Ez(z, Om0, Ode0, w0, wa) ** 2


@jjit
def _Om_at_z(z, Om0, Ode0, w0, wa):
    E = _Ez(z, Om0, Ode0, w0, wa)
    return Om0 * (1.0 + z) ** 3 / E / E


_A = (0, *[None] * 5)
_distance_modulus = jjit(vmap(_distance_modulus_to_z, in_axes=_A))
_luminosity_distance = jjit(vmap(_luminosity_distance_to_z, in_axes=_A))
_angular_diameter_distance = jjit(vmap(_angular_diameter_distance_to_z, in_axes=_A))
_lookback_time = jjit(vmap(_lookback_to_z, in_axes=_A))
_Om = jjit(vmap(_Om_at_z, in_axes=_A[:-1]))


@jjit
def _delta_vir(z, Om0, Ode0, w0, wa):
    x = _Om(z, Om0, Ode0, w0, wa) - 1.0
    Delta = 18 * jnp.pi ** 2 + 82.0 * x - 39.0 * x ** 2
    return Delta


@jjit
def _virial_dynamical_time(z, Om0, Ode0, w0, wa, h):
    """Dynamical time to cross the diameter of a halo at redshift z.
    The pericentric passage time is half this time.
    The orbital time is PI times this time."""
    delta = _delta_vir(z, Om0, Ode0, w0, wa)
    t_cross = 2 ** 1.5 * _hubble_time(z, Om0, Ode0, w0, wa, h) * delta ** -0.5
    return t_cross
