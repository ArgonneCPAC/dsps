"""Kernels calculating distances in flat FLRW cosmologies"""
import typing
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

__all__ = (
    "comoving_distance_to_z",
    "luminosity_distance_to_z",
    "distance_modulus_to_z",
    "angular_diameter_distance_to_z",
    "lookback_to_z",
    "age_at_z0",
    "age_at_z",
    "rho_crit",
    "virial_dynamical_time",
)


class CosmoParams(typing.NamedTuple):
    """NamedTuple storing parameters of a flat w0-wa cdm cosmology"""

    Om0: jnp.float32
    w0: jnp.float32
    wa: jnp.float32
    h: jnp.float32


PLANCK15 = CosmoParams(0.3075, -1.0, 0.0, 0.6774)
WMAP5 = CosmoParams(0.277, -1.0, 0.0, 0.702)
FSPS_COSMO = CosmoParams(0.27, -1.0, 0.0, 0.72)

C_SPEED = 2.99792458e8  # m/s

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
def _Ez(z, Om0, w0, wa):
    zp1 = 1.0 + z
    Ode0 = 1.0 - Om0
    t = Om0 * zp1**3 + Ode0 * _rho_de_z(z, w0, wa)
    E = jnp.sqrt(t)
    return E


@jjit
def _integrand_oneOverEz(z, Om0, w0, wa):
    return 1 / _Ez(z, Om0, w0, wa)


@jjit
def _integrand_oneOverEz1pz(z, Om0, w0, wa):
    return 1.0 / _Ez(z, Om0, w0, wa) / (1.0 + z)


@jjit
def comoving_distance_to_z(redshift, Om0, w0, wa, h):
    """Comoving distance in Mpc

    Parameters
    ----------
    redshift : float

    Om0 : float

    w0 : float

    wa : float

    h : float

    Returns
    -------
    d : float
        comoving distance in Mpc

    """
    z_table = jnp.linspace(0, redshift, 256)
    integrand = _integrand_oneOverEz(z_table, Om0, w0, wa)
    # The 1E-5 factor comes from the conversion between the
    # speed of light in m/s to km/s and H0 = 100 * h.
    return jnp.trapz(integrand, x=z_table) * C_SPEED * 1e-5 / h


@jjit
def luminosity_distance_to_z(redshift, Om0, w0, wa, h):
    """Luminosity distance in Mpc

    Parameters
    ----------
    redshift : float

    Om0 : float

    w0 : float

    wa : float

    h : float

    Returns
    -------
    d : float
        luminosity distance in Mpc

    """
    return comoving_distance_to_z(redshift, Om0, w0, wa, h) * (1 + redshift)


@jjit
def angular_diameter_distance_to_z(redshift, Om0, w0, wa, h):
    """Angular diameter distance in Mpc

    Parameters
    ----------
    redshift : float

    Om0 : float

    w0 : float

    wa : float

    h : float

    Returns
    -------
    d : float
        angular diameter distance in Mpc

    """
    return comoving_distance_to_z(redshift, Om0, w0, wa, h) / (1 + redshift)


@jjit
def distance_modulus_to_z(redshift, Om0, w0, wa, h):
    """Distance modulus, defined as apparent-Absolute magnitude
    for an object at the input redshift

    Parameters
    ----------
    redshift : float

    Om0 : float

    w0 : float

    wa : float

    h : float

    Returns
    -------
    d : float
        distance modulus

    """
    d_lum = luminosity_distance_to_z(redshift, Om0, w0, wa, h)
    mu = 5.0 * jnp.log10(d_lum * 1e5)
    return mu


@jjit
def _hubble_time(z, Om0, w0, wa, h):
    E0 = _Ez(z, Om0, w0, wa)
    htime = 1e-16 * MPC / YEAR / h / E0
    return htime


@jjit
def lookback_to_z(redshift, Om0, w0, wa, h):
    """Lookback time in Gyr

    Parameters
    ----------
    redshift : float

    Om0 : float

    w0 : float

    wa : float

    h : float

    Returns
    -------
    t : float
        lookback time in Gyr

    """
    z_table = jnp.linspace(0, redshift, 512)
    integrand = 1 / _Ez(z_table, Om0, w0, wa) / (1 + z_table)
    res = jnp.trapz(integrand, x=z_table)
    th = _hubble_time(0.0, Om0, w0, wa, h)
    return th * res


@jjit
def age_at_z0(Om0, w0, wa, h):
    """Age of the Universe in Gyr at z=0

    Parameters
    ----------
    Om0 : float

    w0 : float

    wa : float

    h : float

    Returns
    -------
    t0 : float
        Age of the Universe in Gyr at z=0

    """
    z_table = jnp.logspace(0, 3, 512) - 1.0
    integrand = 1 / _Ez(z_table, Om0, w0, wa) / (1 + z_table)
    res = jnp.trapz(integrand, x=z_table)
    th = _hubble_time(0.0, Om0, w0, wa, h)
    return th * res


@jjit
def _age_at_z_kern(redshift, Om0, w0, wa, h):
    t0 = age_at_z0(Om0, w0, wa, h)
    tlook = lookback_to_z(redshift, Om0, w0, wa, h)
    return t0 - tlook


_age_at_z_vmap = jjit(vmap(_age_at_z_kern, in_axes=(0, *[None] * 4)))


@jjit
def age_at_z(redshift, Om0, w0, wa, h):
    """Age of the Universe in Gyr as a function of redshift

    Parameters
    ----------
    redshift : ndarray of shape (n, )

    Om0 : float

    w0 : float

    wa : float

    h : float

    Returns
    -------
    t : float
        Age of the Universe in Gyr

    """
    return _age_at_z_vmap(jnp.atleast_1d(redshift), Om0, w0, wa, h)


@jjit
def rho_crit(redshift, Om0, w0, wa, h):
    """Critical density in units of physical Msun/kpc**3

    Parameters
    ----------
    redshift : float

    Om0 : float

    w0 : float

    wa : float

    h : float

    Returns
    -------
    rho_crit : float
        critical density in units of physical Msun/kpc**3

    """
    rho_crit0 = RHO_CRIT0_KPC3_UNITY_H * h * h
    rho_crit = rho_crit0 * _Ez(redshift, Om0, w0, wa) ** 2
    return rho_crit


@jjit
def _Om_at_z(z, Om0, w0, wa):
    E = _Ez(z, Om0, w0, wa)
    return Om0 * (1.0 + z) ** 3 / E / E


_A = (0, *[None] * 4)
distance_modulus = jjit(vmap(distance_modulus_to_z, in_axes=_A))
luminosity_distance = jjit(vmap(luminosity_distance_to_z, in_axes=_A))
angular_diameter_distance = jjit(vmap(angular_diameter_distance_to_z, in_axes=_A))
lookback_time = jjit(vmap(lookback_to_z, in_axes=_A))
_Om = jjit(vmap(_Om_at_z, in_axes=_A[:-1]))


@jjit
def _delta_vir(z, Om0, w0, wa):
    x = _Om(z, Om0, w0, wa) - 1.0
    Delta = 18 * jnp.pi**2 + 82.0 * x - 39.0 * x**2
    return Delta


@jjit
def virial_dynamical_time(redshift, Om0, w0, wa, h):
    """Dynamical time to cross the diameter of a halo at redshift z.

    Parameters
    ----------
    redshift : float

    Om0 : float

    w0 : float

    wa : float

    h : float

    Returns
    -------
    t_cross : float
        dynamical crossing time

    Notes
    -----
    The pericentric passage time is half this time.
    The orbital time is π times this time.

    """
    delta = _delta_vir(redshift, Om0, w0, wa)
    t_cross = 2**1.5 * _hubble_time(redshift, Om0, w0, wa, h) * delta**-0.5
    return t_cross
