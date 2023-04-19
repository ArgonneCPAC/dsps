# flake8: noqa
"""Globals storing defaults for various DSPS options"""
from .flat_wcdm import PLANCK15 as DEFAULT_COSMOLOGY


from .flat_wcdm import age_at_z0

TODAY = age_at_z0(*DEFAULT_COSMOLOGY)
