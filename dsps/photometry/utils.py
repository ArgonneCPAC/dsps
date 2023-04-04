"""Convenience functions commonly encountered in photometry calculations"""
import numpy as np


def interpolate_filter_trans_curves(wave_filters, trans_filters, n=None):
    """Interpolate a collection of filter transmission curves to a common length.
    Convenience function for analyses vmapping over broadband colors.

    Parameters
    ----------
    wave_filters : sequence of n_filters ndarrays

    trans_filters : sequence of n_filters ndarrays

    n : int, optional
        Desired length of the output transmission curves.
        Default is equal to the smallest length transmission curve

    Returns
    -------
    wave_filters : ndarray of shape (n_filters, n)

    trans_filters : ndarray of shape (n_filters, n)

    """
    wave0 = wave_filters[0]
    wave_min, wave_max = wave0.min(), wave0.max()

    if n is None:
        n = np.min([x.size for x in wave_filters])

    for wave, trans in zip(wave_filters, trans_filters):
        wave_min = min(wave_min, wave.min())
        wave_max = max(wave_max, wave.max())

    wave_collector = []
    trans_collector = []
    for wave, trans in zip(wave_filters, trans_filters):
        wave_min, wave_max = wave.min(), wave.max()
        new_wave = np.linspace(wave_min, wave_max, n)
        new_trans = np.interp(new_wave, wave, trans)
        wave_collector.append(new_wave)
        trans_collector.append(new_trans)
    return np.array(wave_collector), np.array(trans_collector)
