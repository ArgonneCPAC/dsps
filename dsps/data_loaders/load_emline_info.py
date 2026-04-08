""""""

from collections import namedtuple

import numpy as np


def read_emlines_info_fsps(fn, testmode=False):
    """Read the FSPS file emlines_info.dat into a dictionary"""
    line_wave_collector = []
    emline_dict = dict()
    with open(fn, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            _line_wave, _a = line.split(",")
            line_wave = float(_line_wave)
            _b = _a.replace("-", "_").replace(" ", "_").replace(".", "p")
            line_name = _b.replace("[", "").replace("]", "")

            emline_dict[line_name] = line_wave
            line_wave_collector.append(line_wave)

    if testmode:
        line_wave_arr = np.array(line_wave_collector)
        return emline_dict, line_wave_arr
    else:
        return emline_dict


def get_subset_emline_data(ssp_data, emline_names):
    """Construct a new ssp_data based only a subset of the emission lines

    Parameters
    ----------
    ssp_data : namedtuple

    emline_names : list of strings
        Each entry should appear in ssp_data.ssp_emline_wave._fields

    Returns
    -------
    ssp_data : namedtuple
        The ssp_emline_wave and ssp_emline_luminosity fields will be replaced
        with only information about lines in emline_names, preserving order.

    """
    indx_lines = np.array(
        [ssp_data.ssp_emline_wave._fields.index(name) for name in emline_names]
    )
    line_wave = np.array(ssp_data.ssp_emline_wave)[indx_lines]
    EmLineWave = namedtuple("EmLineWave", emline_names)
    line_wave = EmLineWave(*line_wave)

    line_lum = ssp_data.ssp_emline_luminosity[:, :, indx_lines]

    ssp_data = ssp_data._replace(
        ssp_emline_wave=line_wave, ssp_emline_luminosity=line_lum
    )

    return ssp_data
