""""""

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
