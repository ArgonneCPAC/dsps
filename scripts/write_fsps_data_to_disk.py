"""
usage: 
py write_fsps_data_to_disk.py <.h5 output filename> --gas_logz 0.0 --gas_logu -2.0
"""
import argparse

import h5py
import numpy as np

from dsps.data_loaders import retrieve_ssp_data_from_fsps

from .defaults import DEFAULT_SSP_KEYS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outname", help="Name of the output file")

    parser.add_argument(
        "--gas_logz", type=float, default=None, help="Gas-phase metallicity (log Z)"
    )
    parser.add_argument(
        "--gas_logu", type=float, default=None, help="Ionization parameter (log U)"
    )
    args = parser.parse_args()

    kwargs = {}
    if args.gas_logz is not None:
        kwargs["gas_logz"] = args.gas_logz
    if args.gas_logu is not None:
        kwargs["gas_logu"] = args.gas_logu

    ssp_data = retrieve_ssp_data_from_fsps(**kwargs)
    with h5py.File(args.outname, "w") as hdf:
        for key, arr in zip(DEFAULT_SSP_KEYS, ssp_data):
            hdf[key] = arr
            if "ssp_emlines" in ssp_data._fields:
                ssp_emline_name = list(ssp_data.ssp_emlines._fields)
                ssp_emline_wave = [
                    getattr(ssp_data.ssp_emlines, name).emline_wave
                    for name in ssp_emline_name
                ]
                ssp_emline_luminosity = [
                    getattr(ssp_data.ssp_emlines, name).emline_luminosity
                    for name in ssp_emline_name
                ]
                ssp_emline_luminosity = np.stack(ssp_emline_luminosity, axis=-1)

                grp = hdf.create_group("ssp_emlines")

                dt = h5py.string_dtype(encoding="utf-8")
                grp.create_dataset("ssp_emline_name", data=ssp_emline_name, dtype=dt)

                grp.create_dataset("ssp_emline_wave", data=ssp_emline_wave)
                grp.create_dataset("ssp_emline_luminosity", data=ssp_emline_luminosity)
