"""
usage: 
py write_fsps_data_to_disk.py <.h5 output filename> --gas_logz 0.0 --gas_logu -2.0
"""
import argparse

import h5py

from dsps.data_loaders import retrieve_ssp_data_from_fsps

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
        for key, arr in zip(ssp_data._fields, ssp_data):
            hdf[key] = arr
