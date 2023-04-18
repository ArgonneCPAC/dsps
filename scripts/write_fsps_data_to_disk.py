"""
"""
import argparse
from dsps.data_loaders import retrieve_ssp_data_from_fsps
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outname", help="Name of the output file")
    args = parser.parse_args()

    ssp_data = retrieve_ssp_data_from_fsps()

    with h5py.File(args.outname, "w") as hdf:
        for key, arr in zip(ssp_data._fields, ssp_data):
            hdf[key] = arr
