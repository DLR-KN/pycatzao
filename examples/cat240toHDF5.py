# noqa: D100

import argparse
import itertools
import pathlib

import h5py
import numpy as np
from tqdm import tqdm

import pycatzao


def toHDF(*, cat240_file, hdf5_file, n_read_binning=100_000_000, verbose=True):
    """Converts an Asterix CAT240 into an HDF file.

    Args:
        cat240_file (str | pathlib.Path):
            Filename of the cat240 file.
        hdf5_file (str | pathlib.Path):
            Filename of the HDF5 file.
        n_read_binning (int):
            Max. number of bins to read for inferring binning scheme.
        verbose (bool):
            Print status to console.
    """
    log = print if verbose else lambda *args, **kwargs: None

    df = dict()
    for blocks in tqdm(
        itertools.batched(
            pycatzao.decode_file(cat240_file, size=-1, buffer_size=100_000),
            1_000_000,
        ),
        desc=f"Decoding CAT240 data from {cat240_file}",
    ):
        cols = pycatzao.join_blocks(blocks)
        for k in cols:
            df[k] = np.append(df[k], cols[k]) if k in df else cols[k]

    log("\nFinding cycles ...")
    daz = np.diff(df["az"])
    df["cycle"] = np.zeros_like(df["az"], dtype=np.uint32)
    df["cycle"][1:] = np.cumsum(daz < 0)
    log(f" * found {df['cycle'].max() + 1} cycles in {df['cycle'].size:,} rows")

    log("\nInferring binning scheme ...")
    with open(cat240_file, "rb") as f:
        bins, _ = pycatzao.infer_bin_edges(f.read(n_read_binning))

    df["az_edges"] = np.linspace(**bins["az"])
    df["r_edges"] = np.arange(**bins["r"], stop=df["r"].max() + bins["r"]["step"])

    df["az"] = np.searchsorted(df["az_edges"], df["az"]).astype(np.uint16) - 1
    df["r"] = np.searchsorted(df["r_edges"], df["r"]).astype(np.uint16) - 1

    if verbose:
        bins = df["az_edges"]
        log(
            f" * az(imuth) bin edges ({bins.size - 1} bins): "
            f"{bins[0]:.2f}° .. {bins[-1]:.2f}°"
        )

        bins = df["r_edges"]
        log(
            f" *   r(ange) bin edges ({bins.size - 1} bins): "
            f"{bins[0]:.2f}m .. {bins[-1]:.2f}m\n"
        )

    with h5py.File(hdf5_file, "w") as f:
        for k in tqdm(df, desc=f"Writing data to {hdf5_file}", disable=not verbose):
            f.create_dataset(k, data=df[k], compression="gzip")

        f["tod"].attrs["desc"] = "Time of Day (UTC)"
        f["tod"].attrs["unit"] = "second"

        f["az"].attrs["desc"] = "Index of azimuth bins"
        f["r"].attrs["desc"] = "Index of range bins"
        f["amp"].attrs["desc"] = "Amplitude"
        f["cycle"].attrs["desc"] = "Number of full rotations of the radar antenna"

        f["az_edges"].attrs["desc"] = "Bin edges of clockwise azimuth"
        f["az_edges"].attrs["unit"] = "degree"

        f["r_edges"].attrs["desc"] = "Bin edges of range"
        f["r_edges"].attrs["unit"] = "meter"


def main():  # noqa: D103
    parser = argparse.ArgumentParser(
        prog="cat240toHDF5",
        description="A handy tool that converts Asterix CAT240 data into an HDF5 file.",
    )
    parser.add_argument(
        "--cat240_file",
        type=pathlib.Path,
        required=True,
        help="CAT240 input file",
    )
    parser.add_argument(
        "--hdf5_file",
        type=pathlib.Path,
        required=True,
        help="HDF5 output file",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite of existing HDF5 file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print status to console",
    )
    p = parser.parse_args()

    if not p.cat240_file.is_file():
        parser.error(f"Cannot open data file '{p.cat240_file}'")

    if p.hdf5_file.exists() and not p.force:
        parser.error(
            f"HDF5 file '{p.hdf5_file}' already exists. "
            "Use --force to enforce overwrite."
        )

    toHDF(cat240_file=p.cat240_file, hdf5_file=p.hdf5_file, verbose=p.verbose)


if __name__ == "__main__":
    main()
