# noqa: D100

import argparse
import functools
import itertools
import pathlib

import h5py
import numpy as np
from tqdm import tqdm

import pycatzao


def _hour(*, tod):
    return int(tod // 3600) % 24


def _join_blocks(blocks, chunk, verbose=False):
    t, r1, r2, az1, az2, amp = [], [], [], [], [], []

    tod = -1
    for block in tqdm(blocks, desc="Decoding blocks", disable=not verbose):
        if block["az_cell_size"] > 0 and block["amp"].size > 0:
            if block["tod"] < tod:
                raise ValueError(
                    "ToD of previous block: {tod} < {block['tod']=}. "
                    "ToD values of subsequent blocks have to increase."
                )

            if chunk and _hour(tod=tod) != _hour(tod=block["tod"]):
                if len(t) > 0:
                    yield (
                        _hour(tod=tod),
                        dict(
                            tod=np.concatenate(t),
                            r1=np.concatenate(r1),
                            r2=np.concatenate(r2),
                            az1=np.concatenate(az1),
                            az2=np.concatenate(az2),
                            amp=np.concatenate(amp),
                        ),
                    )

                tod = block["tod"]
                t, r1, r2, az1, az2, amp = [], [], [], [], [], []

            r = block["r"]
            dr = block["r_cell_size"]
            r1.append(r - dr / 2)
            r2.append(r + dr / 2)

            az = block["az"]
            daz = block["az_cell_size"]
            az1.append(np.full_like(r, az - daz / 2))
            az2.append(np.full_like(r, az + daz / 2))

            amp.append(block["amp"])

            t.append(np.full_like(r, block["tod"]))

    if len(t) > 0:
        yield (
            _hour(tod=tod),
            dict(
                tod=np.concatenate(t),
                r1=np.concatenate(r1),
                r2=np.concatenate(r2),
                az1=np.concatenate(az1),
                az2=np.concatenate(az2),
                amp=np.concatenate(amp),
            ),
        )


def _angle_diff_sign(diff):
    return (diff < -180) | ((diff > 0) & (diff < 180))


def _find_cycle_start(az, *, verbose=False):
    first = np.zeros(az.shape, dtype=np.uint32)
    for i in tqdm(range(1, az.size), desc="Finding cycle lengths", disable=not verbose):
        m = _angle_diff_sign(az[first[i - 1] : i][::-1] - az[i]) > 0
        q = np.flatnonzero(m[:-1] & ~m[1:])
        first[i] = i - q[0] - 1 if q.size > 0 else first[i - 1]

    return first


def _make_cycle_lookup(blocks, *, verbose=False):
    first = _find_cycle_start(blocks["az1"], verbose=verbose)
    last = np.arange(first.size)

    t = blocks["tod"]
    mask = np.full(t.shape, True, dtype=np.bool)
    mask[:-1] = t[1:] > t[:-1]

    return {"cycle/first": first[mask], "cycle/last": last[mask], "cycle/tod": t[mask]}


def toHDF(
    *,
    cat240_files,
    hdf5_file_pattern,
    chunk,
    force_overwrite,
    verbose=True,
):
    """Converts an Asterix CAT240 into an HDF file.

    Args:
        cat240_files (list[str | pathlib.Path]):
            Filename of the cat240 file.
        hdf5_file_pattern (Callable[[datetime.date], str | pathlib.Path]):
            Pattern that takes a date and returns the HDF5 file name.
        chunk (bool):
            Write to separate HDF5 files per hour.
        force_overwrite (bool):
            Overwrite HDF5 files.
        verbose (bool):
            Print status to console.

    Returns:
        (list[pathlib.Path]): List of generated HDF5 files.
    """
    log = print if verbose else lambda *args, **kwargs: None

    hdf5_files = []

    f, path = None, None
    decode_file = functools.partial(pycatzao.decode_file, buffer_size=100_000_000)
    for h, df in _join_blocks(
        itertools.chain.from_iterable(map(decode_file, cat240_files)),
        chunk=chunk,
        verbose=verbose,
    ):
        df |= _make_cycle_lookup(df, verbose=verbose)

        if (p := pathlib.Path(hdf5_file_pattern(h)).resolve()) != path:
            if f is not None:
                f.close()

            path = p

            if path.is_file() and not force_overwrite:
                raise ValueError(
                    f"HDF5 file '{path}' already exists. "
                    "Use --force to enforce overwrite."
                )

            f = h5py.File(path, "w")
            hdf5_files.append(path)

        assert f is not None

        for k in df:
            f.create_dataset(k, data=df[k], compression="gzip")

        f["tod"].attrs["desc"] = "Time of Day (UTC)"
        f["tod"].attrs["unit"] = "second"

        f["az1"].attrs["desc"] = "Low edge of azimuth cell"
        f["az1"].attrs["unit"] = "degree"

        f["az2"].attrs["desc"] = "High edge of azimuth cell"
        f["az2"].attrs["unit"] = "degree"

        f["r1"].attrs["desc"] = "Low edge of range cell"
        f["r1"].attrs["unit"] = "meter"

        f["r2"].attrs["desc"] = "High edge of range cell"
        f["r2"].attrs["unit"] = "meter"

        f["amp"].attrs["desc"] = "Amplitude"

        f["cycle/tod"].attrs["desc"] = "Time of Day (UTC)"
        f["cycle/first"].attrs["desc"] = "Index of cycle start"
        f["cycle/last"].attrs["desc"] = "Index of cycle end"

    if f is not None:
        f.close()

    log("done.")

    return hdf5_files


def main():  # noqa: D103
    parser = argparse.ArgumentParser(
        prog="cat240toHDF5",
        description="A handy tool that converts Asterix CAT240 data into HDF5 files.",
    )
    parser.add_argument(
        "cat240_files",
        type=pathlib.Path,
        nargs="+",
        help="Path(s) to the CAT240 file(s)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix of HDF5 file names",
    )
    parser.add_argument(
        "--hourly",
        action="store_true",
        help="Write to separate HDF5 files per hour instead of a single file.",
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

    for f in p.cat240_files:
        if not f.is_file():
            parser.error(f"Cannot open data file '{f}'")

    def _pattern(prefix, hour):
        return f"{prefix}{'' if hour is None else f'{hour:02}_UTC'}.hdf5"

    hdf5_files = toHDF(
        cat240_files=p.cat240_files,
        hdf5_file_pattern=lambda h: _pattern(p.prefix, h if p.hourly else None),
        chunk=p.hourly,
        force_overwrite=p.force,
        verbose=p.verbose,
    )

    if p.verbose:
        print("Results are written to: " + ", ".join(map(str, hdf5_files)))


if __name__ == "__main__":
    main()
