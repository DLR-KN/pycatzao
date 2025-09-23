import cat240toHDF5
import h5py
import numpy as np
import pytest
from helpers import test_utils

import pycatzao


def _join_blocks(blocks):
    t, r1, r2, az1, az2, amp = [], [], [], [], [], []

    for block in blocks:
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

    return dict(
        tod=np.concatenate(t),
        r1=np.concatenate(r1),
        r2=np.concatenate(r2),
        az1=np.concatenate(az1),
        az2=np.concatenate(az2),
        amp=np.concatenate(amp),
    )


def test_cat240toHDF5(tmp_path):
    rng = np.random.default_rng(0)

    input_files = []
    msgs = []
    for i, hours in enumerate([[1.1, 1.2, 1.3], [1.4, 1.5], [2.1, 2.2]], start=1):
        file_name = tmp_path / f"f{i}.cat240"
        input_files.append(file_name)

        encoded = [
            test_utils.random_type2_message(
                rng, n_max=10, dtype=np.uint8, tod=h * 60 * 60, compress=False
            )[0]
            for h in hours
        ]
        msgs += encoded

        with open(file_name, "bw") as f:
            f.write(b"".join(encoded))

    hdf5_files = cat240toHDF5.toHDF(
        cat240_files=input_files,
        hdf5_file_pattern=lambda h: tmp_path / f"output_{h:02}.hdf5",
        chunk=True,
        force_overwrite=True,
        verbose=False,
    )
    assert hdf5_files == [tmp_path / "output_01.hdf5", tmp_path / "output_02.hdf5"]

    for f, idx in zip(hdf5_files, [[0, 1, 2, 3, 4], [5, 6]], strict=True):
        expected, _ = pycatzao.decode(b"".join([msgs[i] for i in idx]))
        expected = _join_blocks(expected)

        with h5py.File(f, "r") as f:
            for k, v in expected.items():
                assert f[k][:] == pytest.approx(v), k
