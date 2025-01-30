import pathlib
import tempfile
from itertools import accumulate

import numpy as np
import pytest
from helpers import test_utils

import pycatzao


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("n_msg", [1, 5, 10])
@pytest.mark.parametrize("limit_size", [False, True])
@pytest.mark.parametrize("buffer_size", [1, 10, 100, -1])
@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("tod", [True, False])
@pytest.mark.parametrize("tail", [False, True])
def test_decode_file(seed, n_msg, limit_size, buffer_size, compress, tod, tail):
    encoded = [
        pycatzao.encode(
            pycatzao.make_summary(summary="foobar"), sac=1, sic=2, tod=100.0
        )
    ]

    rng = np.random.default_rng(seed)
    encoded += [
        test_utils.random_type2_message(
            rng, n_max=100, dtype=np.uint8, compress=compress, tod=tod
        )[0]
        for _ in range(n_msg)
    ]

    raw = b"".join(encoded)
    if tail:
        n_bytes = rng.integers(4, 512).item()
        n = rng.integers(0, n_bytes - 3)
        assert n + 3 < n_bytes

        raw += b"\xf0" + n_bytes.to_bytes(2, byteorder="big") + rng.bytes(n)

    with tempfile.NamedTemporaryFile() as f:
        f.write(raw)
        f.flush()

        acclen = list(accumulate(map(len, encoded)))
        size = acclen[-1]
        if limit_size:
            size = rng.integers(0, size + 1)

        decoded = list(
            pycatzao.decode_file(
                str(f.name) if seed % 2 == 0 else pathlib.Path(f.name),
                size=size,
                buffer_size=buffer_size,
            )
        )

    # assume that pycatazo.decode() works for a single message
    expected = [
        pycatzao.decode(msg)[0][0] for n, msg in zip(acclen, encoded) if n <= size
    ]

    assert len(decoded) == len(expected)

    if len(expected) > 0:
        for block1, block2 in zip(decoded, expected):
            assert block1["sac"] == block2["sac"]
            assert block1["sic"] == block2["sic"]
            assert block1["type"] == block2["type"]
            if tod:
                assert block1["tod"] == block2["tod"]

        assert decoded[0]["summary"] == expected[0]["summary"]
        for block1, block2 in zip(decoded[1:], expected[1:]):
            assert block1["idx"] == block2["idx"]
            assert block1["az"] == block2["az"]
            assert block1["az_cell_size"] == block2["az_cell_size"]
            assert np.allclose(block1["r"], block2["r"])
            assert np.allclose(block1["r_cell_size"], block2["r_cell_size"])
            assert np.allclose(block1["amp"], block2["amp"])
