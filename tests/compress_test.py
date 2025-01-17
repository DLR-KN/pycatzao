import pathlib
import tempfile

import numpy as np
import pytest
from helpers import test_utils

import pycatzao


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("n_msg", [1, 5, 10])
@pytest.mark.parametrize("buffer_size", [1, 10, 100, -1])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
@pytest.mark.parametrize("tod", [True, False])
@pytest.mark.parametrize("compress_twice", [False, True])
def test_compress_file(seed, n_msg, buffer_size, dtype, tod, compress_twice):
    encoded = [
        pycatzao.encode(
            pycatzao.make_summary(summary="foobar"), sac=1, sic=2, tod=100.0
        )
    ]

    rng = np.random.default_rng(seed)
    encoded += [
        test_utils.random_type2_message(
            rng, n_max=100, dtype=dtype, compress=compress_twice, tod=tod
        )[0]
        for _ in range(n_msg)
    ]
    raw = b"".join(encoded)

    with tempfile.NamedTemporaryFile() as f:
        f.write(raw)
        f.flush()

        encoded_compressed = list(
            pycatzao.compress_file(
                str(f.name) if seed % 2 == 0 else pathlib.Path(f.name),
                buffer_size=buffer_size,
            )
        )

    decoded, _ = pycatzao.decode(b"".join(encoded_compressed))
    expected, _ = pycatzao.decode(b"".join(encoded))
    assert len(decoded) == len(expected)

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
        assert np.allclose(block1["r"], block2["r"])
        assert np.allclose(block1["amp"], block2["amp"])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_compression_ratio(dtype):
    block = pycatzao.encode(
        pycatzao.make_video_message(
            np.array([0] * 100 + [1] * 100, dtype=dtype),
            msg_index=42,
            header=pycatzao.make_video_header(
                start_az=45.0,
                end_az=45.0,
                cell_offset=9,
                cell_width=3,
            ),
            compress=False,
        ),
        sac=1,
        sic=2,
    )

    compressed, tail = pycatzao.compress(block)
    assert tail == b""
    assert len(block) // 3 > len(compressed[0])
