import numpy as np
import pytest
from helpers import test_utils

import pycatzao


@pytest.mark.parametrize("summary, sac, sic", [["Lorem ipsum dolor sit", 1, 2]])
@pytest.mark.parametrize("tod", [-1, 42, 24 * 60 * 60])
def test_single_type1_message(summary, sac, sic, tod):
    encoded = pycatzao.encode(pycatzao.make_summary(summary), sac=sac, sic=sic, tod=tod)
    decoded, tail = pycatzao.decode(encoded)
    assert len(decoded) == 1
    assert tail == b""

    decoded = decoded[0]

    assert decoded["sac"] == sac
    assert decoded["sic"] == sic
    assert decoded["type"] == 1
    assert decoded["summary"] == summary

    if tod >= 0:
        assert decoded["tod"] == pytest.approx(tod)


@pytest.mark.parametrize("seed", list(range(100)))
@pytest.mark.parametrize("n_max", [1, 2, 10, 100, 1000])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("tod", [True, False])
def test_single_type2_message(seed, n_max, dtype, compress, tod):
    rng = np.random.default_rng(seed)
    encoded, msg = test_utils.random_type2_message(
        rng, n_max=n_max, dtype=dtype, compress=compress, tod=tod
    )

    decoded, tail = pycatzao.decode(encoded)
    assert len(decoded) == 1
    assert tail == b""

    decoded = decoded[0]

    assert decoded["sac"] == msg["sac"]
    assert decoded["sic"] == msg["sic"]
    assert decoded["type"] == 2

    assert decoded["idx"] == msg["idx"]

    assert decoded["az"] == pytest.approx(msg["az"], abs=0.01)

    non_zero = msg["amp"] > 0
    assert len(decoded["r"]) == np.sum(non_zero)

    assert msg["amp"].ndim == 1
    idx = np.arange(len(msg["amp"]))[non_zero]
    assert decoded["r"] == pytest.approx(
        msg["cell_offset"] + idx * msg["cell_width"], rel=1e-3
    )

    assert decoded["amp"] == pytest.approx(msg["amp"][non_zero])
    assert decoded["amp"].dtype == dtype

    if tod:
        assert decoded["tod"] == pytest.approx(msg["tod"], abs=0.01)
