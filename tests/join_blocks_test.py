from collections import defaultdict

import numpy as np
import pytest
from helpers import test_utils

import pycatzao


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("n_msg", [1, 5, 10])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
@pytest.mark.parametrize("add_summary", [False, True])
@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("tod", [True, False])
def test_join_blocks(seed, n_msg, dtype, add_summary, compress, tod):
    rng = np.random.default_rng(seed)
    encoded = [
        test_utils.random_type2_message(
            rng, n_max=100, dtype=dtype, compress=compress, tod=tod
        )[0]
        for _ in range(n_msg)
    ]

    expected = defaultdict(list)
    for block in pycatzao.decode(b"".join(encoded))[0]:
        for r, a in zip(block["r"], block["amp"], strict=True):
            if tod:
                expected["tod"].append(block["tod"])

            expected["az"].append(block["az"])
            expected["r"].append(r)
            expected["amp"].append(a)

    if add_summary:
        summary = pycatzao.encode(
            pycatzao.make_summary(summary="foobar"), sac=1, sic=2, tod=100.0
        )

        i = rng.integers(0, len(encoded) + 1)
        if i == len(encoded):
            encoded.append(summary)

        encoded = encoded[:i] + [summary] + encoded[i:]

    table = pycatzao.join_blocks(pycatzao.decode(b"".join(encoded))[0])

    if tod:
        assert table["tod"].tolist() == pytest.approx(expected["tod"])
    else:
        assert np.all(np.isnan(table["tod"]))

    assert table["amp"].dtype == dtype
    assert table["amp"].tolist() == expected["amp"]

    assert table["az"].tolist() == pytest.approx(expected["az"])
    assert table["r"].tolist() == pytest.approx(expected["r"])


@pytest.mark.parametrize("add_summary", [False, True])
def test_join_zero_blocks(add_summary):
    encoded = []
    if add_summary:
        summary = pycatzao.encode(
            pycatzao.make_summary(summary="foobar"), sac=1, sic=2, tod=100.0
        )
        encoded.append(summary)

    table = pycatzao.join_blocks(pycatzao.decode(b"".join(encoded))[0])
    assert table["tod"].tolist() == []
    assert table["az"].tolist() == []
    assert table["r"].tolist() == []
    assert table["amp"].tolist() == []
