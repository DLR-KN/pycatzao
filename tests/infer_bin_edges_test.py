import numpy as np
import pytest

import pycatzao


@pytest.mark.parametrize("n_msg", [0, 1, 5])
@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("tod", [True, False])
def test_too_few_messages(n_msg, compress, tod):
    sac, sic = 0, 0
    encoded = [
        pycatzao.encode(pycatzao.make_summary(summary="FoObaR"), sac=sac, sic=sic),
    ]

    # Using the same start/end values for all iterations will result in too few values
    # for inferring the azimuth bins
    start_az = 42
    end_az = 45

    for i in range(n_msg):
        encoded += [
            pycatzao.encode(
                pycatzao.make_video_message(
                    np.zeros(3, dtype=np.uint8),
                    msg_index=i,
                    header=pycatzao.make_video_header(
                        start_az=start_az,
                        end_az=end_az,
                        cell_offset=0,
                        cell_width=1,
                    ),
                    compress=compress,
                ),
                sac=sac,
                sic=sic,
                tod=tod,
            )
        ]

    raw = b"".join(encoded)
    bins, data = pycatzao.infer_bin_edges(raw)

    assert bins is None
    assert data == raw


@pytest.mark.parametrize(
    "az_start,az_end,az_binning_scheme",
    [
        (
            [0.0, 0.0, 6, 18, 42],
            [0.0, 0.0, 18, 30, 54],
            [-6, 366, 32],
        )
    ],
)
@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("tod", [False, True])
@pytest.mark.parametrize("tail", [False, True])
def test_binning_inference(az_start, az_end, az_binning_scheme, compress, tod, tail):
    sac, sic = 0, 0

    encoded = [
        pycatzao.encode(pycatzao.make_summary(summary="FoObaR"), sac=sac, sic=sic),
    ]

    for i, (az1, az2, cell_offset) in enumerate(
        zip(
            az_start,
            az_end,
            (1 + np.arange(len(az_start))) * 100,
            strict=True,
        )
    ):
        encoded += [
            pycatzao.encode(
                pycatzao.make_video_message(
                    np.zeros(3, dtype=np.uint8),
                    msg_index=i,
                    header=pycatzao.make_video_header(
                        start_az=az1,
                        end_az=az2,
                        cell_offset=cell_offset,
                        cell_width=10,
                    ),
                    compress=compress,
                ),
                sac=sac,
                sic=sic,
                tod=tod,
            )
        ]

    raw = b"".join(encoded)
    if tail:
        rng = np.random.default_rng(0)
        n_bytes = rng.integers(4, 512).item()
        n = rng.integers(0, n_bytes - 3)
        assert n + 3 < n_bytes

        tail = b"\xf0" + n_bytes.to_bytes(2, byteorder="big") + rng.bytes(n)
    else:
        tail = b""

    bins, data = pycatzao.infer_bin_edges(raw + tail)
    assert data == tail

    assert bins["r"]["start"] == pytest.approx(95)
    assert bins["r"]["step"] == pytest.approx(10)

    az_start, az_stop, az_num = az_binning_scheme
    assert bins["az"]["start"] == pytest.approx(az_start, abs=0.01)
    assert bins["az"]["stop"] == pytest.approx(az_stop, abs=0.1)
    assert bins["az"]["num"] == az_num
