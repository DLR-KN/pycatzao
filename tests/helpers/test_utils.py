import numpy as np

import pycatzao


def random_type2_message(rng, *, n_max, dtype, tod, compress):
    sac = rng.integers(0, 256).item()
    sic = rng.integers(0, 256).item()
    msg_index = rng.integers(0, 0xFFFFFFFF).item()

    start_az = rng.uniform(0, 360)
    end_az = rng.uniform(0, 360)

    cell_width = rng.uniform(0, 1_000)
    cell_offset = rng.integers(0, 4).item() * cell_width

    n = rng.integers(1, n_max + 1)
    amp = rng.integers(0, np.iinfo(dtype).max, size=n).astype(dtype)

    tod = rng.uniform(0, 24 * 60 * 60) if tod else -1

    az1 = np.deg2rad(start_az)
    az2 = np.deg2rad(end_az)
    az12 = np.arctan2(np.sin(az1) + np.sin(az2), np.cos(az1) + np.cos(az2))
    az12 = np.rad2deg(az12)

    if az12 < 0:
        az12 += 360

    return pycatzao.encode(
        pycatzao.make_video_message(
            amp,
            msg_index=msg_index,
            header=pycatzao.make_video_header(
                start_az=start_az,
                end_az=end_az,
                cell_offset=cell_offset,
                cell_width=cell_width,
            ),
            compress=compress,
        ),
        sac=sac,
        sic=sic,
        tod=tod,
    ), {
        "sac": sac,
        "sic": sic,
        "idx": msg_index,
        "az": az12,
        "cell_offset": cell_offset,
        "cell_width": cell_width,
        "amp": amp,
        "tod": tod,
    }
