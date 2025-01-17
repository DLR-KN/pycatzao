"""Helper functions for dealing with decoded Asterix CAT240 data.

If you need the decoded data more structured as a table, use :func:`join_blocks` for
this! This function will join (type `002`) messages by repeating scalar values and
return a single `dict` that can be fed directly to, e.g., a `pandas.DataFrame`.

Typically, the azimuth and range values follow an implicit binning scheme. Sometimes,
this scheme can be inferred by parsing a handful of messages. We implement a simple
heuristic that tries to infer this scheme in :func:`infer_bin_edges`. Note that the
azimuth bins are circular and that due to this periodicity, the bin content of the first
and last bin might have to be superimposed depending on the context.
"""

import numpy as np
from tqdm import tqdm

from pycatzao import _utils


def _cell_size(data):
    uap1 = data[3]
    m040 = uap1 & 0x08 == 0x08
    mFX = uap1 & 0x01 == 0x01

    i = 7 if mFX else 6
    if data[i] == 1:  # message type 1
        return None

    az_scale = 360 / 2**16
    t_scale = 1e-9 if m040 else 1e-15

    start_az = int.from_bytes(data[i + 5 : i + 7], byteorder="big") * az_scale
    end_az = int.from_bytes(data[i + 7 : i + 9], byteorder="big") * az_scale
    start_rg = int.from_bytes(data[i + 9 : i + 13], byteorder="big")
    cell_dur = int.from_bytes(data[i + 13 : i + 17], byteorder="big") * t_scale

    c = 299_792_458
    cell_width = cell_dur * c / 2  # in meters

    return _utils._circular_mean(start_az, end_az), start_rg * cell_width, cell_width


def infer_bin_edges(data):
    r"""Infers binning scheme from CAT240 data.

    This function infers the range and azimuthal binning scheme of the given CAT240 data
    using simple heuristics. The result can directly be fed into :func:`numpy.arange`
    and :func:`numpy.linspace` to get the bin edges in meters and degrees, respectively.
    The azimuth bins are assuemd to be cyclic whereas for the range bins no upper limit
    (aka `stop`) is infered; this value has to be set manually:

    Example:
        >>> data = b'\\xf0\\x00\\x13\\xd1...'
        >>> bins, _ = pycatzao.infer_bin_edges(data)
        >>> r_edges = np.arange(**bins["r"], stop=3_000)
        >>> az_edges = np.linspace(**bins["az"])

    Note that due to the monotonicity of :func:`numpy.arange` and the periodicity of the
    azimuth bins, the first bin edge is negative and the last is larger than 360.
    Depending on the context one might have to superimpose the corresponding bin
    entries!

    Args:
        data (bytes):
            Encoded CAT240 messages as a binary blob that start with a new message.

    Returns:
        tuple[dict | None, bytes]:
            The first return value is a dictionary with the inferred binning scheme for
            range (key `"r"`) and azimuth (key `"az"`). If an insufficient number of
            bytes were provided, `None` and the full input data are returned. The second
            return value carries the state of the decoder and should be prepended to the
            input data for the subsequent call to :func:`infer_bin_edges`. Note that the
            inferred binning scheme of previous calls are not taken into account for the
            current inference.
    """
    cell_sizes, tail = _utils._map_blocks(data, func=_cell_size)
    cell_sizes = list(zip(*filter(lambda s: s is not None, cell_sizes)))
    if len(cell_sizes) == 0:
        return None, data

    az, r0, dr = cell_sizes

    az, c = np.unique(az, return_counts=True)
    if az.size < 2:
        return None, data

    az0 = az[np.argmax(c)]
    daz = np.diff(az)

    daz_min = 0.01  # magic value... in the end this is just a heuristic:)
    daz = np.median(daz[np.abs(daz) > daz_min])
    az_start = (az0 + np.sign(180 - az0) * daz / 2) % daz - daz
    az_end = az_start + np.ceil((360 - az_start) / daz) * daz

    if az_end - daz > 360:
        az_end -= daz

    assert az_start <= 0, az_start
    assert az_end >= 360, az_end

    r0 = min(r0)
    dr = np.median(dr).item()

    return {
        "az": {
            "start": az_start.item(),
            "num": round((az_end - az_start) / daz),
            "stop": az_end.item(),
        },
        "r": {
            "start": r0 - dr / 2,
            "step": dr,
        },
    }, tail


def join_blocks(blocks, *, show_progress=False):
    r"""Join decoded CAT240 blocks into columns.

    Joins the fields `tod`, `az`, `r`, and `amp` of decoded Asterix CAT240 blocks into a
    rectangular table by repeating the scalar values (`tod` and `az`). The result can
    directly be fed to, e.g., `pandas.DataFrame`. Messages of type `001` are skipped.

    Example:
        >>> data = b'\\xf0\\x00\\x13\\xd1...'
        >>> blocks, _ = pycatzao.decode(data)
        >>> pycatzao.join_blocks(blocks)
        { 'tod': array([100.0, 100.0, 100.1, 100.1, 100.1, 100.2, ...], dtype=float32),
           'az': array([  5.9,   5.9,   6.4,   6.4,   6.4,   6.9, ...], dtype=float32),
            'r': array([ 94.2,  94.2,  89.2,  89.2,  89.2,  94.2, ...], dtype=float32),
          'amp': array([248,   250,   127,   125,   130,   255,   ...], dtype=uint8)}


    Args:
        blocks (Iterable[dict]):
            Decoded blocks as returned by, e.g, :func:`decode` or :func:`decode_file`.

        show_progress (bool):
            Show progress bar.

    Returns:
        dict:
            Columns of a (rectangular) table.

    """
    blocks = [block for block in blocks if block["type"] == 2]
    if len(blocks) == 0:
        return {
            "tod": np.empty(0, dtype=np.float32),
            "az": np.empty(0, dtype=np.float32),
            "r": np.empty(0, dtype=np.float32),
            "amp": np.empty(0, dtype=np.uint8),
        }

    n = np.cumsum([len(block["amp"]) for block in blocks])
    n = np.insert(n, 0, 0)
    n_max = n[-1]

    tod = np.empty(n_max, dtype=np.float32)
    az = np.empty(n_max, dtype=np.float32)
    rg = np.empty(n_max, dtype=np.float32)
    amp = np.empty(n_max, dtype=blocks[0]["amp"].dtype)

    for block, i, j in tqdm(
        zip(blocks, n[:-1], n[1:]),
        total=len(blocks),
        disable=not show_progress,
        desc="Joining blocks",
    ):
        tod[i:j] = block.get("tod", np.nan)
        az[i:j] = block["az"]
        rg[i:j] = block["r"]
        amp[i:j] = block["amp"]

    return {
        "tod": tod,
        "az": az,
        "r": rg,
        "amp": amp,
    }
