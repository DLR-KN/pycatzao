"""Decoding of Asterix CAT240 messages.

Use :func:`decode` for decoding a byte sequence of encoded Asterix CAT240 messages. This
function will return trailing bytes that were not decoded such that one can easily
implement loops without the need to split CAT240 messages beforehand. Below, we show a
simple example how to decode data from a file in chunks. (Note that this is just a dull
example to demonstrate how to utilize the returned state, checkout :func:`decode_file`
if you actually want to decode a file!)

Example:
    .. code-block:: python

        with open("my-cat240-data.bin", "rb") as f:
            state = b""
            while chunk := f.read(1024)) != b"":
                blocks, state = pycatzao.decode(state + chunk)
                for block in blocks:
                    process_blocks(block)  # do somehting meaningful with the data

Here, a decoded `block` is a `dict` that contains the payload of the respective CAT240
message, i.e.,

 - `summary`: ASCII string to deliver stream meta data and/or labels

for blocks of message type `001` (Video Summary message), and

 - `idx`: Message sequence identifier
 - `az`: The azimuth in degrees. This value is calculated as the circular mean of `START_AZ` and `END_AZ` as both values are often set to non-meaningful values and it is very easy to confuse their range with the sensor resolution. In contrast, their circular mean is less ambiguous and can be used safely for downstream tasks.
 - `r`: Range of the data points in meters
 - `amp`: Video signal amplitude (aka the "data points")

for message type `002` (Video message), as well as

 - `sac`: System Area Code (SAC)
 - `sic`: System Identification Code (SIC)
 - `type`: Type of the message, `1` (Video summary message) or `2` (Video message)
 - `tod` (if present): Time of Day in seconds (We literally just decode the encoded value. It is beyond the scope of this library to interpret this value, e.g., as an absolute UTC time stamping.)

in both messages.
"""  # noqa: E501

import zlib

import numpy as np

from pycatzao import _utils


def _decode_block(data):
    uap1 = data[3]
    m040 = uap1 & 0x08 == 0x08
    mFX = uap1 & 0x01 == 0x01

    i, uap2 = (5, data[4]) if mFX else (4, 0)
    m050 = uap2 & 0x40 == 0x40
    m051 = uap2 & 0x20 == 0x20
    m052 = uap2 & 0x10 == 0x10
    m140 = uap2 & 0x08 == 0x08

    msg = {
        "sac": data[i],  # I240/010
        "sic": data[i + 1],  # I240/010
        "type": data[i + 2],  # I240/000
    }
    i += 3

    match msg["type"]:
        case 1:
            # I240/030
            n = data[i]
            msg["summary"] = data[i + 1 : i + 1 + n].decode("ascii")
            i += 1 + n

        case 2:
            # I240/020
            msg["idx"] = int.from_bytes(data[i : i + 4], byteorder="big")
            i += 4

            # I240/040 or I240/041
            az_scale = 360 / 2**16
            t_scale = 1e-9 if m040 else 1e-15

            start_az = int.from_bytes(data[i : i + 2], byteorder="big") * az_scale
            end_az = int.from_bytes(data[i + 2 : i + 4], byteorder="big") * az_scale
            start_rg = int.from_bytes(data[i + 4 : i + 8], byteorder="big")
            cell_dur = int.from_bytes(data[i + 8 : i + 12], byteorder="big") * t_scale
            i += 12

            msg["az"] = _utils._circular_mean(start_az, end_az)

            # I240/048
            compression = data[i] & 0x80 == 0x80
            res = 2 ** (data[i + 1] - 1)
            i += 2

            # I240/049
            nb_vb = int.from_bytes(data[i : i + 2], byteorder="big")
            nb_cells = int.from_bytes(data[i + 2 : i + 5], byteorder="big")
            i += 5

            # I240/050 or I240/051 or I240/052
            if m050:
                n = 4
            elif m051:
                n = 64
            elif m052:
                n = 256
            else:  # pragma: no cover
                raise ValueError("Invalid Asterix CAT240 message.")

            n *= data[i]
            amp = data[i + 1 : i + n + 1][:nb_vb]
            i += n + 1

            if compression:
                try:
                    amp = zlib.decompress(amp, wbits=0)
                except zlib.error as e:  # pragma: no cover
                    raise ValueError("Video blocks not compressed with zlib.") from e

            if res == 8:
                dtype = np.uint8
            elif res == 16:
                dtype = np.uint16
            elif res == 32:
                dtype = np.uint32
            else:  # pragma: no cover
                raise NotImplementedError(
                    f"{res} bit resolution is not yet implemented."
                )

            msg["amp"] = np.frombuffer(amp, dtype=dtype)[:nb_cells]

            c = 299_792_458
            msg["r"] = cell_dur * (start_rg + np.arange(nb_cells)) * c / 2

            mask = msg["amp"] > 0
            msg["amp"] = msg["amp"][mask]
            msg["r"] = msg["r"][mask].astype(np.float32)

    # I240/140
    if m140:
        msg["tod"] = int.from_bytes(data[i : i + 3], byteorder="big") / 128
        i += 3

    if i != len(data):  # pragma: no cover
        raise ValueError("Invalid Asterix CAT240 message.")

    return msg


def decode(data):
    r"""Decode CAT240 data.

    This functions decodes a given binary blob of encoded Asterix CAT240 messages (type
    `001` or `002`). The data have to start with a new CAT240 message but can end in
    the middle of one. The bytes of the incomplete message at the end of the input data
    are returned such that a chunking a long byte sequence by subsequent calls to
    :func:`decode` becomes trivial.

    Example:
        >>> data = b'\\xf0\\x00\\x13\\xd1...'
        >>> blocks, state = pycatzao.decode(data[:128])
        >>> blocks
        [{'sac': 7, 'sic': 42, 'type': 1, 'summary': '...', 'tod': 100.0},
         {'sac': 7, 'sic': 42, 'type': 2, 'idx': 4711, 'az': 5.9, 'amp': array([248, ...], dtype=uint8), 'r': array([ 94.2, ...], dtype=float32), 'tod': 100.0},
         {'sac': 7, 'sic': 42, 'type': 2, 'idx': 4712, 'az': 6.4, 'amp': array([127, ...], dtype=uint8), 'r': array([ 89.2, ...], dtype=float32), 'tod': 100.1}]
        >>> blocks, state = pycatzao.decode(state + data[128:])
        >>> blocks
        [{'sac': 7, 'sic': 42, 'type': 2, 'idx': 4713, 'az': 6.9, 'amp': array([255, ...], dtype=uint8), 'r': array([ 94.2, ...], dtype=float32), 'tod': 100.2}, ...]

    Args:
        data (bytes):
            Encoded CAT240 messages as a binary blob that start with a new message.

    Returns:
        tuple[dict, bytes]:
            Decoded message and trailing bytes. The latter carries the state of the
            decoder and should be prepended to the input data for the subsequent call
            to :func:`decode`.
    """  # noqa: E501
    return _utils._map_blocks(data, func=_decode_block)


def decode_file(file_name, *, size=-1, buffer_size=-1):
    """Decode a CAT240 file.

    This is a helper function that reads and decodes a file with (binary) Asterix
    CAT240 data.

    Args:
        file_name (str | pathlib.Path):
            Name of the file.
        size (int):
            Maximum number of bytes to read from the file. If negative, the entire file
            will be read.
        buffer_size (int):
            Process file in chunks of this size. If negative, the entire file will be
            loaded to RAM at once.

    Returns:
        typing.Generator[dict, None, None]:
            Decoded messages (type of generated items is the same as the first return
            type of :func:`decode`.)
    """
    yield from _utils._map_file(
        file_name, func=decode, size=size, buffer_size=buffer_size
    )
