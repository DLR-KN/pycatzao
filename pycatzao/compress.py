"""Compression of Asterix CAT240 messages.

Use :func:`compress` to compress encoded Asterix CAT240 messages or
:func:`compress_file` to compress an entire file. These operations do not change type
`001` messages or already compressed messages of type `002`.

Examples:
    .. code-block:: python

        # generate uncompressed messages
        blocks = [
            pycatzao.encode(
                pycatzao.make_video_message(..., compress=False),
                ...
            ),
        ]

        # compress encoded messages
        compressed, _ = pycatzao.compress(blocks)

    .. code-block:: python

        # compress an uncompressed CAT240 file
        with open("compressed.cat240", "wb") as f:
            for msg in tqdm(
                pycatzao.compress_file("uncompressed.cat240", buffer_size=100_000),
                desc="Compressing data",
            ):
                f.write(msg)
"""

import zlib

from pycatzao import _utils as utils


def _compress_block(data):
    data = bytearray(data)

    uap1 = data[3]
    mFX = uap1 & 0x01 == 0x01

    i, uap2 = (5, data[4]) if mFX else (4, 0)
    m050 = uap2 & 0x40 == 0x40
    m051 = uap2 & 0x20 == 0x20
    m052 = uap2 & 0x10 == 0x10
    m140 = uap2 & 0x08 == 0x08

    if data[i + 2] == 1:  # message type 1
        return bytes(data)

    i += 2  # I240/010
    i += 1  # I240/000
    i += 4  # I240/020
    i += 12  # I240/040 or I240/041

    # I240/04
    if data[i] == 0x80:  # already compressed
        return bytes(data)

    data[i] = 0x80  # set compression bit
    i += 2

    # I240/049
    nb_vb = int.from_bytes(data[i : i + 2], byteorder="big")

    # I240/050 or I240/051 or I240/052
    if m050:
        n = 4
    elif m051:
        n = 64
    elif m052:
        n = 256
    else:  # pragma: no cover
        raise ValueError("Invalid Asterix CAT240 message.")

    m = n * data[i + 5]
    amp = zlib.compress(data[i + 6 : i + m + 6][:nb_vb])
    tod = data[i + m + 6 : i + m + 9]

    # set NB_VB to the number of bytes after compression
    data[i : i + 2] = len(amp).to_bytes(2, byteorder="big")

    # pad to a multiple of `n`
    amp += b"\x00" * ((1 + (len(amp) - 1) // n) * n - len(amp))
    m = len(amp)
    assert m % n == 0

    # update REP and VIDEO BLOCK
    data[i + 5] = m // n
    data[i + 6 : i + m + 6] = amp

    i += 5  # I240/049
    i += m + 1  # I240/05x

    # I240/140
    if m140:
        data[i : i + 3] = tod
        i += 3

    data[1:3] = i.to_bytes(2, byteorder="big")

    return bytes(data[:i])


def compress(data):
    """Compress Asterix CAT240 data.

    Compresses non-compressed Asterix CAT240 type `002` messages. Other messages are
    returned unchanged. The data have to start with a new CAT240 message but can end in
    the middle of one. This requirement and the return type is identical to
    :func:`pycatzao.decoder.decode`.

    Args:
        data (bytes):
            Encoded Asterix CAT240 messages.

    Returns:
        tuple[bytes, bytes]:
            Compressed messages and state (see :func:`pycatzao.decoder.decode` for
            details on how to use the latter.)

    """
    return utils._map_blocks(data, func=_compress_block)


def compress_file(file_name, *, buffer_size=-1):
    """Compress an Asterix CAT240 file.

    This is a helper function that compresses a file with (binary) Asterix CAT240 data.

    Args:
        file_name (str | pathlib.Path):
            Name of the file.
        buffer_size (int):
            Process file in chunks of this size. If negative, the entire file will be
            loaded to RAM at once.

    Returns:
        typing.Generator[bytes, None, None]:
            Compressed messages.
    """
    yield from utils._map_file(
        file_name, func=compress, size=-1, buffer_size=buffer_size
    )
