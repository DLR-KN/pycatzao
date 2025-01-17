import numpy as np


def _circular_mean(a, b):
    a = np.deg2rad(a)
    b = np.deg2rad(b)
    mean = np.arctan2(np.sin(a) + np.sin(b), np.cos(a) + np.cos(b))
    return np.rad2deg(mean if mean >= 0 else mean + 2 * np.pi).item()


def _map_blocks(data, func):
    blocks = []

    i = 0
    done = False
    while i + 3 < len(data) and not done:
        if data[i] != 240:  # pragma: no cover
            raise ValueError("Invalid Asterix CAT240 message.")

        length = int.from_bytes(data[i + 1 : i + 3], byteorder="big")
        if not (done := i + length > len(data)):
            blocks.append(func(data[i : i + length]))
            i += length

    return blocks, data[i:] if i < len(data) else b""


def _map_file(file_name, func, *, size, buffer_size):
    if buffer_size <= 0:
        buffer_size = -1

    if size > 0 > buffer_size:
        buffer_size = size

    n = 0
    data = b""
    with open(file_name, "rb") as f:
        while (size < 0 or n < size) and (
            chunk := f.read(buffer_size if size < 0 else min(buffer_size, size - n))
        ) != b"":
            n += len(chunk)
            blocks, data = func(data + chunk)
            yield from blocks
