"""Pycatzao is a pure Python library for encoding, decoding and compressing Asterix CAT240 messages.

Below, find the documentation of the methods exposed by the API of Pycatzao grouped
under the topics "Decoder", exposed as:

 - `pycatzao.decode`
 - `pycatzao.decode_file`

for decoding already existing bytestreams of Asterix CAT240 messsages, "Encoder",
exposed as

 - `pycatzao.encode`
 - `pycatzao.make_summary`
 - `pycatzao.make_video_header`
 - `pycatzao.make_video_message`

for compiling new messages, and "Compressor", exposed as:

 - `pycatzao.compress`
 - `pycatzao.compress_file`

for compressing (encoded) messages.

Typically, no interactions between methods of different topics is necessary and one can
safely study their documentation separately.

Convenient helper functions for inferring the implicit binning scheme or for stacking
decoded block are exposed as "Utilities":

 - `pycatzao.infer_bin_edges`
 - `pycatzao.join_blocks`

Example:
    .. code-block:: python

        buffer_size = 100_000
        join_n_blocks = 1_000_000

        # compress data
        data_file = "my-data.cat240"
        with open("compressed.cat240", "wb") as f:
            for msg in pycatzao.compress_file(data_file, buffer_size=buffer_size):
                f.write(msg)

        # convert to csv
        for i, blocks in enumerate(
            itertools.batched(
                pycatzao.decode_file("compressed.cat240", buffer_size=buffer_size),
                join_n_blocks,
            )
        ):
            pd.DataFrame(pycatzao.join_blocks(blocks)).to_csv(
                f"csv-{i + 1}.csv.gz", header=True, index=False, float_format="%.2f"
            )

"""  # noqa: E501

from . import _version

__version__ = _version.get_versions()["version"]

# export public API of the package
from .compress import (
    compress,  # noqa: F401
    compress_file,  # noqa: F401
)
from .decoder import (
    decode,  # noqa: F401
    decode_file,  # noqa: F401
)
from .encoder import (
    encode,  # noqa: F401
    make_summary,  # noqa: F401
    make_video_header,  # noqa: F401
    make_video_message,  # noqa: F401
)
from .utils import (
    infer_bin_edges,  # noqa: F401
    join_blocks,  # noqa: F401
)
