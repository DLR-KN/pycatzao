#!/usr/bin/env python

import os
import sys

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(here)

import versioneer  # noqa: E402

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="pycatzao",
    description="A pure Python library for encoding, decoding and compressing Asterix"
    " CAT240 messages.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DLR-KN/pycatzao",
    author="Nis Meinert",
    author_email="nis.meinert@dlr.de",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(exclude=["docs", "tests"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pytest",
            "pytest-cov",
            "sphinx",
        ],
    },
)
