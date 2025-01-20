[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyPI](https://img.shields.io/pypi/v/pycatzao)](https://pypi.org/project/pycatzao/)
[![Documentation](https://github.com/DLR-KN/pycatzao/actions/workflows/deploy_docs.yml/badge.svg)](https://DLR-KN.github.io/pycatzao/)
[![Test coverage](https://codecov.io/gh/DLR-KN/pycatzao/graph/badge.svg?token=LJZ8G0DCHS)](https://codecov.io/gh/DLR-KN/pycatzao)
[![Unit tests](https://github.com/DLR-KN/pycatzao/actions/workflows/run_tests.yml/badge.svg)](https://codecov.io/gh/DLR-KN/pycatzao)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[[Documentation]](https://DLR-KN.github.io/pycatzao/)

# Pycatzao

Pycatzao is a pure Python library for encoding, decoding and compressing Asterix CAT240 messages.

## Usage & Documentation

Pycatzao releases are available as wheel packages for macOS, Windows and Linux on [PyPI](https://pypi.org/project/pycatzao/).
Install it using pip:
```bash
$ pip install pycatzao
```

If you prefer installing this library from source, run:
```bash
# optional
$ pip install -r requirements.txt
$ pip install -e .
```

After installing, simply import `pycatzao` and start fooling around with its API:

```python
import pycatzao

print("Version:", pycatzao.__version__)

for block in pycatzao.decode_file("my-cat240-data.bin"):
    process_cat240_block(
        range=block["r"],
        azimuth=block["az"],
        amplitude=block["amp"],
        ...
    )  # do something meaningful with the data
```

See [examples/cat240toHDF5.py](https://github.com/DLR-KN/pycatzao/blob/main/examples/cat240toHDF5.py) for a more elaborated example or [our documentation](https://DLR-KN.github.io/pycatzao) for a full list of available functions. Note, however, that you won't find an introduction into Asterix CAT240 here since we assume that you know the basics of the standard. If you need such an introduction though, simply search for _Asterix CAT240_ with a search engine of your choice or directly consult [https://www.eurocontrol.int/asterix](https://www.eurocontrol.int/asterix).

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to submit a pull request.

### Development Setup
To set up your development environment, follow these steps:

1. Clone the repository:
   ```bash
   $ git clone git@github.com:DLR-KN/pycatzao.git
   ```

2. Change to the project directory:
   ```bash
   $ cd pycatzao
   ```

3. Install the development dependencies using `pip`:
   ```bash
   $ pip install -e .[dev]
   ```

### Pre-Commit-Hooks
To maintain code quality and avoid pushing invalid commits, we recommend using pre-commit hooks. These hooks perform automated checks before commits are made. To set up pre-commit hooks, follow these steps:

1. Install the pre-commit package (if not already installed):
   ```bash
   $ pip install pre-commit
   ```

2. Install the hooks:
   ```bash
   $ pre-commit install
   ```

Now, each time you commit changes, the pre-commit hooks will run checks such as formatting, linting, and more. If any issues are found, they will be flagged before the commit is made.

### Running Tests
You can run tests using the following command:
```bash
$ pytest --cov=pycatzao
```

Make sure to run tests before submitting a pull request to ensure that everything is functioning as expected.

### Generate documentation (optional)
If you like, you can generate the documentation locally by navigating to the `docs/` folder and running:
```bash
$ cd docs && make html
[...]
The HTML pages are in build/html.
```
This will generate an HTML documentation under `docs/build/html/`. Open the generated `index.html` with a browser and start reading🤓 
