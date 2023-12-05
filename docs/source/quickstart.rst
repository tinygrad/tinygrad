Installation
============

The current recommended way to install tinygrad is from source.

1. Clone the repository

.. code-block:: bash

    git clone https://github.com/tinygrad/tinygrad.git
    cd tinygrad
    
2. Create a python virtual environment

.. code-block:: bash
    
    python3 -m venv <env_name>
    source <env_name>/bin/activate
    pip install --upgrade pip

3. Install tinygrad from source

.. code-block:: bash

    python3 -m pip install -e .

Contributing
=============

There has been a lot of interest in Tinygrad lately. Here are some basic guidelines for contributing:

- Bug fixes are highly valued and always welcome! Like `this one <https://github.com/tinygrad/tinygrad/pull/421/files>`_.
- If you don't understand the code you are changing, it's advised not to modify it.
- Code golf pull requests will be closed, but `conceptual cleanups <https://github.com/tinygrad/tinygrad/pull/372/files>`_ are appreciated.
- Features are welcome, but if you are adding a feature, ensure that you include tests.
- Improving test coverage is encouraged, with a focus on creating reliable and non-brittle tests.

Additional guidelines can be found in `CONTRIBUTING.md <https://github.com/tinygrad/tinygrad/blob/master/CONTRIBUTING.md>`_.

Running Tests
-------------

For more examples on how to run the full test suite, please refer to the `CI workflow <https://github.com/tinygrad/tinygrad/blob/master/.github/workflows/test.yml>`_.

Some examples:

.. code-block:: bash

    python3 -m pip install -e '.[testing]'
    python3 -m pytest
    python3 -m pytest -v -k TestTrain
    python3 ./test/models/test_train.py TestTrain.test_efficientnet