#!/bin/bash -e
rm -rf dist
ipython3 -m build . --wheel --sdist
twine upload dist/*

