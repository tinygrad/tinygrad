#!/bin/bash -e
rm -rf dist
python -m build
twine upload dist/*

