# tinygrad Documentation

The tinygrad documentation is built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## Installation

Install `mkdocs-material` and its dependencies in a separate virtual environment.
It also requires an installation of `tinygrad` to execute the `abstractions.py` notebook.

```sh
python -m venv venv
source venv/bin/activate
pip install mkdocs-material mkdocs-jupyter
pip install -e ../
```

## Development

Serve the documentation with live-reloading on [`localhost:8000`](https://localhost:8000).

```sh
python -m mkdocs serve
```

## Build

Build the documentation so that it can be deployed as a static site. 

```sh
python -m mkdocs build
```
