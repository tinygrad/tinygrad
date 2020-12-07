SHELL := /bin/bash

.PHONY: docs docs-serve

## Build documentation site
docs:
	make -C ./docs docs

## Serve the documentation site localy
docs-serve:
	make -C ./docs docs-serve
