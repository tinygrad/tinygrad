#!/usr/bin/env bash

source venv/bin/activate && \
PYTHONPATH=. WEBGPU=1 python examples/compile_efficientnet.py && \
PYTHONPATH=. python examples/tinychat/tinychat-browser/compile.py && \
pytest test/web/test_browser_apps.py
