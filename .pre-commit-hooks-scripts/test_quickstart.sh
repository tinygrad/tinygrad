#!/bin/bash -e
awk '/```python/{flag=1;next}/```/{flag=0}flag' docs/quickstart.md > quickstart.py &&  PYTHONPATH=. python3 quickstart.py && rm quickstart.py