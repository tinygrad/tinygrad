#!/bin/bash
find tinygrad -type f -name "*.py" -exec sed -i '' 's/ *$//' '{}' ';'
