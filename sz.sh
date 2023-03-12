#!/bin/bash
# switched to cloc due to https://github.com/boyter/scc/issues/379
cloc --by-file tinygrad/* | grep "tinygrad"
# also some sloccount for a dir summary
sloccount tinygrad | grep "python"
