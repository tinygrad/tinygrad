# Viz

The `tinygrad/viz/` directory contains tools for visualizing the execution graph and profiling data.

## `serve.py`

Starts a local web server to serve the visualization interface.
It reads profiling data (pickled `RewriteTrace` or `ExecItem` list) and serves it to the frontend.

## `index.html`, `js/`, `assets/`

The frontend code for the visualizer. It displays the graph rewrite steps, kernel execution timelines, and other debugging info.
