# Helpers

`tinygrad/helpers.py` contains utility functions and classes used throughout the library.

## Context Variables
Using `ContextVar`, tinygrad manages global configuration and flags.
- `DEBUG`: Controls logging level.
- `IMAGE`, `WINO`: Control feature flags.
- `JIT`: Controls JIT compilation.

## Utilities
- `prod(x)`: Product of elements in an iterable.
- `flatten(l)`: Flattens nested lists.
- `argsort(x)`: Returns indices that would sort the array.
- `colored(st, color)`: Adds ANSI color codes.
- `diskcache`: Decorator for caching results to disk (sqlite).
- `fetch(url)`: Downloads files from the web with caching.

## Profiling
- `Profiling`: Context manager for Python cProfile.
- `Timing`: Context manager for measuring execution time.

## `GlobalCounters`
Tracks global statistics like total operations, memory usage, and kernel counts.
