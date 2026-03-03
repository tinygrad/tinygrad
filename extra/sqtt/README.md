# SQTT Profiling

## Getting SQ Thread Trace

`VIZ=2` to enable SQTT profiling.

`SQTT_BUFFER_SIZE=X` to change size of SQTT buffer (per shader engine, 6 SEs on 7900xtx) in megabytes, default 256.

`SQTT_ITRACE_SE_MASK=X` to select for which shader engines instruction tracing will be enabled, -1 is all, 0 is none (instruction tracing disabled), >0 is
bitfield/mask for SEs to enable instruction tracing on, default 0b11 (first two shader engines).

## Viewing the traces

- Web UI: `tinygrad/viz/serve.py`
- Command line: `python -m tinygrad.renderer.amd.sqtt`
