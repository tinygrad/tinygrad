A command line tool for exploring the VIZ trace.

# Lightweight tracing

Supported on all backends.

Flags: VIZ=-1 to only save the trace to a file, VIZ=1 also launches a web server.

1. Set VIZ to -1 to save the trace.
2. Use `extra/viz/cli.py` to inspect the trace files. Set NO_COLOR=1 to disable colored output.

## Inspect runtime profiling

Use `extra/viz/cli.py --profile` to list all sources.

List top slowest kernels on a source: `--profile -s "AMD"`
List samples of a kernel on a source: `--profile -s "AMD" -i E_3 | head 4`

## Inspect codegen and PatternMatcher

Use `extra/viz/cli.py --rewrites` to list all sources.

List all codegen steps for a kernel: `--rewrites -s E_3`
Get source code: `--rewrites -s E_3 -i "View Source"`
Inspect a graph rewrite: `--rewrites -s E_3 -i "initial symbolic"`

## SQTT tracing

Supported on AMD for RDNA3 and RDNA4 (best) and CDNA (developing).

Flags: VIZ=-2 to save SQTT trace to a file. VIZ=2 also launches a web server. View other flags in tinygrad/runtime/ops_amd.py to configure SQTT as needed.

Use `extra/viz/cli.py --profile | grep SQTT` to view all available SQTT traces.
You can select a specific trace with --source, Example workflow:

```bash
# Run amd_asm_matmul with VIZ=-2 to capture the trace
VIZ=-2 python extra/gemm/amd_asm_matmul.py

# View barriers
extra/viz/cli.py --profile -s "kernel SQTT SE:0 PKTS" | rg BARRIER | head -10

# Get bank conflicts from performance counters

python extra/viz/cli.py -p -s "kernel PMC" -i "SQC_LDS_BANK_CONFLICT"

# Find the EXEC corresponding to a DISPATCH at cycle 410
extra/viz/cli.py --profile -s "kernel SQTT SE:0 PKTS" | awk '/EXEC/ && $1 - $5 == 410'
```
