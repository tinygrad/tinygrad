VIZ is a tool for inspecting tinygrad's rewrites and runtime profiling.

to use:
1. Run tinygrad with VIZ=1 (this saves the pkls and launches the server in interactive shells)
2. That's it!

# VIZ in the command line

Use `python -m tinygrad.viz.cli` (add --json for scripting) to view the full timeline of events.

### Environment variables

Setting DEBUG includes the following data in the output stream. These show up as raw `{"value": "..."}` lines when using `--json`.

| DEBUG | Includes |
|------:|----------|
| 3 | Base AST |
| 4 | Generated source |
| 5 | Rewrite steps and kernel graph |
| 6 | All UOp graphs |
| 7 | All rewrites |


VIZ defaults to colored output. Set `NO_COLOR=1` to disable colors.

### Profiling Examples

Get kernel times and ASTs

```bash
DEBUG=3 python -m tinygrad.viz.cli --json > /tmp/events.jsonl
```

Select kernel times after a marker

Note: markers are set by the `profile_marker` helper in the user code, to find them:
```
python -m tinygrad.viz.cli | rg MARKER
```
Then:
```
python -m tinygrad.viz.cli --interval "train @ 2" "train @ 3"
```

Set `-t` to aggregate events.

### Rewrites debugging example

First, find the rewrite you are looking for. This can be a schedule or kernel:

```bash
python -m tinygrad.viz.cli -s TINY | rg Schedule
python -m tinygrad.viz.cli -s TINY | rg E_3
```

List all rewrite passes:

Note: the names come from `graph_rewrite(..., name="...")` in user code.
```bash
python -m tinygrad.viz.cli -s TINY "Schedule 6 Kernels n1" --ls
```

Get input graph to all passes

```bash
DEBUG=6 python -m tinygrad.viz.cli -s TINY "Schedule 6 Kernels n1"
```

Get all rewrites

```bash
# for the entire scheduler
DEBUG=7 python -m tinygrad.viz.cli -s TINY "Schedule 6 Kernels n1"

# or for a specific pass
DEBUG=7 python -m tinygrad.viz.cli -s TINY "Schedule 6 Kernels n1" "earliest rewrites"
```

# SQTT / PMC profiling (DEV=AMD only)

SQTT has additional overhead. Set VIZ=2 to include it in pkls.

Examples:

Get all SQTT packets
```bash
python -m tinygrad.viz.cli -s "kernel SQTT SE:0 PKTS" --json
```

Get bank conflicts:
```bash
python -m tinygrad.viz.cli -s "gemm PMC" | rg -A 16 SQC_LDS_BANK_CONFLICT
```
