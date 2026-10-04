# GPT-OSS HCQ2 scheduling

`HCQ_SCHEDULE=extra.gptoss_kernels.hcq_schedule` enables the measured order;
`HCQ_SCHEDULE=0` uses stock order. Both GPT-OSS launchers enable it by default.
This package only reorders existing physical calls. HCQ2 still builds the
queues, waits, signals and profiling records. There are no runtime imports from
experiment directories, GPU operations here, or per-replay scheduling work.

## Current training recipe

`wide16_routing_scan.plan.json` matches the full 24-layer, DP8, two-update graph
with stable routing maps and direct FC1 gradient stores: 114965 calls and
678247 physical dependencies. Its fingerprint is
`ceab0cb4d7e371efd9e293520ec37134cb682b7497ce22320592a1443a0044a5`.
The three routing phases replace seven integer-map kernels and two checkpoint
copies per layer, removing 2304 calls from the captured graph. All unrelated
source/binary/geometry and argument signatures, every physical dependency and
the old copy-queue FIFO are preserved. The 576-update C/A/C measured
544.447/544.403/545.253 ms/update: the 0.447 ms bracketed saving is smaller than
the 0.806 ms control drift. This is a cleaner routing path and an isolated
compute win, not an established full-step wall win or full convergence result.
Evidence: `.codex_artifacts/routing_production_20260928/README.md`.

## Previous training recipes

`wide16_fc1_direct.plan.json` matches the full 24-layer, DP8, two-update graph
with direct FC1 gradient stores: 117269 calls and 680724 physical dependencies.
Its fingerprint is
`c8a9e631285c48524617a557c719b9cffd08213993cbaeabca69e18b991e9d4d`.
It removes384 placement kernels without changing another compute binary. The
576-update control/candidate/control comparison measured548.786/548.773/550.240
ms/update: a0.740ms bracketed saving, with1.454ms control drift. All metrics were
finite; this recipe has not yet completed a full convergence run. Every new
physical dependency and the old copy-queue FIFO order were preserved.

The preceding `wide16_router_gather.plan.json` matches the pre-direct-store graph:
117653 calls and 681134 physical dependencies. Its fingerprint is
`6987b67551d1a365ad4bf4f1989b15b6c9fa1938936ea05ab27ce0278ac14d9c`.

With router/gather fusion, training-only directional AQL fences and the program
image arena, the bracketed 576-update comparison measured 550.981 ms/update versus
557.297/557.296 ms controls. The combined configuration subsequently converged at
update 12288, eval loss 3.3340, in 118m51.609s from launch. Clean process exit took
119m15.710s. These are combined results, not an isolated scheduler speedup.

The later copy-lowering optimization retained this same graph and permutation;
its native startup comparison saved 9.050s. That change was not part of the full
convergence measurement above.

The recipe's original name and `qualification` label are retained as historical
build metadata. The current qualification evidence is:

- `.codex_artifacts/stack_integration_20260927/result.md`
- `.codex_artifacts/end_to_end_stack_20260927/result.md`
- `.codex_artifacts/copy_lower_native_20260927/result.md`
- `.codex_artifacts/fc1_direct_integration_20260928/README.md`

These files are development evidence, not runtime dependencies.

## Matching and fallback

Each recipe contains an exact permutation bound to complete kernel source and
binary hashes, launch geometry, devices/queues, read/write argument slots,
shapes/dtypes, physical allocation/lane/range identities, and every RAW/WAR/WAW
edge. Generated names and timing estimates do not identify operations.
All actual dependencies are independently checked before applying a recipe.

Different shapes, layer counts or binaries retain the original batch and use
stock order unless another exact recipe matches. Full-size matches/fallbacks
are logged even at `DEBUG=0`; unrelated eager/eval sizes are quiet. In
particular, a two-layer visualization does not use the full 24-layer recipe.

The three older recipes remain for exact older/override configurations:

| Recipe | Calls | Historical qualification |
| --- | ---: | --- |
| `wide16_raw` | 118421 | Paired GPU order test; loaded FC1 image equivalence audited |
| `wide16_packed` | 118037 | CPU-only candidate; no combined performance claim |
| `wide16_packed_score_fma` | 118037 | Finite 577-update C/A/C; 555.805→554.621 ms bracketed |

They cannot match the current114965-call graph. Their historical timings are
not additive to the current combined result.

## Building a recipe

Capture the actual post-allocation batch with `capture(batch)` to get
`rows, programs`. Derive and validate a permutation offline, then run:

```sh
python -m extra.gptoss_kernels.hcq_schedule.build \
  --graph graph.json --programs programs.json --order order.json --name candidate_name
```

The command prints a compressed, self-contained recipe marked `cpu_candidate`.
It checks physical dependencies, not numerical correctness or performance.
GPU qualification must follow before promotion. Never replace source/binary
hashes merely to force a match; changed programs need their own validation.
