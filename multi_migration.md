# Multi-device op migration: MULTI/MSELECT/MSTACK → PAD / WHERE / STACK+INDEX

## Status (updated)

- **Stage 0 — DONE.** Internal `Ops.PAD` fills **Invalid** (`schedule/indexing.py:104`, bool keeps 0-fill); external `Tensor.pad`/`pad_to` always emit an explicit fill mask (`mixin/op.py:289`, `mixin/movement.py:267`) — required because a bare Invalid-pad leaks through elementwise ALU (`pad(x)+1` would read 0 instead of 1 in pad regions). REDUCE inputs with Invalid contribute the reduce identity (`pm_invalid_reduce_identity` in `uop/symbolic.py`, run in `get_kernel_graph` after gate lifting in `schedule/rangeify.py`) — only WHERE-alt gates whose condition involves a reduce range are rewritten, so gather-with-Invalid-index still poisons whole lanes. Same-condition nested where collapse rule added (`c?(c?t:f):f2 -> c?t:f2`) so the mask form folds to a single gate. All suites green (`test/unit`, `test/null`, `test/backend`, `test/external/external_test_schedule_scaling.py`, mypy, ruff).
- **Stage 1 — representation in place behind `SYMBOLIC_MULTI`.** `symbolic_multi_pm` (`schedule/multi.py`) converts `MULTI→_unshard` (raw Invalid pad), `MSELECT→dnum.eq(i).where(x, Invalid)`, `MSTACK→STACK.index(dnum)` + INDEX(STACK,var)→nested-where lowering. `_unshard` uses the raw Invalid pad; `_unshard_fill` (0-fill) is used for the ALU allreduce in `copy_multi` because gated stores leave stale pad regions (the ALU-sum path can't use the identity rule). Basic shard ops work; full parity is Stage 2.
- **Stage 2 — remaining.** Buffer level, reduce/allreduce split for shard-axis reduces, API surface.
- **Stage 3 — remaining.**

Notes: `test_schedule.py:test_pad_reduce_unsafe_multiview_st` went 4→5 kernels (pad now materializes an explicit mask; the mask form is also what makes the previously-wrong masked-pad+hazard case correct). `test_jit_footguns.py:test_symbolic_pad_view_frozen` went 2→4: the explicit mask recomputes from the symbolic shape, fixing the frozen-pad footgun. Also fixed a latent infinite loop: `(x+y) !=/< c → x !=/< c-y` collapse rules in `codegen/simplify.py` now only fire when the remaining side still contains the range (they previously shuffled constants forever when both sides were range-free).

## Goal

Replace the three multi-device UOps with a symbolic `_device_num` representation:

| Old op | New form |
|---|---|
| `MULTI(x, axis)` | `x._unshard(axis)` — PAD with `_device_num`-dependent bounds back to full shape (helper already exists at `tinygrad/uop/ops.py:704-707`) |
| `MSELECT(x, i)` | `dnum.eq(i).where(x, x.const_like(Invalid))` |
| `MSTACK(s0..sn)` | `UOp(Ops.STACK, src=srcs).index(dnum)` — leading device axis, indexed per-device |

where `dnum = UOp.variable("_device_num", 0, ndev-1)`. The per-device specialization mechanism already exists: `unwrap_multi` (`tinygrad/engine/realize.py:148-153`) binds `_device_num` per device at exec time.

**Key semantic decision (approved):** internal `Ops.PAD` produces **Invalid** in padded regions; external `Tensor.pad` API still pads with 0. Staged migration: introduce the new representation first, keep old ops working, migrate call sites incrementally, delete old ops last.

## Background: current design

- `Ops.MULTI(src, axis)` (`tinygrad/uop/__init__.py:100`) — per-shard graph marker. Eliminated by `multi_pm` (`tinygrad/schedule/multi.py:162-195`) as the first step of `get_kernel_graph` (`tinygrad/schedule/rangeify.py:548`). Shape/axis tracking: `UOp.axis`/`UOp.bounds` (`tinygrad/uop/ops.py:667-702`).
- `Ops.MSELECT(x, i)` / `Ops.MSTACK(srcs)` (`__init__.py:96`) — buffer-level ops. Spec at `tinygrad/uop/spec.py:181-184`; device prop `ops.py:816-819`; per-kernel PARAMs via debuf (`rangeify.py:474`); per-device dependency states (`tinygrad/schedule/__init__.py:11-17`); `MultiBuffer` (`ops.py:904-930`, `tinygrad/device.py:88-99`); only MSTACK can be `realized` (`ops.py:920-930`).
- `_shard`/`_unshard` (`ops.py:704-714`) already emit symbolic SHRINK/PAD bounds with `_device_num`.
- Naive allreduce already uses the target pattern: `dnum.eq(i).where(buf, state)` (`tinygrad/schedule/allreduce.py:27-33`).

## Existing Invalid machinery (rely on this)

- `pm_data_invalid` (`tinygrad/uop/symbolic.py:71-92`): Invalid poisons ALU (ops move inside the gate); gated LOAD folds to alt/0, gated STORE folds to NOOP.
- `pm_remove_invalid` (`symbolic.py:94-96`): leftover Invalid → 0 in final codegen (`codegen/__init__.py:345`). Spec forbids Invalid in final programs (`spec.py:217`), so materialized Invalid regions read as 0.
- STORE of CONST(Invalid) → NOOP (`rangeify.py:423-424`).
- `identity_element(op, dtype)` exists (`ops.py:51`): ADD→0, MUL→1, MAX→dtype.min.
- `found_after` (`rangeify.py:26`) already matches `WHERE(cond, PAD(x), Invalid)`.

## Stage 0 — internal PAD = Invalid; external pad = explicit 0

1. `tinygrad/schedule/indexing.py:100-104` (`convert_pad_to_where_to_keep_behavior_local`): fill value `0` → `UOp.const(x.dtype, Invalid)`, **except `dtypes.bool` keeps 0-fill** (False is the bool-reduce identity, and the external-pad mask below needs it).
2. `tinygrad/mixin/op.py:282-290` (`_pad_constant`): **remove the `if value == 0: return base` shortcut** — always emit `pad(bool_ones).where(base, value)`. Required because bare Invalid-pad leaks through elementwise ALU: `pad(x)+1` gate-lifts to `where(valid, x+1, Invalid)` and reads 0 instead of 1 in pad regions. The mask lowers to a pure index expression (`valid.where(1,0)`), no extra kernel. External behavior unchanged for all `value`.
3. **New rule**: `REDUCE(where(c, x, Invalid), op)` → `REDUCE(where(c, x, identity_element(op, dtype)), op)`. Must fire in rangeify/symbolic *before* codegen builds the accumulator loop — otherwise `pm_data_invalid` gate-lifts `acc + where(c,x,Invalid)` into `where(c, acc+x, Invalid)` and one invalid lane poisons the whole reduction. Placement (symbolic.py vs the reduce path in indexing.py) TBD at implementation; verify with `Tensor.pad(...).sum()/max()` tests.
4. Audit: schedule tests with kernel counts involving pads; circular/reflect/replicate pads don't use PAD fill (verified, `op.py:292-312`) — unaffected; `allreduce.py:59,76` usum-of-padded-chunks gets *more* correct (disjoint regions).

## Stage 1 — new representation behind env flag

New `symbolic_multi_pm` PatternMatcher (in `schedule/multi.py` or new file), gated by env (e.g. `SYMBOLIC_MULTI`), run in `get_kernel_graph` right after `multi_pm`:

- `MULTI(x, axis)` → `x._unshard(axis)`
- `MSELECT(x, i)` → `dnum.eq(i).where(x, x.const_like(Invalid))` (Invalid from `tinygrad.dtype`)
- `MSTACK(srcs)` → `STACK(*srcs).index(dnum)`, plus new lowering `INDEX(STACK(vals), var)` → nested `var.eq(k).where(src_k, Invalid)` (analogous to `convert_stack_to_where`, `indexing.py:113-121`; must fire before `validate_index` spec, `spec.py:118-122`)

Flag off = zero behavior change; flag on = new forms flow through rangeify and specialize per device at exec.

## Stage 2 — migrate producers/consumers (one commit each, independently testable)

1. `UOp.shard` (`ops.py:715-717`): emit symbolic `_shard`+`_unshard` full-shape form directly instead of `.multi(axis)`; delete movement-op `multi_pm` rules that PAD subsumes (`pad_multi`, `permute_multi`, `expand_multi`, `reshape_multi`, `flip_multi`, `shrink_multi` — `multi.py:93-125`).
2. ALU/STACK: `alu_multi`/`shard_srcs`/`stack_multi` (`multi.py:55-78,127-131`) become plain elementwise on full-shape padded tensors. `reduce_multi` (`multi.py:80-91`) keeps the shard-axis → local-reduce + ALLREDUCE split; Invalid-pad + identity rule replaces neutral-pad-value reasoning.
3. allreduce (`schedule/allreduce.py`): naive path already matches; migrate ring/all2all MSELECT/MSTACK scratch-buffer assembly (lines 35-76) to WHERE/STACK+INDEX forms.
4. Buffer level: debuf (`rangeify.py:474`), `_states`/`_unwrap_src` (`schedule/__init__.py:11-17`), `_collect_bufs` (`schedule/memory.py:9`), `unwrap_multi` (`realize.py:148-153`), JIT (`jit.py:127-130, 237`), callify (`callify.py:52-95`), `buffer`/`realized`/`buf_uop`/`has_buffer_identity` (`ops.py:841-930`).
5. API surface: `UOp.multi/mselect/mstack` (`ops.py:662-725`), `Tensor.shard` (`tensor.py:333-347`), gradient (`mixin/gradient.py:72`), `_multi_like` (`mixin/creation.py:16-20`), embedding backward (`nn/__init__.py:309-354`), `copy_to_device(arg=)` MSELECT path (`ops.py:719-723`).

## Stage 3 — removal

Delete `Ops.MULTI/MSELECT/MSTACK` from the enum (`uop/__init__.py:96,100`), spec rules, viz colors (`viz/serve.py:51,56`), `UOp.axis`/`bounds` machinery (`ops.py:667-702`), remaining `multi_pm` rules, and `MultiBuffer` if fully subsumed. Flip flag default-on, then delete the flag.

## Open implementation details

- REDUCE-identity rule placement (must precede codegen accumulator construction).
- INDEX(STACK, var) spec timing — the value-STACK INDEX violates the pointer-INDEX spec until lowered.
- Whether `MultiBuffer`/tuple-`device` survives as the runtime container, or buffers become single-device with the device axis explicit in shape — decides how much of Stage 2.4 is rewrite vs delete.
- Bool carve-out in Stage 0.1: verify no internal consumer needs Invalid-filled bool pads.

## Verification (run at each stage)

```bash
python -m pytest test/unit/test_multitensor.py test/unit/test_allreduce.py test/null/test_multitensor.py test/unit/test_call.py -x -q -n12
python -m pytest test/external/external_test_schedule_scaling.py -x -q   # test_concat_scaling
python -m mypy tinygrad/
python -m ruff check .
```

Also pad/reduce numeric tests after Stage 0 (`test_ops` pad tests, `Tensor.pad(...).sum()/max()`).
