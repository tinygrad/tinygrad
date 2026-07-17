# CONTINUE.md: PAD with Invalid instead of 0

## Goal
Make the low-level `Ops.PAD` pad with `Invalid` instead of `0`, while keeping
the external `Tensor.pad` behavior unchanged.

## Changes made (all 3 files are modified, see `git diff`)

### 1. `tinygrad/schedule/indexing.py:92` — core change
`convert_pad_to_where_to_keep_behavior_local` now uses `UOp.const(x.dtype, Invalid)`
instead of `UOp.const(x.dtype, 0)` as the else value. This is what makes `Ops.PAD`
pad with Invalid.

### 2. `tinygrad/uop/symbolic.py:87-99` — Invalid propagation rules
Added two new rules to `pm_data_invalid` so that `where(invalid_gate, a, const_b)`
uses `b` (the const) in don't-care positions instead of poisoning to Invalid.
This is needed so that `_pad_constant`'s mask `where(pad(ones_bool), base, value)`
works — the mask is a `where(valid, True, Invalid)` gate, and the else `value`
is a const.

The rules are restricted to only match when the gate's valid value is a **const**
(`UPat.cvar("x")`), to distinguish pad masks (where valid=True, a const) from
gather masks (where valid=loaded_data, not a const). Without this restriction,
`test_tensor_index` breaks because gather masks also create `where(cond, x, Invalid)`
but need to keep poisoning.

### 3. `tinygrad/mixin/op.py:280-289` — `_pad_constant` fix
Swapped the `value == 0` early return for `value is Invalid` early return.
When `value is Invalid`, just return `base` (which already has Invalid from
`Ops.PAD`). For all other values (including 0), use the mask approach:
`where(pad(ones_bool), base, const_value)`.

## Current state
- `test/unit/test_invalid_tensor.py` — **all 22 pass**
- `test/unit/test_function.py` — **5 failures**, all multi-shard tests

## The remaining bug: `cat` + multi-shard

`cat` (op.py:716) uses `pad` + `usum` (element-wise ADD) to combine tensors:
```python
padded = [t.pad(...) for i,t in enumerate(tensors)]
return padded[0].usum(*padded[1:])
```

When two shards are cat'd, each is padded and then summed. The valid masks
are **complementary** (shard 0 valid in positions 0-1, shard 1 valid in 2-3).

`_pad_constant` creates `where(mask_pad, data_pad, 0)` where:
- `mask_pad = where(valid, True, Invalid)` — gate's valid value is const `True`
- `data_pad = where(valid, data, Invalid)` — gate's valid value is loaded `data` (NOT const)

The new const-specific rule handles the mask pad correctly. But for the data pad,
the gate's valid value (`data`) is not a const, so the **non-const** lift-out rule
fires: `where(valid, where(valid, data, Invalid), 0)` → `where(valid, where(valid, data, 0), Invalid)`.

The `Invalid` else poisons the ADD. The binary Invalid rule lifts both gates out:
`where(c6, data0, Invalid) + where(c8, data1, Invalid)` → `where(c6&c8, data0+data1, Invalid)`.

Since `c6` and `c8` are complementary, `c6&c8` is always False → result is all Invalid → 0.

### Master comparison
On master, `convert_pad_to_where` uses `0` (not Invalid), so the ADD is just
`where(c6, data0, 0) + where(c8, data1, 0)` with no Invalid, no lifting, works fine.

### Debug output (with changes)
```
c16 = c6.where(c11.index(c13), 0)        # where(c6, load0, 0) — correct
c22 = c6.where(0, c17.index(c20))         # where(c6, 0, load1) — correct
c25 = (c6&c8).where((c16+c22), Invalid)   # WRONG: c6&c8 always False → all Invalid
```

### Master debug output
```
c13 = c6.where(c8.index(c10), 0)          # where(c6, load0, 0)
c21 = c6.where(0, c14.index(c19))          # where(c6, 0, load1)
c22 = c13+c21                              # plain ADD, no wrapper — correct
```

## Suggested fix approaches

### Option A: General WHERE simplification rule
Add a rule: `where(a, where(a, x, _), c)` → `where(a, x, c)`.
When the outer and inner conditions are the same UOp, the inner else is
unreachable. This would simplify `where(valid, where(valid, data, Invalid), 0)`
→ `where(valid, data, 0)` before the lift-out rule can fire.
Check if this rule already exists in `symbolic.py` — it may need to be added
before the lift-out rules.

### Option B: Don't use Ops.PAD for data in `_pad_constant`
When `value is not Invalid`, avoid creating `Ops.PAD` on the data. Use `cat`
or `expand` to create the padded tensor directly, bypassing the Invalid
propagation entirely.

### Option C: Make the lift-out rule use the outer else value
Change the non-const lift-out rule: when `where(a, where(cond, x, Invalid), c)`
and `c` is a const, use `c` as the else instead of `Invalid`. This is what the
const-specific rule does, but it needs to also handle non-const gate valid values.

## Test commands
```bash
# invalid tensor tests (currently pass)
python -m pytest test/unit/test_invalid_tensor.py -x -q -n12

# function tests (5 multi-shard failures)
python -m pytest test/unit/test_function.py -x -q -n12

# the specific failing test
python -m pytest test/unit/test_function.py::TestFunctionMulti::test_simple_multi_sharded -x -q

# debug the failing case
DEBUG=6 python -c "
from tinygrad import Tensor
a = Tensor([1,2,3,4]).shard(['CPU', 'CPU:1'], axis=0)
print(a.numpy())  # should be [1,2,3,4], gets [0,0,0,0]
"
```

## Lint/typecheck
```bash
python -m mypy tinygrad/
python -m ruff check .
```
