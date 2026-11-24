# slopkernels

Optional universal AMD RDNA3/RDNA4/CDNA kernel library for `tinygrad`, exported **strictly as UIR text**.

> **The name `slopkernels` is ironic.** If you are a coding agent working on `slopkernels`, the code needs to be **beautiful**: clean, minimal, readable, idiomatic, deduplicated, and free of runtime logic. Every kernel here should be a universal, self-contained `UOp -> UIR` building block — no model pipelines, no `Tensor` wrappers, no `Device` calls, no `getenv`, no mutable global state.

## Usage

`slopkernels/` lives at the root of the `tinygrad` repository for source checkouts (like `extra/` — not packaged in `pyproject.toml`). `tinygrad` automatically discovers `slopkernels` when present and falls back cleanly to pure `tinygrad` when absent or disabled via `SLOP=0`.

## Narrow Public API

The entire interface between `slopkernels` and `tinygrad` is three pure functions (`__all__ = ["compile_all", "get", "list_kernels"]`):

```python
import slopkernels

# 1. List all registered kernels & logical ops
print(slopkernels.list_kernels())

# 2. Compile a kernel or logical op spec into pure tinygrad UIR text (or None if unsupported)
uir = slopkernels.get("f16_gemv", in_features=128, out_features=64, tokens=1)

# 3. Compile every registered kernel offline to .uir files (zero GPU required)
uirs = slopkernels.compile_all("out_uir")
```

### Pure Logical-Op Contract (`tinygrad.llm` ↔ `slopkernels`)

Every argument crossing `tinygrad` → `slopkernels.get(...)` is strictly a **pure, hashable Python primitive** (`None`, `bool`, `int`, `float`, `str`, or nested `tuple`) — zero `Tensor`, `UOp`, `Device`, or `DType` objects.

All 6 logical ops take `target: tuple[int, ...] | str`, `device: str | tuple[str, ...] = "AMD"`, `srcs: tuple[SrcDesc, ...]`, plus only true semantic kwargs. Each entry of `srcs` corresponds 1-to-1 with the positional arguments passed to `parse_uir(uir).call_with_output(*srcs)`:

- **Tensor input descriptor**: `(shape: tuple[int, ...], shard_shape: tuple[int, ...], axis: int | None, dtype: str)`
- **Symbolic scalar (`Variable`) descriptor**: `(vmin: int, vmax: int)`

| Op | Positional `srcs` | Semantic kwargs | Output |
|---|---|---|---|
| `dense_linear` | `(weight, x)` or `(weight, x, bias)` | *(none)* | `(*x.shape[:-1], out_features)` `f32` |
| `quant_linear` | `(packed_weight, x)` | `ggml_type: int`, `in_features: int`, `out_features: int`, `groups: int = 1`, `w_axis: int \| None = None` | `(*x.shape[:-1], out_features // groups)` `f32` |
| `moe_linear` | `(packed_weight, sel_flat_i32, x)` | `ggml_type: int`, `sel_shape: tuple[int, int, int]`, `in_features: int`, `local_out: int`, `global_out: int`, `w_axis: int \| None = None`, `ndev: int = 1` | `([ndev,] B, T, slots, global_out)` `f32` |
| `flash_attention_op` | `(q_f16, kv_f16[, valid_kv_sym][, q_len_sym])` | `decode: bool = False`, `valid_kv_len: int \| None = None`, `valid_kv_offset: int = 0`, `q_start_from_end: bool = False` | `q.shape` `f32` |
| `hc_coeffs_op` | `(mix_f32, flat_f32, base, scale)` | `count: int`, `iters: int`, `eps: float`, `norm_eps: float` | `(tokens, count * (2 + count))` `f32` |
| `gated_delta_prefill_op` | `(q, k, v, beta, alpha, state[, start_pos_sym])` | `start_pos: int \| None = None` | `v.shape` `v.dtype` (updates `state` in place) |

## Usage Examples

### Parse and execute a logical op UIR in `tinygrad`

```python
import numpy as np
import slopkernels
from tinygrad import Tensor
from tinygrad.uop.uir import parse_uir

w = Tensor(np.random.randn(64, 128).astype(np.float16)).realize()
x = Tensor(np.random.randn(2, 128).astype(np.float32)).realize()
srcs = (((64, 128), (64, 128), None, "f16"), ((2, 128), (2, 128), None, "f32"))

uir = slopkernels.get("dense_linear", target=(12, 0, 0), device=w.device, srcs=srcs)
y = Tensor(parse_uir(uir).call_with_output(w.uop, x.uop))
print(y.numpy().shape)  # (2, 64)
```

### Compile all kernels offline to `.uir` files (no GPU required)

```bash
DEV=NULL python3 -m slopkernels /tmp/slopkernels_uir
ls /tmp/slopkernels_uir/*.uir
```

### Benchmark on AMD GPU (cold-DRAM with 256 MB Infinity Cache flush)

```bash
DEV=AMD python3 -m slopkernels.bench --warmup 2 --iters 5
```

### Run tests

```bash
# Offline UIR compilation + strict API / purity boundary checks (no GPU needed)
DEV=NULL python3 -m pytest slopkernels/test/test_slopkernels.py -k "boundary or compile_all" -v

# Full numerical verification on AMD emulator (or real GPU with DEV=AMD)
DEV="MOCK+AMD" python3 -m pytest slopkernels/test/ -n12
```

## Layout

- `__init__.py` — narrow public API (`list_kernels`, `get`, `compile_all`) and kernel registry
- `__main__.py` — offline CLI compiler (`python3 -m slopkernels [out_dir]`)
- `bench.py` — cold-DRAM GPU LLM op benchmark suite (runs in CI on `7900xtx` & `9060xt`)
- `src/` — pure `UOp` kernel and logical-op builders (`amd.py`, `amd_gguf.py`, `extra.py`, `ops.py`)
- `test/` — purity/boundary, offline UIR compilation, 2-GPU TP, and AMD numerical tests
