# slopkernels

Optional AMD RDNA3/RDNA4/CDNA templated UIR kernels for `tinygrad.llm` (`SLOP=0` disables and falls back to pure `tinygrad`).

## API

```python
import slopkernels

k = slopkernels.get("rdna4_gemm_q4_k_f16")  # writes fresh slopkernels/uir/<name>.uir, returns parsed templated UOp
uirs = slopkernels.generate_all()           # generates all static templated .uir files in slopkernels/uir/
```

## CLI & Tests

```bash
DEV=NULL python3 -m slopkernels
DEV="MOCKKFD+AMD::gfx1201" PARALLEL=0 python3 -m pytest slopkernels/test/ -n12
DEV=AMD python3 -m slopkernels.bench --fast --warmup 2 --iters 5
```
