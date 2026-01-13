# WMMA GEMM Kernels

**Test after every small change** to `wmma_uop_helpers.py`:

```bash
PYTHONPATH=. CUDA=1 UOPS=1 DTYPE_IN=half DTYPE_OUT=float DTYPE_ACC=float GEMM_VARIATION=flat_smem_input INPUT=RAND CNT=1024 python3 ./extra/gemm/max_matmul.py
PYTHONPATH=. CUDA=1 UOPS=1 DTYPE_IN=half DTYPE_OUT=float DTYPE_ACC=float GEMM_VARIATION=max INPUT=IDENTITY CNT=1024 python3 ./extra/gemm/max_matmul.py
PYTHONPATH=. CUDA=1 UOPS=1 DTYPE_IN=half DTYPE_OUT=half DTYPE_ACC=half GEMM_VARIATION=max INPUT=RAND ATOL=5 CNT=1024 python3 ./extra/gemm/max_matmul.py
```

Variants: `flat_smem_input` (1-stage), `max` with fp32 acc (2-stage), `max` with fp16 acc (3-stage).
