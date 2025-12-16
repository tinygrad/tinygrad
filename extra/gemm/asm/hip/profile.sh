PYTHONPATH=. rocprofv3 \
  --kernel-trace \
  --stats \
  --output-format csv \
  --output-directory /tmp/rocprof \
  -- python extra/gemm/asm/hip/test.py
