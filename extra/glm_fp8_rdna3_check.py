#!/usr/bin/env python3
"""RDNA3 SEQ_FP8 building blocks: fp16 versus e4m3 indexer and MLA outputs (no model load).
Run on one 7900 XTX: DEV=AMD PYTHONPATH=. python3 extra/glm_fp8_rdna3_check.py
"""
import functools
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.amd_glm import amd_flash_attention_mla_sparse, fp8_rows, indexer_scores, _indexer_scores_decode_kernel
from tinygrad.uop.ops import UOp

rng = np.random.default_rng(23)


def fp8_index_rows(k:Tensor) -> Tensor:
  rows, dims = k.shape
  assert dims == 128
  scale = k.float().abs().max(-1, keepdim=True).maximum(1e-20) / 448
  q = (k.float() / scale).cast(dtypes.fp8e4m3).bitcast(dtypes.uint8).reshape(rows, dims).contiguous().bitcast(dtypes.uint32)
  return q.cat(scale.bitcast(dtypes.uint32), dim=-1).contiguous()


def compare(name:str, ref:Tensor, got:Tensor) -> None:
  a, b = ref.numpy(), got.numpy()
  assert np.isfinite(b).all(), f"{name}: non-finite output"
  scale = max(float(np.sqrt(np.mean(a*a))), 1e-12)
  max_rel = float(np.abs(a-b).max() / max(float(np.abs(a).max()), 1e-12))
  rmse = float(np.sqrt(np.mean((a-b)**2)) / scale)
  print(f"{name}: max|diff|/max|ref|={max_rel:.5f}, RMSE/RMSF={rmse:.5f}", flush=True)
  assert max_rel < 0.25 and rmse < 0.10, f"{name}: FP8 layout/path error, not ordinary quantization: {max_rel=}, {rmse=}"


def main() -> None:
  T, H, D, P = 16, 4, 128, 64
  q = Tensor((rng.standard_normal((T,H,D), dtype=np.float32)*0.1).astype(np.float16), device="AMD").realize()
  k = Tensor((rng.standard_normal((P,D), dtype=np.float32)*0.1).astype(np.float16), device="AMD").realize()
  w = Tensor((rng.standard_normal((T,H), dtype=np.float32)*0.01).astype(np.float32), device="AMD").realize()
  k8 = fp8_index_rows(k).realize()
  compare("indexer prefill", indexer_scores(q,k,w,P), indexer_scores(q,k8,w,P))

  def decode(keys:Tensor) -> Tensor:
    out = Tensor.empty(T,P,dtype=dtypes.float32,device="AMD")
    fxn = functools.partial(_indexer_scores_decode_kernel, pools=UOp.const(P,dtypes.weakint), rdna4=False)
    return Tensor.custom_kernel(out,q,k if keys is k else k8,w,fxn=fxn)[0]
  compare("indexer decode", decode(k), decode(k8))

  N = 64
  mq = Tensor((rng.standard_normal((1,16,1,512), dtype=np.float32)*0.1).astype(np.float16), device="AMD").realize()
  cache = Tensor((rng.standard_normal((N,512), dtype=np.float32)*0.1).astype(np.float16), device="AMD").realize()
  index = np.arange(N+1,dtype=np.int32).reshape(1,N+1)
  index[0,:N] = rng.permutation(N)
  index[0,N] = N
  ti = Tensor(index,device="AMD").realize()
  compare("MLA sparse", amd_flash_attention_mla_sparse(mq,cache,ti,512**-0.5,N),
          amd_flash_attention_mla_sparse(mq,fp8_rows(cache).contiguous(),ti,512**-0.5,N))
  print("RDNA3 FP8 CHECK PASS", flush=True)


if __name__ == "__main__": main()
