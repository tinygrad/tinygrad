#!/usr/bin/env python3
"""Cheap preflight for an official moonshotai/Kimi-K3 checkout. Does not load model weights."""
import argparse, json, pathlib, shutil
from tinygrad.llm.kimi_k3 import KIMI_K3_TP8_BYTES_PER_GPU, audit_kimi_k3_checkpoint

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("model_dir", type=pathlib.Path)
  parser.add_argument("--metadata-only", action="store_true", help="permit absent weight shards")
  parser.add_argument("--context", type=int, default=4096, help="context length used for the memory estimate")
  args = parser.parse_args()
  stats = audit_kimi_k3_checkpoint(args.model_dir, require_shards=not args.metadata_only)
  if not 1 <= args.context <= 1_048_576: raise ValueError("--context must be between 1 and 1048576")

  # K3 has 24 MLA layers. Each token stores the 512-value compressed latent plus 64 RoPE values in BF16.
  per_gpu_weights = KIMI_K3_TP8_BYTES_PER_GPU
  mla_cache = 24 * args.context * (512 + 64) * 2
  hbm = 288_000_000_000
  print(json.dumps(stats, indent=2))
  print(f"exact text weights/GPU under this TP8 layout: {per_gpu_weights/1e9:.2f} GB ({per_gpu_weights/2**30:.2f} GiB)")
  print(f"replicated MLA cache/GPU at {args.context:,} tokens: {mla_cache/1e9:.2f} GB ({mla_cache/2**30:.2f} GiB)")
  print(f"nominal MI350X headroom before runtime buffers: {(hbm-per_gpu_weights-mla_cache)/1e9:.2f} GB")
  if not args.metadata_only:
    usage = shutil.disk_usage(args.model_dir)
    print(f"filesystem free space: {usage.free/1e9:.2f} GB")

if __name__ == "__main__": main()
