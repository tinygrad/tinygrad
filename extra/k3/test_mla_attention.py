#!/usr/bin/env python3
# DEV=AMD python3 extra/k3/test_mla_attention.py: amd.mla_decode_attention (flash decoding on the mla latent cache) against numpy
import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.llm.kernels.amd import mla_decode_attention

rng = np.random.default_rng(0)
H, D, DV, N, SCALE = 24, 576, 512, 131072, 192 ** -0.5
cache = rng.standard_normal((1, 1, N, D)).astype(np.float16)  # (the model's cache is half)
t_cache = Tensor(cache).realize()
cache = cache.astype(np.float32)
for S, T, masked in ((200, 1, False), (201, 2, False), (3000, 2, True), (3999, 4, False), (4096, 1, True), (24000, 2, False), (24577, 1, True),
                     (131072, 2, False), (131000, 4, True)):
  q = rng.standard_normal((1, H, T, D)).astype(np.float32)
  s = (q @ cache[0, 0, :S].T) * SCALE  # (1, H, T, S)
  if masked: mask = np.where(np.arange(S) % 5 == 1, -np.inf, 0).astype(np.float32) * np.ones((1, 1, T, 1), np.float32)  # (keys dropped)
  else: mask = np.triu(np.full((T, S), -np.inf, np.float32), S - T + 1)[None, None]
  p = np.exp(s + mask - (s + mask).max(-1, keepdims=True))
  want = (p / p.sum(-1, keepdims=True)) @ cache[0, 0, :S, :DV]
  valid = UOp.variable("valid", 1, N).bind(S)
  got = mla_decode_attention(Tensor(q), t_cache, valid, DV, SCALE, Tensor(mask) if masked else None).numpy()
  np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
  print(f"{S} keys, {T} tokens{', masked' if masked else ''}: ok, max |d| {np.abs(got - want).max():.2e}")

# amd.mla_prefill_attention: a chunk of T tokens at start_pos (causal, the cache holding its keys), 16 heads, against numpy; start_pos
# a bound variable in a jit (two calls of other values), else an int
from tinygrad import TinyJit
from tinygrad.llm.kernels.amd import mla_prefill_attention
H, SCALE = 16, D ** -0.5
jits:dict[int, TinyJit] = {}
for T, S, sym in ((1, 0, False), (7, 100, False), (64, 0, True), (64, 31, True), (300, 4096, True), (512, 3000, False), (100, 131072 - 100, False)):
  q = rng.standard_normal((1, H, T, D)).astype(np.float16)
  s = (q.astype(np.float32) @ cache[0, 0, :S + T].T) * SCALE
  mask = np.triu(np.full((T, S + T), -np.inf, np.float32), S + 1)
  p = np.exp(s + mask - (s + mask).max(-1, keepdims=True))
  want = (p / p.sum(-1, keepdims=True)) @ cache[0, 0, :S + T, :DV]
  if sym:
    jit = jits.setdefault(T, TinyJit(lambda q, sp: mla_prefill_attention(q, t_cache, sp, DV, SCALE).realize()))
    got = jit(Tensor(q), UOp.variable("start_pos", 0, N - 512).bind(S)).numpy()
  else: got = mla_prefill_attention(Tensor(q), t_cache, S, DV, SCALE).numpy()
  np.testing.assert_allclose(got, want, rtol=1e-3, atol=2e-3)
  print(f"prefill {T} tokens at {S}{' (jit)' if sym else ''}: ok, max |d| {np.abs(got - want).max():.2e}")
