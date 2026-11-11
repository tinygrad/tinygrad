#!/usr/bin/env python3
"""SEQ_IDX_SHARD exactness: the indexer's lists from pooled keys split over the devices (pool_index_shard) against the whole keys on every
device (pool_index_rows), same random fp8 keys by pool. decode rows (indexer_decode_index_rows) and a prefill block (indexer_scores) at a
few positions, with ties (TIES=1: some keys zero, their scores 0). GPU box only, all devices:
  DEV=AMD PYTHONPATH=. python3 extra/glm_idxshard_check.py
"""
import os
import numpy as np
from tinygrad import Tensor, Device, UOp
from tinygrad.llm.kernels import amd_glm
from tinygrad.llm.kernels.amd_glm import SEQ_BLK, indexer_decode_index_rows, indexer_scores, pool_index_rows, pool_index_shard

N, H, D, KP, TOP = int(os.environ.get("NDEV", 6)), 32, 128, 4, 512
PT = int(os.environ.get("PT", 66048))  # a region's pools (264,192 rows / 4)
devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(N))
rng = np.random.default_rng(int(os.environ.get("SEED", 0)))
key = (rng.integers(0, 0x7F, size=(PT, 128), dtype=np.uint8) | (rng.integers(0, 2, size=(PT, 128), dtype=np.uint8) << 7)).view(np.uint32)
key = np.concatenate([key, np.full((PT, 1), np.array([1e-3/256], dtype=np.float32).view(np.uint32)[0], dtype=np.uint32)], axis=1)
if int(os.environ.get("TIES", 1)): key[rng.random(PT) < 0.3, :32] = 0  # (zero keys: scores 0, ties at the k-th)
PB = SEQ_BLK // KP
# (e4m3 values of the bytes)
_b = np.arange(256)
FP8 = np.where((_b >> 3) & 15, (1 + (_b & 7)/8) * 2.0**(((_b >> 3) & 15) - 7), (_b & 7)/8 * 2.0**-6) * np.where(_b >> 7, -1, 1)
FP8[[0x7F, 0xFF]] = np.nan
loc = np.arange(PT // N)
k_rep = Tensor(np.ascontiguousarray(key)).shard(devs).realize()
k_sh = Tensor(np.ascontiguousarray(np.concatenate([key[(loc//PB)*(PB*N) + d*PB + loc%PB] for d in range(N)]))).shard(devs, axis=0).realize()
assert amd_glm._idx_shard(k_sh) == N and not amd_glm._idx_shard(k_rep)

def lists(o:np.ndarray) -> list:
  return [(int(r[-1]), r[:r[-1]].tobytes()) for r in o]

fails = 0
CASES = [(3, 4000), (3, 12000), (3, 30000), (3, 65000), (3, 200000), (3, PT*KP - 8), (64, 30000), (64, 131000), (64, 261000), (512, 200000),
         (512, 262144 - 600)]
for T, pos in CASES:
  q = Tensor((rng.standard_normal((T, H, D)) * 0.5).astype(np.float16)).shard(devs).realize()
  w = Tensor((rng.standard_normal((T, H)) * 0.1).astype(np.float32)).shard(devs).realize()
  qs = UOp.variable("start_pos", 0, PT*KP).bind(pos)
  pools = (pos + T + KP - 1) // KP
  if T <= 8:
    a = indexer_decode_index_rows(q, k_rep, w, qs, pools, TOP, KP, width=PT)
    b = indexer_decode_index_rows(q, k_sh, w, qs, pools, TOP, KP, width=PT)
  else:
    if int(os.environ.get("SCORES", 1)):
      # (the split scores against the whole's, every row's pools before its query: bit-equal?)
      sw = indexer_scores(q, k_rep, w, pools, KP, width=PT).to(devs[0]).numpy()
      ss_t = indexer_scores(q, k_sh, w, pools, KP, width=PT).realize()
      ss = ss_t.numpy().reshape(N, T, -1)
      PL = ss.shape[2]
      lp = np.arange(PL)
      for d in range(N):
        gp = (lp//PB)*(PB*N) + d*PB + lp%PB
        for r in range(T):
          ok = (gp + 1)*KP <= pos + r + 1
          diff = np.nonzero(sw[r, gp[ok]].view(np.uint32) != ss[d, r, lp[ok]].view(np.uint32))[0]
          if len(diff) and (r < 12 or r % 97 == 0):
            print(f"  scores dev {d} row {r}: {len(diff)} of {ok.sum()} differ, first pools {gp[ok][diff[:4]]} "
                                         f"whole {sw[r, gp[ok][diff[:2]]]} split {ss[d, r, lp[ok][diff[:2]]]}")
      # (every device's candidates: its top TOP of a row's pools before the query, against numpy's of the split scores)
      import functools
      cand = Tensor.empty(N*T, 2*TOP, 2, dtype=amd_glm.dtypes.float32, device=devs, axis=0)
      cand = Tensor.custom_kernel(cand, Tensor(ss_t.uop.after(qs)), amd_glm._device_index(devs), fxn=functools.partial(amd_glm._pool_cand_kernel,
        q_start=amd_glm._unbind(qs), pools=UOp.const(pools, amd_glm.dtypes.weakint), k=TOP, kpool=KP, n=N, ties=True))[0].numpy()
      cand = cand.reshape(N, T, 2*TOP, 2)
      for d in range(N):
        gp = (lp//PB)*(PB*N) + d*PB + lp%PB
        for r in range(T):
          ok = (gp + 1)*KP <= pos + r + 1
          v, g = ss[d, r, lp[ok]], gp[ok]
          kth = np.sort(v)[-TOP] if len(v) >= TOP else -np.inf
          above = cand[d, r, :TOP][cand[d, r, :TOP, 1] >= 0]
          tie = cand[d, r, TOP:][cand[d, r, TOP:, 1] >= 0]
          want_above, want_tie = set(g[v > kth].tolist()), set(g[v == kth][:TOP].tolist())
          have_above, have_tie = set(above[:, 1].astype(np.int64).tolist()), set(tie[:, 1].astype(np.int64).tolist())
          if want_above != have_above or want_tie != have_tie:
            print(f"  cand dev {d} row {r}: above {len(have_above)} want {len(want_above)}, ties {len(have_tie)} want {len(want_tie)}")
    a = pool_index_rows(indexer_scores(q, k_rep, w, pools, KP, width=PT), qs, pools, TOP, KP)
    b = pool_index_shard(indexer_scores(q, k_sh, w, pools, KP, width=PT), qs, pools, TOP, KP, width=PT)
  la, lb = lists(a.to(devs[0]).numpy()), lists(b.to(devs[0]).numpy())
  bad = [i for i, (x, y) in enumerate(zip(la, lb)) if x != y]
  fails += len(bad)
  print(f"T={T} pos={pos}: rows {len(la)} differ {len(bad)} counts {[x[0] for x in la[:3]]} vs {[y[0] for y in lb[:3]]}", flush=True)
  for i in bad[:2]:
    xa, xb = np.frombuffer(la[i][1], np.int32)//KP, np.frombuffer(lb[i][1], np.int32)//KP
    sa, sb = set(xa.tolist()), set(xb.tolist())
    print(f"  row {i}: only whole {sorted(sa-sb)[:8]} only split {sorted(sb-sa)[:8]} same set {sa==sb}")
    # (a float64 reference of the row's scores: which list has the top pools)
    qp, npool = pos + i, (pos + i + 1) // KP
    kv = FP8[key[:npool, :32].view(np.uint8)].astype(np.float64) * np.float64(np.float32(1e-3/256)) * 256
    qn, wn = q.to(devs[0]).numpy()[i].astype(np.float64), w.to(devs[0]).numpy()[i].astype(np.float64)
    ref = (np.maximum(kv @ qn.T, 0) * wn).sum(-1)
    kth = np.sort(ref)[-TOP]
    print(f"  ref k-th {kth:.6g}: whole picks >= it {int((ref[[p for p in sa if p < npool]] >= kth).sum())}, split "
          f"{int((ref[[p for p in sb if p < npool]] >= kth).sum())}; split-only scores {ref[sorted(sb-sa)[:4]]} q {qp}")
print("IDXSHARD_CHECK", "PASS" if fails == 0 else f"FAIL {fails}")
