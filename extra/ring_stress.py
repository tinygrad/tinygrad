#!/usr/bin/env python3
"""
the ring collectives (amd_qwen's all gather and int8 reduce scatter around the devices' peer to peer ring) alone, many times: a hang or a wrong
word of one (a lost or late write of a peer) without the model. env: N (devices, 6), WORDS (an all gather's words a device, 147456: a prefill
chunk's part), RS (a reduce scatter's floats, 0: none), ITERS (exchanges a launch), LAUNCHES, CHECK (every that many launches the values checked),
LANE (FLAG_LANE: the mailbox), on a hang: the devices' queues and the mailboxes' epochs and flags (comm_state_report)
"""
import time, sys
import numpy as np
from tinygrad import Tensor, Device, TinyJit
from tinygrad.helpers import getenv
from tinygrad.llm.kernels import amd_qwen as aq

N, WORDS, RS, ITERS, LAUNCHES, CHECK = getenv("N", 6), getenv("WORDS", 147456), getenv("RS", 0), getenv("ITERS", 32), getenv("LAUNCHES", 2000), \
  getenv("CHECK", 50)
devs = tuple(f"AMD:{i}" if i else "AMD" for i in range(N))
aq.FLAG_LANE[0] = getenv("LANE", 0)
rng = np.random.default_rng(0)
xs = rng.integers(0, 2**32, size=(N, WORDS), dtype=np.uint32)
x = Tensor(xs).shard(devs, axis=0).realize()
p = Tensor(rng.standard_normal((N, RS), dtype=np.float32)).shard(devs, axis=0).realize() if RS else None

@TinyJit
def step(x:Tensor, p:Tensor|None) -> list[Tensor]:
  outs = []
  for _ in range(ITERS):
    g = aq.ring_all_gather_words(x)
    assert g is not None, "no ring all gather of these"
    outs.append(g)
    if p is not None:
      r = aq.ring_reduce_scatter_q8(p, RS)
      assert r is not None, "no ring reduce scatter of these"
      outs.append(r)
  # (every one an output: none left out of the launch)
  return [o.contiguous() for o in outs]

def report(e:BaseException):
  print(f"FAIL: {type(e).__name__}: {e}", flush=True)
  for d in devs:
    try: print("hang:", Device[d].iface.hang_report(), flush=True)
    except Exception as e2: print(f"hang: {d}: {e2}")
  print(aq.comm_state_report(), flush=True)

st = time.perf_counter()
want = xs.reshape(-1)
for i in range(LAUNCHES):
  try:
    outs = step(x, p)
    if i % CHECK == 0 or i == LAUNCHES - 1:
      got = outs[0 if p is None else -2].to(devs[-1]).numpy()
      if not np.array_equal(got, want):
        bad = np.nonzero(got != want)[0]
        print(f"WRONG at launch {i}: {len(bad)} words, first {bad[:8].tolist()}", flush=True)
        sys.exit(2)
      for d in devs: Device[d].synchronize()
      el = time.perf_counter() - st
      print(f"launch {i}: {(i+1)*ITERS} exchanges ok, {el:.1f} s, {(i+1)*ITERS*N*WORDS*4/el/1e9:.2f} GB/s gathered", flush=True)
  except Exception as e:
    report(e)
    sys.exit(3)
print("PASS", flush=True)
