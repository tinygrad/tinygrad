#!/usr/bin/env python3
"""
a memory test of AM devices' VRAM by kernels: as many buffers of GB GB as fit (up to MAXGB) on every device, filled with a pattern of the word's
index (hash ^ seed, every other pass inverted), read back after DELAY s by a kernel that counts the words that differ and the last one: its
physical address (an AM_BAD_PAGES entry, see am/amdev.py bad_pages). env: DEVS (6), GB (1), MAXGB (23), PASSES (4), DELAY (0)
"""
import time
from typing import Any, cast
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import Buffer
from tinygrad.helpers import getenv

DEVS, GB, MAXGB, PASSES, DELAY = getenv("DEVS", 6), getenv("GB", 1), getenv("MAXGB", 23), getenv("PASSES", 4), getenv("DELAY", 0.0)
devs = [f"AMD:{i}" if i else "AMD" for i in range(DEVS)]
n = GB * (1 << 30) // 4
def pattern(dev:str, k:int, seed:int) -> Tensor:
  i = Tensor.arange(n, dtype=dtypes.uint32).to(dev) + (k * n) % (1 << 32)
  return (i * 2654435761 + seed) ^ (0xffffffff if seed & 1 else 0)
bufs:dict[str, list[Tensor]] = {d: [] for d in devs}
for d in devs:
  for k in range(MAXGB // GB):
    try: bufs[d].append(Tensor.empty(n, device=d, dtype=dtypes.uint32).realize())
    except Exception as e:
      print(f"{d}: {k} buffers of {GB} GB ({type(e).__name__})", flush=True)
      break
  print(f"{d}: {len(bufs[d])} GB", flush=True)
for ps in range(PASSES):
  seed = (0x9e3779b9 * (ps + 1) & 0xfffffffe) | (ps & 1)
  st = time.perf_counter()
  for d in devs:
    for k, b in enumerate(bufs[d]): b.assign(pattern(d, k, seed)).realize()
  for d in devs: Device[d].synchronize()
  time.sleep(DELAY)
  for d in devs:
    for k, b in enumerate(bufs[d]):
      bad = (b != pattern(d, k, seed))
      if (cnt:=int(bad.sum().item())):
        first = int((bad.cast(dtypes.int32) * Tensor.arange(n, dtype=dtypes.int32).to(d)).max().item())
        m, off, pa = cast(Buffer, b.uop.buffer).get_storage(d).meta.mapping, first * 4, -1
        for p, sz in m.paddrs:
          if off < sz:
            pa = p + off
            break
          off -= sz
        bus = cast(Any, Device[d]).iface.pci_dev.pcibus
        print(f"BAD {d} buffer {k}: {cnt} words differ, last at word {first} = {bus}@{pa:#x} (pass {ps})", flush=True)
  print(f"pass {ps} seed {seed:#x} done {time.perf_counter()-st:.1f} s", flush=True)
print("DONE", flush=True)
