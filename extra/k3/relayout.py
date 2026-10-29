#!/usr/bin/env python3
# lays the store's experts out again with another split of their units over the gpus (default even): every expert is read from its
# parts, put back together and split again into a new region of the disks with its own manifest (store.py open_store(at=)). the old
# layout and the trunk stay where they are.
# PYTHONPATH=. python3 extra/k3/relayout.py [units] [at] [n]
import sys, time
import numpy as np
from tinygrad.runtime.support.nvme import PRPList, OP_READ, OP_WRITE
from extra.k3.store import open_store, host_io, write_manifest, expert_parts, EXPERTS

B = 64  # experts per batch
disks, m = open_store()
units = [int(u) for u in (sys.argv[1] if len(sys.argv) > 1 else "768,768,768,768").split(",")]
at = int(sys.argv[2]) if len(sys.argv) > 2 else 512 << 30
(blk, tsz), lat, ff = m["block"], m["lat"], m["ff"]
row_up, n_all = lat // blk * tsz, len(m["layers"]) * m["n_experts"]
n_total = int(sys.argv[3]) if len(sys.argv) > 3 else n_all
assert sum(units) == ff and all(u % blk == 0 for u in units), f"units {units} must split {ff} in multiples of {blk}"
parts = expert_parts(units, lat, blk, tsz)
# the new region starts past everything of the old layout: its experts on every disk and the trunk on disk 0
assert at >= m["trunk_off"] + m["trunk_bytes"] and all(at >= m["experts_off"] + n_all * p for p in m["part_bytes"]), "would overwrite"
assert all(at + EXPERTS + n_all * p <= sum(d.size for d in disk.drives) for disk, p in zip(disks, parts)), "doesn't fit"

def bufs(sizes):
  return [(mem, PRPList(disk.drives[0], addrs)) for disk, size in zip(disks, sizes) for mem, addrs in [disk.drives[0].pci_dev.alloc_sysmem(B * size)]]
old, new = bufs(m["part_bytes"]), bufs(parts)

def cuts(us):  # per gpu: its first unit, units, first down block column
  return [(sum(us[:g]), u, sum(us[:g]) // blk * tsz) for g, u in enumerate(us)]

st = time.perf_counter()
gate, up, down = np.empty((B, ff, row_up), np.uint8), np.empty((B, ff, row_up), np.uint8), np.empty((B, lat, ff // blk * tsz), np.uint8)
for x0 in range(0, n_total, B):
  n = min(B, n_total - x0)
  for (u0, u, c0), disk, size, (mem, prps) in zip(cuts(m["units"]), disks, m["part_bytes"], old):
    host_io(disk, OP_READ, m["experts_off"] + x0 * size, prps, n * size)
    a, gs, cs = np.frombuffer(mem.mv, np.uint8)[:n * size].reshape(n, size), u * row_up, u // blk * tsz
    gate[:n, u0:u0 + u], up[:n, u0:u0 + u] = a[:, :gs].reshape(n, u, row_up), a[:, gs:2 * gs].reshape(n, u, row_up)
    down[:n, :, c0:c0 + cs] = a[:, 2 * gs:2 * gs + lat * cs].reshape(n, lat, cs)
  for (u0, u, c0), disk, size, (mem, prps) in zip(cuts(units), disks, parts, new):
    b, gs, cs = np.frombuffer(mem.mv, np.uint8)[:n * size].reshape(n, size), u * row_up, u // blk * tsz
    b[:, :gs], b[:, gs:2 * gs] = gate[:n, u0:u0 + u].reshape(n, -1), up[:n, u0:u0 + u].reshape(n, -1)
    b[:, 2 * gs:2 * gs + lat * cs], b[:, 2 * gs + lat * cs:] = down[:n, :, c0:c0 + cs].reshape(n, -1), 0
    host_io(disk, OP_WRITE, at + EXPERTS + x0 * size, prps, n * size)
  if (x0 // B) % 64 == 0: print(f"{x0 + n}/{n_total} experts, {(x0 + n) * sum(parts) / 1e9 / (time.perf_counter() - st):.2f} GB/s", flush=True)
write_manifest(disks[0], {**m, "units": units, "part_bytes": parts, "experts_off": at + EXPERTS, "crc": [],
                          "drives": [[d.pci_dev.pcibus for d in disk.drives] for disk in disks]}, at)
print(f"relaid {n_total} experts as {units} in {time.perf_counter() - st:.0f} s", flush=True)
for disk in disks:
  for d in disk.drives: d.fini()
