#!/usr/bin/env python3
# python3 extra/nvme_driver/bench.py: nvme drives stream into gpu vram. the drives are split into 4 sets in pci order and set g feeds gpu g
# (on tinys3 that keeps each gpu fed by the 4 drives closest to it). GPUS=0,2 picks sets, PAIRS=0:1,1:2 maps sets to gpus,
# FILL=1 first writes the test region. the test region is the last GB GiB of each drive: the data before it stays.
import os, time
from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.helpers import getenv
from tinygrad.runtime.support.system import PCIDevice
from tinygrad.runtime.support.nvme import NVMeDev, PRPList, OP_READ, OP_WRITE, PAGE

QD, GB, SLOTS = getenv("QD", 16), getenv("GB", 4), getenv("SLOTS", 256)
# PAIRS=set:gpu,... (default: set g -> gpu g for GPUS)
PAIRS = [tuple(map(int, x.split(":"))) for x in getenv("PAIRS", ",".join(f"{g}:{g}" for g in getenv("GPUS", "0,1,2,3").split(","))).split(",")]
bdfs = sorted(d for d in os.listdir("/sys/bus/pci/devices") if open(f"/sys/bus/pci/devices/{d}/class").read().startswith("0x0108"))
per_gpu = len(bdfs) // 4  # drive sets in pci order, set g feeds gpu g
drives = [NVMeDev(PCIDevice("nvme", b)) for g, _ in PAIRS for b in bdfs[g * per_gpu:(g + 1) * per_gpu]]
chunk = drives[0].max_xfer
nchunks = GB * (1 << 30) // chunk
base = {d: (d.size - GB * (1 << 30)) >> d.lba_shift for d in drives}

if getenv("FILL"):
  for d in drives:
    src, addrs = d.pci_dev.alloc_sysmem(chunk)
    src.mv[:] = os.urandom(chunk)
    p1, p2 = PRPList(d, addrs)(0, chunk)
    for i in range(nchunks):
      d.io(OP_WRITE, base[d] + ((i * chunk) >> d.lba_shift), chunk, p1, p2)
      d.io_q.kick()
      while d.io_q.inflight >= QD: d.io_q.reap()
    d.io_q.kick()
    d.io_q.wait(60000)
  print(f"filled {GB} GiB on {len(drives)} drives")

# per drive: the prps of SLOTS vram chunks of its gpu, computed once
work = []
for k, (_, g) in enumerate(PAIRS):
  dev, buf = Device[f"AMD:{g}"], Buffer(f"AMD:{g}", SLOTS * chunk).allocate()
  bar, bar_size = dev.iface.pci_dev.bar_info(dev.iface.vram_bar)
  assert all(p + sz <= bar_size for p, sz in buf.get_storage().meta.mapping.paddrs), "vram past the bar"
  pages = [bar + p + o for p, sz in buf.get_storage().meta.mapping.paddrs for o in range(0, sz, PAGE)]
  for d in drives[k * per_gpu:(k + 1) * per_gpu]:
    prps = PRPList(d, pages)
    work.append((d, [prps(s * chunk, chunk) for s in range(SLOTS)], buf))

st, pending, done = time.perf_counter(), [[d, d.io_q, prps, 0] for d, prps, _ in work], {}
while pending:
  for w in pending:
    d, q, prps, i = w
    n = min(QD - q.inflight, nchunks - i)
    for j in range(i, i + n): d.io(OP_READ, base[d] + ((j * chunk) >> d.lba_shift), chunk, *prps[j % SLOTS])
    w[3] = i + n
    q.kick()
    q.reap()
  for w in pending:
    if w[3] == nchunks and not w[1].inflight: done[w[0].pci_dev.pcibus] = time.perf_counter() - st
  pending = [w for w in pending if w[3] < nchunks or w[1].inflight]
# drives finish at different times (cross-die traffic is slower), so the aggregate is the sum of the per drive rates
rates = {b: nchunks * chunk / t / 1e9 for b, t in done.items()}
lo, hi = min(rates.values()), max(rates.values())
print(f"{len(work)} drives, set:gpu {PAIRS}: {sum(rates.values()):.1f} GB/s aggregate, per drive {lo:.2f}-{hi:.2f} GB/s, qd {QD}")
if getenv("PER_DRIVE"):
  for b, r in sorted(rates.items()): print(f"  {b}: {r:.2f} GB/s")
for d in drives: d.fini()
