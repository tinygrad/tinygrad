#!/usr/bin/env python3
# runs on the box with the drives: receives extra/k3/pack.py's stream and writes the K3 weights onto the raw nvme disks (see store.py)
# PYTHONPATH=. python3 extra/k3/recv.py [port] [at]: at a region past the trunk snapshot (another expert quant, its own manifest; the stream
# has no trunk then, the manifest keeps the base store's trunk)
import json, socket, struct, sys, time, zlib
from tinygrad.helpers import round_up
from tinygrad.runtime.support.nvme import PRPList, OP_WRITE
from extra.k3.store import open_disks, open_store, host_io, write_manifest, crc, EXPERTS, STRIPE

BATCH = 16  # experts staged per write

def recv_into(sock, mv):
  got = 0
  while got < len(mv):
    if not (n:=sock.recv_into(mv[got:])): raise ConnectionError("stream ended early")
    got += n

srv = socket.create_server(("0.0.0.0", int(sys.argv[1]) if len(sys.argv) > 1 else 7777))
print("waiting for the packer", flush=True)
sock, _ = srv.accept()
recv_into(sock, memoryview(hdr:=bytearray(8)))
recv_into(sock, memoryview(raw:=bytearray(struct.unpack("<Q", hdr)[0])))
m = json.loads(raw)
parts, at = m["part_bytes"], int(sys.argv[2]) if len(sys.argv) > 2 else 0
# a region writes in the drive order that reads back the base store, and keeps its trunk
disks, base = open_store() if at else (open_disks(len(m["units"])), {})
# (the trunk snapshot is at 1 TiB, ~10 GB a disk)
assert at == 0 or (at >= 5 << 38 and not m["trunk_bytes"]), "a region past the trunk snapshot, without a trunk"
staging = []
for disk, size in zip(disks, parts):
  mem, addrs = disk.drives[0].pci_dev.alloc_sysmem(BATCH * size)
  staging.append((mem, PRPList(disk.drives[0], addrs)))

n_total, crcs, st = len(m["layers"]) * m["n_experts"], [[] for _ in parts], time.perf_counter()
for x0 in range(0, n_total, BATCH):
  n = min(BATCH, n_total - x0)
  for k in range(n):
    for g, size in enumerate(parts):
      view = staging[g][0].mv[k * size:(k + 1) * size]
      recv_into(sock, view)
      crcs[g].append(crc(view))
  for g, (disk, size) in enumerate(zip(disks, parts)): host_io(disk, OP_WRITE, at + EXPERTS + x0 * size, staging[g][1], n * size)
  if (x0 // BATCH) % 64 == 0:
    gb = (x0 + n) * sum(parts) / 1e9
    print(f"{x0 + n}/{n_total} experts, {gb:.1f} GB, {gb / (time.perf_counter() - st):.3f} GB/s", flush=True)

# the trunk gguf, raw, after the experts of disk 0
trunk_off, left, buf_mem, buf_prps = round_up(EXPERTS + n_total * parts[0], 1 << 30), m["trunk_bytes"], *staging[0]
off, tcrc = trunk_off, 0
while left:
  n = min(left, len(buf_mem.mv))
  recv_into(sock, buf_mem.mv[:n])
  tcrc = zlib.crc32(buf_mem.mv[:n], tcrc)
  host_io(disks[0], OP_WRITE, off, buf_prps, round_up(n, 4096))
  off, left = off + n, left - n
m.update({"experts_off": at + EXPERTS, "stripe": STRIPE, "drives": [[d.pci_dev.pcibus for d in disk.drives] for disk in disks], "crc": crcs,
          "trunk_off": trunk_off, "trunk_crc": tcrc})
if at: m.update({k: v for k, v in base.items() if k.startswith("trunk_")})
write_manifest(disks[0], m, at)
sock.sendall(b"ok")
print(f"wrote {n_total} experts and the trunk in {time.perf_counter() - st:.0f} s", flush=True)
for disk in disks:
  for d in disk.drives: d.fini()
