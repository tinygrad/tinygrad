# K3 weights on raw nvme: each gpu owns a set of drives used as a raid0 (1 MiB stripes) "disk", at offset 0 of the first gpu's disk a json
# manifest. every routed expert is split along its intermediate units: gpu g holds gate/up rows and down columns of its units, so a slot
# of gpu g is one contiguous, 4k aligned part of that disk at EXPERTS + index * part_size[g]. the trunk gguf is stored raw on disk 0.
import itertools, json, os, zlib
import numpy as np
from tinygrad.runtime.support.nvme import NVMeDev, PRPList, OP_READ, OP_WRITE, PAGE
from tinygrad.runtime.support.system import PCIDevice
from tinygrad.helpers import round_up

STRIPE, MANIFEST, EXPERTS = 1 << 20, 0, 16 << 20  # manifest gets the first 16 MiB of disk 0

def nvme_bdfs() -> list[str]:
  return sorted(d for d in os.listdir("/sys/bus/pci/devices") if open(f"/sys/bus/pci/devices/{d}/class").read().startswith("0x0108"))

class Disk:
  """raid0 over drives in STRIPE units: offsets are bytes of the disk"""
  def __init__(self, drives:list[NVMeDev]): self.drives = drives
  def cmds(self, off:int, size:int):
    """(drive, lba, nbytes, offset into the transfer) for [off, off+size), split at stripe boundaries"""
    while size:
      s, n = off // STRIPE, min(size, STRIPE - off % STRIPE)
      d = self.drives[s % len(self.drives)]
      yield d, ((s // len(self.drives)) * STRIPE + off % STRIPE) >> d.lba_shift, n
      off, size = off + n, size - n

def open_disks(ngpus:int=4, queues:int=1) -> list[Disk]:
  bdfs = nvme_bdfs()
  per = len(bdfs) // ngpus
  return [Disk([NVMeDev(PCIDevice("nvme", b), queues=queues) for b in bdfs[g * per:(g + 1) * per]]) for g in range(ngpus)]

def expert_parts(units:list[int], lat:int, blk:int, tsz:int) -> list[int]:
  """bytes of each gpu's part of one expert: gate+up rows (lat inputs) and down columns (rows of lat outputs), padded to 4k"""
  return [round_up(2 * u * (lat // blk * tsz) + lat * (u // blk * tsz), PAGE) for u in units]

def host_io(disk:Disk, opcode:int, off:int, prps:PRPList, size:int, qd:int=32):
  """write/read size bytes at disk offset off from/to the host buffer behind prps"""
  done = 0
  for d, lba, n in disk.cmds(off, size):
    d.io(opcode, lba, n, *prps(done, n))
    d.io_q.kick()
    while d.io_q.inflight >= qd: d.io_q.reap()
    done += n
  for d in disk.drives: d.io_q.wait(60000)

def read_manifest(disk0:Disk, at:int=MANIFEST) -> dict:
  d = disk0.drives[0]
  mem, addrs = d.pci_dev.alloc_sysmem(EXPERTS)
  host_io(disk0, OP_READ, at, PRPList(d, addrs), EXPERTS)
  raw = bytes(mem.mv)
  return json.loads(raw[:raw.index(b"\0")])

def open_store(queues:int=1, at:int=MANIFEST) -> tuple[list[Disk], dict]:
  """the disks of the store and its manifest. bus numbers change between boots: each disk gets the order of its drives that reads back
  the manifest (disk 0) and its first two experts as sane blocks, every block's fp16 scale under 1. a stripe out of place is off the
  block grid (1 MiB % 98 != 0): its "scales" are quant bytes"""
  disks, m = open_disks(queues=queues), {}
  def reads_back(disk:Disk, g:int) -> bool:
    nonlocal m
    try:
      if g == 0: m = read_manifest(disk, at)
    except ValueError: return False
    (blk, tsz), u, size = m["block"], m["units"][g], m["part_bytes"][g]
    n = 2 * u * (m["lat"] // blk * tsz) + m["lat"] * (u // blk * tsz)
    mem, addrs = disk.drives[0].pci_dev.alloc_sysmem(2 * size)
    host_io(disk, OP_READ, m["experts_off"], PRPList(disk.drives[0], addrs), 2 * size)
    raw = np.frombuffer(mem.mv, np.uint8)
    return all((np.abs(raw[i * size:i * size + n].reshape(-1, tsz)[:, :2].copy().view(np.float16)) < 1).all() for i in range(2))
  for g, disk in enumerate(disks):
    if (order:=next((p for p in itertools.permutations(disk.drives) if reads_back(Disk(list(p)), g)), None)) is None:
      raise RuntimeError(f"disk {g}: no order of its drives reads back the store")
    disk.drives = list(order)
  return disks, m

def write_manifest(disk0:Disk, manifest:dict, at:int=MANIFEST):
  raw = json.dumps(manifest).encode()
  assert len(raw) < EXPERTS, "manifest too big"
  d = disk0.drives[0]
  mem, addrs = d.pci_dev.alloc_sysmem(round_up(len(raw) + 1, STRIPE))
  mem.mv[:] = raw.ljust(len(mem.mv), b"\0")
  host_io(disk0, OP_WRITE, at, PRPList(d, addrs), len(mem.mv))

def crc(b) -> int: return zlib.crc32(b)
