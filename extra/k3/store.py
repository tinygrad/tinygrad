# the experts of a streamed moe on the nvme drives. gpu g owns a "disk": its drives (NVME:n, nd of them in bus order under it) as a raid0
# of 1 MiB stripes. at the store's offset of disk 0 a json manifest (in 16 MiB), the experts after it: every expert split along its
# intermediate units, gpu g's part of the expert code (layer * E + expert) is part_bytes[g] at experts_off + code * part_bytes[g] of its
# disk. the order of a disk's drives is found by reading the store back (bus numbers change between boots). reads here are master's
# static copies (a drive's tensor to a gpu), the expert reads of a step are the host's (host.py)
import itertools, json
from typing import cast
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.runtime.ops_nvme import NVME_IDS, NVMEDevice
from tinygrad.runtime.support.system import System

STRIPE, EXPERTS = 1 << 20, 16 << 20

def pieces(off:int, size:int, nd:int) -> list[tuple[int, int, int]]:
  """(drive in the disk's order, offset on the drive, bytes) of the disk's [off, off + size), split at the stripes"""
  out = []
  while size:
    s, n = off // STRIPE, min(size, STRIPE - off % STRIPE)
    out.append((s % nd, s // nd * STRIPE + off % STRIPE, n))
    off, size = off + n, size - n
  return out

def part_off(m:dict, g:int, code:int) -> int: return m["experts_off"] + code * m["part_bytes"][g]

def disks(ngpu:int) -> list[list[str]]: # the drives under each gpu, in bus order
  nd = len(System.list_devices(*NVME_IDS)) // ngpu
  return [[f"NVME:{g * nd + i}" for i in range(nd)] for g in range(ngpu)]

def read(reads:list[tuple[str, int, int]], dev:str) -> list[np.ndarray]:
  """the bytes of reads (drive, offset, bytes) through dev: static copies"""
  ts = [Tensor.empty(cast(NVMEDevice, Device[d]).nvme.size, dtype=dtypes.uint8, device=d)[o:o + n].to(dev) for d, o, n in reads]
  Tensor.realize(*ts)
  return [t.numpy() for t in ts]

def read_disk(drives:list[str], off:int, size:int, dev:str) -> np.ndarray: # the disk's [off, off + size)
  return np.concatenate(read([(drives[d], o, n) for d, o, n in pieces(off, size, len(drives))], dev))

def sane(m:dict, g:int, part:np.ndarray) -> bool:
  """a part's gate and up rows as blocks whose scale is under 1 (an fp16, mxfp4's an e8m0). a stripe out of place is off the block grid
  (1 MiB % 98 != 0, % 17 != 0): its "scales" are quant bytes"""
  (blk, tsz), mxfp4 = m["block"], m.get("expert_type") == "MXFP4"
  b = part[:2 * m["units"][g] * (m["lat"] // blk * tsz)].reshape(-1, tsz)
  return bool((b[:, 0] < 127).all() if mxfp4 else (np.abs(b[:, :2].copy().view(np.float16)) < 1).all())

def in_orders(drives:list[str], off:int, size:int, dev:str) -> dict[tuple[int, ...], np.ndarray]:
  """the disk's [off, off + size) in every order of its drives: each drive's span of the range read once"""
  ps = pieces(off, size, len(drives))
  lo, hi = min(o for _, o, _ in ps), max(o + n for _, o, n in ps)
  local = read([(d, lo, hi - lo) for d in drives], dev)
  return {order: np.concatenate([local[order[s]][o - lo:o - lo + n] for s, o, n in ps]) for order in itertools.permutations(range(len(drives)))}

def manifest(raw:bytes) -> dict|None:
  try: return json.loads(raw[:raw.index(b"\0")])
  except ValueError: return None

def open_store(gpus:list[str], at:int=0) -> tuple[list[list[str]], dict]:
  """each gpu's drives in the store's order and the manifest at disk 0's offset at: disk 0's order reads back the manifest, every disk's
  its first two experts sane"""
  out:list[list[str]] = []
  for g, drives in enumerate(disks(len(gpus))):
    if g == 0:
      found = {o: mf for o, raw in in_orders(drives, at, EXPERTS, gpus[g]).items() if (mf:=manifest(raw.tobytes())) is not None}
      if not found: raise RuntimeError(f"no manifest at {at:#x} of disk 0 ({', '.join(drives)})")
      m = next(iter(found.values()))
    parts = in_orders(drives, part_off(m, g, 0), 2 * m["part_bytes"][g], gpus[g])
    ok = [o for o, raw in parts.items() if (g or o in found) and all(sane(m, g, p) for p in raw.reshape(2, -1))]
    if len(ok) != 1: raise RuntimeError(f"disk {g} ({', '.join(drives)}): {len(ok)} orders of its drives read back the store")
    out.append([drives[i] for i in ok[0]])
  return out, m
