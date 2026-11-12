#!/usr/bin/env python3
"""
the AM driver's live physical (VRAM) allocations by call site, every device: where a device's memory goes that no live Buffer counts
(tinygrad's mem_used counts host buffers and not the driver's own). usage: python3 extra/mem_trace.py <script.py> [args...] (or
install() before the devices open and report() later). the report prints at the script's end or at its exception (an OOM's): per device
the counted Buffers (VRAM and host), the physical used, and the sites (MEM_TRACE_TOP, default 25) with the largest difference of device
MEM_TRACE_A (default 0) and MEM_TRACE_B (default 1).
"""
import sys, os, traceback, collections, runpy, gc

_live:dict[int, dict[int, tuple[int, str]]] = collections.defaultdict(dict)  # (id of a MemoryManager: paddr -> (size, site))
_mms:dict[int, object] = {}
_SKIP = ("runtime/support/memory.py", "runtime/support/system.py", "runtime/support/am/",
         "/contextlib.py", "/runpy.py", "mem_trace.py", "tinygrad/device.py")

def _site() -> str:
  # (the innermost frames outside the allocator: three of the runtime's and the first of the model's or a script's)
  fr = [f for f in traceback.extract_stack(limit=40)[:-2] if not any(s in f.filename for s in _SKIP)][::-1]
  def fmt(f) -> str: return f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno} {f.name}"
  inner = [fmt(f) for f in fr[:3]]
  outer = next((fmt(f) for f in fr[3:] if "/tinygrad/llm/" in f.filename or "/extra/" in f.filename or "/tinygrad/" not in f.filename), "")
  return " < ".join(inner + ([outer] if outer else []))

def install() -> None:
  from tinygrad.runtime.support.memory import MemoryManager
  if getattr(MemoryManager, "_mem_trace", False): return
  palloc, pfree = MemoryManager.palloc, MemoryManager.pfree
  def tpalloc(self, size:int, align:int=0x1000, zero=True, boot=False, ptable=False) -> int:
    paddr = palloc(self, size, align, zero, boot, ptable)
    _mms[id(self)] = self
    _live[id(self)][paddr] = (size, "ptable" if ptable else "boot" if boot else _site())
    return paddr
  def tpfree(self, paddr:int, ptable=False):
    _live[id(self)].pop(paddr, None)
    return pfree(self, paddr, ptable)
  MemoryManager.palloc, MemoryManager.pfree, MemoryManager._mem_trace = tpalloc, tpfree, True  # type: ignore[method-assign,attr-defined]

def report(tag:str="end") -> None:
  from tinygrad.device import Device, Buffer
  devs = [d for d in Device._opened_devices if d.split(":")[0] == "AMD"] if hasattr(Device, "_opened_devices") else []
  if not devs: devs = sorted({b.device for b in gc.get_objects() if isinstance(b, Buffer) and str(b.device).startswith("AMD")})
  if not devs: return
  vram, host = collections.Counter(), collections.Counter()
  for o in gc.get_objects():
    if isinstance(o, Buffer) and o._base is None and isinstance(o.device, str) and o.is_allocated():
      (host if o.options is not None and o.options.host else vram)[o.device] += o.nbytes
  sites:dict[str, collections.Counter] = {}
  for d in devs:
    try: mm = Device[d].iface.dev_impl.mm  # type: ignore[attr-defined]
    except Exception: continue
    free = sum(sz for sz, _, _, f in mm.pa_allocator.blocks.values() if f)
    traced = sum(s for s, _ in _live[id(mm)].values())
    lru = sum(sz * len(v) for (sz, o), v in Device[d].allocator.cache.items() if not (o is not None and o.host))  # type: ignore[attr-defined]
    print(f"MEM_TRACE {tag} {d}: phys_used={(mm.vram_size - free)/1e9:.3f} free={free/1e9:.3f} counted_vram={vram[d]/1e9:.3f} "
          f"counted_host={host[d]/1e9:.3f} lru_cached_vram={lru/1e9:.3f} traced_live={traced/1e9:.3f} n={len(_live[id(mm)])}", flush=True)
    if lru:
      by = collections.Counter({sz: len(v) for (sz, o), v in Device[d].allocator.cache.items() if v and not (o is not None and o.host)})  # type: ignore[attr-defined]
      print(f"MEM_TRACE {tag} {d}: lru " + " ".join(f"{n}x{sz/1e6:.2f}MB" for sz, n in sorted(by.items(), key=lambda x: -x[0]*x[1])[:16]), flush=True)
    c = sites[d] = collections.Counter()
    for s, site in _live[id(mm)].values(): c[site] += s
  a, b = [devs[int(os.environ.get(f"MEM_TRACE_{k}", i))] for k, i in (("A", 0), ("B", 1))] if len(devs) > 1 else (devs[0], devs[0])
  if a not in sites or b not in sites: return
  keys = set(sites[a]) | set(sites[b])
  for k in sorted(keys, key=lambda k: -abs(sites[a][k] - sites[b][k]))[:int(os.environ.get("MEM_TRACE_TOP", 25))]:
    print(f"MEM_TRACE {tag} diff {(sites[a][k]-sites[b][k])/1e6:9.1f} MB  {a} {sites[a][k]/1e6:9.1f}  {b} {sites[b][k]/1e6:9.1f}  {k}", flush=True)
  for k, v in sites[a].most_common(int(os.environ.get("MEM_TRACE_TOP", 25))):
    print(f"MEM_TRACE {tag} top {a} {v/1e6:9.1f} MB  {k}", flush=True)

if __name__ == "__main__":
  install()
  script, sys.argv = sys.argv[1], sys.argv[1:]
  sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
  try: runpy.run_path(script, run_name="__main__")
  except BaseException:
    report("exception")
    raise
  report("end")
