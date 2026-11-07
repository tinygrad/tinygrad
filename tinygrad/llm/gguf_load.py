# loading a gguf model's tensors on tensor parallel devices straight from the files (the qwen4exp model's, see gguf_shard_units): parallel reads
# into host memory of the devices, the strided layouts (expert units, splits) gathered by kernels there. the decoded weights remember their
# packed bytes (the quantized kernels read those)
import weakref, functools, struct
from typing import Any, Callable, cast
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, round_up, Context, unwrap, dedup
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_NATIVE, _GGML_QUANT

# the dequantized weights the loader made: the expression (its storage views unwrapped) -> (ggml type, the packed bytes). the quantized
# kernels use the packed bytes (find_packed) without making the expression again to match it
decoded_weights:weakref.WeakKeyDictionary[UOp, tuple[int, UOp]] = weakref.WeakKeyDictionary()

def _remember_decoded(decoded:Tensor, typ:int, packed:UOp):
  from tinygrad.llm.kernels.amd_qwen import unwrap_storage_views
  decoded_weights[unwrap_storage_views(decoded.uop)] = (typ, packed)

def gguf_pad(entry:tuple[Tensor, tuple[int, ...], int], axis:int, size:int) -> tuple[Tensor, tuple[int, ...], int]:
  """
  Zero pads a parsed `(data, shape, ggml_type)` entry on `axis` to `size`, in its packed form. Zero bytes decode to zero weights.
  """
  data, shape, typ = entry
  if shape[axis] == size: return entry
  block, nbytes = (1, _GGML_NATIVE[typ].itemsize) if typ in _GGML_NATIVE else _GGML_QUANT[typ]
  if axis == len(shape)-1:
    if shape[axis] % block or size % block: raise ValueError(f"can't pad {shape} on axis {axis} to {size} in blocks of {block}")
    data = data.to("CPU").reshape(prod(shape[:-1]), shape[-1]//block*nbytes).pad(((0, 0), (0, (size-shape[-1])//block*nbytes)))
  else: data = data.to("CPU").reshape(prod(shape[:axis]), shape[axis], -1).pad(((0, 0), (0, size-shape[axis]), (0, 0)))
  return data.flatten(), shape[:axis] + (size,) + shape[axis+1:], typ

@functools.cache
def _file_mmap(path:str) -> memoryview:
  import mmap
  with open(path, "rb") as f: return memoryview(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))

def _disk_location(data:Tensor) -> tuple[str, int, int]|None:
  """the file, offset and size of the bytes of a parsed tensor in a gguf file, None if it isn't a view of one"""
  from tinygrad.device import Buffer
  try: buf = data.uop.buffer
  except Exception: return None
  if not isinstance(buf, Buffer) or not buf.device.startswith("DISK:") or buf.device.startswith("DISK:shm:"): return None
  return buf.device[5:], buf.offset, buf.nbytes

def _disk_bytes(data:Tensor) -> memoryview|None:
  # the bytes of a parsed tensor in the mmapped file, None if it isn't a view of a gguf file
  from tinygrad.device import Buffer
  try: buf = data.uop.buffer
  except Exception: return None
  if not isinstance(buf, Buffer) or not buf.device.startswith("DISK:") or buf.device.startswith("DISK:shm:"): return None
  return _file_mmap(buf.device[5:])[buf.offset:buf.offset+buf.nbytes]

def _gather_word(*nbytes:int) -> Any:
  return next(dt for dt in (dtypes.uint32, dtypes.uint16, dtypes.uint8) if all(n % dt.itemsize == 0 for n in nbytes))

def _units_gather_kernel(dst:UOp, src:UOp, units:UOp, rows:int, slices:int) -> UOp:
  # dst (units, rows, L) words: unit i (expert e, slice j) is the column block j of the rows of expert e in src (experts, rows, slices, L)
  # (one index, the words innermost: a wave's loads of consecutive words, src can be host memory)
  from tinygrad.uop.ops import AxisType, KernelInfo
  L = dst.shape[0] // (units.shape[0] * rows)
  g = UOp.range(dst.shape[0], 0, AxisType.GLOBAL)
  k, r, u = g % L, (g // L) % rows, units[g // (L*rows)].load().cast(dtypes.weakint)
  return dst[g].store(src[((u // slices * rows + r) * slices + u % slices) * L + k].load()).end(g).sink(arg=KernelInfo(name="units_gather"))

def _rows_gather_kernel(dst:UOp, src:UOp, units:UOp, parts:int, chunk:int) -> UOp:
  # dst (units, chunks, parts, chunk) words: the chunks of unit i of every part interleaved. src (parts, units', unit): a unit's rows contiguous
  from tinygrad.uop.ops import AxisType, KernelInfo
  unit, part = dst.shape[0] // (units.shape[0] * parts), src.shape[0] // parts
  g = UOp.range(dst.shape[0], 0, AxisType.GLOBAL)
  w, p, c, u = g % chunk, (g // chunk) % parts, (g // (chunk*parts)) % (unit // chunk), units[g // (unit*parts)].load().cast(dtypes.weakint)
  return dst[g].store(src[p*part + u*unit + c*chunk + w].load()).end(g).sink(arg=KernelInfo(name="rows_gather"))

def _part_gather_kernel(dst:UOp, src:UOp, outer:int, off:int) -> UOp:
  # dst (outer, P) words: the words off.. of every row of src (outer, R), zero past the row
  from tinygrad.uop.ops import AxisType, KernelInfo
  P, R = dst.shape[0] // outer, src.shape[0] // outer
  g = UOp.range(dst.shape[0], 0, AxisType.GLOBAL)
  r, c = g // P, off + g % P
  return dst[g].store((c < R).where(src[r*R + c.minimum(R-1)].load(), UOp.const(0, src.dtype))).end(g).sink(arg=KernelInfo(name="part_gather"))

def _gather_call(fxn:Callable, *bufs:UOp) -> UOp:
  # the kernel on the buffers, a call (compiled once: the kernel is the same for buffers of the same sizes, no schedule)
  return fxn(*[UOp.placeholder_like(b, slot=i) for i,b in enumerate(bufs)]).call(*bufs)

def _merge_reads(reads:list[tuple]) -> list[tuple]:
  """
  the reads (fd, src, [(at, len)]) as few as can be: the ones of contiguous bytes of a file one read (up to IOV_MAX pieces), its pieces
  contiguous in the host buffer one piece. a fill thread holds the gil between its reads (a read of many small ones waits for it every time)
  """
  out:list[list] = []
  for fd, src, iov in sorted(reads, key=lambda r: (r[0], r[1])):
    if out and out[-1][0] == fd and out[-1][1] + out[-1][3] == src and len(out[-1][2]) + len(iov) <= 1024:
      cur = out[-1]
    else: out.append(cur:=[fd, src, [], 0])
    for at, ln in iov:
      if cur[2] and cur[2][-1][0] + cur[2][-1][1] == at: cur[2][-1] = (cur[2][-1][0], cur[2][-1][1] + ln)
      else: cur[2].append((at, ln))
      cur[3] += ln
  return [(fd, src, iov) for fd, src, iov, _ in out]

def _preadv_exact(fd:int, views:list[memoryview], offset:int):
  """Fill every view, handling short reads and raising instead of spinning forever at EOF."""
  import os
  while views:
    if (count:=os.preadv(fd, views, offset)) == 0: raise EOFError(f"short read at file offset {offset}")
    offset += count
    while views and count >= len(views[0]): count -= len(views.pop(0))
    if views and count: views[0] = views[0][count:]

def _parallel_load(entries:dict[str, tuple[Tensor, tuple[int, ...], int]], devices:tuple[str, ...], shard_map:dict[str, int],
                   units:dict[str, tuple[int, list[list[int]]]], pads:dict[str, int], fuse:dict[str, str],
                   defer:list[Callable[[], None]]|None=None,
                   row_sel:dict[str, tuple[int, list[tuple[int, int]]]]|None=None) -> dict[str, Tensor]|None:
  """
  every device's tensors are file reads (os.preadv, threads) into host buffers of the device, then copies to its
  buffer, the devices in turn. expert tensors and column-split weights are read once into shared host buffers; device kernels gather each
  shard directly from host memory
  defer: the device memory is allocated (the tensors are made), the bytes are read by the function added to it
  """
  import os, time, concurrent.futures, contextlib
  from tinygrad.device import Buffer, BufferSpec, MultiBuffer
  n = len(devices)
  if not all(d.split(":")[0] == "AMD" for d in devices): return None
  locs:dict[str, tuple[str, int, int]] = {}  # the file, offset and size of every tensor's bytes
  for name, (data, _, _) in entries.items():
    try: buf = data.uop.buffer
    except Exception: return None
    if not isinstance(buf, Buffer) or not buf.device.startswith("DISK:") or buf.device.startswith("DISK:shm:"): return None
    locs[name] = (buf.device[5:], buf.offset, buf.nbytes)
  # every device's part of every tensor: (nbytes, fill). a fill is ("segs", segments, zero ranges) or ("gather", kind, args). a segment
  # (src, length, dst, piece, stride): the bytes src.. split in pieces, piece p to dst + p*stride
  plan:dict[str, list[tuple[int, tuple]]] = {}
  fused_up = set(fuse.values())
  for name, (data, shape, typ) in entries.items():
    _, off, nb = locs[name]
    if name in fused_up: continue
    if row_sel is not None and name in row_sel:
      # every device its rows of the file's tensor (src_rows of them): one read each
      src_rows, sel = row_sel[name]
      rb = nb // src_rows
      plan[name] = [(cnt*rb, ("segs", [(off + r0*rb, cnt*rb, 0, cnt*rb, 0)], [])) for r0, cnt in sel]
      continue
    if name in units:
      slices, per_device = units[name]
      E, rows, cols = shape
      unit = nb // (E*slices)
      if name.endswith("down_exps.weight"):
        plan[name] = [(unit*len(us), ("gather", "units", (E, rows, slices, unit // rows, us))) for us in per_device]
      else:
        # a unit is rows of an expert. fused (fuse_gate_up): a unit's gate and up rows interleaved by 16, the chunks of each every 2*chunk
        nparts, chunk = (2, unit // (rows // slices) * 16) if name in fuse else (1, unit)
        plan[name] = [(nparts*unit*len(us), ("gather", "rows", (E, nparts, slices, unit, chunk, us))) for us in per_device]
    elif (axis:=shard_map.get(name)) is None: plan[name] = [(nb, ("segs", [(off, nb, 0, nb, 0)], []))]*n
    else:
      outer, size = prod(shape[:axis]), pads.get(name, shape[axis])
      if size % n or (axis == len(shape)-1 and size//n % _GGML_QUANT.get(typ, (1,))[0]):
        raise ValueError(f"{name}: can't split {shape} on axis {axis} over {n} devices")
      row = nb // outer
      part = row * size // shape[axis] // n
      if outer == 1:
        plan[name] = [(part, ("segs", [(off + i*part, real, 0, real, 0)] if (real:=max(0, min(part, row - i*part))) else [],
                              [(real, part)] if real < part else [])) for i in range(n)]
      else: plan[name] = [(part*outer, ("gather", "part", (outer, row, part, i))) for i in range(n)]
  # the offsets of every tensor in every device's buffer
  align, offsets, totals = 256, cast(dict[str, list[int]], {}), [0]*n
  for name, parts in plan.items():
    offsets[name] = []
    for i,(nb, _) in enumerate(parts):
      offsets[name].append(totals[i])
      totals[i] += round_up(nb, align)
  STAGE, K = 256 << 20, 3
  assert STAGE > 0 and K > 0
  fds = {path: os.open(path, os.O_RDONLY) for path in dedup(path for path, _, _ in locs.values())}
  # the gathered tensors (split by columns: every device a part of the same bytes) are read once into a host buffer mapped on every device,
  # every device's gather kernel reads its part of it over its link. shared[name]: the reads, the bytes, {device: (kind, args, offset, nb)}
  shared:dict[str, tuple[list[tuple], int, dict[int, tuple], dict[int, int]|None]] = {}
  # every device's jobs in order: ("copy", reads, zeros, size, offset) a host buffer filled by reads (fd, src, [(at, len)]) and zero ranges,
  # then copied to the device's buffer
  def device_jobs(i:int) -> list[tuple]:
    jobs:list[tuple] = []
    start, end = 0, 0
    reads:list[tuple] = []
    zeros:list[tuple[int, int]] = []
    def flush():
      nonlocal reads, zeros
      if (reads or zeros) and end > start: jobs.append(("copy", _merge_reads(reads), zeros, end - start, start))
      reads, zeros = [], []
    def place(lo:int, hi:int):
      # a window of the buffer covering lo..hi (a new one starts at lo)
      nonlocal start, end
      if (not reads and not zeros) or lo < start or hi > start + STAGE:
        flush()
        start = end = lo
      end = max(end, hi)
    for name, parts in plan.items():
      nb, fill = parts[i]
      fd, base = fds[locs[name][0]], offsets[name][i]
      if fill[0] == "segs":
        for src, length, dst, piece, stride in fill[1]:
          if piece == length:  # contiguous: split in windows
            for o in range(0, length, STAGE):
              place(base + dst + o, base + dst + (e:=min(length, o + STAGE)))
              reads.append((fd, src + o, [(base + dst + o - start, e - o)]))
          else:
            place(base + dst, base + dst + (length // piece - 1) * stride + piece)
            reads.append((fd, src, [(base + dst + p*stride - start, piece) for p in range(length // piece)]))
        for z0, z1 in fill[2]:
          place(base + z0, base + z1)
          zeros.append((base + z0 - start, z1 - z0))
        continue
      # (a window doesn't cover a gathered tensor: its copy would write over what the gather writes)
      flush()
      _, src_off, src_nb = locs[name]
      if name not in shared:
        if fill[1] in ("units", "rows"):
          # only the experts some device uses (contiguous in the file), compacted (a fused up's after the gate's), the units renumbered in them
          E, slices = fill[2][0], fill[2][2]
          ebytes, ex = src_nb // E, sorted(set(u // slices for _, p in plan[name] for u in p[2][-1]))
          sreads:list[tuple] = []
          for pi, so in enumerate([src_off] + ([locs[fuse[name]][1]] if name in fuse else [])):
            for j, e in enumerate(ex):  # runs of consecutive experts in one read
              if j and e == ex[j-1] + 1: sreads[-1] = (fd, sreads[-1][1], [(sreads[-1][2][0][0], sreads[-1][2][0][1] + ebytes)])
              else: sreads.append((fd, so + e*ebytes, [((pi*len(ex) + j)*ebytes, ebytes)]))
          shared[name] = (sreads, (2 if name in fuse else 1)*len(ex)*ebytes, {}, {e: j for j, e in enumerate(ex)})
        else: shared[name] = ([(fd, src_off, [(0, src_nb)])], src_nb, {}, None)
      args, pos = fill[2], shared[name][3]
      if pos is not None: args = (*args[:-1], [pos[u // args[2]]*args[2] + u % args[2] for u in args[-1]])
      shared[name][2][i] = (fill[1], args, base, nb)
    flush()
    return jobs
  jobs = [device_jobs(i) for i in range(n)]
  def fill(mv:memoryview, reads:list, zeros:list):
    for fd, src, iov in reads:
      _preadv_exact(fd, [mv[at:at+ln] for at, ln in iov if ln], src)
    for at, ln in zeros: mv[at:at+ln] = bytes(ln)
  def pieces(reads:list[tuple], size:int) -> list[list[tuple]]:
    # the reads of one piece (a read) in parts of up to size bytes: fills in parallel
    return [[(fd, src + o, [(iov[0][0] + o, min(size, iov[0][1] - o))])] for fd, src, iov in reads for o in range(0, iov[0][1], size)]
  with Context(ALLOW_DEVICE_USAGE=1): bufs = [Buffer(d, t).ensure_allocated() for d,t in zip(devices, totals)]
  def fill_all():
    import sys
    # (the fill threads wait for the gil after every read: handed over sooner)
    switch = sys.getswitchinterval()
    sys.setswitchinterval(50 / 1e6)
    try: _fill_all()
    finally:
      sys.setswitchinterval(switch)
      for fd in fds.values(): os.close(fd)
  def _fill_all():
    from tinygrad.device import BufferStorage
    with Context(ALLOW_DEVICE_USAGE=1), contextlib.ExitStack() as cleanup:
      for d in devices: cleanup.callback(Device[d].synchronize)
      # Small loads need small staging buffers, not 3 x 256MB per GPU regardless of the actual copy jobs.
      stages = [[Buffer(d, max(j[3] for j in jobs[i]), options=BufferSpec(host=True, cpu_access=True)).ensure_allocated()
                 for _ in range(min(K, len(jobs[i])))] for i, d in enumerate(devices)]
      mvs = [[s.as_memoryview(allow_zero_copy=True) for s in ss] for ss in stages]
      # the shared host buffers: on the first device, mapped on the others at the same address
      KS, sh_items = 3, list(shared)
      sh_size = round_up(max([v[1] for v in shared.values()], default=0), align)
      sbufs = [Buffer(devices[0], sh_size, options=BufferSpec(host=True, cpu_access=True)).ensure_allocated() for _ in range(KS if shared else 0)]
      smvs = [b.as_memoryview(allow_zero_copy=True) for b in sbufs]
      def alias(b:Buffer, d:str) -> Buffer:
        return Buffer(d, b.nbytes, opaque=BufferStorage(va:=b.get_buf(d)), options=BufferSpec(external_ptr=va)).ensure_allocated()
      salias = [[b] + [alias(b, d) for d in devices[1:]] for b in sbufs]
      pool = cleanup.enter_context(concurrent.futures.ThreadPoolExecutor(16))
      # per device: the next job to fill, the next job to issue, the timeline (the device signals it: the value of the last work issued),
      # per host buffer: its job and fill, or the timeline value of its copy (free once reached). per shared buffer: free, its tensor and
      # fills, or the timeline values of the gathers from it. nothing waits holding the gil (fills run meanwhile)
      nxt, done = [0]*n, [0]*n
      tls = [Device[d].timeline.host.view(fmt='Q') for d in devices]
      fills:list[list[Any]] = [[None]*len(ss) for ss in stages]
      busy:list[list[int]] = [[0]*len(ss) for ss in stages]
      snext = 0
      sstate:list[Any] = [None]*len(sbufs)
      unit_tables:dict[tuple, Buffer] = {}
      from tinygrad.engine.realize import run_linear
      from tinygrad.uop.ops import Ops
      def launch(call:UOp): run_linear(UOp(Ops.LINEAR, src=(call,)), update_stats=False)
      def gather(i:int, src_buf:Buffer, src_nb:int, kind:str, args:tuple, dst0:int, nb:int) -> UOp:
        word = _gather_word(nb, src_nb, *((args[3],) if kind == "units" else args[3:5] if kind == "rows" else (args[1], args[2])))
        dst = UOp.from_buffer(bufs[i].view(nb, dst0).ensure_allocated(), word, devices[i])
        src = UOp.from_buffer(src_buf.view(src_nb, 0).ensure_allocated(), word, devices[i])
        if kind in ("units", "rows"):
          us = args[-1]
          if (ut:=unit_tables.get((i, tuple(us)))) is None:
            ut = unit_tables[(i, tuple(us))] = Buffer(devices[i], 4*len(us), initial_value=struct.pack(f"<{len(us)}i", *us)).ensure_allocated()
          fxn = functools.partial(_units_gather_kernel, rows=args[1], slices=args[2]) if kind == "units" else \
            functools.partial(_rows_gather_kernel, parts=args[1], chunk=args[4] // word.itemsize)
          return _gather_call(fxn, dst, src, UOp.from_buffer(ut, dtypes.int32, devices[i]))
        outer, row, part, di = args
        return _gather_call(functools.partial(_part_gather_kernel, outer=outer, off=di*part//word.itemsize), dst, src)
      while any(done[i] < len(jobs[i]) for i in range(n)) or snext < len(sh_items) or any(s is not None for s in sstate):
        progressed = False
        for i in range(n):
          cur = tls[i][0]
          # fill the free host buffers with the next copy jobs
          for k in range(len(stages[i])):
            if fills[i][k] is None and cur >= busy[i][k] and nxt[i] < len(jobs[i]):
              fills[i][k], nxt[i], progressed = (job:=jobs[i][nxt[i]], pool.submit(fill, mvs[i][k], job[1], job[2])), nxt[i] + 1, True
          if done[i] >= len(jobs[i]): continue
          job = jobs[i][done[i]]
          kf = next((k for k in range(len(stages[i])) if fills[i][k] is not None and fills[i][k][0] is job), None)
          if kf is None or not fills[i][kf][1].done(): continue
          fills[i][kf][1].result()
          dst = UOp.from_buffer(bufs[i].view(job[3], job[4]).ensure_allocated(), dtypes.uint8)
          src = UOp.from_buffer(stages[i][kf].view(job[3], 0).ensure_allocated(), dtypes.uint8)
          launch(dst.store_call(src))
          fills[i][kf], busy[i][kf] = None, tls[i][1]
          done[i], progressed = done[i] + 1, True
        for k, st in enumerate(sstate):
          if st is None and snext < len(sh_items):
            name = sh_items[snext]
            futs = [pool.submit(fill, smvs[k], p, []) for p in pieces(shared[name][0], 64 << 20)]
            sstate[k], snext, progressed = ("fill", name, futs), snext + 1, True
          elif st is not None and st[0] == "fill" and all(f.done() for f in st[2]):
            for f in st[2]: f.result()
            targets = [0]*n
            for i, g in shared[st[1]][2].items(): launch(gather(i, salias[k][i], shared[st[1]][1], *g))
            for i in shared[st[1]][2]: targets[i] = tls[i][1]
            sstate[k], progressed = ("busy", targets), True
          elif st is not None and st[0] == "busy" and all(tls[i][0] >= st[1][i] for i in range(n)): sstate[k], progressed = None, True
        if not progressed:
          pending = [x[1] for row in fills for x in row if x is not None and not x[1].done()] + \
                    [f for s in sstate if s is not None and s[0] == "fill" for f in s[2] if not f.done()]
          if pending: concurrent.futures.wait(pending, timeout=0.0005, return_when=concurrent.futures.FIRST_COMPLETED)
          else: time.sleep(0.0002)
      for d in devices: Device[d].synchronize()
      pool.shutdown()
      mvs.clear()
      stages.clear()
      smvs.clear()
      salias.clear()
      sbufs.clear()
  if defer is not None: defer.append(fill_all)
  else: fill_all()
  packed:dict[str, Tensor] = {}
  for name, parts in plan.items():
    (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = [bufs[i].view(nb, offsets[name][i]).ensure_allocated() for i,(nb,_) in enumerate(parts)]
    buf = UOp.from_buffer(mb, dtypes.uint8, devices)
    packed[name] = Tensor(buf if name not in units and shard_map.get(name) is None else buf.reshape(1, -1).unshard(0))
  return packed

_staging:dict[str, Any] = {}

def _gather_units(data:Tensor, E:int, rows:int, slices:int, units:list[int], by_rows:bool, device:str) -> Tensor:
  """the bytes of `units` of the (E, rows, cols) packed tensor `data` on `device`, (1, bytes). a unit is the rows (or the column blocks) of a slice"""
  from tinygrad.device import Buffer, BufferSpec
  buf = data.uop.buffer
  assert isinstance(buf, Buffer) and buf.device.startswith("DISK:"), "units are gathered from the gguf file"
  src = _file_mmap(buf.device[5:])[buf.offset:buf.offset+buf.nbytes]
  unit = buf.nbytes // (E*slices)
  nbytes = unit * len(units)
  # a host buffer of the device: written here, then one copy to its memory
  if (stage:=_staging.get(device)) is None or stage.nbytes < nbytes:
    stage = _staging[device] = Buffer(device, nbytes, options=BufferSpec(host=True, cpu_access=True)).ensure_allocated()
  dst = stage.as_memoryview(allow_zero_copy=True)[:nbytes]
  if by_rows:
    for i,u in enumerate(units): dst[i*unit:(i+1)*unit] = src[u*unit:(u+1)*unit]
  else:
    # unit (e, j) is the j-th column block of every row of expert e: strided copies of words
    block = unit // rows
    item = next(i for i in (8, 4, 2, 1) if block % i == 0 and buf.offset % i == 0)
    fmt = {8: "Q", 4: "I", 2: "H", 1: "B"}[item]
    s, d, words, row_words = src.cast(cast(Any, fmt)), dst.cast(cast(Any, fmt)), block // item, block * slices // item
    for i,u in enumerate(units):
      e, j = divmod(u, slices)
      base, out = e*rows*row_words + j*words, d[i*rows*words:(i+1)*rows*words]
      for t in range(words): out[t::words] = s[base+t:base+rows*row_words:row_words]
  out_buf = Buffer(device, nbytes).ensure_allocated().copy_from(stage.view(nbytes, 0).ensure_allocated() if stage.nbytes != nbytes else stage)
  Device[device].synchronize()
  return Tensor(UOp.from_buffer(out_buf, dtypes.uint8, device)).reshape(1, -1)

def gguf_shard_units(entries:dict[str, tuple[Tensor, tuple[int, ...], int]], devices:tuple[str, ...],
               shard_map:dict[str, int]|None=None, repeats:dict[str, int]|None=None, units:dict[str, tuple[int, list[list[int]]]]|None=None,
               pads:dict[str, int]|None=None, fuse:dict[str, str]|None=None, defer:list[Callable[[], None]]|None=None,
               row_sel:dict[str, tuple[int, list[tuple[int, int]]]]|None=None) -> dict[str, Tensor]:
  """
  Loads the parsed `entries` on `devices`: a tensor in `shard_map` is sharded on the given axis, the others are copied to every device.
  a tensor (E, rows, cols) in `units` (slices, per device unit lists) is cut in E*slices units: unit e*slices+j is expert e's rows j*rows/slices..
  (its columns for the down projection). every device gets its units in order, a tensor (devices*units, unit rows, unit cols)
  pads: a sharded tensor is zero padded on its shard axis to this size (gguf_pad)
  fuse: gate -> up tensors in units to load as one gate tensor, their unit rows interleaved by 16 (if loaded straight from the file)
  defer: the bytes are read later, by the function added to it (if loaded straight from the file)
  row_sel: a tensor (its entry's shape the loaded one: devices*count rows, sharded on axis 0) is every device's rows (start, count) of the file's
  tensor of src_rows rows: (src_rows, [(start, count) of every device])
  repeats: a tensor first cut into len(devices)//repeats row blocks that are each repeated `repeats` times
  """
  pads, fuse = pads or {}, fuse or {}
  n, packed, shard_map, units, shapes = len(devices), {}, shard_map or {}, units or {}, {}
  for name, r in (repeats or {}).items():
    data, shape, typ = entries[name]
    if n % r or shape[0] % (n//r): raise ValueError(f"{name}: can't repeat {shape} {r} times over {n} devices")
    entries[name] = (data.to("CPU").reshape(n//r, -1).repeat_interleave(r, 0).flatten(), (shape[0]*r, *shape[1:]), typ)
  for name, (_, shape, _) in entries.items():
    if name in units: shapes[name] = (n*len(units[name][1][0]), shape[1] // units[name][0], shape[2]) if not name.endswith("down_exps.weight") else \
      (n*len(units[name][1][0]), shape[1], shape[2] // units[name][0])
  # the tensors of the files are read straight into the devices' memory, the others (made from them: a model's layout of them) copied
  row_sel = row_sel or {}
  disk = {k:v for k,v in entries.items() if _disk_bytes(v[0]) is not None}
  direct = _parallel_load(disk, devices, shard_map, units, pads, fuse, defer, row_sel) if disk else None
  if direct is not None:
    packed = direct
    entries_left = {k:gguf_pad(v, unwrap(shard_map.get(k)), pads[k]) if k in pads else v for k,v in entries.items() if k not in disk}
    for g, u in fuse.items():
      shapes[g] = (shapes[g][0], 2*shapes[g][1], shapes[g][2])
      del entries[u]
  else: entries_left = {k:gguf_pad(v, unwrap(shard_map.get(k)), pads[k]) if k in pads else v for k,v in entries.items()}
  entries = {k:(d, s[:shard_map[k]] + (pads[k],) + s[shard_map[k]+1:] if k in pads else s, t) for k,(d,s,t) in entries.items()}
  for name, (data, shape, typ) in entries_left.items():
    if name in row_sel:
      src_rows, sel = row_sel[name]
      rb = int(data.shape[0]) // src_rows
      parts = [data[r0*rb:(r0+cnt)*rb].reshape(1, -1).to(d).contiguous() for (r0, cnt), d in zip(sel, devices)]
      Tensor.realize(*parts)
      packed[name] = Tensor(UOp.mstack(*[p.uop.base for p in parts]).reshape(parts[0].shape).unshard(0))
      continue
    if name in units:
      slices, per_device = units[name]
      E, rows, cols = shape
      by_rows = not name.endswith("down_exps.weight")
      parts = [_gather_units(data, E, rows, slices, us, by_rows, d) for us,d in zip(per_device, devices)]
      packed[name] = Tensor(UOp.mstack(*[p.uop.base for p in parts]).reshape(parts[0].shape).unshard(0))
      shapes[name] = (n*len(per_device[0]), rows // slices, cols) if by_rows else (n*len(per_device[0]), rows, cols // slices)
      continue
    if (axis:=shard_map.get(name)) is None: packed[name] = data.shard(devices)
    else:
      if shape[axis] % n or (axis == len(shape)-1 and shape[axis]//n % _GGML_QUANT.get(typ, (1,))[0]):
        raise ValueError(f"{name}: can't split {shape} on axis {axis} over {n} devices")
      # every device copies its own part: (devices, part bytes) sharded on the devices
      outer = prod(shape[:axis])
      if outer == 1: parts = [data[i*(sz:=data.shape[0]//n):(i+1)*sz] for i in range(n)]
      else: parts = [x[:, i].contiguous() for x in [data.to("CPU").reshape(outer, n, -1)] for i in range(n)]
      Tensor.realize(*(parts:=[p.reshape(1, -1).to(d).contiguous() for p,d in zip(parts, devices)]))
      packed[name] = Tensor(UOp.mstack(*[p.uop.base for p in parts]).reshape(parts[0].shape).unshard(0))
  if entries_left:
    for t in packed.values(): t.realize()
  for d in devices: Device[d].allocator.free_cache()
  def decode(name:str, shape:tuple[int, ...], typ:int) -> Tensor:
    if name in shapes: shape = shapes[name]
    decoded = ggml_data_to_tensor(packed[name].flatten(), prod(shape), typ)
    _remember_decoded(decoded, typ, packed[name].uop)
    if (axis:=0 if name in shapes else shard_map.get(name)) is None: return decoded.reshape(shape)
    local = decoded.reshape(n, *shape[:axis], shape[axis]//n, *shape[axis+1:])
    return local.permute(*range(1, axis+1), 0, *range(axis+1, local.ndim)).reshape(shape)
  return {name: decode(name, shape, typ) for name, (_, shape, typ) in entries.items()}
