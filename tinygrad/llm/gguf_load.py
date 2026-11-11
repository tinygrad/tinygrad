# loading a gguf model's tensors on tensor parallel devices straight from the files (the qwen4exp model's, see gguf_shard_units): parallel reads
# into host memory of the devices, the strided layouts (expert units, splits) gathered by kernels there. the decoded weights remember their
# packed bytes (the quantized kernels read those)
import weakref, functools, struct
from typing import Any, Callable, cast
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, round_up, Context, unwrap, dedup, DEBUG, getenv
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_NATIVE, _GGML_QUANT
from tinygrad.llm.direct_read import DirectFile, read_many, close_rings, slow_bytes, PAGE

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

COPY_THREADS, COPY_UNROLL, COPY_ITERS = 256, 4, 1024
COPY_PIECE = COPY_THREADS * COPY_UNROLL * COPY_ITERS  # (the words of a row of a copy at most, see _ranges_copy_kernel)

def _ranges_copy_kernel(dst:UOp, src:UOp, table:UOp) -> UOp:
  """
  table (rows*6,) int32, a row: the source word (-1: zeros) and the destination word of `count` ranges of `words`, the next range's source and
  destination `sstride` and `dstride` words on (words*count COPY_PIECE at most). a workgroup a row
  """
  from tinygrad.uop.ops import AxisType, KernelInfo
  g, t = UOp.range(table.shape[0] // 6, 0, AxisType.GLOBAL), UOp.range(COPY_THREADS, 1, AxisType.LOCAL)
  so, do, nw, cnt, ss, ds = (table[g*6 + k].load().cast(dtypes.weakint) for k in range(6))
  j = UOp.range(COPY_ITERS, 2, AxisType.LOOP)
  stores = []
  for u in range(COPY_UNROLL):
    w = (j*COPY_UNROLL + u)*COPY_THREADS + t
    i, k = w // nw, w % nw
    ok = w < nw*cnt
    val = (so >= 0).where(src[(so + i*ss + k).valid(ok & (so >= 0))].load(), UOp.const(0, src.dtype))
    stores.append(dst[(do + i*ds + k).valid(ok)].store(val))
  return UOp.group(*stores).end(j).end(g, t).sink(arg=KernelInfo(name="ranges_copy", opts_to_apply=()))

@functools.cache
def _gather_ast(fxn:Callable, *params:tuple) -> UOp: return fxn(*[UOp.placeholder(*p) for p in params])
def _gather_call(fxn:Callable, *bufs:UOp) -> UOp:
  # the kernel on the buffers, a call (compiled once: the kernel is the same for buffers of the same sizes, no schedule. its ast made once)
  return _gather_ast(fxn, *[(b.max_shard_shape, b.dtype, i) for i, b in enumerate(bufs)]).call(*bufs)

# (the bytes of a chunk of the files at most, the host buffers they're read into, the bytes of a device's buffer a copy kernel writes at most,
# the bytes of the copies' tables of a chunk)
CHUNK, HOST_BUFS, COPY_WINDOW, TABLES = 2 << 30, 4, 1 << 30, 16 << 20

def _parallel_load(entries:dict[str, tuple[Tensor, tuple[int, ...], int]], devices:tuple[str, ...], shard_map:dict[str, int],
                   units:dict[str, tuple[int, list[list[int]]]], pads:dict[str, int], fuse:dict[str, str],
                   row_sel:dict[str, tuple[int, list[tuple[int, int]]]]|None=None, pending:list|None=None) -> dict[str, Tensor]|None:
  """
  the files read in chunks (in their order, io_uring and O_DIRECT, see direct_read) into host buffers every device maps, every device takes its
  parts of a chunk (its tensors, its experts' units, its columns: a copy kernel of ranges from the host buffer) while the next ones are read
  pending: the reads not done now (the buffers made): their function added to it, called later (e.g. while a model's jits compile)
  """
  import time
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
  with Context(ALLOW_DEVICE_USAGE=1): bufs = [Buffer(d, t).ensure_allocated() for d,t in zip(devices, totals)]
  # the files in chunks (in their order, every tensor whole in one, a fused gate with its up): a chunk read once (io_uring, O_DIRECT: the drives'
  # rate) into a host buffer every device maps, the devices take their parts of it (a kernel of the copies a device, the gathers) while the
  # next ones are read
  span:dict[str, tuple[str, int, int]] = {}
  for name in plan:
    path, lo, nb = locs[name]
    hi = lo + nb
    if name in fuse: lo, hi = min(lo, locs[fuse[name]][1]), max(hi, locs[fuse[name]][1] + locs[fuse[name]][2])
    span[name] = (path, lo, hi)
  chunks:list[tuple[str, int, int, list[str]]] = []
  for name in sorted(plan, key=lambda nm: span[nm]):
    path, lo, hi = span[name]
    if chunks and chunks[-1][0] == path and max(chunks[-1][2], hi) - chunks[-1][1] <= CHUNK:
      chunks[-1] = (path, chunks[-1][1], max(chunks[-1][2], hi), chunks[-1][3] + [name])
    else: chunks.append((path, lo, hi, [name]))
  files = {path: DirectFile(path) for path in dedup(path for path, _, _ in locs.values())}
  def fill_all():
    try: _fill_all()
    finally:
      for f in files.values(): f.close()
      close_rings()
  def _fill_all():
    import threading, queue
    from tinygrad.device import BufferStorage
    from tinygrad.engine.realize import run_linear
    from tinygrad.uop.ops import Ops
    st, slow0 = time.perf_counter(), slow_bytes[0]
    size = max(round_up(hi, PAGE) - (lo - lo % PAGE) for _, lo, hi, _ in chunks)
    # (LOAD_HOST_BUFS: the host buffers the reads go through, fewer when the reads are deferred: they're made while the jits compile)
    hosts = [Buffer(devices[0], size, options=BufferSpec(host=True, cpu_access=True)).ensure_allocated()
             for _ in range(min(getenv("LOAD_HOST_BUFS", 2 if pending is not None else HOST_BUFS), len(chunks)))]
    def alias(b:Buffer, d:str) -> Buffer:
      return Buffer(d, b.nbytes, opaque=BufferStorage(va:=b.get_buf(d)), options=BufferSpec(external_ptr=va)).ensure_allocated()
    views = [[h] + [alias(h, d) for d in devices[1:]] for h in hosts]
    # (the copies' tables of a chunk: in host memory every device maps, written here)
    tabs = [Buffer(devices[0], TABLES, options=BufferSpec(host=True, cpu_access=True)).ensure_allocated() for _ in hosts]
    tviews = [[t] + [alias(t, d) for d in devices[1:]] for t in tabs]
    # (their memory, taken now: as_memoryview waits for the devices)
    hmvs, tmvs = [h.as_memoryview(allow_zero_copy=True) for h in hosts], [t.as_memoryview(allow_zero_copy=True) for t in tabs]
    # the reader: the chunks in order into the free host buffers
    free:queue.Queue[int] = queue.Queue()
    ready:queue.Queue[tuple[int, int]|BaseException] = queue.Queue()
    for k in range(len(hosts)): free.put(k)
    def reader():
      try:
        for c, (path, lo, hi, _) in enumerate(chunks):
          k, base = free.get(), lo - lo % PAGE
          read_many([(files[path], base, hmvs[k][:min(round_up(hi, PAGE), files[path].size) - base])])
          ready.put((c, k))
      except BaseException as e: ready.put(e)
    threading.Thread(target=reader, daemon=True, name="weights_reader").start()
    tls = [Device[d].timeline.host.view(fmt='Q') for d in devices]
    def ranges_of(name:str, i:int, base:int) -> list[tuple[int, int, int, int, int, int]]:
      """the bytes device i takes of the tensor (in the chunk from base): (source (-1: zeros), destination, bytes, count, strides) ranges"""
      nb, fill = plan[name][i]
      at = offsets[name][i]
      if fill[0] == "segs":
        return [(src - base, at + dst, length, 1, 0, 0) for src, length, dst, _, _ in fill[1]] + \
               [(-1, at + z0, z1 - z0, 1, 0, 0) for z0, z1 in fill[2]]
      kind, args, src0 = fill[1], fill[2], locs[name][1] - base
      if kind == "units":
        # unit u (expert u // slices, its column block u % slices): its L bytes of every row
        E, rows, slices, L, us = args
        return [(src0 + (u // slices * rows * slices + u % slices) * L, at + ui * rows * L, L, rows, slices * L, L) for ui, u in enumerate(us)]
      if kind == "rows":
        # unit u: its rows (unit bytes), of every part (the gate's and the fused up's) chunk bytes in turn
        E, nparts, slices, unit, chunk, us = args
        srcs = [src0] + ([locs[fuse[name]][1] - base] if nparts == 2 else [])
        return [(sp + u * unit, at + ui * nparts * unit + p * chunk, chunk, unit // chunk, chunk, nparts * chunk)
                for ui, u in enumerate(us) for p, sp in enumerate(srcs)]
      # the columns di*part.. of every row (zeros past the row)
      outer, row, part, di = args
      real = max(0, min(part, row - di * part))
      return ([(src0 + di * part, at, real, outer, row, part)] if real else []) + \
             ([(-1, at + real, part - real, outer, 0, part)] if real < part else [])
    def calls_of(c:int, k:int) -> list[UOp]:
      path, lo, _, names = chunks[c]
      base, calls, tat, tmv = lo - lo % PAGE, [], 0, tmvs[k]
      for i, d in enumerate(devices):
        # every range in words of the largest size that divides it, a copy kernel a size (and a window of the device's buffer)
        by_word:dict[Any, list[tuple[int, int, int, int, int, int]]] = {}
        for name in names:
          for r in ranges_of(name, i, base):
            if r[2] and r[3]: by_word.setdefault(_gather_word(r[1], r[2], r[4], r[5], *([r[0]] if r[0] >= 0 else [])), []).append(r)
        for word, rs in by_word.items():
          # (the kernels of a few shapes: compiled once. a window of the device's buffer, the whole host buffer, the table padded)
          w, window = word.itemsize, min(COPY_WINDOW, totals[i])
          groups:list[list[tuple[int, int, int, int, int, int]]] = []
          def end(r): return r[1] + (r[3] - 1) * r[5] + r[2]
          for r in sorted(rs, key=lambda x: x[1]):
            if groups and end(r) - (groups[-1][0][1] - groups[-1][0][1] % align) <= window: groups[-1].append(r)
            else: groups.append([r])
          for g in groups:
            v0 = min(g[0][1] - g[0][1] % align, (totals[i] - window) // align * align)
            rows = []
            for so, do, ln, cnt, ss, ds in g:
              so, do, ln, ss, ds = (so // w if so >= 0 else -1), (do - v0) // w, ln // w, ss // w, ds // w
              if cnt == 1:  # (a long range in pieces)
                rows += [(so + o if so >= 0 else -1, do + o, min(COPY_PIECE, ln - o), 1, 0, 0) for o in range(0, ln, COPY_PIECE)]
              elif ln > COPY_PIECE: rows += [(so + n*ss if so >= 0 else -1, do + n*ds, ln, 1, 0, 0) for n in range(cnt)]
              else:
                per = COPY_PIECE // ln
                rows += [(so + n*ss if so >= 0 else -1, do + n*ds, ln, min(per, cnt - n), ss, ds) for n in range(0, cnt, per)]
            rows += [(-1, 0, 1, 0, 0, 0)] * (max(64, 1 << (len(rows) - 1).bit_length()) - len(rows))
            assert tat + 24*len(rows) <= TABLES, "the copies' tables of a chunk don't fit"
            struct.pack_into(f"<{6*len(rows)}i", tmv, tat, *[x for r in rows for x in r])
            table = tviews[k][i].view(24*len(rows), tat).ensure_allocated()
            tat += round_up(24*len(rows), 256)
            calls.append(_gather_call(_ranges_copy_kernel, UOp.from_buffer(bufs[i].view(window, v0).ensure_allocated(), word, d),
                                      UOp.from_buffer(views[k][i], word, d), UOp.from_buffer(table, dtypes.int32, d)))
      return calls
    busy:list[tuple[int, list[int]]] = []  # (the host buffer, every device's timeline value its work is done at)
    with Context(ALLOW_DEVICE_USAGE=1):
      for _ in range(len(chunks)):
        while True:
          for b in [b for b in busy if all(tls[i][0] >= v for i, v in enumerate(b[1]))]:
            busy.remove(b)
            free.put(b[0])
          try: got = ready.get(timeout=0.0005)
          except queue.Empty: continue
          break
        if isinstance(got, BaseException): raise got
        c, k = got
        run_linear(UOp(Ops.LINEAR, src=tuple(calls_of(c, k))), update_stats=False)
        busy.append((k, [int(tl[1]) for tl in tls]))
      for d in devices: Device[d].synchronize()
    if DEBUG >= 1:
      total, dt = sum(hi - lo for _, lo, hi, _ in chunks), time.perf_counter() - st
      print(f"weights read: {total/1e9:.1f} GB in {dt:.2f} s ({total/1e9/dt:.1f} GB/s), {len(chunks)} chunks, "
            f"{(slow_bytes[0]-slow0)/1e6:.1f} MB through the page cache")
  if pending is not None: pending.append(fill_all)
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
               pads:dict[str, int]|None=None, fuse:dict[str, str]|None=None,
               row_sel:dict[str, tuple[int, list[tuple[int, int]]]]|None=None, pending:list|None=None) -> dict[str, Tensor]:
  """
  pending: the file's direct reads left to do (see _parallel_load), their function added to it
  Loads the parsed `entries` on `devices`: a tensor in `shard_map` is sharded on the given axis, the others are copied to every device.
  a tensor (E, rows, cols) in `units` (slices, per device unit lists) is cut in E*slices units: unit e*slices+j is expert e's rows j*rows/slices..
  (its columns for the down projection). every device gets its units in order, a tensor (devices*units, unit rows, unit cols)
  pads: a sharded tensor is zero padded on its shard axis to this size (gguf_pad)
  fuse: gate -> up tensors in units to load as one gate tensor, their unit rows interleaved by 16 (if loaded straight from the file)
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
  direct = _parallel_load(disk, devices, shard_map, units, pads, fuse, row_sel, pending) if disk else None
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
