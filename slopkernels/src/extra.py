"""Additional standalone AMD UOp kernel builders (FWHT, Q8/IQ4_XS embeddings, Q8 pack/unpack/sum, Q8_K quantize,
row gather/scatter, tiled argmax, nucleus round, MoE softmax router, mHC Sinkhorn coeffs).

Pure UOp builders only — no runtime logic, no Device access, no getenv.
"""
from __future__ import annotations

import functools
import math
from typing import cast

from tinygrad import UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import round_up
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from slopkernels.src.amd import LOG2E, WARP_SIZE, _lane_xor, warp_reduce


def _nearbyintf(a:UOp, b:UOp, mul:bool=False) -> UOp:
  fmt = "__builtin_nearbyintf({0}*{1})" if mul else "__builtin_nearbyintf({0}/{1})"
  return UOp.native(fmt, dtype=dtypes.float, fmt=True).call(a, b)


def _fwht_kernel(out:UOp, x:UOp, sign:UOp, *, pre:bool) -> UOp:
  # out (n, D) = x @ Hadamard(sign) (pre=True) or x @ Hadamard(sign).T (pre=False). One wave per row, D = 32*2^k.
  D = cast(int, out.shape[-1])
  out, x = out.reshape(-1, D), x.reshape(-1, D)
  n, E = cast(int, out.shape[0]), D // WARP_SIZE
  assert D & (D - 1) == 0 and E >= 1
  r, lane = UOp.range(n, 0, AxisType.GLOBAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  vals = [x[r, e * WARP_SIZE + lane].load().float() for e in range(E)]
  if pre: vals = [v * sign[e * WARP_SIZE + lane].load().float() for e, v in enumerate(vals)]
  for b in range(5):
    m, hi = 1 << b, (lane & (1 << b)).ne(0)
    vals = [hi.where(o - v, v + o) for v, o in zip(vals, [_lane_xor(v, m) for v in vals])]
  for b in range(E.bit_length() - 1):
    m = 1 << b
    vals = [vals[e ^ m] - vals[e] if e & m else vals[e] + vals[e ^ m] for e in range(E)]
  inv = D ** -0.5
  vals = [v * inv if pre else v * inv * sign[e * WARP_SIZE + lane].load().float() for e, v in enumerate(vals)]
  return UOp.group(*[out[r, e * WARP_SIZE + lane].store(v.cast(out.dtype)) for e, v in enumerate(vals)]).end(r, lane).sink(
    arg=KernelInfo(name="fwht", opts_to_apply=()))


def _rows_gather_kernel(out:UOp, table:UOp, idx:UOp) -> UOp:
  # out[r] (n, F) <- table[idx[r]] (P, F), cast to out's dtype
  n, F = cast(tuple[int, int], out.shape)
  P, L = cast(int, table.shape[0]), math.gcd(F, 256)
  r, blk, lane = UOp.range(n, 0, AxisType.GLOBAL), UOp.range(F // L, 1, AxisType.GLOBAL), UOp.range(L, 2, AxisType.LOCAL)
  f, row = blk * L + lane, idx[r].load().cast(dtypes.weakint).maximum(0).minimum(P - 1)
  return out[r, f].store(table[row, f].load().cast(out.dtype)).end(r, blk, lane).sink(arg=KernelInfo(name="rows_gather", opts_to_apply=()))


def _rows_scatter_kernel(buf:UOp, rows:UOp, idx:UOp) -> UOp:
  # buf[idx[r]] (P, F) <- rows[r] (n, F), out-of-range indices skipped
  n, F = cast(tuple[int, int], rows.shape)
  P, L = cast(int, buf.shape[0]), math.gcd(F, 256)
  r, blk, lane = UOp.range(n, 0, AxisType.GLOBAL), UOp.range(F // L, 1, AxisType.GLOBAL), UOp.range(L, 2, AxisType.LOCAL)
  f, row = blk * L + lane, idx[r].load().cast(dtypes.weakint)
  return buf[row.valid((row >= 0) & (row < P)), f].store(rows[r, f].load().cast(buf.dtype)).end(r, blk, lane).sink(
    arg=KernelInfo(name="rows_scatter", opts_to_apply=()))


def _q8_embed_kernel(out:UOp, raw:UOp, ids:UOp) -> UOp:
  # out (R, D) <- rows ids (R,) of a Q8_0 table raw (bytes: 34 per block of 32), dequantized in float then cast to half->out.dtype
  R, D = cast(tuple[int, int], out.shape)
  r, blk, lane = UOp.range(R, 0, AxisType.GLOBAL), UOp.range(D // 32, 1, AxisType.GLOBAL), UOp.range(32, 2, AxisType.LOCAL)
  base = ids[r].load().cast(dtypes.weakint) * (D // 32 * 34) + blk * 34
  d = (raw[base].load().cast(dtypes.uint16) | (raw[base + 1].load().cast(dtypes.uint16) << 8)).bitcast(dtypes.half).float()
  q = raw[base + 2 + lane].load().bitcast(dtypes.int8).float()
  return out[r, blk * 32 + lane].store((d * q).cast(dtypes.half).cast(out.dtype)).end(r, blk, lane).sink(
    arg=KernelInfo(name="q8_embed", opts_to_apply=()))


def _iq4xs_value(raw:UOp, base:UOp, ib:UOp, lane:UOp) -> UOp:
  from tinygrad.runtime.autogen.ggml_common import kvalues_iq4nl
  d = (raw[base].load().cast(dtypes.uint16) | (raw[base + 1].load().cast(dtypes.uint16) << 8)).bitcast(dtypes.half).float()
  sh = raw[base + 2].load().cast(dtypes.uint32) | (raw[base + 3].load().cast(dtypes.uint32) << 8)
  sl = (raw[base + 4 + ib // 2].load().cast(dtypes.uint32) >> ((ib % 2) * 4).cast(dtypes.uint32)) & 15
  scale = (sl | (((sh >> (ib * 2).cast(dtypes.uint32)) & 3) << 4)).cast(dtypes.int32) - 32
  nib = (raw[base + 8 + ib * 16 + lane % 16].load().cast(dtypes.uint32) >> ((lane // 16) * 4).cast(dtypes.uint32)) & 15
  val = UOp.const(float(kvalues_iq4nl[15]), dtypes.float)
  for i in range(14, -1, -1): val = nib.eq(i).where(UOp.const(float(kvalues_iq4nl[i]), dtypes.float), val)
  return ((d * scale.float()) * val).cast(dtypes.half).float()


def _iq4xs_embed_kernel(out:UOp, raw:UOp, ids:UOp) -> UOp:
  # out (R, D) <- rows ids (R,) of an IQ4_XS table raw (bytes: 136 per block of 256)
  R, D = cast(tuple[int, int], out.shape)
  r, blk, ib, lane = (UOp.range(R, 0, AxisType.GLOBAL), UOp.range(D // 256, 1, AxisType.GLOBAL), UOp.range(8, 2, AxisType.GLOBAL),
                      UOp.range(32, 3, AxisType.LOCAL))
  base = ids[r].load().cast(dtypes.weakint) * (D // 256 * 136) + blk * 136
  return out[r, blk * 256 + ib * 32 + lane].store(_iq4xs_value(raw, base, ib, lane).cast(out.dtype)).end(r, blk, ib, lane).sink(
    arg=KernelInfo(name="iq4xs_embed", opts_to_apply=()))


def _q8k_quantize_kernel(q:UOp, scale:UOp, x:UOp, tokens:int, in_features:int) -> UOp:
  # int8 groups of 32 with one scale per 256 inputs (q8_K): one wave per (token, block), 8 inputs per lane
  blocks = in_features // 256
  token_block, lane = UOp.range(tokens * blocks, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, block = token_block // blocks, token_block % blocks
  values = [x.reshape(tokens, in_features)[token, block * 256 + lane * 8 + i].load().float() for i in range(8)]
  local = functools.reduce(lambda a, b: a.maximum(b), [v.abs() for v in values])
  d = (warp_reduce(local, Ops.MAX, full_wave=True) / 127.0).maximum(1e-8)
  quants = [_nearbyintf(v, d).clip(-127, 127).cast(dtypes.int8) for v in values]
  words = [functools.reduce(lambda a, b: a | b, [quants[w * 4 + i].cast(dtypes.uint8).cast(dtypes.uint32) << (8 * i) for i in range(4)])
           for w in range(2)]
  group = block * 8 + lane // 4
  stores = [q[token, group, (lane % 4) * 2 + w].store(words[w]) for w in range(2)] + [scale[token, block.valid(lane.eq(0))].store(d)]
  return UOp.group(*stores).end(token_block, lane).sink(arg=KernelInfo(name="q8k_quantize", opts_to_apply=()))


def _q8_unpacked(pk:UOp, token:UOp, group:UOp, sub:UOp) -> list[UOp]:
  d = pk[token, group, 8].load().bitcast(dtypes.float32)
  words = [pk[token, group, sub * 2 + w].load() for w in range(2)]
  return [(words[i // 4] >> (8 * (i % 4))).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d for i in range(8)]


def _q8_pack_values(values:list[UOp]) -> tuple[list[UOp], UOp]:
  mx = functools.reduce(lambda a, b: a.maximum(b), [v.abs() for v in values])
  for off in (1, 2): mx = mx.maximum(_lane_xor(mx, off))
  d = (mx / 127.0).maximum(1e-8)
  quants = [_nearbyintf(v, d).clip(-127, 127).cast(dtypes.int8) for v in values]
  words = [functools.reduce(lambda a, b: a | b, [quants[w * 4 + i].cast(dtypes.uint8).cast(dtypes.uint32) << (8 * i) for i in range(4)])
           for w in range(2)]
  return words, d


def _q8_pack_kernel(out:UOp, x:UOp, *add:UOp, row0:int=0) -> UOp:
  T, G = cast(tuple[int, int, int], out.shape)[:2]
  assert G % 8 == 0
  tg8, lane = UOp.range(T * (G // 8), 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group, sub = tg8 // (G // 8), (tg8 % (G // 8)) * 8 + lane // 4, lane % 4
  values = [x.reshape(-1, G * 32)[row0 + token, group * 32 + sub * 8 + i].load().float() for i in range(8)]
  if add: values = [a + b for a, b in zip(_q8_unpacked(add[0], token, group, sub), values)]
  words, d = _q8_pack_values(values)
  stores = [out[token, group, sub * 2 + w].store(words[w]) for w in range(2)]
  stores.append(out[token, group, (sub * 0 + 8).valid(sub.eq(0))].store(d.bitcast(dtypes.uint32)))
  return UOp.group(*stores).end(tg8, lane).sink(arg=KernelInfo(name="q8_pack", opts_to_apply=()))


def _q8_unpack_kernel(out:UOp, *parts:UOp) -> UOp:
  T, D = cast(tuple[int, int], out.shape)
  G, rows = D // 32, T // len(parts)
  assert G % 8 == 0
  tg8, lane = UOp.range(T * (G // 8), 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group, sub = tg8 // (G // 8), (tg8 % (G // 8)) * 8 + lane // 4, lane % 4
  def load(w:int) -> UOp:
    if len(parts) == 1: return parts[0][token, group, w].load()
    pt, r = token // rows, token % rows
    return functools.reduce(lambda a, b: a + b, [(pt.eq(i)).where(p.reshape(rows, G, 9)[r.valid(pt.eq(i)), group, w].load(), 0)
                                                 for i, p in enumerate(parts)])
  d = load(8).bitcast(dtypes.float32)
  words = [load(sub * 2 + w) for w in range(2)]
  vals = [((words[i // 4] >> (8 * (i % 4))).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d).cast(out.dtype) for i in range(8)]
  stores = [out[token, group * 32 + sub * 8 + i].store(vals[i]) for i in range(8)]
  return UOp.group(*stores).end(tg8, lane).sink(arg=KernelInfo(name="q8_unpack", opts_to_apply=()))


def _q8_sum_kernel(out:UOp, mine:UOp, *packed:UOp, row0:int=0) -> UOp:
  T, D = cast(tuple[int, int], out.shape)
  G = D // 32
  assert G % 8 == 0
  tg8, lane = UOp.range(T * (G // 8), 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group, sub = tg8 // (G // 8), (tg8 % (G // 8)) * 8 + lane // 4, lane % 4
  acc = [UOp.const(0.0, dtypes.float)] * 8
  for pk in packed: acc = [a + b for a, b in zip(acc, _q8_unpacked(pk, token, group, sub))]
  acc = [acc[i] + mine.reshape(-1, D)[row0 + token, group * 32 + sub * 8 + i].load().float() for i in range(8)]
  stores = [out[token, group * 32 + sub * 8 + i].store(acc[i].cast(out.dtype)) for i in range(8)]
  return UOp.group(*stores).end(tg8, lane).sink(arg=KernelInfo(name="q8_sum", opts_to_apply=()))


def _local_argmax_kernel(out:UOp, x:UOp, count:UOp) -> UOp:
  B, V = cast(tuple[int, int], x.shape)
  T = 256
  row, t = UOp.range(B, 0, AxisType.GLOBAL), UOp.range(T, 1, AxisType.LOCAL)
  n = count[0].load()
  best, idx = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG), UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
  best = best.after(best[0].store(UOp.const(-math.inf, dtypes.float32)))
  idx = idx.after(idx[0].store(UOp.const(0, dtypes.int32)))
  i = UOp.range((V + T - 1) // T, 2, AxisType.LOOP)
  col = i * T + t
  v = (col < n).where(x[row, col.minimum(V - 1)].load().float(), UOp.const(-math.inf, dtypes.float32))
  cb, ci = best.after(i)[0].load(), idx.after(i)[0].load()
  better = v > cb
  upd = UOp.group(best[0].store(better.where(v, cb)), idx[0].store(better.where(col.cast(dtypes.int32), ci))).end(i)
  vb, vi = best.after(upd)[0].load(), idx.after(upd)[0].load()
  lds = UOp.alloc((T, 2), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds = lds.after(UOp.group(lds[t, 0].store(vb), lds[t, 1].store(vi.bitcast(dtypes.float32))).barrier())
  win_v, win_i = lds[0, 0].load(), lds[0, 1].load().bitcast(dtypes.int32)
  for j in range(1, T):
    v2, i2 = lds[j, 0].load(), lds[j, 1].load().bitcast(dtypes.int32)
    take = (v2 > win_v) | (v2.eq(win_v) & (i2 < win_i))
    win_v, win_i = take.where(v2, win_v), take.where(i2, win_i)
  return UOp.group(out[row, UOp.const(0, dtypes.weakint).valid(t.eq(0))].store(win_v),
                   out[row, UOp.const(1, dtypes.weakint).valid(t.eq(0))].store(win_i.bitcast(dtypes.float32))).end(row, t).sink(
    arg=KernelInfo(name="local_argmax", opts_to_apply=()))


def _argmax_chunks_kernel(out:UOp, x:UOp, count:UOp) -> UOp:
  B, V = cast(tuple[int, int], x.shape)
  C, T = int(out.shape[1]), 256
  CW = round_up((V + C - 1) // C, T)
  row, c, t = UOp.range(B, 0, AxisType.GLOBAL), UOp.range(C, 1, AxisType.GLOBAL), UOp.range(T, 2, AxisType.LOCAL)
  n = count[0].load()
  best, idx = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG), UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
  best = best.after(best[0].store(UOp.const(-math.inf, dtypes.float32)))
  idx = idx.after(idx[0].store(UOp.const(0, dtypes.int32)))
  i = UOp.range(CW // T, 3, AxisType.LOOP)
  col = c * CW + i * T + t
  v = ((col < n) & (col < V)).where(x[row, col.minimum(V - 1)].load().float(), UOp.const(-math.inf, dtypes.float32))
  cb, ci = best.after(i)[0].load(), idx.after(i)[0].load()
  better = v > cb
  upd = UOp.group(best[0].store(better.where(v, cb)), idx[0].store(better.where(col.cast(dtypes.int32), ci))).end(i)
  vb, vi = best.after(upd)[0].load(), idx.after(upd)[0].load()
  lds = UOp.alloc((T, 2), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds = lds.after(UOp.group(lds[t, 0].store(vb), lds[t, 1].store(vi.bitcast(dtypes.float32))).barrier())
  win_v, win_i = lds[0, 0].load(), lds[0, 1].load().bitcast(dtypes.int32)
  for j in range(1, T):
    v2, i2 = lds[j, 0].load(), lds[j, 1].load().bitcast(dtypes.int32)
    take = (v2 > win_v) | (v2.eq(win_v) & (i2 < win_i))
    win_v, win_i = take.where(v2, win_v), take.where(i2, win_i)
  return UOp.group(out[row, c, UOp.const(0, dtypes.weakint).valid(t.eq(0))].store(win_v),
                   out[row, c, UOp.const(1, dtypes.weakint).valid(t.eq(0))].store(win_i.bitcast(dtypes.float32))).end(row, c, t).sink(
    arg=KernelInfo(name="argmax_chunks", opts_to_apply=()))


def _argmax_final_kernel(out:UOp, parts:UOp) -> UOp:
  B, C = int(parts.shape[0]), int(parts.shape[1])
  row = UOp.range(B, 0, AxisType.GLOBAL)
  win_v, win_i = parts[row, 0, 0].load(), parts[row, 0, 1].load()
  for c in range(1, C):
    v2, i2 = parts[row, c, 0].load(), parts[row, c, 1].load()
    take = v2 > win_v
    win_v, win_i = take.where(v2, win_v), take.where(i2, win_i)
  return UOp.group(out[row, 0].store(win_v), out[row, 1].store(win_i)).end(row).sink(arg=KernelInfo(name="argmax_final", opts_to_apply=()))


def _nucleus_round_kernel(out:UOp, z:UOp, e:UOp, t:UOp) -> UOp:
  B, _, C, c = cast(tuple[int, int, int, int], z.shape)
  G, n = int(t.shape[-1]), C * c
  assert n > 0 and G > 0
  row, tid = UOp.range(B, 0, AxisType.GLOBAL), UOp.range(256, 1, AxisType.LOCAL)
  ts = [t[row, g].load() for g in range(G)]
  acc = UOp.alloc((G,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(UOp.group(*[acc[g].store(UOp.const(0.0, dtypes.float32)) for g in range(G)]))
  i = UOp.range((n + 255) // 256, 2, AxisType.LOOP)
  pos = i * 256 + tid
  pos_safe = pos.minimum(n - 1) if n % 256 else pos
  zv, ev = z[row, 0, pos_safe // c, pos_safe % c].load(), e[row, 0, pos_safe // c, pos_safe % c].load()
  if n % 256: ev = (pos < n).where(ev, 0.0)
  upd = UOp.group(*[acc[g].store(acc.after(i)[g].load() + (zv >= ts[g]).where(ev, 0.0)) for g in range(G)]).end(i)
  lds = UOp.alloc((256, G), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds = lds.after(UOp.group(*[lds[tid, g].store(acc.after(upd)[g].load()) for g in range(G)]).barrier())
  total = functools.reduce(lambda a, b: a + b, [lds[k, tid.minimum(G - 1)].load() for k in range(256)])
  return out[0, row, tid.valid(tid < G)].store(total).end(row, tid).sink(arg=KernelInfo(name="nucleus_round", opts_to_apply=()))


def _moe_softmax_router_kernel(sel:UOp, probs:UOp, logits:UOp, *table:UOp, k:int, scale:float) -> UOp:
  n = cast(int, logits.shape[-1])
  tokens, width = cast(int, logits.numel()) // n, cast(int, table[0].shape[-1]) if table else 1
  assert n % 32 == 0 and 0 < k <= n and k * width <= 32
  logits, sel, probs = logits.reshape(tokens, n), sel.reshape(tokens, k * width), probs.reshape(tokens, k * width)
  token, lane = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  es = [(lane + 32 * m).cast(dtypes.int32) for m in range(n // 32)]
  scores = [logits[token, e].load().float() for e in es]
  taken = [UOp.const(False, dtypes.bool)] * len(es)
  def better(a:UOp, i:UOp, b:UOp, j:UOp) -> UOp: return (a > b) | (a.eq(b) & (i < j))
  won = []
  for _ in range(k):
    best, index = UOp.const(-math.inf, dtypes.float32), UOp.const(1 << 30, dtypes.int32)
    for e, sc, used in zip(es, scores, taken):
      value, idx = used.where(-math.inf, sc), used.where(1 << 30, e)
      choose = better(value, idx, best, index)
      best, index = choose.where(value, best), choose.where(idx, index)
    for off in (16, 8, 4, 2, 1):
      other, oi = _lane_xor(best, off), _lane_xor(index.bitcast(dtypes.float32), off).bitcast(dtypes.int32)
      choose = better(other, oi, best, index)
      best, index = choose.where(other, best), choose.where(oi, index)
    won.append((index, best))
    taken = [used | e.eq(index) for e, used in zip(es, taken)]
  slots, slot = won[::-1], lane // width
  expert, score = slots[0]
  for i, (e, sc) in enumerate(slots[1:], 1): expert, score = slot.eq(i).where(e, expert), slot.eq(i).where(sc, score)
  p = ((score - won[0][1]) * LOG2E).exp2()
  total = warp_reduce(((lane < k * width) & (lane % width).eq(0)).where(p, 0.0), full_wave=True)
  selected = table[0].reshape(-1, width)[expert.cast(dtypes.weakint).maximum(0).minimum(n - 1), lane % width].load() if table else expert
  at = lane.valid(lane < k * width)
  return UOp.group(sel[token, at].store(selected), probs[token, at].store(p / total * scale)).end(token, lane).sink(
    arg=KernelInfo(name="moe_softmax_router", opts_to_apply=()))


def _hc_coeffs_kernel(out:UOp, mix:UOp, sq:UOp, base:UOp, scale:UOp, count:int, iters:int, eps:float, norm_eps:float, features:int) -> UOp:
  assert count == 4, "the swizzles assume 4 streams"
  token, lane = UOp.range(out.shape[0], 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  groups = sq.shape[1]
  total = sum((sq[token, (lane + 32 * i).valid(lane + 32 * i < groups)].load() for i in range((groups + 31) // 32)), UOp.const(0.0, dtypes.float))
  rms = (warp_reduce(total, full_wave=True) * (1.0 / features) + norm_eps).rsqrt()
  is_comb = lane < count * count
  idx = is_comb.where(lane + 2 * count, (lane - count * count).minimum(2 * count - 1).maximum(0))
  which = is_comb.where(2, (lane < count * count + count).where(0, 1))
  s = which.eq(0).where(scale[0].load().float(), which.eq(1).where(scale[1].load().float(), scale[2].load().float()))
  v = mix[token, idx].load() * rms * s + base[idx].load().float()
  sig = (1.0 + (v * -LOG2E).exp2()).reciprocal()
  prepost = which.eq(0).where(sig + eps, sig * 2.0)
  def rowsum(x:UOp, op=lambda a, b: a + b) -> UOp: return op(y := op(x, _lane_xor(x, 1)), _lane_xor(y, 2))
  def colsum(x:UOp) -> UOp: return (y := x + _lane_xor(x, 4)) + _lane_xor(y, 8)
  e = ((v - rowsum(v, UOp.maximum)) * LOG2E).exp2()
  c = e * rowsum(e).reciprocal() + eps
  for it in range(2 * iters - 1): c = c * ((colsum(c) if it % 2 == 0 else rowsum(c)) + eps).reciprocal()
  valid = lane < count * count + 2 * count
  out_idx = is_comb.where(lane + 2 * count, lane - count * count)
  return out[token, out_idx.valid(valid)].store(is_comb.where(c, prepost)).end(token, lane).sink(
    arg=KernelInfo(name="hc_coeffs", opts_to_apply=()))
