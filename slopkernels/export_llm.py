"""Offline exporter for pregenerated templated LLM UIR kernels (`tinygrad/llm/kernels/*.uir`).

Offline-only tooling: generates compact, color-free, `AddrSpace.TEMPLATE`-parameterized `.uir` files
plus a verifiable `manifest.json` so `tinygrad.llm` runs 100% from static UIR artifacts with zero
runtime imports of `slopkernels`.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import threading
from dataclasses import replace
from typing import Any

from tinygrad import UOp, dtypes
from tinygrad.dtype import AddrSpace, DType
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, ParamArg, graph_rewrite
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.uir import parse_uir, render_uir

from slopkernels.src import amd as _amd
from slopkernels.src import amd_gguf as _gguf
from slopkernels.src import extra as _extra

_DTYPES: dict[str, DType] = {v.name: v for v in vars(type(dtypes)).values() if isinstance(v, DType)}
_IQ_GRID_LENS: dict[int, int] = {
  _gguf.IQ2_XXS: 512, _gguf.IQ2_XS: 1024, _gguf.IQ1_S: 4096,
  _gguf.IQ2_S: 2048, _gguf.IQ3_XXS: 256, _gguf.IQ3_S: 512,
}
_WMMA_TABLE_LENS: dict[int, int] = {
  _gguf.IQ4_XS: 256, _gguf.IQ3_XXS: 512, _gguf.IQ2_S: 4096,
  _gguf.IQ2_XXS: 512, _gguf.IQ2_XS: 1024, _gguf.IQ1_S: 4096, _gguf.IQ3_S: 512,
}


def _word_dt(ggml_type:int) -> DType:
  return dtypes.uint8 if ggml_type == _gguf.MXFP4 else dtypes.uint16 if ggml_type in _gguf.HALFWORD_QUANTS else dtypes.uint32


def _p(slot:int, shape:tuple[int|UOp, ...], dt:str|DType) -> UOp:
  return UOp.param(slot, dt if isinstance(dt, DType) else _DTYPES[dt], shape)


def _v(slot:int, vmin:int, vmax:int, name:str) -> UOp:
  return UOp.variable(name, vmin, vmax).replace(arg=ParamArg(slot, dtypes.weakint, (vmin, vmax), name, AddrSpace.ALU, None))


def _canonical_uir(u:UOp) -> str:
  u = graph_rewrite(u, symbolic_simple, enter_calls=True)
  allocs = [x for x in u.toposort(enter_calls=True) if x.op is Ops.ALLOC]
  if allocs: u = u.substitute({b: b.replace(arg=replace(b.arg, slot=i)) for i, b in enumerate(allocs)}, enter_calls=True)
  txt = render_uir(u, color=False)
  assert parse_uir(txt) is u, "exported templated UIR failed lossless round-trip"
  return txt


def _quant_base_t(output:UOp, group:UOp, in_features:UOp, ggml_type:int, itemsize:int) -> UOp:
  if ggml_type == _gguf.NVFP4: return (output * (in_features // 64) + (group // 8) * 4) * (36 // itemsize)
  if ggml_type in _gguf.SMALL_BLOCK_QUANTS: return (output * (in_features // 32) + group) * _gguf.SMALL_BLOCK_QUANTS[ggml_type]
  return (output * (in_features // 256) + group // 8) * (_gguf.QUANT_SIZES[ggml_type] // itemsize)


def _row_words_t(in_features:UOp, ggml_type:int, itemsize:int) -> UOp:
  if ggml_type == _gguf.NVFP4: return (in_features // 64) * (36 // itemsize)
  if ggml_type in _gguf.SMALL_BLOCK_QUANTS: return (in_features // 32) * _gguf.SMALL_BLOCK_QUANTS[ggml_type]
  return (in_features // 256) * (_gguf.QUANT_SIZES[ggml_type] // itemsize)


# ******** Templated Kernel Builders ********

def _build_q8_quantize(in_dtype:str) -> UOp:
  tokens = UOp.template("tokens", (1, 1 << 20))
  in_features = UOp.template("in_features", (32, 1 << 20))
  g = in_features // 32
  return _gguf._q8_quantize_kernel(_p(0, (tokens, g, 8), "u32"), _p(1, (tokens, g), "f32"),
                                   _p(2, (tokens, g, 2), "f32"), _p(3, (tokens, in_features), in_dtype),
                                   tokens=tokens, in_features=in_features)


def _build_dense_gemv(w_dtype:str, x_dtype:str, val_chunk:int) -> UOp:
  tokens = UOp.template("tokens", (1, 15))
  out_features = UOp.template("out_features", (1, 1 << 20))
  in_features = UOp.template("in_features", (32 * val_chunk, 1 << 20))
  out = _p(0, (tokens, out_features), "f32")
  w = _p(1, (out_features, in_features), w_dtype)
  x = _p(2, (tokens, in_features), x_dtype)
  lanes, rows = _amd.WARP_SIZE, 2
  out_blks = (out_features + rows - 1) // rows
  tok_blk = UOp.range(tokens * out_blks, 0, AxisType.GLOBAL)
  wave = UOp.range(rows, 3, AxisType.LOCAL)
  lane = UOp.range(lanes, 2, AxisType.LOCAL)
  tok, out_blk = tok_blk // out_blks, tok_blk % out_blks
  out_row = out_blk * rows + wave
  active, safe_row = out_row < out_features, out_row.minimum(out_features - 1)
  per = in_features // (lanes * val_chunk)
  w3 = w.reshape((out_features, per, lanes * val_chunk))
  x3 = x.reshape((tokens, per, lanes * val_chunk))
  zero = UOp.const(0.0, dtypes.float32)
  acc = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc[0].store(zero))
  i_r = UOp.range(per, 4)
  wv = _amd._vec_load(w3[safe_row, i_r, lane * val_chunk], val_chunk)
  xv = _amd._vec_load(x3[tok, i_r, lane * val_chunk], val_chunk)
  dot = sum((wv[j] * xv[j] for j in range(val_chunk)), zero)
  upd = acc[0].store(acc.after(i_r)[0].load() + dot).end(i_r)
  total = _amd.warp_reduce(acc.after(upd)[0].load(), full_wave=True)
  return out[tok, safe_row.valid(active & lane.eq(0))].store(total).end(tok_blk, wave, lane).sink(
    arg=KernelInfo(name="linear_f16_gemv", opts_to_apply=()))


def _build_quant_gemv(ggml_type:int, cdna:bool) -> UOp:
  w_dt = _word_dt(ggml_type)
  tokens = UOp.template("tokens", (1, 1 << 20))
  out_features = UOp.template("out_features", (1, 1 << 20))
  in_features = UOp.template("in_features", (32, 1 << 20))
  chunks = UOp.template("chunks", (1, 1024))
  raw_len = UOp.template("raw_len", (1, 1 << 30))
  group_rows = UOp.template("group_rows", (0, 1 << 20))
  g = in_features // 32
  out = _p(0, (tokens, out_features, chunks), "f32")
  raw = _p(1, (raw_len,), w_dt)
  xq = _p(2, (tokens, g, 8), "u32")
  xd = _p(3, (tokens, g), "f32")
  xs = _p(4, (tokens, g, 2), "f32")
  grid_in = _p(5, (_IQ_GRID_LENS[ggml_type],), "u32") if ggml_type in _IQ_GRID_LENS else None
  rows = 4
  out_blks = (out_features + rows - 1) // rows
  row = UOp.range(tokens * out_blks, 0, AxisType.GLOBAL)
  chunk = UOp.range(chunks, 1, AxisType.GLOBAL)
  wave = UOp.range(rows, 3, AxisType.LOCAL)
  lane = UOp.range(32, 2, AxisType.LOCAL)
  grid = _gguf._stage_iq_grid(grid_in, ggml_type, wave, lane, rows)
  tok, out_blk = row // out_blks, row % out_blks
  output = out_blk * rows + wave
  active, safe_out = output < out_features, output.minimum(out_features - 1)
  steps = (g + chunks * 32 - 1) // (chunks * 32)
  words = out_features * _row_words_t(in_features, ggml_type, w_dt.itemsize)
  zero = UOp.const(0.0, dtypes.float32)
  acc = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc[0].store(zero))
  s = UOp.range(steps, 4)
  group = lane + (chunk + s * chunks) * 32
  g_safe = group.minimum(g - 1)
  base = _quant_base_t(safe_out, g_safe, in_features, ggml_type, w_dt.itemsize)
  base = base + (group_rows > 0).where((tok // group_rows.maximum(1)) * words, 0)
  xwords = _amd._amd_load(xq[tok, g_safe, 0], 8)
  dot = _gguf._quant_group_dot(raw, base, g_safe % 8, xwords, xd[tok, g_safe],
                                lambda h: xs[tok, g_safe, h].load(), ggml_type, cdna, grid)
  upd = acc[0].store(acc.after(s)[0].load() + (group < g).where(dot, zero)).end(s)
  val = _amd.warp_reduce(acc.after(upd)[0].load(), full_wave=True)
  return out[tok, safe_out.valid(active & lane.eq(0)), chunk].store(val).end(row, wave, chunk, lane).sink(
    arg=KernelInfo(name=f"linear_{_gguf.QUANT_NAMES[ggml_type]}", opts_to_apply=()))


def _build_quant_wmma(ggml_type:int, kgroups:int, lanes:int) -> UOp:
  w_dt = _word_dt(ggml_type)
  tokens = UOp.template("tokens", (16, 1 << 20))
  out_features = UOp.template("out_features", (16, 1 << 20))
  in_features = UOp.template("in_features", (32, 1 << 20))
  raw_len = UOp.template("raw_len", (1, 1 << 30))
  out = _p(0, (tokens, out_features), "f32")
  raw = _p(1, (raw_len,), w_dt)
  x = _p(2, (tokens, in_features), "f16")
  table_in = _p(3, (_WMMA_TABLE_LENS[ggml_type],), "u32") if ggml_type in _WMMA_TABLE_LENS else None
  token_tile, output_tiles, output_waves = 32, 2, 2
  tile_n = 16 * output_tiles * output_waves
  token_block = UOp.range((tokens + token_tile - 1) // token_tile, 0, AxisType.GLOBAL)
  output_block = UOp.range((out_features + tile_n - 1) // tile_n, 1, AxisType.GLOBAL)
  lane = UOp.range(lanes, -1, axis_type=AxisType.WARP)
  wave = UOp.range(output_waves, 3, axis_type=AxisType.LOCAL)
  col, lane_group, rows = lane % 16, lane // 16, 256 // lanes
  outputs_raw = tuple((output_block * output_waves + wave) * (16 * output_tiles) + tile * 16 + col for tile in range(output_tiles))
  outputs_safe = tuple(o.minimum(out_features - 1) for o in outputs_raw)
  input_tokens = tuple((token_block * token_tile + tile * 16 + col).minimum(tokens - 1) for tile in range(token_tile // 16))
  token_rows = tuple(tuple(token_block * token_tile + tile * 16 + (i * 2 + lane_group if kgroups == 1 else lane_group * rows + i)
                           for i in range(rows))
                     for tile in range(token_tile // 16))
  word_indices = _gguf._wmma_words(lane_group, kgroups)
  table = table_in
  if table_in is not None:
    n_tbl = _WMMA_TABLE_LENS[ggml_type]
    local_tbl = UOp.alloc((n_tbl,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    tid, threads = wave * lanes + lane, output_waves * lanes
    table = local_tbl.after(*(local_tbl[tid + i * threads].store(table_in[tid + i * threads]) for i in range(n_tbl // threads)))
  dequant = _gguf._make_wmma_dequant(raw, ggml_type, word_indices, table)
  type_words = _gguf.QUANT_SIZES[ggml_type] // w_dt.itemsize
  output_words = _row_words_t(in_features, ggml_type, w_dt.itemsize)
  accs = tuple(tuple(UOp.alloc((len(token_rows[0]),), dtypes.float32, addrspace=AddrSpace.REG)
                     for _ in range(token_tile // 16)) for _ in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  ks = tuple(word * 4 + b for word in word_indices for b in range(4))
  group = UOp.range(in_features // 32, 4)
  block, subgroup = group // 8, group % 8
  wmma_accs = [list(output_accs) for output_accs in accs]
  for half in range(2):
    afrags = tuple(_gguf._load_afrag_f16(x, inp_tok, group * 32 + half * 16, ks, kgroups) for inp_tok in input_tokens)
    for output_tile, output in enumerate(outputs_safe):
      bfrag = UOp.stack(*dequant(output * output_words + block * type_words, subgroup, half))
      for tile, afrag in enumerate(afrags):
        prev = accs[output_tile][tile].after(group) if half == 0 else wmma_accs[output_tile][tile]
        wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, prev, _gguf.WMMA_ARG[0], lanes)
  update = UOp.group(*(acc.store(value) for output_accs, output_values in zip(accs, wmma_accs)
                       for acc, value in zip(output_accs, output_values))).end(group)
  stores: list[UOp] = []
  for output_tile, o_raw in enumerate(outputs_raw):
    o_safe = o_raw.minimum(out_features - 1)
    o_ok = o_raw < out_features
    for tile, tile_tokens in enumerate(token_rows):
      acc = accs[output_tile][tile].after(update)
      for i, tok in enumerate(tile_tokens):
        t_safe = tok.minimum(tokens - 1)
        stores.append(out[t_safe.valid((tok < tokens) & o_ok), o_safe].store(acc[i].load()))
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(
    arg=KernelInfo(name=f"linear_{_gguf.QUANT_NAMES[ggml_type]}_f16_wmma", opts_to_apply=()))


def _build_moe_decode(ggml_type:int, cdna:bool) -> UOp:
  w_dt = _word_dt(ggml_type)
  rows_t = UOp.template("rows", (1, 1 << 20))
  x_rows = UOp.template("x_rows", (1, 1 << 20))
  out_features = UOp.template("out_features", (1, 1 << 20))
  in_features = UOp.template("in_features", (32, 1 << 20))
  chunks = UOp.template("chunks", (1, 1024))
  raw_len = UOp.template("raw_len", (1, 1 << 30))
  x_share = UOp.template("x_share", (1, 1024))
  ndev = UOp.template("ndev", (1, 64))
  g = in_features // 32
  out = _p(0, (rows_t, out_features, chunks), "f32")
  raw = _p(1, (raw_len,), w_dt)
  xq = _p(2, (x_rows, g, 8), "u32")
  xd = _p(3, (x_rows, g), "f32")
  xs = _p(4, (x_rows, g, 2), "f32")
  sel = _p(5, (rows_t,), "i32")
  grid_in = _p(6, (_IQ_GRID_LENS[ggml_type],), "u32") if ggml_type in _IQ_GRID_LENS else None
  rows = 4
  out_blks = (out_features + rows - 1) // rows
  row = UOp.range(rows_t * out_blks, 0, AxisType.GLOBAL)
  chunk = UOp.range(chunks, 1, AxisType.GLOBAL)
  wave = UOp.range(rows, 3, AxisType.LOCAL)
  lane = UOp.range(32, 2, AxisType.LOCAL)
  grid = _gguf._stage_iq_grid(grid_in, ggml_type, wave, lane, rows)
  tok, out_blk = row // out_blks, row % out_blks
  output = out_blk * rows + wave
  active, safe_out = output < out_features, output.minimum(out_features - 1)
  steps = (g + chunks * 32 - 1) // (chunks * 32)
  expert_words = out_features * _row_words_t(in_features, ggml_type, w_dt.itemsize)
  local_experts = raw_len // expert_words
  dev_r = UOp.range(ndev, 4, AxisType.DEVICE)
  expert = sel[tok].load().cast(dtypes.weakint) - dev_r * local_experts
  in_dev = (expert >= 0) & (expert < local_experts)
  exp_safe = expert.maximum(0).minimum(local_experts - 1)
  xtok = tok // x_share
  zero = UOp.const(0.0, dtypes.float32)
  acc = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc[0].store(zero))
  s = UOp.range(steps, 4)
  group = lane + (chunk + s * chunks) * 32
  g_safe = group.minimum(g - 1)
  base = (_quant_base_t(safe_out, g_safe, in_features, ggml_type, w_dt.itemsize) + exp_safe * expert_words).valid(in_dev)
  xwords = _amd._amd_load(xq[xtok, g_safe, 0], 8)
  dot = _gguf._quant_group_dot(raw, base, g_safe % 8, xwords, xd[xtok, g_safe],
                                lambda h: xs[xtok, g_safe, h].load(), ggml_type, cdna, grid)
  upd = acc[0].store(acc.after(s)[0].load() + (group < g).where(dot, zero)).end(s)
  val = _amd.warp_reduce(acc.after(upd)[0].load(), full_wave=True)
  return out[tok, safe_out.valid(active & lane.eq(0)), chunk].store(val).end(row, wave, chunk, lane).sink(
    arg=KernelInfo(name=f"moe_{_gguf.QUANT_NAMES[ggml_type]}", opts_to_apply=()))


def _build_flash_decode_combine() -> UOp:
  B = UOp.template("B", (1, 1 << 16))
  H = UOp.template("H", (1, 1 << 16))
  C = UOp.template("C", (1, 1024))
  d_blks = UOp.template("d_blks", (1, 64))
  vkl_off = UOp.template("valid_kv_offset", (0, 1 << 30))
  D = d_blks * 64
  o = _p(0, (B, H, 1, D), "f32")
  partial = _p(1, (B, H, C, D), "f32")
  stats = _p(2, (B, H, C, 2), "f32")
  vkl = _v(3, 0, 1 << 30, "valid_kv_len")
  return _amd._amd_flash_decode_combine(o, partial, stats, valid_kv_len=vkl + vkl_off)


def _build_flash_decode_partial(G:int, D:int) -> UOp:
  waves = 16
  while waves * G * ((D + 4) * 2 + 8) > 65536: waves //= 2
  B = UOp.template("B", (1, 1 << 16))
  H_KV = UOp.template("H_KV", (1, 1 << 16))
  n_blks = UOp.template("n_blks", (1, 1 << 20))
  C = UOp.template("C", (1, 1024))
  vkl_off = UOp.template("valid_kv_offset", (0, 1 << 30))
  H, N = H_KV * G, n_blks * 64
  out = _p(0, (B, H, C, D), "f32")
  stats = _p(1, (B, H, C, 2), "f32")
  q = _p(2, (B, H, 1, D), "f16")
  kv = _p(3, (2, B, H_KV, N, D), "f16")
  vkl = _v(4, 0, 1 << 30, "valid_kv_len")
  return _amd._amd_flash_attention_decode_partial(out, stats, q, kv, valid_kv_len=vkl + vkl_off,
                                                  max_kv_len=N, block_n=64, waves=waves, group=G)


def _build_flash_prefill(D:int, rdna4:bool) -> UOp:
  BH = UOp.template("BH", (1, 1 << 16))
  m_blks = UOp.template("m_blks", (1, 1 << 16))
  B = UOp.template("B", (1, 1 << 16))
  H_KV = UOp.template("H_KV", (1, 1 << 16))
  n_blks = UOp.template("n_blks", (1, 1 << 20))
  vkl_off = UOp.template("valid_kv_offset", (0, 1 << 30))
  from_end = UOp.template("q_start_from_end", (0, 1))
  M, N = m_blks * 32, n_blks * 32
  o = _p(0, (BH, M, D), "f32")
  q = _p(1, (BH, M, D), "f16")
  kv = _p(2, (2, B, H_KV, N, D), "f16")
  vkl_in = _v(3, 0, 1 << 30, "valid_kv_len")
  q_len = _v(4, 1, 1 << 20, "q_len")
  vkl = from_end.eq(0).where(vkl_in + vkl_off + q_len, vkl_in + vkl_off)
  qs = from_end.eq(0).where(vkl_in + vkl_off, vkl_in + vkl_off - q_len)
  return _amd._amd_flash_attention(o, q, kv, valid_kv_len=vkl, q_start=qs, rdna4=rdna4)


def _build_hc_coeffs(base_dtype:str, scale_dtype:str) -> UOp:
  tokens = UOp.template("tokens", (1, 1 << 20))
  features = UOp.template("features", (32, 1 << 20))
  iters = UOp.template("iters", (1, 64))
  eps = UOp.template("eps", (0.0, 1.0), dtypes.float32)
  norm_eps = UOp.template("norm_eps", (0.0, 1.0), dtypes.float32)
  count, n = 4, 24
  out = _p(0, (tokens, n), "f32")
  mix = _p(1, (tokens, n), "f32")
  sq = _p(2, (tokens, 32), "f32")
  base = _p(3, (n,), base_dtype)
  scale = _p(4, (3,), scale_dtype)
  token, lane = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  total = sq[token, lane].load()
  rms = (_amd.warp_reduce(total, full_wave=True) * features.cast(dtypes.float32).reciprocal() + norm_eps).rsqrt()
  is_comb = lane < count * count
  idx = is_comb.where(lane + 2 * count, (lane - count * count).minimum(2 * count - 1).maximum(0))
  which = is_comb.where(2, (lane < count * count + count).where(0, 1))
  s = which.eq(0).where(scale[0].load().float(), which.eq(1).where(scale[1].load().float(), scale[2].load().float()))
  v = mix[token, idx].load() * rms * s + base[idx].load().float()
  sig = (1.0 + (v * -_extra.LOG2E).exp2()).reciprocal()
  prepost = which.eq(0).where(sig + eps, sig * 2.0)
  def rowsum(x:UOp, op=lambda a, b: a + b) -> UOp: return op(y := op(x, _amd._lane_xor(x, 1)), _amd._lane_xor(y, 2))
  def colsum(x:UOp) -> UOp: return (y := x + _amd._lane_xor(x, 4)) + _amd._lane_xor(y, 8)
  e = ((v - rowsum(v, UOp.maximum)) * _extra.LOG2E).exp2()
  c_init = e * rowsum(e).reciprocal() + eps
  c_reg = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  c_reg = c_reg.after(c_reg[0].store(c_init))
  it_r = UOp.range(2 * iters - 1, 1)
  cv = c_reg.after(it_r)[0].load()
  denom = (it_r % 2).eq(0).where(colsum(cv), rowsum(cv)) + eps
  c_upd = c_reg[0].store(cv * denom.reciprocal()).end(it_r)
  c = c_reg.after(c_upd)[0].load()
  valid = lane < count * count + 2 * count
  out_idx = is_comb.where(lane + 2 * count, lane - count * count)
  return out[token, out_idx.valid(valid)].store(is_comb.where(c, prepost)).end(token, lane).sink(
    arg=KernelInfo(name="hc_coeffs", opts_to_apply=()))


def _build_gated_delta(key_dim:int, col_alpha:bool) -> UOp:
  batch = UOp.template("batch", (1, 1 << 16))
  heads = UOp.template("heads", (1, 1 << 16))
  tokens = UOp.template("tokens", (1, 1 << 20))
  v_tiles = UOp.template("v_tiles", (1, 1 << 16))
  row_tile = 4
  value_dim = v_tiles * row_tile
  alpha_dim = key_dim if col_alpha else 1
  core = _p(0, (batch, heads, tokens, value_dim), "f32")
  q = _p(1, (batch, heads, tokens, key_dim), "f32")
  k = _p(2, (batch, heads, tokens, key_dim), "f32")
  v = _p(3, (batch, heads, tokens, value_dim), "f32")
  beta = _p(4, (batch, heads, tokens), "f32")
  alpha = _p(5, (batch, heads, tokens, key_dim) if col_alpha else (batch, heads, tokens), "f32")
  state = _p(6, (batch, heads, value_dim, key_dim), "f32")
  kq = _p(7, (batch, heads, tokens), "f32")
  start_pos = _v(8, 0, 1 << 30, "start_pos")
  core, v = core.reshape((batch * heads, tokens, value_dim)), v.reshape((batch * heads, tokens, value_dim))
  q, k = q.reshape((batch * heads, tokens, key_dim)), k.reshape((batch * heads, tokens, key_dim))
  beta, kq = beta.reshape((batch * heads, tokens)), kq.reshape((batch * heads, tokens))
  alpha = alpha.reshape((batch * heads, tokens, alpha_dim))
  state = state.reshape((batch * heads, value_dim, key_dim))
  bh_row = UOp.range(batch * heads * v_tiles, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, axis_type=AxisType.LOCAL)
  bh, row_base = bh_row // v_tiles, (bh_row % v_tiles) * row_tile
  rows = tuple(row_base + i for i in range(row_tile))
  cols = tuple(lane + i * 32 for i in range(key_dim // 32))
  current = UOp.alloc((row_tile * key_dim // 32,), dtypes.float32, addrspace=AddrSpace.REG)
  initial = start_pos.eq(0)
  init_vals = tuple(initial.where(0.0, state[bh, row, col].load().float()) for row in rows for col in cols)
  current = current.after(current.store(UOp.stack(*init_vals)))
  token = UOp.range(tokens, 2)
  keys = tuple(k[bh, token, col].load() for col in cols)
  queries = tuple(q[bh, token, col].load() for col in cols)
  updates, stores = [], []
  for row_idx, row in enumerate(rows):
    previous = tuple(current.after(token)[row_idx * (key_dim // 32) + i].load() for i in range(key_dim // 32))
    decayed = tuple(x * alpha[bh, token, col if col_alpha else 0].load() for x, col in zip(previous, cols))
    bv = beta[bh, token].load()
    state_k = _amd.warp_reduce(sum((x * y for x, y in zip(decayed, keys)), UOp.const(0.0, dtypes.float32)), full_wave=True)
    state_q = _amd.warp_reduce(sum((x * y for x, y in zip(decayed, queries)), UOp.const(0.0, dtypes.float32)), full_wave=True)
    delta = (v[bh, token, row].load() - state_k) * bv
    updates += [x + delta * y for x, y in zip(decayed, keys)]
    stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q + delta * kq[bh, token].load()))
  step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(token)
  state_stores = (state[bh, row, col].store(current.after(step)[row_idx * (key_dim // 32) + i].load())
                  for row_idx, row in enumerate(rows) for i, col in enumerate(cols))
  return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))


# ******** Catalog & Exporter ********

def llm_kernel_catalog() -> dict[str, UOp]:
  """Return `{kernel_name: templated_UOp}` for all pregenerated LLM UIR kernels."""
  cat: dict[str, UOp] = {}
  # 1. Q8 activation quantize
  for dt in ("f32", "f16", "bf16"):
    cat[f"amd_q8_quantize_{dt}"] = _build_q8_quantize(dt)
  # 2. Dense GEMV (dot_dt == f32)
  for w_dt, x_dt in (("f16", "f32"), ("bf16", "f32"), ("f32", "f32"), ("f32", "f16"), ("f32", "bf16")):
    cat[f"amd_gemv_{w_dt}_{x_dt}"] = _build_dense_gemv(w_dt, x_dt, val_chunk=4)
    cat[f"amd_gemv_{w_dt}_{x_dt}_v1"] = _build_dense_gemv(w_dt, x_dt, val_chunk=1)
  # 3. Quantized GGUF decode GEMV, WMMA GEMM, and MoE decode
  for ggml_type, qname in sorted(_gguf.QUANT_NAMES.items()):
    cat[f"rdna_gemv_{qname}"] = _build_quant_gemv(ggml_type, cdna=False)
    cat[f"cdna_gemv_{qname}"] = _build_quant_gemv(ggml_type, cdna=True)
    cat[f"rdna4_gemm_{qname}_f16"] = _build_quant_wmma(ggml_type, kgroups=2, lanes=32)
    cat[f"rdna3_gemm_{qname}_f16"] = _build_quant_wmma(ggml_type, kgroups=1, lanes=32)
    cat[f"cdna_gemm_{qname}_f16"] = _build_quant_wmma(ggml_type, kgroups=4, lanes=64)
    cat[f"rdna_moe_{qname}"] = _build_moe_decode(ggml_type, cdna=False)
    cat[f"cdna_moe_{qname}"] = _build_moe_decode(ggml_type, cdna=True)
  # 4. Flash Attention (decode combine + decode partial + prefill)
  cat["amd_flash_decode_combine"] = _build_flash_decode_combine()
  for d in (64, 128, 256):
    for g in (1, 2, 4, 8):
      cat[f"amd_flash_decode_partial_g{g}_d{d}"] = _build_flash_decode_partial(g, d)
    cat[f"rdna4_flash_attn_d{d}"] = _build_flash_prefill(d, rdna4=True)
    cat[f"rdna3_flash_attn_d{d}"] = _build_flash_prefill(d, rdna4=False)
  # 5. Hyper-Connections Sinkhorn coefficients
  for b_dt in ("f16", "f32"):
    for s_dt in ("f16", "f32"):
      cat[f"amd_hc_coeffs_{b_dt}_{s_dt}"] = _build_hc_coeffs(b_dt, s_dt)
  # 6. Gated Delta Net recurrent prefill
  for k in (32, 64, 128, 256):
    cat[f"amd_gated_delta_k{k}_a1"] = _build_gated_delta(k, col_alpha=False)
    cat[f"amd_gated_delta_k{k}_ak"] = _build_gated_delta(k, col_alpha=True)
  return cat


def export_llm_kernels(out_dir:pathlib.Path) -> dict[str, Any]:
  """Generate all pregenerated templated LLM `.uir` files + `manifest.json` in `out_dir`."""
  out_dir = out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  cat = llm_kernel_catalog()
  artifacts: list[dict[str, Any]] = []
  for name in sorted(cat):
    u = cat[name]
    uir = _canonical_uir(u)
    raw = (uir + "\n").encode("utf-8")
    dst = out_dir / f"{name}.uir"
    dst.write_bytes(raw)
    artifacts.append({
      "name": name,
      "kind": "llm_template",
      "path": f"{name}.uir",
      "templates": sorted({t.arg.name for t in u.templates()}),
      "size": len(raw),
      "sha256": hashlib.sha256(raw).hexdigest(),
    })
  manifest = {"version": 1, "kind": "llm_kernels", "artifacts": artifacts}
  (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  return manifest


def _atomic_write(dst:pathlib.Path, raw:bytes) -> None:
  dst.parent.mkdir(parents=True, exist_ok=True)
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_bytes(raw)
  os.replace(tmp, dst)


def verify_llm_manifest(manifest_path:pathlib.Path, materialize:bool=False) -> tuple[int, int]:
  """Regenerate every artifact in an `llm_kernels` manifest and verify declared size + SHA-256."""
  manifest_path = manifest_path.resolve()
  data = json.loads(manifest_path.read_text(encoding="utf-8"))
  cat = llm_kernel_catalog()
  total_bytes = 0
  for entry in data["artifacts"]:
    name = entry["name"]
    if name not in cat: raise KeyError(f"unknown llm kernel {name!r} in {manifest_path}")
    raw = (_canonical_uir(cat[name]) + "\n").encode("utf-8")
    size, digest = len(raw), hashlib.sha256(raw).hexdigest()
    if size != entry["size"]: raise ValueError(f"{name}: regenerated size {size} != manifest size {entry['size']}")
    if digest != entry["sha256"]: raise ValueError(f"{name}: regenerated sha256 {digest} != manifest sha256 {entry['sha256']}")
    on_disk = manifest_path.parent / entry["path"]
    if materialize:
      if not on_disk.is_file() or len(on_disk.read_bytes()) != size or hashlib.sha256(on_disk.read_bytes()).hexdigest() != digest:
        _atomic_write(on_disk, raw)
    elif on_disk.is_file():
      disk_raw = on_disk.read_bytes()
      if len(disk_raw) != entry["size"] or hashlib.sha256(disk_raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"{name}: on-disk artifact {on_disk} does not match manifest size/sha256")
    total_bytes += size
  return len(data["artifacts"]), total_bytes


DEFAULT_LLM_KERNELS_DIR = pathlib.Path(__file__).resolve().parents[1] / "tinygrad" / "llm" / "kernels"


def generate_all() -> None:
  """Generate all templated LLM `.uir` files on disk in `tinygrad/llm/kernels/` (race-safe, verified against manifest.json)."""
  manifest_path = DEFAULT_LLM_KERNELS_DIR / "manifest.json"
  if not manifest_path.is_file(): return
  def _already_valid() -> bool:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    for e in data.get("artifacts", []):
      p = DEFAULT_LLM_KERNELS_DIR / e["path"]
      if not p.is_file(): return False
      raw = p.read_bytes()
      if len(raw) != e["size"] or hashlib.sha256(raw).hexdigest() != e["sha256"]: return False
    return True
  def _run() -> None:
    if not _already_valid(): verify_llm_manifest(manifest_path, materialize=True)
  try:
    import fcntl
    with open(DEFAULT_LLM_KERNELS_DIR / ".lock", "w") as lf:
      fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
      try: _run()
      finally: fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
  except ImportError:
    _run()
