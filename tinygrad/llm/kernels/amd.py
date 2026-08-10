from __future__ import annotations
import functools
from typing import cast
from tinygrad import UOp
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.dtype import AddrSpace, dtypes

def warp_reduce(val:UOp, full_wave:bool=False, maximum:bool=False) -> UOp:
  for offset in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = UOp(Ops.CUSTOM, dtypes.float, (val,), arg=
      f"__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | offset<<10}))")
    val = val.maximum(other) if maximum else val + other
  return val

@functools.cache
def _mxfp8_qdq_kernel(out:UOp, x:UOp) -> UOp:
  """Software OCP E4M3/E8M0 round trip, one wave per 32-value MX block."""
  groups = cast(int, x.shape[-1])//32
  outer = x.numel()//cast(int, x.shape[-1])
  block, lane = UOp.range(outer*groups, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  value = x.reshape(outer, groups, 32)[block//groups, block%groups, lane].float()
  amax = warp_reduce(value.abs(), full_wave=True, maximum=True)
  exponent = (amax.maximum(1e-38)/448.0).log2().round().maximum(-127.0).minimum(127.0)
  block_scale = amax.eq(0).where(1.0, exponent.exp2())
  normalized = value/block_scale
  magnitude = normalized.abs().minimum(448.0)
  elem_exp = magnitude.maximum(2**-9).log2().floor().maximum(-6.0).minimum(8.0)
  quantum = (elem_exp-3.0).exp2()
  quantized = (magnitude/quantum).round()*quantum
  quantized = (normalized < 0).where(-quantized, quantized).maximum(-448.0).minimum(448.0)
  store = out.reshape(outer, groups, 32)[block//groups, block%groups, lane].store((quantized*block_scale).cast(out.dtype))
  return store.end(lane, block).sink(arg=KernelInfo(name="mxfp8_qdq", opts_to_apply=()))

def _mxfp4_value(code:UOp) -> UOp:
  """Decode one OCP E2M1 nibble without a lookup-table memory access."""
  magnitude = code & 7
  value = magnitude.eq(7).where(6.0, magnitude.eq(6).where(4.0, magnitude.eq(5).where(3.0, magnitude.float()*0.5)))
  return (code & 8).ne(0).where(-value, value)

@functools.cache
def _kda_qkv_kernel(qout:UOp, kout:UOp, vout:UOp, x:UOp, qw:UOp, kw:UOp, vw:UOp) -> UOp:
  """Fused BF16 decode projection for equal-sized KDA Q/K/V tensors."""
  batch, tokens, out_features = cast(tuple[int, int, int], qout.shape)
  in_features, output_tile = cast(int, x.shape[-1]), 1
  assert qout.shape == kout.shape == vout.shape and out_features % output_tile == 0 and in_features % 32 == 0
  row, lane = UOp.range(batch*tokens*(out_features//output_tile), 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  token, output_block = row // (out_features//output_tile), row % (out_features//output_tile)
  outputs = tuple(output_block*output_tile+i for i in range(output_tile))
  acc = UOp.placeholder((3, output_tile), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0.0)))
  group = UOp.range(in_features//32, 2, AxisType.REDUCE)
  activation = x.reshape(batch*tokens, in_features)[token, group*32+lane].float()
  updates = [acc.after(group)[p, i].load()+activation*w[output, group*32+lane].float()
             for p,w in enumerate((qw, kw, vw)) for i,output in enumerate(outputs)]
  update = acc.store(UOp.stack(*updates).reshape(3, output_tile)).end(group)
  outs = (qout, kout, vout)
  stores = (outs[p].reshape(batch*tokens, out_features)[token, output.valid(lane.eq(0))].store(
    warp_reduce(acc.after(update)[p, i], full_wave=True).cast(outs[p].dtype))
    for p in range(3) for i,output in enumerate(outputs))
  return UOp.group(*stores).end(lane, row).sink(arg=KernelInfo(name="kda_qkv", opts_to_apply=()))

@functools.cache
def _mxfp4_expert_linear_kernel(out:UOp, sel:UOp, x:UOp, weight:UOp, scale:UOp) -> UOp:
  """Wave32 decode GEMM which consumes selected experts directly from packed MXFP4 storage."""
  batch, tokens, topk, out_features = cast(tuple[int, int, int, int], out.shape[:4])
  partials = cast(int, out.shape[4]) if len(out.shape) == 5 else 1
  output_tile = 1
  assert out_features % output_tile == 0
  in_features = cast(int, weight.shape[-1])*2
  assert in_features % 32 == 0 and x.shape[-1] == in_features and sel.shape == (batch, tokens, topk)
  xchoices = cast(int, x.shape[-2])
  assert xchoices in (1, topk)
  row, lane = UOp.range(batch*tokens*topk*(out_features//output_tile)*partials, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  partial, output_block, route = row % partials, (row//partials) % (out_features//output_tile), \
    row // ((out_features//output_tile)*partials)
  outputs = tuple(output_block*output_tile+i for i in range(output_tile))
  token, choice = route // topk, route % topk
  expert = sel.reshape(batch*tokens, topk)[token, choice]
  xv = x.reshape(batch*tokens, xchoices, in_features)
  acc = UOp.placeholder((output_tile,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0.0)))
  group = UOp.range(in_features//32, 2, AxisType.REDUCE)
  activation = xv[token, 0 if xchoices == 1 else choice, group*32+lane].float()
  updates = []
  for i,output in enumerate(outputs):
    packed = weight[expert, output, group*16 + lane//2]
    code = (packed >> ((lane&1)*4).cast(dtypes.uint8)) & 15
    w = _mxfp4_value(code) * (scale[expert, output, group].float()-127.0).exp2()
    updates.append(acc.after(group)[i].load()+activation*w)
  update = acc.store(UOp.stack(*updates)).end(group)
  out = out.reshape(batch*tokens, topk, out_features, partials)
  stores = (out[token, choice, output, partial.valid(lane.eq(0))].store(warp_reduce(acc.after(update)[i], full_wave=True).cast(out.dtype))
            for i,output in enumerate(outputs))
  return UOp.group(*stores).end(lane, row).sink(arg=KernelInfo(name="mxfp4_expert_linear", opts_to_apply=()))

@functools.cache
def _bf16_partial_linear_kernel(out:UOp, x:UOp, weight:UOp) -> UOp:
  """Per-device BF16 down projection; its dummy final axis is reduced after combining TP partials."""
  batch, tokens, out_features, partials = cast(tuple[int, int, int, int], out.shape)
  in_features, output_tile = cast(int, x.shape[-1]), 1
  assert out_features % output_tile == 0 and in_features % 16 == 0
  row, lane = UOp.range(batch*tokens*(out_features//output_tile)*partials, 0), UOp.range(16, 1, axis_type=AxisType.LOCAL)
  partial, output_block, token = row%partials, (row//partials)%(out_features//output_tile), row//(partials*(out_features//output_tile))
  outputs = tuple(output_block*output_tile+i for i in range(output_tile))
  acc = UOp.placeholder((output_tile,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0.0)))
  chunk, group = in_features//16, UOp.range(in_features//16, 2, AxisType.REDUCE)
  input_idx = lane*chunk+group
  activation = x.reshape(batch*tokens, in_features)[token, input_idx].float()
  update = acc.store(UOp.stack(*(acc.after(group)[i].load()+(activation*weight[output, input_idx].float()).cast(dtypes.bfloat16).float()
    for i,output in enumerate(outputs)))).end(group)
  local = UOp.placeholder((output_tile, 16), dtypes.float32, slot=1, addrspace=AddrSpace.LOCAL)
  barrier = UOp.group(*(local[i, lane].store(acc.after(update)[i]) for i in range(output_tile))).barrier()
  stores = (out.reshape(batch*tokens, out_features, partials)[token, output, partial.valid(lane.eq(0))].store(
    sum((local.after(barrier)[i, j] for j in range(16)), UOp.const(0, dtypes.float32))) for i,output in enumerate(outputs))
  return UOp.group(*stores).end(lane, row).sink(arg=KernelInfo(name="bf16_partial_linear", opts_to_apply=()))

@functools.cache
def _bf16_matvec_kernel(out:UOp, x:UOp, weight:UOp) -> UOp:
  batch, tokens, out_features = cast(tuple[int, int, int], out.shape)
  in_features = cast(int, x.shape[-1])
  assert in_features % 16 == 0
  row, lane = UOp.range(batch*tokens*out_features, 0), UOp.range(16, 1, axis_type=AxisType.LOCAL)
  token, output = row//out_features, row%out_features
  acc = UOp.placeholder((), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(0.0))
  chunk, group = in_features//16, UOp.range(in_features//16, 2, AxisType.REDUCE)
  input_idx = lane*chunk+group
  product = (x.reshape(batch*tokens, in_features)[token, input_idx].float()*weight[output, input_idx].float()).cast(dtypes.bfloat16).float()
  update = acc.store(acc.after(group)+product).end(group)
  local = UOp.placeholder((16,), dtypes.float32, slot=1, addrspace=AddrSpace.LOCAL)
  barrier = local[lane].store(acc.after(update)).barrier()
  total = sum((local.after(barrier)[j] for j in range(16)), UOp.const(0, dtypes.float32))
  return out.reshape(batch*tokens, out_features)[token, output.valid(lane.eq(0))].store(total.cast(out.dtype)).end(lane, row).sink(
    arg=KernelInfo(name="bf16_matvec", opts_to_apply=()))

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, next_state:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp) -> UOp:
  batch, heads, tokens, value_dim, row_tile = *core.shape, 4
  key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
  assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim)) and key_dim % 32 == 0 and value_dim % row_tile == 0
  batch, heads, tokens, value_dim, key_dim = cast(tuple[int, int, int, int, int], (batch, heads, tokens, value_dim, key_dim))
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha = alpha.reshape(batch*heads, tokens, alpha_dim)
  state, next_state = (x.reshape(batch*heads, value_dim, key_dim) for x in (state, next_state))
  bh_row, lane = UOp.range(batch*heads*value_dim//row_tile, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  bh, row_base = bh_row // (value_dim//row_tile), (bh_row % (value_dim//row_tile))*row_tile
  rows = tuple(row_base+i for i in range(row_tile))
  cols = tuple(lane + i*32 for i in range(key_dim//32))
  current = UOp.placeholder((row_tile*key_dim//32,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  current = current.after(current.store(UOp.stack(*(state[bh, row, col].float() for row in rows for col in cols))))
  token = UOp.range(tokens, 2, AxisType.REDUCE)
  keys = tuple(k[bh, token, col].load() for col in cols)
  queries = tuple(q[bh, token, col].load() for col in cols)
  updates:list[UOp] = []
  stores:list[UOp] = []
  for row_idx,row in enumerate(rows):
    previous = tuple(current.after(token)[row_idx*key_dim//32+i].load() for i in range(key_dim//32))
    av = tuple(alpha[bh, token, col if alpha_dim > 1 else 0].load() for col in cols)
    bv = beta[bh, token].load()
    state_k = warp_reduce(sum((x*a*y for x,a,y in zip(previous, av, keys)), UOp.const(0, dtypes.float32)), full_wave=True)
    state_q = warp_reduce(sum((x*a*y for x,a,y in zip(previous, av, queries)), UOp.const(0, dtypes.float32)), full_wave=True)
    delta = (v[bh, token, row].load() - state_k) * bv
    updates += [x*a + delta*y for x,a,y in zip(previous, av, keys)]
    stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q + delta*kq[bh, token]))
  step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(token)
  state_stores = (next_state[bh, row, col].store(current.after(step)[row_idx*key_dim//32+i].load().cast(next_state.dtype))
                  for row_idx,row in enumerate(rows) for i,col in enumerate(cols))
  return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))
