from collections import Counter
from dataclasses import dataclass, replace

from tinygrad.dtype import dtypes
from tinygrad.renderer import Estimates, Renderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, ProgramInfo, UOp, ssimplify


@dataclass(frozen=True)
class ChannelReduceMatch:
  kind: str
  params: tuple[UOp, ...]
  channels: int
  spatial: int


def _match_channel_reduce(ast:UOp, device:str, arch:str) -> ChannelReduceMatch|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1 or not isinstance(ast.arg, KernelInfo): return None
  end = ast.src[0]
  if end.op is not Ops.END or len(end.src) != 3 or end.src[0].op is not Ops.STORE: return None
  store, channel, feature = end.src
  if any(x.op is not Ops.RANGE or x.arg[1] is not AxisType.LOOP for x in (channel, feature)): return None
  channels, features = int(channel.vmax)+1, int(feature.vmax)+1
  if channels not in (64, 256) or features != 256 or store.src[0].op is not Ops.INDEX: return None

  reduce = store.src[1]
  if reduce.op is not Ops.REDUCE or reduce.arg != (Ops.ADD, 0) or len(reduce.src) != 4: return None
  batch, ry, rx = reduce.src[1:]
  if any(x.op is not Ops.RANGE or x.arg[1] is not AxisType.REDUCE for x in (batch, ry, rx)): return None
  spatial = int(ry.vmax)+1
  if (int(batch.vmax)+1, spatial, int(rx.vmax)+1) != (6, spatial, spatial) or channels*spatial*spatial != 16384: return None

  out_idx = ssimplify(channel*features+feature)
  data_idx = ssimplify((feature*6+batch)*16384+channel*spatial*spatial+ry*spatial+rx)
  if ssimplify(store.src[0].src[1].get_idx()) is not out_idx: return None
  params = tuple(sorted((x for x in ast.toposort() if x.op is Ops.PARAM), key=lambda x:x.arg.slot))
  total = features*6*16384
  signature = tuple((x.arg.slot, x.dtype, int(x.src[0].arg)) for x in params)
  signatures = {
    "activation": ((0, dtypes.float, channels*features), (1, dtypes.half, total), (2, dtypes.float, channels),
                   (3, dtypes.float, channels), (4, dtypes.float, total), (5, dtypes.half, total)),
    "centered": ((0, dtypes.float, channels*features), (1, dtypes.half, total), (2, dtypes.float, channels),
                 (3, dtypes.float, channels), (4, dtypes.float, total)),
    "scale": ((0, dtypes.float, channels*features), (1, dtypes.float, channels), (2, dtypes.float, channels),
              (3, dtypes.float, total)),
  }
  kind = next((k for k,v in signatures.items() if signature == v), None)
  if kind is None: return None

  indexes = [(x.src[0].arg.slot, ssimplify(x.src[1].get_idx())) for x in reduce.src[0].toposort() if x.op is Ops.INDEX]
  expected_indexes = {
    "activation": ((1, data_idx), (2, channel), (3, channel), (4, data_idx), (5, data_idx)),
    "centered": ((1, data_idx), (2, channel), (3, channel), (4, data_idx)),
    "scale": ((1, channel), (2, channel), (3, data_idx)),
  }[kind]
  if len(indexes) != len(expected_indexes) or any(slot != eslot or idx is not eidx for (slot,idx),(eslot,eidx) in zip(indexes, expected_indexes)):
    return None
  op_signature = Counter(x.op for x in reduce.src[0].toposort())
  expected_ops = {
    "activation": {Ops.CONST:13, Ops.PARAM:5, Ops.RANGE:5, Ops.MUL:15, Ops.ADD:11, Ops.INDEX:5, Ops.CAST:3,
                   Ops.EXP2:1, Ops.RECIPROCAL:1},
    "centered": {Ops.CONST:10, Ops.PARAM:4, Ops.RANGE:5, Ops.MUL:8, Ops.ADD:6, Ops.INDEX:4, Ops.CAST:1},
    "scale": {Ops.CONST:10, Ops.PARAM:3, Ops.RANGE:5, Ops.INDEX:3, Ops.RECIPROCAL:1, Ops.SQRT:1, Ops.MUL:8, Ops.ADD:5},
  }[kind]
  if op_signature != Counter(expected_ops): return None
  return ChannelReduceMatch(kind, params, channels, spatial)


def _sum16(term:str) -> str:
  lanes = tuple(f"v{i}.{lane}" for i in range(4) for lane in "xyzw")
  return "+".join(term.format(v=x) for x in lanes)

def _use_wave_reduce(m:ChannelReduceMatch) -> bool:
  return m.kind != "scale" or m.channels == 64

def _wave_group_size(m:ChannelReduceMatch) -> int:
  return 256 if m.kind == "scale" else 1024

def _render_wave_reduce(m:ChannelReduceMatch, name:str) -> str:
  group_size = _wave_group_size(m)
  param_decls = ", ".join(f'{"half" if x.dtype == dtypes.half else "float"}* p{x.arg.slot}' for x in m.params)
  if m.kind == "activation":
    value = """half z=(half)p4[base], g=p5[base];
      half sig=(half)1.0/((half)1.0+__builtin_elementwise_exp2(z*(half)-2.4554669595930156));
      half ag=sig*g+(half)1.702*z*g*sig*((half)1.0-sig);
      v=((float)p1[base]-p2[channel])*(float)ag;"""
    out = "acc*p3[channel]"
  elif m.kind == "centered": value, out = "v=((float)p1[base]-p2[channel])*p4[base];", "acc*p3[channel]"
  else: value, out = "v=p3[base];", "-(acc*p1[channel]*__builtin_elementwise_sqrt(1.0f/p2[channel]))"
  return f'''#define half _Float16
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size({group_size},{group_size}))) {name}({param_decls}) {{
  int tid=__builtin_amdgcn_workitem_id_x(), lane=tid&31, wave=tid>>5;
  int oi=__builtin_amdgcn_workgroup_id_x()*{group_size//32}+wave, channel=oi/256, feature=oi%256;
  float acc=0.0f;
  for (int r=lane; r<{6*m.spatial*m.spatial}; r+=32) {{
    float v=0.0f;
    int base=(feature*6+r/{m.spatial*m.spatial})*16384+channel*{m.spatial*m.spatial}+r%{m.spatial*m.spatial};
    {value}
    acc+=v;
  }}
  unsigned acc_bits=__builtin_bit_cast(unsigned, acc);
  if (lane==0) {{
    float total=acc;
    #pragma unroll
    for (int i=1; i<32; i++) total+=__builtin_bit_cast(float, __builtin_amdgcn_readlane(acc_bits, i));
    p0[channel*256+feature]={out.replace("acc", "total")};
  }}
}}'''

def _render_channel_reduce(m:ChannelReduceMatch, name:str) -> str:
  if _use_wave_reduce(m): return _render_wave_reduce(m, name)
  param_decls = ", ".join(f'{"half" if x.dtype == dtypes.half else "float"}* p{x.arg.slot}' for x in m.params)
  lines = [
    "#define half _Float16",
    "typedef half half4 __attribute__((ext_vector_type(4)));",
    "typedef float float4 __attribute__((ext_vector_type(4)));",
  ]
  lines += [
    f'extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(128, 128))) {name}({param_decls}) {{',
    "  int feature=__builtin_amdgcn_workgroup_id_x()*16+__builtin_amdgcn_workitem_id_y();",
    "  int channel=(__builtin_amdgcn_workgroup_id_y()*8+__builtin_amdgcn_workitem_id_x())*4;",
    "  float acc0=0.0f, acc1=0.0f, acc2=0.0f, acc3=0.0f;",
  ]
  if m.kind in ("activation", "centered"): lines.append("  float4 mean=*((float4*)(p2+channel));")
  lines += ["  for (int batch=0; batch<6; batch++) {", f"    for (int chunk=0; chunk<{m.spatial*m.spatial//16}; chunk++) {{"]
  for q in range(4):
    lines.append("      {")
    lines.append(f"      int base{q}=(feature*6+batch)*16384+(channel+{q})*{m.spatial*m.spatial}+chunk*16;")
    data_slot = 3 if m.kind == "scale" else 1
    data_type = "float4" if m.kind == "scale" else "half4"
    for i in range(4): lines.append(f"      {data_type} v{i}=*((({data_type}*)(p{data_slot}+base{q}+{i*4})));" )
    if m.kind == "activation":
      for i in range(4): lines.append(f"      float4 z{i}=*((float4*)(p4+base{q}+{i*4})); half4 g{i}=*((half4*)(p5+base{q}+{i*4}));")
      for i in range(4):
        lines.append(f"      half4 zh{i}=__builtin_convertvector(z{i},half4);")
        lines.append(f"      half4 sig{i}=(half4)1.0/((half4)1.0+__builtin_elementwise_exp2(zh{i}*(half)-2.4554669595930156));")
        lines.append(f"      half4 ag{i}=sig{i}*g{i}+(half)1.702*zh{i}*g{i}*sig{i}*((half)1.0-sig{i});")
      term = f"(((float){{v}}-mean.{ 'xyzw'[q]})*(float)ag{{i}}.{{lane}})"
      terms = []
      for i in range(4):
        for lane in "xyzw": terms.append(term.format(v=f"v{i}.{lane}", i=i, lane=lane))
      lines.append(f"      acc{q} += "+"+".join(terms)+";")
    elif m.kind == "centered":
      for i in range(4): lines.append(f"      float4 g{i}=*((float4*)(p4+base{q}+{i*4}));")
      terms = [f"(((float)v{i}.{lane}-mean.{'xyzw'[q]})*g{i}.{lane})" for i in range(4) for lane in "xyzw"]
      lines.append(f"      acc{q} += "+"+".join(terms)+";")
    else: lines.append(f"      acc{q} += "+_sum16("{v}")+";")
    lines.append("      }")
  lines += ["    }", "  }"]
  for q in range(4):
    out = f"p0[(channel+{q})*256+feature]"
    if m.kind in ("activation", "centered"): rhs = f"acc{q}*p3[channel+{q}]"
    else: rhs = f"-(acc{q}*p1[channel+{q}]*__builtin_elementwise_sqrt(1.0f/p2[channel+{q}]))"
    lines.append(f"  {out}={rhs};")
  lines.append("}")
  return "\n".join(lines)


def channel_reduce_program(ast:UOp, renderer:Renderer, compile_binary:bool) -> UOp|None:
  if (m:=_match_channel_reduce(ast, renderer.target.device, renderer.target.arch)) is None: return None
  name = f"channel_reduce_{m.kind}_{m.channels}_{m.spatial}"
  source = _render_channel_reduce(m, name)
  slots = tuple(x.arg.slot for x in m.params)
  elements, reduce_size = m.channels*256, 6*m.spatial*m.spatial
  estimates = Estimates(elements*reduce_size*(18 if m.kind == "activation" else 3),
                        elements*(reduce_size*(8 if m.kind == "activation" else 6)+4), sum(int(x.src[0].arg)*x.dtype.itemsize for x in m.params))
  sink = ast.replace(arg=replace(ast.arg, name=name, estimates=estimates))
  if _use_wave_reduce(m):
    group_size = _wave_group_size(m)
    global_size, local_size = (m.channels*256//(group_size//32), 1, 1), (group_size, 1, 1)
  else: global_size, local_size = (16, m.channels//32, 1), (8, 16, 1)
  info = ProgramInfo(name=name, global_size=global_size, local_size=local_size, globals=slots, outs=(0,), ins=slots[1:])
  src:tuple[UOp, ...] = (sink, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=source))
  if compile_binary: src += (UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source)),)
  return UOp(Ops.PROGRAM, src=src, arg=info)
