from collections import Counter
from dataclasses import dataclass, replace

from tinygrad.dtype import dtypes
from tinygrad.renderer import Estimates, Renderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, ProgramInfo, UOp, ssimplify


@dataclass(frozen=True)
class ChannelReduceMatch:
  kind: str
  params: tuple[UOp, ...]
  groups: int
  channels: int
  spatial: int

@dataclass(frozen=True)
class DualChannelReduceMatch:
  groups:int
  channels:int
  spatial:int

@dataclass(frozen=True)
class DualActivationReduceElementwiseMatch:
  groups:int
  channels:int
  spatial:int

@dataclass(frozen=True)
class ActivationVarGradElementwiseSumDMeanMatch:
  batch:int

@dataclass(frozen=True)
class DualMoments512Match:
  batch:int

@dataclass(frozen=True)
class DualBNGradDMean512Match:
  batch:int

@dataclass(frozen=True)
class Col2ImMatch:
  params: tuple[UOp, ...]
  fused_activation: bool
  batch: int
  channels: int
  spatial: int

@dataclass(frozen=True)
class Im2ColMatch:
  params: tuple[UOp, ...]
  batch: int
  channels: int
  spatial: int

def _match_moment_mean_512(ast:UOp, device:str, arch:str) -> int|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM), key=lambda u:u.arg.slot))
  reduce = next((u for u in ast.toposort() if u.op is Ops.REDUCE), None)
  if reduce is None or reduce.arg != (Ops.ADD,0) or len(reduce.src) != 4: return None
  batch, sy, sx = tuple(int(r.vmax)+1 for r in reduce.src[1:])
  if batch not in (1024,1280,1536) or (sy,sx) != (4,4) or tuple((u.arg.slot,u.dtype,u.max_numel()) for u in params) != \
     ((0,dtypes.float,512),(1,dtypes.half,batch*8192)): return None
  value = reduce.src[0]
  if value.op is not Ops.CAST or value.dtype != dtypes.float or value.src[0].op is not Ops.INDEX or value.src[0].src[0] is not params[1]: return None
  return batch if Counter(u.op for u in ast.toposort()) in (Counter({Ops.CONST:10,Ops.MUL:5,Ops.RANGE:4,Ops.ADD:4,Ops.PARAM:2,Ops.INDEX:2,
    Ops.CAST:1,Ops.REDUCE:1,Ops.STORE:1,Ops.END:1,Ops.SINK:1}), Counter({Ops.CONST:8,Ops.MUL:4,Ops.RANGE:4,Ops.ADD:3,Ops.PARAM:2,
    Ops.INDEX:2,Ops.CAST:1,Ops.REDUCE:1,Ops.STORE:1,Ops.END:1,Ops.SINK:1})) else None

def _match_moment_var_512(ast:UOp, device:str, arch:str) -> int|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM), key=lambda u:u.arg.slot))
  reduce = next((u for u in ast.toposort() if u.op is Ops.REDUCE), None)
  if reduce is None or reduce.arg != (Ops.ADD,0) or len(reduce.src) != 4: return None
  batch, sy, sx = tuple(int(r.vmax)+1 for r in reduce.src[1:])
  if batch not in (1024,1280,1536) or (sy,sx) != (4,4) or tuple((u.arg.slot,u.dtype,u.max_numel()) for u in params) != \
     ((0,dtypes.float,512),(1,dtypes.half,batch*8192),(2,dtypes.float,512)): return None
  value = reduce.src[0]
  if value.op is not Ops.MUL or value.src[0] is not value.src[1] or value.src[0].op is not Ops.CAST or \
     value.src[0].src[0].op is not Ops.INDEX or value.src[0].src[0].src[0] is not params[1]: return None
  return batch if Counter(u.op for u in ast.toposort()) in (Counter({Ops.CONST:12,Ops.MUL:8,Ops.ADD:6,Ops.RANGE:4,Ops.PARAM:3,Ops.INDEX:3,
    Ops.CAST:1,Ops.REDUCE:1,Ops.STORE:1,Ops.END:1,Ops.SINK:1}), Counter({Ops.CONST:10,Ops.MUL:7,Ops.ADD:5,Ops.RANGE:4,Ops.PARAM:3,
    Ops.INDEX:3,Ops.CAST:1,Ops.REDUCE:1,Ops.STORE:1,Ops.END:1,Ops.SINK:1})) else None

def moments_512_program(ast:UOp, renderer:Renderer, compile_binary:bool) -> UOp|None:
  if not isinstance(ast.tag,DualMoments512Match) or renderer.target.device != "AMD" or not renderer.target.arch.startswith("gfx11"): return None
  batch = ast.tag.batch
  name = f"channel_moments_{batch}_512_4_{ast.key.hex()[:8]}"
  source = f'''#define half _Float16
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(128,128))) {name}(
    float* p0, float* p1, half* p2) {{
  __attribute__((shared, aligned(32))) float partial_sum[128], partial_sq[128];
  int c=__builtin_amdgcn_workgroup_id_x(), tid=__builtin_amdgcn_workitem_id_x();
  float sum=0.0f, sq=0.0f;
  for (int q=0; q<{batch//8}; q++) {{
    int r=tid+q*128, idx=(r>>4)*8192+c*16+(r&15);
    float v=(float)p2[idx]; sum+=v; sq+=v*v;
  }}
  partial_sum[tid]=sum; partial_sq[tid]=sq;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
  for (int stride=64; stride; stride>>=1) {{
    if (tid<stride) {{ partial_sum[tid]+=partial_sum[tid+stride]; partial_sq[tid]+=partial_sq[tid+stride]; }}
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
  }}
  if (tid==0) {{ float mean=partial_sum[0]*{1/(batch*16)!r}f; p0[c]=mean;
    p1[c]=partial_sq[0]*{1/(batch*16)!r}f-mean*mean+1e-12f; }}
}}'''
  sink = ast.replace(arg=replace(ast.arg,name=name,estimates=Estimates(512*batch*16*2,batch*8192*2,512*128*8)))
  info = ProgramInfo(name=name,global_size=(512,1,1),local_size=(128,1,1),globals=(0,1,2),outs=(0,1),ins=(2,))
  src:tuple[UOp, ...] = (sink,UOp(Ops.LINEAR),UOp(Ops.SOURCE,arg=source))
  if compile_binary: src += (UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source)),)
  return UOp(Ops.PROGRAM,src=src,arg=info)

def _match_bn_var_512(ast:UOp, device:str, arch:str) -> int|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM),key=lambda u:u.arg.slot))
  reduce = next((u for u in ast.toposort() if u.op is Ops.REDUCE),None)
  if reduce is None or reduce.arg != (Ops.ADD,0) or len(reduce.src) != 4: return None
  batch, sy, sx = tuple(int(r.vmax)+1 for r in reduce.src[1:])
  signature = ((0,dtypes.float,512),(1,dtypes.float,512),(2,dtypes.half,batch*8192),(3,dtypes.float,512),
               (4,dtypes.float,512),(5,dtypes.float,batch*8192))
  return batch if batch in (1024,1280,1536) and (sy,sx) == (4,4) and tuple((u.arg.slot,u.dtype,u.max_numel()) for u in params) == signature and \
    Counter(u.op for u in ast.toposort()) == Counter({Ops.CONST:11,Ops.MUL:10,Ops.PARAM:6,Ops.INDEX:6,Ops.ADD:5,Ops.RANGE:4,
      Ops.RECIPROCAL:1,Ops.SQRT:1,Ops.CAST:1,Ops.REDUCE:1,Ops.STORE:1,Ops.END:1,Ops.SINK:1}) else None

def _match_bn_sum_512(ast:UOp, device:str, arch:str) -> int|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM),key=lambda u:u.arg.slot))
  reduce = next((u for u in ast.toposort() if u.op is Ops.REDUCE),None)
  if reduce is None or reduce.arg != (Ops.ADD,0) or len(reduce.src) != 4: return None
  batch, sy, sx = tuple(int(r.vmax)+1 for r in reduce.src[1:])
  return batch if batch in (1024,1280,1536) and (sy,sx) == (4,4) and tuple((u.arg.slot,u.dtype,u.max_numel()) for u in params) == \
    ((0,dtypes.float,512),(1,dtypes.float,batch*8192),(2,dtypes.float,512)) and \
    Counter(u.op for u in ast.toposort()) == Counter({Ops.CONST:10,Ops.MUL:5,Ops.ADD:5,Ops.RANGE:4,Ops.PARAM:3,Ops.INDEX:3,
      Ops.REDUCE:1,Ops.STORE:1,Ops.END:1,Ops.SINK:1}) else None

def _match_bn_dmean_512(ast:UOp, device:str, arch:str) -> int|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM),key=lambda u:u.arg.slot))
  reduce = next((u for u in ast.toposort() if u.op is Ops.REDUCE),None)
  if reduce is None or reduce.arg != (Ops.ADD,0) or len(reduce.src) != 4: return None
  batch, sy, sx = tuple(int(r.vmax)+1 for r in reduce.src[1:])
  if batch not in (1024,1280,1536) or (sy,sx) != (4,4) or tuple((u.arg.slot,u.dtype,u.max_numel()) for u in params) != \
     ((0,dtypes.float,512),(1,dtypes.float,512),(2,dtypes.float,512),(3,dtypes.float,512),
      (4,dtypes.float,512),(5,dtypes.float,batch*8192)): return None
  expected = {Ops.CONST:12,Ops.MUL:10,Ops.PARAM:6,Ops.INDEX:6,Ops.ADD:5,Ops.RANGE:4,Ops.RECIPROCAL:1,
              Ops.SQRT:1,Ops.REDUCE:1,Ops.STORE:1,Ops.END:1,Ops.SINK:1}
  return batch if Counter(u.op for u in ast.toposort()) == Counter(expected) else None

def bn_grad_512_program(ast:UOp, renderer:Renderer, compile_binary:bool) -> UOp|None:
  if not isinstance(ast.tag,DualBNGradDMean512Match) or renderer.target.device != "AMD" or \
     not renderer.target.arch.startswith("gfx11"): return None
  batch = ast.tag.batch
  name = f"channel_bn_grad_{batch}_512_4_dmean_{ast.key.hex()[:8]}"
  source = f'''#define half _Float16
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(128,128))) {name}(
    float* p0, float* p1, float* p2, float* p3, half* p4, float* p5, float* p6, float* p7, float* p8) {{
  __attribute__((shared, aligned(32))) float partial_cov[128], partial_sum[128];
  int c=__builtin_amdgcn_workgroup_id_x(), tid=__builtin_amdgcn_workitem_id_x();
  float cov=0.0f, sum=0.0f, mean=p5[c], weight=p6[c];
  for (int q=0; q<{batch//8}; q++) {{
    int r=tid+q*128, idx=(r>>4)*8192+c*16+(r&15); float grad=p7[idx];
    cov+=((float)p4[idx]-mean)*weight*grad; sum+=grad;
  }}
  partial_cov[tid]=cov; partial_sum[tid]=sum;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
  for (int stride=64; stride; stride>>=1) {{
    if (tid<stride) {{ partial_cov[tid]+=partial_cov[tid+stride]; partial_sum[tid]+=partial_sum[tid+stride]; }}
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
  }}
  if (tid==0) {{ float inv=1.0f/p3[c], vg=inv*__builtin_elementwise_sqrt(inv)*partial_cov[0]*-0.5f; p0[c]=vg;
    p1[c]=partial_sum[0]+0.018447889655172415f*p8[c];
    p2[c]=(vg*mean*-2.0f-weight*__builtin_elementwise_sqrt(inv)*partial_sum[0])*{1/(batch*16)!r}f; }}
}}'''
  sink = ast.replace(arg=replace(ast.arg,name=name,estimates=Estimates(512*batch*16*5,batch*8192*6,512*128*8)))
  info = ProgramInfo(name=name,global_size=(512,1,1),local_size=(128,1,1),globals=tuple(range(9)),outs=(0,1,2),ins=tuple(range(3,9)))
  src:tuple[UOp, ...] = (sink,UOp(Ops.LINEAR),UOp(Ops.SOURCE,arg=source))
  if compile_binary: src += (UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source)),)
  return UOp(Ops.PROGRAM,src=src,arg=info)

@dataclass(frozen=True)
class MaxPoolMatch:
  params: tuple[UOp, ...]
  batch: int
  channels: int
  spatial: int


def _match_activation_var_grad(ast:UOp, device:str, arch:str) -> int|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  end = ast.src[0]
  if end.op is not Ops.END or len(end.src) != 2 or end.src[0].op is not Ops.STORE: return None
  store, channel = end.src
  if channel.op is not Ops.RANGE or channel.arg[1] is not AxisType.LOOP or int(channel.vmax)+1 != 512: return None
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM), key=lambda u:u.arg.slot))
  signature = tuple((u.arg.slot, u.dtype, u.max_numel()) for u in params)
  if len(params) != 7 or params[2].max_numel() % 8192: return None
  batch = params[2].max_numel()//8192
  if batch not in (1024,1280,1536) or signature != ((0,dtypes.float,512), (1,dtypes.float,512), (2,dtypes.half,batch*8192),
      (3,dtypes.float,512), (4,dtypes.float,512), (5,dtypes.float,batch*8192), (6,dtypes.half,batch*8192)): return None
  expected_ops = {Ops.MUL:17, Ops.CONST:14, Ops.ADD:10, Ops.PARAM:7, Ops.INDEX:7, Ops.RANGE:4, Ops.CAST:3,
                  Ops.RECIPROCAL:2, Ops.SQRT:1, Ops.EXP2:1, Ops.REDUCE:1, Ops.STORE:1, Ops.END:1, Ops.SINK:1}
  if Counter(u.op for u in ast.toposort()) != Counter(expected_ops): return None
  reduce = next((u for u in ast.toposort() if u.op is Ops.REDUCE), None)
  if reduce is None or reduce.arg != (Ops.ADD, 0) or len(reduce.src) != 4 or \
     tuple(int(x.vmax)+1 for x in reduce.src[1:]) != (batch,4,4): return None
  return batch if store.src[0].src[0] is params[0] and ssimplify(store.src[0].src[1].get_idx()) is channel else None

def _match_activation_elementwise_512(ast:UOp, device:str, arch:str) -> int|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM), key=lambda u:u.arg.slot))
  if len(params) != 5 or params[0].max_numel() % 8192: return None
  batch = params[0].max_numel()//8192
  if batch not in (1024,1280,1536) or tuple((u.arg.slot,u.dtype,u.max_numel()) for u in params) != \
     ((0,dtypes.float,batch*8192),(1,dtypes.float,512),(2,dtypes.float,512),(3,dtypes.float,batch*8192),(4,dtypes.half,batch*8192)): return None
  expected_ops = {Ops.MUL:13,Ops.CONST:12,Ops.ADD:9,Ops.PARAM:5,Ops.INDEX:5,Ops.RANGE:4,Ops.RECIPROCAL:2,
                  Ops.CAST:2,Ops.SQRT:1,Ops.EXP2:1,Ops.STORE:1,Ops.END:1,Ops.SINK:1}
  return batch if Counter(u.op for u in ast.toposort()) == Counter(expected_ops) else None

def _match_activation_sum_512(ast:UOp, device:str, arch:str) -> int|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM),key=lambda u:u.arg.slot))
  if len(params) != 4 or params[1].max_numel() % 8192: return None
  batch = params[1].max_numel()//8192
  if batch not in (1024,1280,1536) or tuple((u.arg.slot,u.dtype,u.max_numel()) for u in params) != \
     ((0,dtypes.float,512),(1,dtypes.float,batch*8192),(2,dtypes.half,batch*8192),(3,dtypes.float,512)): return None
  reduce = next((u for u in ast.toposort() if u.op is Ops.REDUCE),None)
  if reduce is None or reduce.arg != (Ops.ADD,0) or len(reduce.src) != 4 or \
     tuple(int(r.vmax)+1 for r in reduce.src[1:]) != (batch,4,4): return None
  expected = {Ops.CONST:13,Ops.MUL:12,Ops.ADD:10,Ops.PARAM:4,Ops.RANGE:4,Ops.INDEX:4,Ops.CAST:2,
              Ops.EXP2:1,Ops.RECIPROCAL:1,Ops.REDUCE:1,Ops.STORE:1,Ops.END:1,Ops.SINK:1}
  return batch if Counter(u.op for u in ast.toposort()) == Counter(expected) else None

def _match_activation_dmean_512(ast:UOp, device:str, arch:str) -> int|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM),key=lambda u:u.arg.slot))
  if len(params) != 4 or params[3].max_numel() % 8192: return None
  batch = params[3].max_numel()//8192
  if batch not in (1024,1280,1536) or tuple((u.arg.slot,u.dtype,u.max_numel()) for u in params) != \
     ((0,dtypes.float,512),(1,dtypes.float,512),(2,dtypes.float,512),(3,dtypes.float,batch*8192)): return None
  reduce = next((u for u in ast.toposort() if u.op is Ops.REDUCE),None)
  if reduce is None or reduce.arg != (Ops.ADD,0) or len(reduce.src) != 4 or \
     tuple(int(r.vmax)+1 for r in reduce.src[1:]) != (batch,4,4): return None
  expected = {Ops.CONST:12,Ops.MUL:8,Ops.ADD:5,Ops.PARAM:4,Ops.RANGE:4,Ops.INDEX:4,
              Ops.REDUCE:1,Ops.STORE:1,Ops.END:1,Ops.SINK:1}
  return batch if Counter(u.op for u in ast.toposort()) == Counter(expected) else None

def activation_var_grad_program(ast:UOp, renderer:Renderer, compile_binary:bool) -> UOp|None:
  if not isinstance(ast.tag,ActivationVarGradElementwiseSumDMeanMatch) or \
     (batch:=_match_activation_var_grad(ast,renderer.target.device,renderer.target.arch)) is None: return None
  name = f"direct_activation_var_grad_{batch}_512_4_elementwise_sum_dmean"
  source = f'''#define half _Float16
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(128,128))) {name}(
    float* p0, float* p1, float* p2, float* p3, float* p4, half* p5, float* p6, float* p7,
    float* p8, half* p9, float* p10) {{
  __attribute__((shared, aligned(32))) float partial[128], partial_sum[128], partial_elem[128];
  int c=__builtin_amdgcn_workgroup_id_x(), tid=__builtin_amdgcn_workitem_id_x();
  float mean=p6[c], norm=p7[c]*__builtin_elementwise_sqrt(1.0f/p4[c]), acc=0.0f, sum=0.0f, elem_sum=0.0f;
  for (int q=0; q<{batch//8}; q++) {{
    int r=tid+q*128, idx=(r>>4)*8192+c*16+(r&15);
    half z=(half)p8[idx], sig=(half)1.0/((half)1.0+__builtin_elementwise_exp2(z*(half)-2.4554669595930156));
    half grad=sig*p9[idx]+(half)1.702*z*p9[idx]*sig*((half)1.0-sig);
    acc+=((float)p5[idx]-mean)*(float)grad;
    float elem=norm*(float)grad; p1[idx]=elem; sum+=(float)grad; elem_sum+=elem;
  }}
  partial[tid]=acc; partial_sum[tid]=sum; partial_elem[tid]=elem_sum;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
  for (int stride=64; stride; stride>>=1) {{
    if (tid<stride) {{ partial[tid]+=partial[tid+stride]; partial_sum[tid]+=partial_sum[tid+stride];
      partial_elem[tid]+=partial_elem[tid+stride]; }}
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
  }}
  if (tid==0) {{ float inv=1.0f/p4[c]; float vg=inv*__builtin_elementwise_sqrt(inv)*partial[0]*p7[c]*-0.5f; p0[c]=vg;
    p2[c]=partial_sum[0]+0.018447889655172415f*p10[c];
    p3[c]=(vg*p6[c]*-2.0f-partial_elem[0])*{1/(batch*16)!r}f; }}
}}'''
  sink = ast.replace(arg=replace(ast.arg,name=name,estimates=Estimates(512*batch*16*20,512*batch*16*12,512*128*4)))
  info = ProgramInfo(name=name,global_size=(512,1,1),local_size=(128,1,1),globals=tuple(range(11)),
                     outs=(0,1,2,3),ins=tuple(range(4,11)))
  src:tuple[UOp, ...] = (sink, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=source))
  if compile_binary: src += (UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source)),)
  return UOp(Ops.PROGRAM, src=src, arg=info)

@dataclass(frozen=True)
class MaxPoolBackwardMatch:
  params: tuple[UOp, ...]
  batch: int
  channels: int
  spatial: int

def _match_maxpool_backward(ast:UOp, device:str, arch:str) -> MaxPoolBackwardMatch|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1 or not isinstance(ast.arg, KernelInfo): return None
  end = ast.src[0]
  if end.op is not Ops.END or len(end.src) != 3 or end.src[0].op is not Ops.STORE: return None
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM), key=lambda u:u.arg.slot))
  if any(not u.src or u.src[0].op is not Ops.CONST or not isinstance(u.src[0].arg, int) for u in params): return None
  ranges = end.src[1:]
  if any(r.op is not Ops.RANGE or r.arg[1] is not AxisType.LOOP for r in ranges): return None
  position_size, channels = tuple(int(r.vmax)+1 for r in ranges)
  spatial = {64:32, 256:16, 512:8}.get(channels)
  if spatial is None or position_size % (spatial*spatial) or end.src[0].src[0].src[0].arg.slot != 0: return None
  batch = position_size//(spatial*spatial)
  if batch not in (1024, 1280, 1536): return None
  output_size, pooled_size = batch*channels*spatial*spatial, batch*channels*(spatial//2)**2
  signature = tuple((u.arg.slot, u.dtype, int(u.src[0].arg)) for u in params)
  expected_signature = ((0, dtypes.half, output_size), (1, dtypes.half, output_size),
                        (2, dtypes.half, pooled_size), (3, dtypes.half, pooled_size), (4, dtypes.half, pooled_size))
  if signature != expected_signature: return None
  expected_ops = {Ops.CONST:{32:19, 16:18, 8:19}[spatial], Ops.PARAM:5, Ops.RANGE:4, Ops.MUL:14, Ops.ADD:12,
                  Ops.INDEX:5, Ops.FLOORDIV:6, Ops.FLOORMOD:6, Ops.CMPLT:6, Ops.AND:10, Ops.WHERE:4,
                  Ops.CMPNE:2, Ops.CAST:3, Ops.RECIPROCAL:1, Ops.REDUCE:1, Ops.STORE:1, Ops.END:1, Ops.SINK:1}
  if Counter(u.op for u in ast.toposort()) != Counter(expected_ops): return None
  reduce = next((u for u in ast.toposort() if u.op is Ops.REDUCE), None)
  if reduce is None or reduce.arg != (Ops.ADD, 0) or tuple(int(r.vmax)+1 for r in reduce.src[1:]) != (3, 3): return None
  return MaxPoolBackwardMatch(params, batch, channels, spatial)

def maxpool_backward_program(ast:UOp, renderer:Renderer, compile_binary:bool) -> UOp|None:
  if (m:=_match_maxpool_backward(ast, renderer.target.device, renderer.target.arch)) is None: return None
  name = f"direct_maxpool_backward_{m.channels}_{m.spatial}"
  pooled_spatial, pooled_size, block = m.spatial//2, m.batch*m.channels*(m.spatial//2)**2, min(m.spatial, 16)
  declarations = "half* p0, half* p1, half* p2, half* p3, half* p4"
  source = f'''#define half _Float16
typedef half half2 __attribute__((ext_vector_type(2)));
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size({block//2*16},{block//2*16}))) {name}(
    {declarations}) {{
  __attribute__((shared, aligned(32))) half tile[{2*block*16}];
  int gx=__builtin_amdgcn_workgroup_id_x(), b=__builtin_amdgcn_workgroup_id_z();
  int px=__builtin_amdgcn_workitem_id_x(), lc=__builtin_amdgcn_workitem_id_y(), cg=__builtin_amdgcn_workgroup_id_y();
  int py=gx/{m.spatial//block}, xb=(gx%{m.spatial//block})*{block}, y=py*2, x=xb+px*2, c=cg*16+lc;
  int ii=(b*{m.channels}+c)*{m.spatial*m.spatial}+y*{m.spatial}+x;
  int pi=(b*{m.channels}+c)*{pooled_spatial*pooled_spatial}+py*{pooled_spatial}+xb/2+px;
  half mx=p2[pi];
  half scale=(half)1.0/p3[pi]*p4[pi];
  half2 row0=*((half2*)(p1+ii)), row1=*((half2*)(p1+ii+{m.spatial}));
  tile[(px*2)*16+lc]=(half)(row0.x==mx)*scale;
  tile[(px*2+1)*16+lc]=(half)(row0.y==mx)*scale;
  tile[({block}+px*2)*16+lc]=(half)(row1.x==mx)*scale;
  tile[({block}+px*2+1)*16+lc]=(half)(row1.y==mx)*scale;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
  int tid=lc*{block//2}+px;
  #pragma unroll
  for (int q=0; q<4; q++) {{
    int e=tid+q*{block//2*16}, pos=e/16, oc=cg*16+e%16;
    int oy=y+pos/{block}, ox=xb+pos%{block};
    p0[((b*{m.spatial}+oy)*{m.spatial}+ox)*{m.channels}+oc]=tile[e];
  }}
}}'''
  output_size = pooled_size*4
  sink = ast.replace(arg=replace(ast.arg, name=name, estimates=Estimates(output_size*2, pooled_size*22, 0)))
  slots = tuple(u.arg.slot for u in m.params)
  info = ProgramInfo(name=name, global_size=(m.spatial//block*pooled_spatial, m.channels//16, m.batch), local_size=(block//2, 16, 1),
                     globals=slots, outs=(0,), ins=slots[1:])
  src:tuple[UOp, ...] = (sink, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=source))
  if compile_binary: src += (UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source)),)
  return UOp(Ops.PROGRAM, src=src, arg=info)

def _match_maxpool(ast:UOp, device:str, arch:str) -> MaxPoolMatch|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 2 or not isinstance(ast.arg, KernelInfo): return None
  if any(end.op is not Ops.END or len(end.src) != 5 or end.src[0].op is not Ops.STORE for end in ast.src): return None
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM), key=lambda u:u.arg.slot))
  if any(not u.src or u.src[0].op is not Ops.CONST or not isinstance(u.src[0].arg, int) for u in params): return None
  loop_ranges = ast.src[0].src[1:]
  if any(r.op is not Ops.RANGE or r.arg[1] is not AxisType.LOOP for r in loop_ranges): return None
  batch, channels, sy, sx = tuple(int(r.vmax)+1 for r in loop_ranges)
  if batch not in (1024, 1280, 1536) or sy != sx or (channels,sy) not in ((64,16),(256,8),(512,4)) or ast.src[1].src[1:] != loop_ranges: return None
  spatial = sy
  out_size = batch*channels*spatial*spatial
  signature = tuple((u.arg.slot, u.dtype, int(u.src[0].arg)) for u in params)
  if signature != ((0,dtypes.half,out_size),(1,dtypes.half,out_size*4),(2,dtypes.half,out_size)): return None
  if tuple(end.src[0].src[0].src[0].arg.slot for end in ast.src) != (0, 2): return None
  expected_ops = {Ops.CONST:13 if spatial == 8 else 14, Ops.MUL:9, Ops.ADD:9, Ops.RANGE:6, Ops.PARAM:3,
                  Ops.INDEX:3, Ops.REDUCE:2, Ops.STORE:2, Ops.END:2, Ops.CMPNE:2, Ops.CAST:1, Ops.SINK:1}
  if Counter(u.op for u in ast.toposort()) != Counter(expected_ops): return None
  return MaxPoolMatch(params, batch, channels, spatial)

def maxpool_program(ast:UOp, renderer:Renderer, compile_binary:bool) -> UOp|None:
  if (m:=_match_maxpool(ast, renderer.target.device, renderer.target.arch)) is None: return None
  name = f"direct_maxpool_{m.channels}_{m.spatial}"
  out_size, in_size = m.batch*m.channels*m.spatial*m.spatial, m.batch*m.channels*m.spatial*m.spatial*4
  source = f'''#define half _Float16
typedef half half2 __attribute__((ext_vector_type(2)));
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1,128))) {name}(half* p0, half* p1, half* p2) {{
  int group=__builtin_amdgcn_workgroup_id_x(), lx=__builtin_amdgcn_workitem_id_x();
  int ly=__builtin_amdgcn_workitem_id_y(), ii=group*512+lx*{m.spatial*4}+ly*2;
  half2 row0=*((half2*)(p1+ii)), row1=*((half2*)(p1+ii+{m.spatial*2}));
  int oi=group*128+lx*{m.spatial}+ly;
  half mx01=row0.x<row0.y?row0.y:row0.x, mx012=mx01<row1.x?row1.x:mx01;
  half mx=mx012<row1.y?row1.y:mx012;
  p0[oi]=mx;
  p2[oi]=(half)(row0.x==mx)+(half)(row0.y==mx)+(half)(row1.x==mx)+(half)(row1.y==mx);
}}'''
  sink = ast.replace(arg=replace(ast.arg, name=name, estimates=Estimates(out_size*3, in_size*2+out_size*4, 0)))
  slots = tuple(u.arg.slot for u in m.params)
  info = ProgramInfo(name=name, global_size=(out_size//128, 1, 1), local_size=(128//m.spatial, m.spatial, 1),
                     globals=slots, outs=(0, 2), ins=(1,))
  src:tuple[UOp, ...] = (sink, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=source))
  if compile_binary: src += (UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source)),)
  return UOp(Ops.PROGRAM, src=src, arg=info)

def _match_im2col(ast:UOp, device:str, arch:str) -> Im2ColMatch|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1 or not isinstance(ast.arg, KernelInfo): return None
  end = ast.src[0]
  if end.op is not Ops.END or len(end.src) != 3 or end.src[0].op is not Ops.STORE: return None
  store, position, patch = end.src
  if any(r.op is not Ops.RANGE or r.arg[1] is not AxisType.LOOP for r in (position, patch)): return None
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM), key=lambda u:u.arg.slot))
  if any(not u.src or u.src[0].op is not Ops.CONST or not isinstance(u.src[0].arg, int) for u in params): return None
  patch_size, position_size = int(patch.vmax)+1, int(position.vmax)+1
  if patch_size % 9: return None
  channels = patch_size//9
  spatial = {32:32, 64:16, 256:8, 512:4}.get(channels)
  if spatial is None or position_size % (spatial*spatial): return None
  batch = position_size//(spatial*spatial)
  if batch not in (1024, 1280, 1536): return None
  input_size = batch*channels*spatial*spatial
  signature = tuple((u.arg.slot, u.dtype, int(u.src[0].arg)) for u in params)
  if signature != ((0,dtypes.half,input_size*9),(1,dtypes.half,input_size)): return None
  if (int(position.vmax)+1, int(patch.vmax)+1) != (batch*spatial*spatial, channels*9): return None
  if ssimplify(store.src[0].src[1].get_idx()) is not ssimplify(position*channels*9+patch): return None
  loads = [u for u in store.src[1].toposort() if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and u.src[0].arg.slot == 1]
  expected_idx = ssimplify(position//(spatial*spatial)*(channels*spatial*spatial) + patch//9*(spatial*spatial) +
                           (position//spatial%spatial+patch//3%3)*spatial + (position%spatial+patch%3)-spatial-1)
  if len(loads) != 1 or ssimplify(loads[0].src[1].get_idx()) is not expected_idx: return None
  expected_ops = {Ops.CONST:15 if spatial != 8 else 14, Ops.ADD:7, Ops.MUL:4, Ops.FLOORDIV:4, Ops.FLOORMOD:4,
                  Ops.CMPLT:4, Ops.AND:3, Ops.PARAM:2, Ops.RANGE:2, Ops.INDEX:2, Ops.CMPNE:2, Ops.WHERE:2,
                  Ops.STORE:1, Ops.END:1, Ops.SINK:1}
  if Counter(u.op for u in ast.toposort()) != Counter(expected_ops): return None
  return Im2ColMatch(params, batch, channels, spatial)

def im2col_program(ast:UOp, renderer:Renderer, compile_binary:bool) -> UOp|None:
  if (m:=_match_im2col(ast, renderer.target.device, renderer.target.arch)) is None: return None
  name = f"direct_im2col_{m.channels}_{m.spatial}"
  xblock = min(8, m.spatial)
  source = f'''#define half _Float16
typedef half half8 __attribute__((ext_vector_type(8)));
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size({xblock*16},{xblock*16}))) {name}(half* p0, half* p1) {{
  int gx=__builtin_amdgcn_workgroup_id_x(), y=__builtin_amdgcn_workgroup_id_y();
  int b=__builtin_amdgcn_workgroup_id_z(), lx=__builtin_amdgcn_workitem_id_x();
  int lc=__builtin_amdgcn_workitem_id_y(), x=(gx%{m.spatial//xblock})*{xblock}+lx;
  int c=(gx/{m.spatial//xblock})*16+lc;
  int ii=((b*{m.channels}+c)*{m.spatial}+y)*{m.spatial}+x;
  int base=((b*{m.spatial*m.spatial}+y*{m.spatial}+x)*{m.channels}+c)*9;
  bool top=y>0, bottom=y<{m.spatial-1}, left=x>0, right=x<{m.spatial-1};
  half v0=(top&&left)?p1[ii-{m.spatial+1}]:(half)0, v1=top?p1[ii-{m.spatial}]:(half)0;
  half v2=(top&&right)?p1[ii-{m.spatial-1}]:(half)0, v3=left?p1[ii-1]:(half)0, v4=p1[ii];
  half v5=right?p1[ii+1]:(half)0, v6=(bottom&&left)?p1[ii+{m.spatial-1}]:(half)0;
  half v7=bottom?p1[ii+{m.spatial}]:(half)0, v8=(bottom&&right)?p1[ii+{m.spatial+1}]:(half)0;
  *((half8*)(p0+base))=(half8){{v0,v1,v2,v3,v4,v5,v6,v7}}; p0[base+8]=v8;
}}'''
  total = m.batch*m.channels*m.spatial*m.spatial
  sink = ast.replace(arg=replace(ast.arg, name=name, estimates=Estimates(0, total*20, 0)))
  slots = tuple(u.arg.slot for u in m.params)
  info = ProgramInfo(name=name, global_size=(m.spatial//xblock*m.channels//16,m.spatial,m.batch), local_size=(xblock,16,1),
                     globals=slots, outs=(0,), ins=(1,))
  src:tuple[UOp, ...] = (sink, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=source))
  if compile_binary: src += (UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source)),)
  return UOp(Ops.PROGRAM, src=src, arg=info)

def _match_col2im(ast:UOp, device:str, arch:str) -> Col2ImMatch|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1 or not isinstance(ast.arg, KernelInfo): return None
  end = ast.src[0]
  if end.op is not Ops.END or len(end.src) != 5 or end.src[0].op is not Ops.STORE: return None
  store, batch, channel, y, x = end.src
  if any(r.op is not Ops.RANGE or r.arg[1] is not AxisType.LOOP for r in (batch, channel, y, x)): return None
  batch_size, channels, sy, sx = tuple(int(r.vmax)+1 for r in (batch, channel, y, x))
  if batch_size not in (1024,1280,1536) or (channels, sy, sx) not in ((64, 16, 16), (256, 8, 8), (512, 4, 4)): return None
  reduces = [u for u in store.src[1].toposort() if u.op is Ops.REDUCE]
  if len(reduces) != 1 or reduces[0].arg != (Ops.ADD, 0) or \
     tuple(int(r.vmax)+1 for r in reduces[0].src[1:]) != (4, 4): return None
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM), key=lambda u:u.arg.slot))
  signature = tuple((u.arg.slot, u.dtype, int(u.src[0].arg)) for u in params)
  output_size, patch_size = batch_size*channels*sy*sx, batch_size*channels*sy*sx*9
  plain = ((0, dtypes.half, output_size), (1, dtypes.half, patch_size))
  fused = ((0, dtypes.float, output_size), (1, dtypes.float, output_size),
           (2, dtypes.half, output_size), (3, dtypes.half, patch_size))
  if signature not in (plain, fused): return None
  expected_ops = ({Ops.CONST:20, Ops.ADD:13, Ops.MUL:11, Ops.AND:8, Ops.RANGE:6, Ops.CMPLT:4, Ops.PARAM:2,
                   Ops.INDEX:2, Ops.FLOORMOD:2, Ops.FLOORDIV:2, Ops.WHERE:2, Ops.CAST:2, Ops.REDUCE:1,
                   Ops.STORE:1, Ops.END:1, Ops.SINK:1} if signature == plain else
                  {Ops.CONST:22 if sy == 4 else 23, Ops.ADD:19, Ops.MUL:18, Ops.AND:8, Ops.RANGE:6, Ops.PARAM:4, Ops.INDEX:4,
                   Ops.CAST:4, Ops.CMPLT:4, Ops.FLOORMOD:2, Ops.FLOORDIV:2, Ops.WHERE:2, Ops.EXP2:1,
                   Ops.RECIPROCAL:1, Ops.REDUCE:1, Ops.STORE:1, Ops.END:1, Ops.SINK:1})
  if Counter(u.op for u in ast.toposort()) != Counter(expected_ops): return None
  return Col2ImMatch(params, signature == fused, batch_size, channels, sy)

def col2im_program(ast:UOp, renderer:Renderer, compile_binary:bool) -> UOp|None:
  if (m:=_match_col2im(ast, renderer.target.device, renderer.target.arch)) is None: return None
  name = f"direct_col2im_{m.channels}_{m.spatial}{'_activation' if m.fused_activation else ''}"
  declarations = ("float* p0, float* p1, half* p2, half* p3" if m.fused_activation else "half* p0, half* p1")
  patch_slot = 3 if m.fused_activation else 1
  if m.fused_activation:
    output = """half grad=p2[oi]+(half)acc, z=(half)p1[oi];
  half sig=(half)1.0/((half)1.0+__builtin_elementwise_exp2(z*(half)-2.4554669595930156));
  p0[oi]=(float)(sig*grad+(half)1.702*z*grad*sig*((half)1.0-sig));"""
  else: output = "p0[oi]=(half)acc;"
  source = f'''#define half _Float16
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(256,256))) {name}({declarations}) {{
  int tile=__builtin_amdgcn_workgroup_id_x(), cg=__builtin_amdgcn_workgroup_id_y();
  int b=__builtin_amdgcn_workgroup_id_z(), c=cg*16+__builtin_amdgcn_workitem_id_x();
  int y=(tile%{m.spatial//4})*4+__builtin_amdgcn_workitem_id_y();
  int x=(tile/{m.spatial//4})*4+__builtin_amdgcn_workitem_id_z();
  float acc=0.0f;
  #pragma unroll
  for (int ky=0; ky<3; ky++) {{
    int oy=y+1-ky;
    #pragma unroll
    for (int kx=0; kx<3; kx++) {{
      int ox=x+1-kx;
      if (oy>=0 && oy<{m.spatial} && ox>=0 && ox<{m.spatial})
        acc+=(float)p{patch_slot}[((((b*{m.spatial}+oy)*{m.spatial}+ox)*{m.channels}+c)*9+ky*3+kx)];
    }}
  }}
  int oi=((b*{m.channels}+c)*{m.spatial}+y)*{m.spatial}+x;
  {output}
}}'''
  output_size = m.batch*m.channels*m.spatial*m.spatial
  sink = ast.replace(arg=replace(ast.arg, name=name, estimates=Estimates(output_size*18, output_size*20, 0)))
  slots = tuple(u.arg.slot for u in m.params)
  info = ProgramInfo(name=name, global_size=((m.spatial//4)**2, m.channels//16, m.batch), local_size=(16, 4, 4),
                     globals=slots, outs=(0,), ins=slots[1:])
  src:tuple[UOp, ...] = (sink, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=source))
  if compile_binary: src += (UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source)),)
  return UOp(Ops.PROGRAM, src=src, arg=info)


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
  groups = int(batch.vmax)+1
  if groups not in (4,5,6) or int(rx.vmax)+1 != spatial or channels*spatial*spatial != 16384: return None

  out_idx = ssimplify(channel*features+feature)
  data_idx = ssimplify((feature*groups+batch)*16384+channel*spatial*spatial+ry*spatial+rx)
  if ssimplify(store.src[0].src[1].get_idx()) is not out_idx: return None
  params = tuple(sorted((x for x in ast.toposort() if x.op is Ops.PARAM), key=lambda x:x.arg.slot))
  total = features*groups*16384
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
  return ChannelReduceMatch(kind, params, groups, channels, spatial)

def _match_activation_sum(ast:UOp, device:str, arch:str) -> tuple[int, int, int]|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  end = ast.src[0]
  if end.op is not Ops.END or len(end.src) != 3 or end.src[0].op is not Ops.STORE: return None
  store, channel, feature = end.src
  if any(x.op is not Ops.RANGE or x.arg[1] is not AxisType.LOOP for x in (channel,feature)): return None
  channels = int(channel.vmax)+1
  if channels not in (64,256) or int(feature.vmax)+1 != 256: return None
  spatial = 16 if channels == 64 else 8
  reduce = store.src[1]
  if reduce.op is not Ops.REDUCE or reduce.arg != (Ops.ADD,0) or len(reduce.src) != 4: return None
  groups, rsy, rsx = tuple(int(x.vmax)+1 for x in reduce.src[1:])
  if groups not in (4,5,6) or (rsy,rsx) != (spatial,spatial): return None
  total = 256*groups*16384
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM), key=lambda u:u.arg.slot))
  if tuple((u.arg.slot,u.dtype,u.max_numel()) for u in params) != \
     ((0,dtypes.float,channels*256), (1,dtypes.float,total), (2,dtypes.half,total)): return None
  batch, y, x = reduce.src[1:]
  out_idx = ssimplify(channel*256+feature)
  data_idx = ssimplify((feature*groups+batch)*16384+channel*spatial*spatial+y*spatial+x)
  if ssimplify(store.src[0].src[1].get_idx()) is not out_idx: return None
  indexes = [(u.src[0].arg.slot,ssimplify(u.src[1].get_idx())) for u in reduce.src[0].toposort() if u.op is Ops.INDEX]
  if len(indexes) != 2 or any(slot != expected or idx is not data_idx for (slot,idx),expected in zip(indexes,(1,2))): return None
  expected_ops = ({Ops.MUL:15,Ops.CONST:13} if channels == 256 else {Ops.MUL:14,Ops.CONST:12}) | \
    {Ops.ADD:13,Ops.RANGE:5,Ops.PARAM:3,Ops.INDEX:3,Ops.CAST:2,Ops.EXP2:1,Ops.RECIPROCAL:1,
     Ops.REDUCE:1,Ops.STORE:1,Ops.END:1,Ops.SINK:1}
  if Counter(u.op for u in ast.toposort()) != Counter(expected_ops): return None
  return channels, spatial, groups

def _match_activation_elementwise(ast:UOp, device:str, arch:str) -> tuple[int, int, int]|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  end = ast.src[0]
  if end.op is not Ops.END or len(end.src) != 5 or end.src[0].op is not Ops.STORE: return None
  store, batch, channel, y, x = end.src
  if any(r.op is not Ops.RANGE or r.arg[1] is not AxisType.LOOP for r in (batch,channel,y,x)): return None
  batch_size, channels, sy, sx = tuple(int(r.vmax)+1 for r in (batch,channel,y,x))
  if batch_size not in (1024,1280,1536) or sy != sx or (channels,sy) not in ((64,16),(256,8)): return None
  total = batch_size*channels*sy*sx
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM),key=lambda u:u.arg.slot))
  if tuple((u.arg.slot,u.dtype,u.max_numel()) for u in params) != \
     ((0,dtypes.float,total),(1,dtypes.float,channels),(2,dtypes.float,channels),
      (3,dtypes.float,total),(4,dtypes.half,total)): return None
  expected = {Ops.MUL:13,Ops.CONST:12,Ops.ADD:9,Ops.PARAM:5,Ops.INDEX:5,Ops.RANGE:4,Ops.RECIPROCAL:2,
              Ops.CAST:2,Ops.SQRT:1,Ops.EXP2:1,Ops.STORE:1,Ops.END:1,Ops.SINK:1}
  if Counter(u.op for u in ast.toposort()) != Counter(expected): return None
  flat = ssimplify(batch*(channels*sy*sx)+channel*(sy*sx)+y*sy+x)
  return (channels,sy,batch_size//256) if store.src[0].op is Ops.INDEX and ssimplify(store.src[0].src[1].get_idx()) is flat else None


def channel_reduce_program(ast:UOp, renderer:Renderer, compile_binary:bool) -> UOp|None:
  if isinstance(am:=ast.tag,DualActivationReduceElementwiseMatch) and renderer.target.device == "AMD" and \
     renderer.target.arch.startswith("gfx11"):
    name = f"channel_reduce_activation_sum_{am.groups}_{am.channels}_{am.spatial}_elementwise"
    source = f'''#define half _Float16
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1024,1024))) {name}(
    float* p0, float* p1, float* p2, half* p3, float* p4, float* p5, float* p6, float* p7, half* p8) {{
  int tid=__builtin_amdgcn_workitem_id_x(), lane=tid&31, wave=tid>>5;
  int oi=__builtin_amdgcn_workgroup_id_x()*32+wave, channel=oi/256, feature=oi%256;
  float centered=0.0f, plain=0.0f, mean=p4[channel];
  for (int r=lane; r<{am.groups*am.spatial*am.spatial}; r+=32) {{
    int base=(feature*{am.groups}+r/{am.spatial*am.spatial})*16384+channel*{am.spatial*am.spatial}+r%{am.spatial*am.spatial};
    half z=(half)p7[base], grad=p8[base];
    half sig=(half)1.0/((half)1.0+__builtin_elementwise_exp2(z*(half)-2.4554669595930156));
    float ag=(float)(sig*grad+(half)1.702*z*grad*sig*((half)1.0-sig));
    centered+=((float)p3[base]-mean)*ag;
    plain+=ag;
    p2[base]=p5[channel]*(__builtin_elementwise_sqrt(1.0f/p6[channel])*ag);
  }}
  unsigned centered_bits=__builtin_bit_cast(unsigned,centered), plain_bits=__builtin_bit_cast(unsigned,plain);
  if (lane==0) {{
    #pragma unroll
    for (int i=1; i<32; i++) {{
      centered+=__builtin_bit_cast(float,__builtin_amdgcn_readlane(centered_bits,i));
      plain+=__builtin_bit_cast(float,__builtin_amdgcn_readlane(plain_bits,i));
    }}
    p0[oi]=centered*p5[channel];
    p1[oi]=plain;
  }}
}}'''
    elements, reduce_size = am.channels*256, am.groups*am.spatial*am.spatial
    sink = ast.replace(arg=replace(ast.arg, name=name, estimates=Estimates(elements*reduce_size*18, elements*reduce_size*8, 0)))
    slots = tuple(range(9))
    info = ProgramInfo(name=name, global_size=(elements//32,1,1), local_size=(1024,1,1), globals=slots,outs=(0,1,2),ins=slots[3:])
    src:tuple[UOp, ...] = (sink,UOp(Ops.LINEAR),UOp(Ops.SOURCE,arg=source))
    if compile_binary: src += (UOp(Ops.BINARY,arg=renderer.compiler.compile_cached(source)),)
    return UOp(Ops.PROGRAM,src=src,arg=info)
  if isinstance(dm:=ast.tag, DualChannelReduceMatch) and renderer.target.device == "AMD" and renderer.target.arch.startswith("gfx11"):
    name = f"channel_reduce_centered_scale_{dm.groups}_{dm.channels}_{dm.spatial}"
    source = f'''#define half _Float16
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1024,1024))) {name}(
    float* p0, float* p1, half* p2, float* p3, float* p4, float* p5, float* p6, float* p7) {{
  int tid=__builtin_amdgcn_workitem_id_x(), lane=tid&31, wave=tid>>5;
  int oi=__builtin_amdgcn_workgroup_id_x()*32+wave, channel=oi/256, feature=oi%256;
  float centered=0.0f, scale=0.0f, mean=p3[channel];
  for (int r=lane; r<{dm.groups*dm.spatial*dm.spatial}; r+=32) {{
    int base=(feature*{dm.groups}+r/{dm.spatial*dm.spatial})*16384+channel*{dm.spatial*dm.spatial}+r%{dm.spatial*dm.spatial};
    float grad=p7[base];
    centered+=((float)p2[base]-mean)*grad;
    scale+=grad;
  }}
  unsigned centered_bits=__builtin_bit_cast(unsigned, centered), scale_bits=__builtin_bit_cast(unsigned, scale);
  if (lane==0) {{
    #pragma unroll
    for (int i=1; i<32; i++) {{
      centered+=__builtin_bit_cast(float, __builtin_amdgcn_readlane(centered_bits, i));
      scale+=__builtin_bit_cast(float, __builtin_amdgcn_readlane(scale_bits, i));
    }}
    p0[oi]=centered*p4[channel];
    p1[oi]=-(scale*p5[channel]*__builtin_elementwise_sqrt(1.0f/p6[channel]));
  }}
}}'''
    elements, reduce_size = dm.channels*256, dm.groups*dm.spatial*dm.spatial
    sink = ast.replace(arg=replace(ast.arg, name=name, estimates=Estimates(elements*reduce_size*7, elements*reduce_size*8, 0)))
    info = ProgramInfo(name=name, global_size=(elements//32,1,1), local_size=(1024,1,1), globals=tuple(range(8)),
                       outs=(0,1), ins=(2,3,4,5,6,7))
    src = (sink, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=source))
    if compile_binary: src += (UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source)),)
    return UOp(Ops.PROGRAM, src=src, arg=info)
  return None
