from collections import Counter
from dataclasses import dataclass, replace
from math import prod
from tinygrad.dtype import dtypes
from tinygrad.renderer import Estimates, Renderer
from tinygrad.uop.ops import AxisType, Ops, ProgramInfo, UOp, ssimplify

WMMA_M = WMMA_N = WMMA_K = 16
BLOCK_M = BLOCK_N = 128
BLOCK_K = 32
THREADS = 128

@dataclass(frozen=True)
class GemmMatch:
  c:UOp
  a:UOp
  b:UOp
  m:int
  n:int
  k:int
  old:UOp|None = None
  scale:float = 0.0
  a_kxm:bool = False
  b_kxn:bool = False

@dataclass(frozen=True)
class BatchedGemmMatch:
  c:UOp
  a:UOp
  b:UOp
  m:int
  n:int
  k:int
  batch:int

@dataclass(frozen=True)
class DirectConvBwdActivationMatch:
  m:int
  cin:int
  cout:int
  spatial:int
  residual:bool = False

@dataclass(frozen=True)
class GemmOutputNCHW:
  spatial:int

@dataclass(frozen=True)
class PartialWeightGradMatch:
  c:UOp
  a:UOp
  b:UOp

def _match_partial_weight_grad(ast:UOp, device:str, arch:str) -> PartialWeightGradMatch|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  end = ast.src[0]
  if end.op is not Ops.END or len(end.src) != 4 or end.src[0].op is not Ops.STORE: return None
  store, m, n, batch = end.src
  if any(x.op is not Ops.RANGE or x.arg[1] is not AxisType.LOOP for x in (m, n, batch)) or \
     tuple(int(x.vmax)+1 for x in (m, n, batch)) != (32, 12, 256): return None
  params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM), key=lambda u:u.arg.slot))
  if any(not u.src or u.src[0].op is not Ops.CONST or not isinstance(u.src[0].arg, int) for u in params): return None
  if tuple((u.arg.slot, u.dtype, int(u.src[0].arg)) for u in params) != \
     ((0, dtypes.float, 98304), (1, dtypes.half, 18874368), (2, dtypes.float, 50331648)): return None
  expected_ops = {Ops.CONST:14, Ops.ADD:14, Ops.MUL:13, Ops.RANGE:6, Ops.PARAM:3, Ops.INDEX:3,
                  Ops.CAST:2, Ops.REDUCE:1, Ops.STORE:1, Ops.END:1, Ops.SINK:1}
  if Counter(u.op for u in ast.toposort()) != Counter(expected_ops): return None
  reduce = next((u for u in ast.toposort() if u.op is Ops.REDUCE), None)
  if reduce is None or reduce.arg != (Ops.ADD, 0) or tuple(int(r.vmax)+1 for r in reduce.src[1:]) != (6, 32, 32): return None
  if store.src[0].src[0] is not params[0] or ssimplify(store.src[0].src[1].get_idx()) is not ssimplify(m*3072+n*256+batch): return None
  return PartialWeightGradMatch(params[0], params[1], params[2])

def partial_weight_grad_program(ast:UOp, renderer:Renderer, compile_binary:bool) -> UOp|None:
  if _match_partial_weight_grad(ast, renderer.target.device, renderer.target.arch) is None: return None
  name = "coop_partial_weight_grad_32_12_6144_256"
  source = f'''#define half _Float16
typedef half half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef float float16 __attribute__((ext_vector_type(16)));
#define WMMA __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(64,64))) {name}(
    float* p0, half* p1, float* p2) {{
  __attribute__((shared, aligned(32))) half As[32*48], Bs[16*48];
  int batch=__builtin_amdgcn_workgroup_id_x(), tid=__builtin_amdgcn_workitem_id_x();
  int wave=tid>>5, lane=tid&31, row=lane&15, halfrow=lane>>4;
  float8 c={{}};
  for (int kt=0; kt<192; kt++) {{
    int kk=kt*32, rb=kk>>10, sp=kk&1023;
    if (tid<32) {{
      long ai=((long)(batch*6+rb)*32+tid)*1024+sp;
      *((half16*)(As+tid*48))=__builtin_convertvector(*((float16*)(p2+ai)),half16);
      *((half16*)(As+tid*48+16))=__builtin_convertvector(*((float16*)(p2+ai+16)),half16);
    }}
    if (tid<16) {{
      half16 v0=(half16){{0}}, v1=(half16){{0}};
      if (tid<12) {{
        long bi=((long)(batch*6+rb)*12+tid)*1024+sp;
        v0=*((half16*)(p1+bi)); v1=*((half16*)(p1+bi+16));
      }}
      *((half16*)(Bs+tid*48))=v0; *((half16*)(Bs+tid*48+16))=v1;
    }}
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
    #pragma unroll
    for (int ki=0; ki<2; ki++) {{
      half16 av=*((half16*)(As+(wave*16+row)*48+ki*16));
      half16 bv=*((half16*)(Bs+row*48+ki*16));
      c=WMMA(av,bv,c);
    }}
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
  }}
  if (row<12) {{
    #pragma unroll
    for (int e=0; e<8; e++) p0[((wave*16+e*2+halfrow)*12+row)*256+batch]=c[e];
  }}
}}'''
  sink = ast.replace(arg=replace(ast.arg, name=name, estimates=Estimates(2*256*32*12*6144, 0, 0)))
  info = ProgramInfo(name=name, global_size=(256, 1, 1), local_size=(64, 1, 1), globals=(0, 1, 2), outs=(0,), ins=(1, 2))
  src:tuple[UOp, ...] = (sink, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=source))
  if compile_binary: src += (UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source)),)
  return UOp(Ops.PROGRAM, src=src, arg=info)

def _match_gemm(ast:UOp, device:str, arch:str) -> GemmMatch|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  end = ast.src[0]
  if end.op is not Ops.END or len(end.src) < 3 or end.src[0].op is not Ops.STORE: return None
  store, ranges = end.src[0], end.src[1:]
  if any(x.op is not Ops.RANGE or x.arg[1] is not AxisType.LOOP for x in ranges) or store.src[0].op is not Ops.INDEX: return None
  value = store.src[1]
  old, old_index, scale = None, None, 0.0
  if value.op is Ops.CAST and value.dtype == dtypes.half and value.src[0].op is Ops.REDUCE: reduce = value.src[0]
  elif value.op is Ops.CAST and value.dtype == dtypes.float and value.src[0].op is Ops.ADD:
    add_lhs, add_rhs = value.src[0].src
    if add_lhs.op is not Ops.CAST or add_lhs.dtype != dtypes.half or add_lhs.src[0].op is not Ops.REDUCE:
      add_lhs, add_rhs = add_rhs, add_lhs
    if add_lhs.op is not Ops.CAST or add_lhs.dtype != dtypes.half or add_lhs.src[0].op is not Ops.REDUCE: return None
    if add_rhs.op is not Ops.MUL: return None
    old_idx = next((x for x in add_rhs.src if x.op is Ops.INDEX and x.dtype == dtypes.half), None)
    scale_uop = next((x for x in add_rhs.src if x.op is Ops.CONST and x.dtype == dtypes.half), None)
    if old_idx is None or scale_uop is None: return None
    reduce, old, old_index, scale = add_lhs.src[0], old_idx.src[0], old_idx.src[1], float(scale_uop.arg)
  else: return None
  if reduce.arg != (Ops.ADD, 0) or len(reduce.src) != 2 or reduce.src[1].op is not Ops.RANGE: return None
  k, product = reduce.src[1], reduce.src[0]
  if product.op is not Ops.CAST or product.dtype != dtypes.float or product.src[0].op is not Ops.MUL: return None
  lhs, rhs = product.src[0].src
  if lhs.op is not Ops.INDEX or rhs.op is not Ops.INDEX: return None
  if any(x.dtype != dtypes.half for x in (lhs.src[0], rhs.src[0])): return None
  if store.src[0].src[0].dtype not in (dtypes.half, dtypes.float): return None

  out_idx = ssimplify(store.src[0].src[1].get_idx())
  if old_index is not None and ssimplify(old_index.get_idx()) is not out_idx: return None
  k_size, total_size = int(k.vmax)+1, prod(int(x.vmax)+1 for x in ranges)
  lhs_idx, rhs_idx = ssimplify(lhs.src[1].get_idx()), ssimplify(rhs.src[1].get_idx())
  for m in ranges:
    m_size, n_size = int(m.vmax)+1, total_size//(int(m.vmax)+1)
    n = ssimplify(out_idx%n_size)
    if ssimplify(out_idx//n_size) is not m or int(n.vmin) != 0 or int(n.vmax) != n_size-1: continue
    n64 = old is None and n_size == 64
    a_mxk_idx, a_kxm_idx = ssimplify(m*k_size+k), ssimplify(k*m_size+m)
    b_nxk_idx, b_kxn_idx = ssimplify(n*k_size+k), ssimplify(k*n_size+n)
    for a_idx, a_buf, b_idx, b_buf in ((lhs_idx, lhs.src[0], rhs_idx, rhs.src[0]), (rhs_idx, rhs.src[0], lhs_idx, lhs.src[0])):
      if not (a_idx is a_mxk_idx or a_idx is a_kxm_idx): continue
      if not (b_idx is b_nxk_idx or b_idx is b_kxn_idx): continue
      if n64 and (a_idx is not a_mxk_idx or b_idx is not b_nxk_idx): continue
      g = GemmMatch(store.src[0].src[0], a_buf, b_buf, m_size, n_size, k_size, old, scale,
                    a_idx is a_kxm_idx, b_idx is b_kxn_idx)
      bm, bn = _gemm_block_m(g), _gemm_block_n(g)
      if m_size % bm or n_size % bn or k_size % BLOCK_K: continue
      if (m_size//bm)*(n_size//bn) < (32 if old is not None else 512): continue
      return g
  return None

def _match_batched_gemm(ast:UOp, device:str, arch:str) -> BatchedGemmMatch|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  end = ast.src[0]
  if end.op is not Ops.END or len(end.src) != 4 or end.src[0].op is not Ops.STORE: return None
  store, ranges = end.src[0], end.src[1:]
  if any(x.op is not Ops.RANGE or x.arg[1] is not AxisType.LOOP for x in ranges) or store.src[0].op is not Ops.INDEX: return None
  reduce = store.src[1]
  if reduce.op is not Ops.REDUCE or reduce.dtype != dtypes.float or reduce.arg != (Ops.ADD, 0) or len(reduce.src) != 2: return None
  k, product = reduce.src[1], reduce.src[0]
  if k.op is not Ops.RANGE or product.op is not Ops.CAST or product.dtype != dtypes.float or product.src[0].op is not Ops.MUL: return None
  lhs, rhs = product.src[0].src
  if lhs.op is not Ops.INDEX or rhs.op is not Ops.INDEX or lhs.dtype != dtypes.half or rhs.dtype != dtypes.half: return None
  out_idx, lhs_idx, rhs_idx = (ssimplify(x.src[1].get_idx()) for x in (store.src[0], lhs, rhs))
  k_size = int(k.vmax)+1
  for m in ranges:
    for n in ranges:
      if n is m: continue
      batch = next(x for x in ranges if x is not m and x is not n)
      m_size, n_size, batch_size = int(m.vmax)+1, int(n.vmax)+1, int(batch.vmax)+1
      if ssimplify(out_idx//(n_size*batch_size)) is not m or ssimplify((out_idx//batch_size)%n_size) is not n or \
         ssimplify(out_idx%batch_size) is not batch: continue
      if m_size % 64 or n_size % 32 or k_size % BLOCK_K: continue
      a_idx, b_idx = ssimplify((batch*k_size+k)*m_size+m), ssimplify((batch*k_size+k)*n_size+n)
      if lhs_idx is a_idx and rhs_idx is b_idx: a, b = lhs.src[0], rhs.src[0]
      elif rhs_idx is a_idx and lhs_idx is b_idx: a, b = rhs.src[0], lhs.src[0]
      else: continue
      return BatchedGemmMatch(store.src[0].src[0], a, b, m_size, n_size, k_size, batch_size)
  return None

def _gemm_block_m(g:GemmMatch) -> int:
  if g.old is None and (g.m,g.n,g.k) in ((393216,576,256), (24576,512,4608)): return 64
  return 32 if g.old is not None and (g.m < 512 or (g.a_kxm and g.b_kxn and g.k >= 65536)) else \
    64 if g.old is not None else BLOCK_M
def _gemm_block_n(g:GemmMatch) -> int:
  if g.old is not None and g.a_kxm and g.b_kxn and (g.m,g.n,g.k) == (256,2304,65536): return 128
  if g.old is not None and g.a_kxm and g.b_kxn and (g.m,g.n,g.k) == (256,2304,98304): return 64
  if g.n == 288 and g.old is None and not g.a_kxm and g.b_kxn: return 96
  if g.n == 576 and g.old is None and not g.a_kxm and g.b_kxn: return 192
  if g.old is not None and g.a_kxm and g.b_kxn and g.m == 256 and g.k >= 65536: return BLOCK_K
  return 64 if g.n == 64 and g.old is None and not g.a_kxm and not g.b_kxn else BLOCK_N
def _gemm_block_k(g:GemmMatch) -> int:
  if g.old is not None and g.a_kxm and g.b_kxn and g.m == 256 and g.k >= 65536: return 64
  if g.old is not None and (g.m < 512 or (g.a_kxm and g.b_kxn and g.k >= 65536)) and g.k % 64 == 0: return 64
  return 144 if _gemm_block_n(g) == 64 and g.k in (288, 576) else BLOCK_K
def _batched_block_k(g:BatchedGemmMatch) -> int:
  if (g.batch, g.m, g.n, g.k) == (192, 64, 288, 8192): return 16
  if (g.batch, g.m, g.n, g.k) == (96, 64, 576, 4096): return BLOCK_K
  return 128 if g.m == 64 and g.batch >= 96 and g.k % 128 == 0 else BLOCK_K
def _batched_block_n(g:BatchedGemmMatch) -> int:
  if (g.batch, g.m, g.n, g.k) == (192, 64, 288, 8192): return 288
  return 192 if g.n == 576 else BLOCK_N
def _batched_threads(g:BatchedGemmMatch) -> int:
  if _batched_block_n(g) == 288: return 192
  return _batched_block_n(g)
def _batched_family_name(g:BatchedGemmMatch) -> str:
  return f"coop_bgemm_bm64_bn{_batched_block_n(g)}_bk{_batched_block_k(g)}_t{_batched_threads(g)}" \
    f"{'_partial_n' if g.n % _batched_block_n(g) else ''}"
def _gemm_threads(g:GemmMatch) -> int:
  if _gemm_block_n(g) == 192: return 192
  if g.old is not None and g.a_kxm and g.b_kxn and (g.m,g.n,g.k) == (256,2304,65536): return 128
  if g.old is not None and g.a_kxm and g.b_kxn and (g.m,g.n,g.k) == (256,2304,98304): return 128
  return THREADS

def _gemm_family_name(g:GemmMatch, output_nchw:GemmOutputNCHW|None) -> str:
  return f"coop_gemm_bm{_gemm_block_m(g)}_bn{_gemm_block_n(g)}_bk{_gemm_block_k(g)}_t{_gemm_threads(g)}" \
    f"{'_kxm' if g.a_kxm else ''}{'_kxn' if g.b_kxn else ''}{'_acc' if g.old is not None else ''}" \
    f"{'_nchw' if output_nchw is not None else ''}"

def _render_gemm(g:GemmMatch, name:str, output_nchw:GemmOutputNCHW|None=None) -> str:
  bm, bn_size, bk, threads = _gemm_block_m(g), _gemm_block_n(g), _gemm_block_k(g), _gemm_threads(g)
  pack_b = g.b_kxn and not g.a_kxm
  as_stride, bs_stride = (bm if g.a_kxm else bk+8), (bn_size if g.b_kxn else bk+8)
  waves_m, waves_n = (2, 3) if threads == 192 else (1, 2) if threads == 64 else \
                     (((1, 8) if bm == 32 else (2, 4) if bm == 64 else (4, 2)) if threads == 256 else
                      ((1, 4) if bm <= 64 else (2, 2)))
  tiles_m, tiles_n = bm//(waves_m*WMMA_M), bn_size//(waves_n*WMMA_N)
  cslot, aslot, bslot = g.c.arg.slot, g.a.arg.slot, g.b.arg.slot
  buffers = (g.c, g.a, g.b) + ((g.old,) if g.old is not None else ())
  params = ', '.join(f'{"float" if x.dtype == dtypes.float else "half"}* p{x.arg.slot}' for x in sorted(buffers, key=lambda x:x.arg.slot))
  params += ', int M, int N, int K' + (', int S' if output_nchw is not None else '')
  lines = [
    '#define half _Float16',
    'typedef half half16 __attribute__((ext_vector_type(16)));',
    'typedef float float8 __attribute__((ext_vector_type(8)));',
    'typedef unsigned uint8 __attribute__((ext_vector_type(8)));',
    'typedef unsigned uint16 __attribute__((ext_vector_type(16)));',
    'typedef unsigned short ushort16 __attribute__((ext_vector_type(16)));',
    '#define HALF_BITS(x) (unsigned short)(x), (unsigned short)((x)>>16)',
    '#define WMMA __builtin_amdgcn_wmma_f32_16x16x16_f16_w32',
    f'extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size({threads}, {threads}))) {name}({params}) {{',
    f'  __attribute__((shared, aligned(32))) half As[{bk*bm if g.a_kxm else bm*as_stride}], '
    f'Bs[{bk*bn_size if g.b_kxn else bn_size*bs_stride}];',
    '  int bn=__builtin_amdgcn_workgroup_id_x(), bm=__builtin_amdgcn_workgroup_id_y(), tid=__builtin_amdgcn_workitem_id_x();',
    f'  int wave=tid>>5, lane=tid&31, wm=wave/{waves_n}, wn=wave%{waves_n}, row=lane&15, halfrow=lane>>4;',
  ]
  lines.append(f'  float8 c[{tiles_m}][{tiles_n}]={{}};')
  lines += [f'  for (int kt=0; kt<K/{bk}; kt++) {{']
  if g.a_kxm:
    aseg_count = bm*bk//16
    lines += ['    #pragma unroll', f'    for (int q=0; q<{(aseg_count+threads-1)//threads}; q++) {{',
              f'      int aseg=tid+q*{threads};']
    if aseg_count % threads: lines.append(f'      if (aseg<{aseg_count}) {{')
    lines += [f'      int ak=aseg/{bm//16}, am=(aseg%{bm//16})*16;',
              f'      *((half16*)(As+ak*{bm}+am))=*((half16*)(p{aslot}+((long)(kt*{bk}+ak)*M)+bm*{bm}+am));']
    if aseg_count % threads: lines.append('      }')
    lines.append('    }')
  else:
    prefix, suffix = (f'    if (tid<{bm}) {{ ', ' }') if threads != bm else ('    ', '')
    lines += [f'{prefix}long ao=((long)(bm*{bm}+tid)*K)+kt*{bk};']
    lines += ['      #pragma unroll', f'      for (int q=0; q<{bk//16}; q++) '
              f'*((half16*)(As+tid*{as_stride}+q*16))=*((half16*)(p{aslot}+ao+q*16));']
    lines[-1] += suffix
  if pack_b:
    bsegs = bn_size//16
    if threads != bsegs*16: lines.append(f'    if (tid<{bsegs*16}) {{')
    lines += ['    #pragma unroll', f'    for (int q=0; q<{bk//32}; q++) {{',
              f'      int bkp=tid/{bsegs}+q*16, bn0=(tid%{bsegs})*16;',
              f'      half16 bv0=*((half16*)(p{bslot}+((long)(kt*{bk}+bkp*2)*N)+bn*{bn_size}+bn0));',
              f'      half16 bv1=*((half16*)(p{bslot}+((long)(kt*{bk}+bkp*2+1)*N)+bn*{bn_size}+bn0));',
              '      ushort16 pb0=__builtin_bit_cast(ushort16,bv0), pb1=__builtin_bit_cast(ushort16,bv1);',
              f'      *((uint16*)(((unsigned*)Bs)+bkp*{bn_size}+bn0))=__builtin_convertvector(pb0,uint16)|'
              '(__builtin_convertvector(pb1,uint16)<<16);', '    }']
    if threads != bsegs*16: lines.append('    }')
  elif g.b_kxn:
    bsegs = bn_size//16
    lines += ['    #pragma unroll', f'    for (int q=0; q<{bn_size*bk//(threads*16)}; q++) {{',
              f'      int bseg=tid+q*{threads}, bki=bseg/{bsegs}, bni=(bseg%{bsegs})*16;',
              f'      *((half16*)(Bs+bki*{bn_size}+bni))=*((half16*)(p{bslot}+'
              f'((long)(kt*{bk}+bki)*N)+bn*{bn_size}+bni));', '    }']
  else:
    prefix, suffix = (f'    if (tid<{bn_size}) {{ ', ' }') if threads != bn_size else ('    ', '')
    lines += [f'{prefix}long bo=((long)(bn*{bn_size}+tid)*K)+kt*{bk};']
    lines += ['      #pragma unroll', f'      for (int q=0; q<{bk//16}; q++) '
              f'*((half16*)(Bs+tid*{bs_stride}+q*16))=*((half16*)(p{bslot}+bo+q*16));']
    lines[-1] += suffix
  lines += ['    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); '
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");']
  lines += ['    #pragma unroll', f'    for (int ki=0; ki<{bk//WMMA_K}; ki++) {{']
  lines += [f'      half16 av[{tiles_m}], bv[{tiles_n}];', '      #pragma unroll', f'      for (int im=0; im<{tiles_m}; im++) {{']
  if g.a_kxm:
    lines += ['        half16 v;', '        #pragma unroll',
              f'        for (int e=0; e<16; e++) v[e]=As[(ki*16+e)*{bm}+(wm*{tiles_m}+im)*16+row];', '        av[im]=v;']
  else:
    lines += [f'        av[im]=*((half16*)(As+(((wm*{tiles_m}+im)*16+row)*{as_stride}+ki*16)));']
  lines += ['      }', '      #pragma unroll', f'      for (int jn=0; jn<{tiles_n}; jn++) {{']
  if pack_b:
    half_bits = ','.join(f'HALF_BITS(bp[{e}])' for e in range(8))
    lines += ['        uint8 bp;', '        #pragma unroll',
              f'        for (int e=0; e<8; e++) bp[e]=((unsigned*)Bs)[(ki*8+e)*{bn_size}+(wn*{tiles_n}+jn)*16+row];',
              f'        ushort16 bb=(ushort16){{{half_bits}}};', '        bv[jn]=__builtin_bit_cast(half16,bb);']
  elif g.b_kxn:
    lines += ['        half16 v;', '        #pragma unroll',
              f'        for (int e=0; e<16; e++) v[e]=Bs[(ki*16+e)*{bn_size}+(wn*{tiles_n}+jn)*16+row];', '        bv[jn]=v;']
  else:
    lines += [f'        bv[jn]=*((half16*)(Bs+(((wn*{tiles_n}+jn)*16+row)*{bs_stride}+ki*16)));']
  lines += ['      }', '      #pragma unroll', f'      for (int im=0; im<{tiles_m}; im++) {{', '        #pragma unroll',
            f'        for (int jn=0; jn<{tiles_n}; jn++) c[im][jn]=WMMA(av[im],bv[jn],c[im][jn]);', '      }', '    }']
  lines += ['    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); '
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");', '  }']
  lines += ['  #pragma unroll', f'  for (int im=0; im<{tiles_m}; im++) {{', '    #pragma unroll',
            f'    for (int jn=0; jn<{tiles_n}; jn++) {{', '      #pragma unroll', '      for (int e=0; e<8; e++) {',
            f'        int om=bm*{bm}+(wm*{tiles_m}+im)*16+e*2+halfrow, on=bn*{bn_size}+(wn*{tiles_n}+jn)*16+row;']
  if output_nchw is not None:
    lines.append('        int spatial_size=S*S;')
    lines.append('        long oi=((long)(om/spatial_size)*N+on)*spatial_size+om%spatial_size;')
  else: lines.append('        long oi=(long)om*N+on;')
  if g.old is None: lines.append(f'        p{cslot}[oi]=(half)c[im][jn][e];')
  else: lines.append(f'        p{cslot}[oi]=(float)((half)c[im][jn][e]+(half){g.scale}*p{g.old.arg.slot}[oi]);')
  lines += ['      }', '    }', '  }']
  lines += ['}']
  return '\n'.join(lines)

def _render_batched_gemm(g:BatchedGemmMatch, name:str) -> tuple[str, int]:
  bm, bn_size, bk, threads, waves_m = 64, _batched_block_n(g), _batched_block_k(g), _batched_threads(g), 1
  waves_n = threads//32
  exact_n = g.n % bn_size == 0
  tiles_m, tiles_n = bm//(waves_m*WMMA_M), bn_size//(waves_n*WMMA_N)
  cslot, aslot, bslot = g.c.arg.slot, g.a.arg.slot, g.b.arg.slot
  params = ', '.join(f'{"float" if x.dtype == dtypes.float else "half"}* p{x.arg.slot}'
                     for x in sorted((g.c, g.a, g.b), key=lambda x:x.arg.slot))
  params += ', int B, int M, int N, int K'
  lines = [
    '#define half _Float16',
    'typedef half half16 __attribute__((ext_vector_type(16)));',
    'typedef float float8 __attribute__((ext_vector_type(8)));',
    '#define WMMA __builtin_amdgcn_wmma_f32_16x16x16_f16_w32',
    f'extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size({threads}, {threads}))) {name}({params}) {{',
    f'  __attribute__((shared, aligned(32))) half As[{bm*bk}], Bs[{bn_size*bk}];',
    '  int bn=__builtin_amdgcn_workgroup_id_x(), bm=__builtin_amdgcn_workgroup_id_y(), '
    'batch=__builtin_amdgcn_workgroup_id_z(), tid=__builtin_amdgcn_workitem_id_x();',
    f'  int wave=tid>>5, lane=tid&31, wm=wave/{waves_n}, wn=wave%{waves_n}, row=lane&15, halfrow=lane>>4;',
  ]
  lines.append(f'  float8 c[{tiles_m}][{tiles_n}]={{}};')
  lines.append(f'  for (int kt=0; kt<K/{bk}; kt++) {{')
  aseg_count = bm*bk//16
  lines += ['    #pragma unroll', f'    for (int q=0; q<{(aseg_count+threads-1)//threads}; q++) {{',
            f'      int aseg=tid+q*{threads};']
  if aseg_count % threads: lines.append(f'      if (aseg<{aseg_count}) {{')
  lines += [f'      int ak=aseg/{bm//16}, am=(aseg%{bm//16})*16;',
            f'      *((half16*)(As+ak*{bm}+am))=*((half16*)(p{aslot}+'
            f'((long)(batch*K+kt*{bk}+ak)*M)+bm*{bm}+am));']
  if aseg_count % threads: lines.append('      }')
  lines.append('    }')
  bsegs = bn_size//16
  lines += ['    #pragma unroll', f'    for (int q=0; q<{bn_size*bk//(threads*16)}; q++) {{',
            f'      int bseg=tid+q*{threads}, bki=bseg/{bsegs}, bni=(bseg%{bsegs})*16;',
            f'      *((half16*)(Bs+bki*{bn_size}+bni))=' + ('' if exact_n else f'bn*{bn_size}+bni<N ? ') +
            f'*((half16*)(p{bslot}+((long)(batch*K+kt*{bk}+bki)*N)+bn*{bn_size}+bni))' +
            (';' if exact_n else ' : (half16){0};'), '    }']
  lines += ['    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); '
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");']
  lines += ['    #pragma unroll', f'    for (int ki=0; ki<{bk//WMMA_K}; ki++) {{',
            f'      half16 av[{tiles_m}], bv[{tiles_n}];', '      #pragma unroll',
            f'      for (int im=0; im<{tiles_m}; im++) {{', '        half16 v;', '        #pragma unroll',
            f'        for (int e=0; e<16; e++) v[e]=As[(ki*16+e)*{bm}+(wm*{tiles_m}+im)*16+row];',
            '        av[im]=v;', '      }', '      #pragma unroll', f'      for (int jn=0; jn<{tiles_n}; jn++) {{',
            '        half16 v;', '        #pragma unroll',
            f'        for (int e=0; e<16; e++) v[e]=Bs[(ki*16+e)*{bn_size}+(wn*{tiles_n}+jn)*16+row];',
            '        bv[jn]=v;', '      }', '      #pragma unroll', f'      for (int im=0; im<{tiles_m}; im++) {{',
            '        #pragma unroll', f'        for (int jn=0; jn<{tiles_n}; jn++) c[im][jn]=WMMA(av[im],bv[jn],c[im][jn]);',
            '      }', '    }']
  lines += ['    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); '
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");', '  }']
  lines += ['  #pragma unroll', f'  for (int im=0; im<{tiles_m}; im++) {{', '    #pragma unroll',
            f'    for (int jn=0; jn<{tiles_n}; jn++) {{',
            f'      int om=bm*{bm}+(wm*{tiles_m}+im)*16+halfrow, on=bn*{bn_size}+(wn*{tiles_n}+jn)*16+row;',
            ('      {' if exact_n else '      if (on<N) {'), '        #pragma unroll',
            f'        for (int e=0; e<8; e++) p{cslot}[((long)(om+e*2)*N+on)*B+batch]=c[im][jn][e];',
            '      }', '    }', '  }']
  lines += ['}']
  return '\n'.join(lines), bm

def _render_direct_conv_bwd_activation(g:DirectConvBwdActivationMatch, name:str) -> str:
  bm, bn, bk, threads = 128, min(g.cin, 64), 32, 128
  persist_weights = g.cin <= 64
  tiles_m = 2
  tiles_n, aslot, bslot = bn//16, (3 if g.residual else 2), (4 if g.residual else 3)
  params = 'float* p0, float* p1, half* p2, half* p3, half* p4' if g.residual else 'float* p0, half* p1, half* p2, half* p3'
  lines = [
    '#define half _Float16',
    'typedef half half16 __attribute__((ext_vector_type(16)));',
    'typedef half half8 __attribute__((ext_vector_type(8)));',
    'typedef float float8 __attribute__((ext_vector_type(8)));',
    'typedef unsigned uint8 __attribute__((ext_vector_type(8)));',
    'typedef unsigned short ushort16 __attribute__((ext_vector_type(16)));',
    '#define HALF_BITS(x) (unsigned short)(x), (unsigned short)((x)>>16)',
    '#define WMMA __builtin_amdgcn_wmma_f32_16x16x16_f16_w32',
    f'extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size({threads},{threads}))) {name}({params}) {{',
    f'  __attribute__((shared, aligned(32))) half As[{bm*max(bk,bn)}], Bs[{bn*(g.cout if persist_weights else bk)}];',
    '  int bn=__builtin_amdgcn_workgroup_id_x(), bm=__builtin_amdgcn_workgroup_id_y(), tid=__builtin_amdgcn_workitem_id_x();',
    '  int wave=tid>>5, lane=tid&31, wm=wave, row=lane&15, halfrow=lane>>4;',
    f'  float8 total[{tiles_m}][{tiles_n}]={{}};',
    '  for (int patch=0; patch<9; patch++) {',
    f'    float8 c[{tiles_m}][{tiles_n}]={{}};',
  ]
  if persist_weights:
    lines += ['    #pragma unroll', f'    for (int q=0; q<{bn*(g.cout//2)//threads}; q++) {{',
              f'      int be=tid+q*{threads}, bp=be/{bn}, ci=be%{bn};',
              f'      half blo=p{bslot}[(bp*2)*{g.cin*9}+(bn*{bn}+ci)*9+patch];',
              f'      half bhi=p{bslot}[(bp*2+1)*{g.cin*9}+(bn*{bn}+ci)*9+patch];',
              f'      ((unsigned*)Bs)[bp*{bn}+ci]=__builtin_bit_cast(unsigned short,blo)|'
              '((unsigned)__builtin_bit_cast(unsigned short,bhi)<<16);', '    }']
    lines += ['    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); '
              '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");']
  lines += [f'    for (int ct=0; ct<{g.cout//bk}; ct++) {{', f'      int co0=ct*{bk};']
  if not persist_weights:
    lines += ['      #pragma unroll', f'      for (int q=0; q<{bn*(bk//2)//threads}; q++) {{',
              f'        int be=tid+q*{threads}, bp=be/{bn}, ci=be%{bn};',
              f'        half blo=p{bslot}[(co0+bp*2)*{g.cin*9}+(bn*{bn}+ci)*9+patch];',
              f'        half bhi=p{bslot}[(co0+bp*2+1)*{g.cin*9}+(bn*{bn}+ci)*9+patch];',
              f'        ((unsigned*)Bs)[bp*{bn}+ci]=__builtin_bit_cast(unsigned short,blo)|'
              '((unsigned)__builtin_bit_cast(unsigned short,bhi)<<16);', '      }']
    lines += ['      __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); '
              '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");']
  lines += [
            '      #pragma unroll', f'      for (int ki=0; ki<{bk//16}; ki++) {{',
            f'        half16 av[{tiles_m}], bv[{tiles_n}];', '        #pragma unroll', f'        for (int im=0; im<{tiles_m}; im++) {{',
            f'          int ami=bm*{bm}+(wm*{tiles_m}+im)*16+row, ab=ami/{g.spatial*g.spatial}, apos=ami%{g.spatial*g.spatial};',
            f'          int ay=apos/{g.spatial}+1-patch/3, ax=apos%{g.spatial}+1-patch%3;',
            f'          av[im]=(ay>=0 && ay<{g.spatial} && ax>=0 && ax<{g.spatial}) ? '
            f'*((half16*)(p{aslot}+((long)ab*{g.spatial*g.spatial}+ay*{g.spatial}+ax)*{g.cout}+co0+ki*16)) : (half16){{0}};',
            '        }',
            '        #pragma unroll', f'        for (int jn=0; jn<{tiles_n}; jn++) {{', '          uint8 bp;', '          #pragma unroll',
            f'          for (int e=0; e<8; e++) bp[e]=((unsigned*)Bs)[({"ct*16+" if persist_weights else ""}ki*8+e)*{bn}+jn*16+row];',
            '          ushort16 bb=(ushort16){HALF_BITS(bp[0]),HALF_BITS(bp[1]),HALF_BITS(bp[2]),HALF_BITS(bp[3]),'
            'HALF_BITS(bp[4]),HALF_BITS(bp[5]),HALF_BITS(bp[6]),HALF_BITS(bp[7])};',
            '          bv[jn]=__builtin_bit_cast(half16,bb);', '        }', '        #pragma unroll',
            f'        for (int im=0; im<{tiles_m}; im++) {{',
            '          #pragma unroll', f'          for (int jn=0; jn<{tiles_n}; jn++) c[im][jn]=WMMA(av[im],bv[jn],c[im][jn]);',
            '        }', '      }']
  if not persist_weights:
    lines += ['      __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); '
              '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");']
  lines += ['    }',
            '    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); '
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");',
            '    #pragma unroll', f'    for (int im=0; im<{tiles_m}; im++) {{', '      #pragma unroll',
            f'      for (int jn=0; jn<{tiles_n}; jn++) total[im][jn]+=__builtin_convertvector('
            '__builtin_convertvector(c[im][jn],half8),float8);', '    }', '  }',
            '  #pragma unroll', f'  for (int im=0; im<{tiles_m}; im++) {{', '    #pragma unroll',
            f'    for (int jn=0; jn<{tiles_n}; jn++) {{', '      #pragma unroll', '      for (int e=0; e<8; e++) {',
            f'        int lm=(wm*{tiles_m}+im)*16+e*2+halfrow, lc=jn*16+row;',
            f'        As[lc*{bm}+lm]=(half)total[im][jn][e];', '      }', '    }', '  }',
            '  __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); '
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");']
  lines += ['  #pragma unroll', f'  for (int q=0; q<{bm*bn//(threads*8)}; q++) {{',
            f'    int oe=tid*{bm*bn//threads}+q*8, lc=oe/{bm}, lm=oe%{bm};',
            f'    int gm=bm*{bm}+lm, ob=gm/{g.spatial*g.spatial}, pos=gm%{g.spatial*g.spatial};',
            f'    long oo=((long)ob*{g.cin}+bn*{bn}+lc)*{g.spatial*g.spatial}+pos;',
            '    half8 grad=*((half8*)(As+oe));']
  if g.residual: lines += ['    grad+=*((half8*)(p2+oo));', '    half8 z=__builtin_convertvector(*((float8*)(p1+oo)),half8);']
  else: lines.append('    half8 z=*((half8*)(p1+oo));')
  lines += ['    half8 sig=(half8)1.0/((half8)1.0+__builtin_elementwise_exp2(z*(half)-2.4554669595930156));',
            '    *((float8*)(p0+oo))=__builtin_convertvector(sig*grad+(half)1.702*z*grad*sig*((half)1.0-sig),float8);', '  }']
  lines.append('}')
  return '\n'.join(lines)

def direct_conv_bwd_activation_program(ast:UOp, renderer:Renderer, compile_binary:bool) -> UOp|None:
  if not isinstance(g:=ast.tag, DirectConvBwdActivationMatch) or renderer.target.device != "AMD" or \
     not renderer.target.arch.startswith("gfx11"): return None
  name = f"coop_direct_conv_bwd_activation_{g.m}_{g.cin}_{g.cout}{'_res' if g.residual else ''}"
  source = _render_direct_conv_bwd_activation(g, name)
  sink = ast.replace(arg=replace(ast.arg, name=name, estimates=Estimates(2*g.m*g.cin*g.cout*9, 0, 0)))
  slots = (0,1,2,3,4) if g.residual else (0,1,2,3)
  bn = min(g.cin, 64)
  info = ProgramInfo(name=name, global_size=((g.cin+bn-1)//bn, g.m//128, 1), local_size=(128,1,1),
                     globals=slots, outs=(0,), ins=slots[1:])
  src:tuple[UOp, ...] = (sink, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=source))
  if compile_binary: src += (UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source)),)
  return UOp(Ops.PROGRAM, src=src, arg=info)


def cooperative_gemm_program(ast:UOp, renderer:Renderer, compile_binary:bool) -> UOp|None:
  if (g:=_match_gemm(ast, renderer.target.device, renderer.target.arch)) is not None:
    output_nchw = ast.tag if isinstance(ast.tag, GemmOutputNCHW) else None
    name = _gemm_family_name(g, output_nchw)
    source = _render_gemm(g, name, output_nchw)
    global_size = (g.n//_gemm_block_n(g), g.m//_gemm_block_m(g), 1)
    local_size = (_gemm_threads(g), 1, 1)
    estimates = Estimates(2*g.m*g.n*g.k, 2*(g.m*g.k+g.n*g.k+g.m*g.n), 2*(g.m*g.k+g.n*g.k+g.m*g.n))
    slots = tuple(sorted((g.c.arg.slot, g.a.arg.slot, g.b.arg.slot) + ((g.old.arg.slot,) if g.old is not None else ())))
    out_slot = g.c.arg.slot
    suffix = f"_{g.m}_{g.n}_{g.k}"
    variables:tuple[UOp, ...] = (UOp.variable(f"M{suffix}", g.m, g.m, dtypes.int),
                                 UOp.variable(f"N{suffix}", g.n, g.n, dtypes.int),
                                 UOp.variable(f"K{suffix}", g.k, g.k, dtypes.int))
    if output_nchw is not None:
      variables += (UOp.variable(f"S{suffix}_{output_nchw.spatial}", output_nchw.spatial, output_nchw.spatial, dtypes.int),)
  elif (bg:=_match_batched_gemm(ast, renderer.target.device, renderer.target.arch)) is not None:
    name = _batched_family_name(bg)
    source, bm = _render_batched_gemm(bg, name)
    global_size = ((bg.n+_batched_block_n(bg)-1)//_batched_block_n(bg), bg.m//bm, bg.batch)
    local_size = (_batched_threads(bg), 1, 1)
    estimates = Estimates(2*bg.batch*bg.m*bg.n*bg.k, 2*bg.batch*(bg.m*bg.k+bg.n*bg.k+2*bg.m*bg.n),
                          2*bg.batch*(bg.m*bg.k+bg.n*bg.k+2*bg.m*bg.n))
    slots, out_slot = tuple(sorted((bg.c.arg.slot, bg.a.arg.slot, bg.b.arg.slot))), bg.c.arg.slot
    suffix = f"_{bg.batch}_{bg.m}_{bg.n}_{bg.k}"
    variables = (UOp.variable(f"B{suffix}", bg.batch, bg.batch, dtypes.int), UOp.variable(f"M{suffix}", bg.m, bg.m, dtypes.int),
                 UOp.variable(f"N{suffix}", bg.n, bg.n, dtypes.int), UOp.variable(f"K{suffix}", bg.k, bg.k, dtypes.int))
  else: return None
  sink = ast.replace(arg=replace(ast.arg, name=name, estimates=estimates))
  info = ProgramInfo(name=name, global_size=global_size, local_size=local_size, vars=variables, globals=slots,
                     outs=(out_slot,), ins=tuple(x for x in slots if x != out_slot))
  src:tuple[UOp, ...] = (sink, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=source))
  if compile_binary: src += (UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source)),)
  return UOp(Ops.PROGRAM, src=src, arg=info)
