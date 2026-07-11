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

def _match_gemm(ast:UOp, device:str, arch:str) -> GemmMatch|None:
  if device != "AMD" or not arch.startswith("gfx11") or len(ast.src) != 1: return None
  end = ast.src[0]
  if end.op is not Ops.END or len(end.src) < 3 or end.src[0].op is not Ops.STORE: return None
  store, ranges = end.src[0], end.src[1:]
  if any(x.op is not Ops.RANGE or x.arg[1] is not AxisType.LOOP for x in ranges) or store.src[0].op is not Ops.INDEX: return None
  out_idx = ssimplify(store.src[0].src[1].get_idx())
  value = store.src[1]
  old, scale = None, 0.0
  if value.op is Ops.CAST and value.dtype == dtypes.half and value.src[0].op is Ops.REDUCE: reduce = value.src[0]
  elif value.op is Ops.CAST and value.dtype == dtypes.float and value.src[0].op is Ops.ADD:
    add_lhs, add_rhs = value.src[0].src
    if add_lhs.op is not Ops.CAST or add_lhs.dtype != dtypes.half or add_lhs.src[0].op is not Ops.REDUCE:
      add_lhs, add_rhs = add_rhs, add_lhs
    if add_lhs.op is not Ops.CAST or add_lhs.dtype != dtypes.half or add_lhs.src[0].op is not Ops.REDUCE: return None
    if add_rhs.op is not Ops.MUL: return None
    old_idx = next((x for x in add_rhs.src if x.op is Ops.INDEX and x.dtype == dtypes.half), None)
    scale_uop = next((x for x in add_rhs.src if x.op is Ops.CONST and x.dtype == dtypes.half), None)
    if old_idx is None or scale_uop is None or ssimplify(old_idx.src[1].get_idx()) is not out_idx: return None
    reduce, old, scale = add_lhs.src[0], old_idx.src[0], float(scale_uop.arg)
  else: return None
  if reduce.arg != (Ops.ADD, 0) or len(reduce.src) != 2 or reduce.src[1].op is not Ops.RANGE: return None
  k, product = reduce.src[1], reduce.src[0]
  if product.op is not Ops.CAST or product.dtype != dtypes.float or product.src[0].op is not Ops.MUL: return None
  lhs, rhs = product.src[0].src
  if lhs.op is not Ops.INDEX or rhs.op is not Ops.INDEX: return None
  if any(x.dtype != dtypes.half for x in (lhs.src[0], rhs.src[0])): return None
  if store.src[0].src[0].dtype not in (dtypes.half, dtypes.float): return None

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
  return 32 if g.old is not None and (g.m < 512 or (g.a_kxm and g.b_kxn and g.k >= 65536)) else \
    64 if g.old is not None or _gemm_block_n(g) == 64 else BLOCK_M
def _gemm_block_n(g:GemmMatch) -> int:
  if g.n == 288 and g.old is None and not g.a_kxm and g.b_kxn: return 96
  if g.n == 576 and g.old is None and not g.a_kxm and g.b_kxn: return 192
  if g.old is not None and g.a_kxm and g.b_kxn and g.m == 256 and g.k >= 65536: return 64
  return 64 if g.n == 64 and g.old is None and not g.a_kxm and not g.b_kxn else BLOCK_N
def _gemm_block_k(g:GemmMatch) -> int:
  if g.old is not None and g.a_kxm and g.b_kxn and g.m == 256 and g.k >= 65536: return BLOCK_K
  if g.old is not None and (g.m < 512 or (g.a_kxm and g.b_kxn and g.k >= 65536)) and g.k % 64 == 0: return 64
  return 96 if _gemm_block_n(g) == 64 and g.k == 288 else BLOCK_K
def _batched_block_k(g:BatchedGemmMatch) -> int:
  return 128 if g.m == 64 and g.batch >= 96 and g.k % 128 == 0 else BLOCK_K
def _batched_block_n(g:BatchedGemmMatch) -> int:
  return 192 if g.n == 576 else BLOCK_N
def _batched_threads(g:BatchedGemmMatch) -> int:
  return _batched_block_n(g)
def _gemm_threads(g:GemmMatch) -> int:
  if _gemm_block_n(g) == 192: return 192
  return 64 if g.old is not None and g.a_kxm and g.b_kxn and g.m >= 256 and g.k >= 65536 else THREADS
def _render_gemm(g:GemmMatch, name:str) -> str:
  bm, bn_size, bk, threads = _gemm_block_m(g), _gemm_block_n(g), _gemm_block_k(g), _gemm_threads(g)
  waves_m, waves_n = (2, 3) if threads == 192 else (1, 2) if threads == 64 else \
                     (((1, 8) if bm == 32 else (2, 4) if bm == 64 else (4, 2)) if threads == 256 else
                      ((1, 4) if bm <= 64 else (2, 2)))
  tiles_m, tiles_n = bm//(waves_m*WMMA_M), bn_size//(waves_n*WMMA_N)
  cslot, aslot, bslot = g.c.arg.slot, g.a.arg.slot, g.b.arg.slot
  buffers = (g.c, g.a, g.b) + ((g.old,) if g.old is not None else ())
  params = ', '.join(f'{"float" if x.dtype == dtypes.float else "half"}* p{x.arg.slot}' for x in sorted(buffers, key=lambda x:x.arg.slot))
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
    f'  __attribute__((shared, aligned(32))) half As[{bm*bk}], Bs[{bn_size*bk}];',
    '  int bn=__builtin_amdgcn_workgroup_id_x(), bm=__builtin_amdgcn_workgroup_id_y(), tid=__builtin_amdgcn_workitem_id_x();',
    f'  int wave=tid>>5, lane=tid&31, wm=wave/{waves_n}, wn=wave%{waves_n}, row=lane&15, halfrow=lane>>4;',
  ]
  lines.append(f'  float8 c[{tiles_m}][{tiles_n}]={{}};')
  lines += [f'  for (int kt=0; kt<{g.k//bk}; kt++) {{']
  if g.a_kxm:
    aseg_count = bm*bk//16
    for q in range((aseg_count+threads-1)//threads):
      prefix, suffix = (f'    if (tid+{q*threads}<{aseg_count}) {{ ', ' }') if aseg_count % threads else ('    ', '')
      lines += [f'{prefix}int aseg{q}=tid+{q*threads}, ak{q}=aseg{q}/{bm//16}, am{q}=(aseg{q}%{bm//16})*16;',
                f'      *((half16*)(As+ak{q}*{bm}+am{q}))=*((half16*)(p{aslot}+'
                f'((long)(kt*{bk}+ak{q})*{g.m})+bm*{bm}+am{q}));{suffix}']
  else:
    prefix, suffix = (f'    if (tid<{bm}) {{ ', ' }') if threads != bm else ('    ', '')
    lines += [f'{prefix}long ao=((long)(bm*{bm}+tid)*{g.k})+kt*{bk};']
    for q in range(bk//16): lines.append(f'      *((half16*)(As+tid*{bk}+{q*16}))=*((half16*)(p{aslot}+ao+{q*16}));')
    lines[-1] += suffix
  if g.b_kxn and not g.a_kxm:
    bsegs = bn_size//16
    for q in range(bk//32):
      prefix, suffix = (f'    if (tid<{bsegs*16}) {{ ', ' }') if threads != bsegs*16 else ('    ', '')
      lines += [f'{prefix}int bkp{q}=tid/{bsegs}+{q*16}, bn0{q}=(tid%{bsegs})*16;',
                f'    half16 bv{q}0=*((half16*)(p{bslot}+((long)(kt*{bk}+bkp{q}*2)*{g.n})+bn*{bn_size}+bn0{q}));',
                f'    half16 bv{q}1=*((half16*)(p{bslot}+((long)(kt*{bk}+bkp{q}*2+1)*{g.n})+bn*{bn_size}+bn0{q}));',
                f'    ushort16 pb{q}0=__builtin_bit_cast(ushort16,bv{q}0), pb{q}1=__builtin_bit_cast(ushort16,bv{q}1);',
                f'    *((uint16*)(((unsigned*)Bs)+bkp{q}*{bn_size}+bn0{q}))=__builtin_convertvector(pb{q}0,uint16)|'
                f'(__builtin_convertvector(pb{q}1,uint16)<<16);{suffix}']
  elif g.b_kxn:
    bsegs = bn_size//16
    for q in range(bn_size*bk//(threads*16)):
      lines += [f'    int bseg{q}=tid+{q*threads}, bk{q}=bseg{q}/{bsegs}, bn{q}=(bseg{q}%{bsegs})*16;',
                f'    *((half16*)(Bs+bk{q}*{bn_size}+bn{q}))=*((half16*)(p{bslot}+'
                f'((long)(kt*{bk}+bk{q})*{g.n})+bn*{bn_size}+bn{q}));']
  else:
    prefix, suffix = (f'    if (tid<{bn_size}) {{ ', ' }') if threads != bn_size else ('    ', '')
    lines += [f'{prefix}long bo=((long)(bn*{bn_size}+tid)*{g.k})+kt*{bk};']
    for q in range(bk//16): lines.append(f'      *((half16*)(Bs+tid*{bk}+{q*16}))=*((half16*)(p{bslot}+bo+{q*16}));')
    lines[-1] += suffix
  lines += ['    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); '
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");']
  lines += ['    #pragma unroll', f'    for (int ki=0; ki<{bk//WMMA_K}; ki++) {{',
            f'      half16 av[{tiles_m}], bv[{tiles_n}];', '      #pragma unroll', f'      for (int im=0; im<{tiles_m}; im++) {{']
  if g.a_kxm:
    lines += ['        half16 v;', '        #pragma unroll',
              f'        for (int e=0; e<16; e++) v[e]=As[(ki*16+e)*{bm}+(wm*{tiles_m}+im)*16+row];', '        av[im]=v;']
  else:
    lines += [f'        av[im]=*((half16*)(As+(((wm*{tiles_m}+im)*16+row)*{bk}+ki*16)));']
  lines += ['      }', '      #pragma unroll', f'      for (int jn=0; jn<{tiles_n}; jn++) {{']
  if g.b_kxn and not g.a_kxm:
    half_bits = ','.join(f'HALF_BITS(bp[{e}])' for e in range(8))
    lines += ['        uint8 bp;', '        #pragma unroll',
              f'        for (int e=0; e<8; e++) bp[e]=((unsigned*)Bs)[(ki*8+e)*{bn_size}+(wn*{tiles_n}+jn)*16+row];',
              f'        ushort16 bb=(ushort16){{{half_bits}}};', '        bv[jn]=__builtin_bit_cast(half16,bb);']
  elif g.b_kxn:
    lines += ['        half16 v;', '        #pragma unroll',
              f'        for (int e=0; e<16; e++) v[e]=Bs[(ki*16+e)*{bn_size}+(wn*{tiles_n}+jn)*16+row];', '        bv[jn]=v;']
  else:
    lines += [f'        bv[jn]=*((half16*)(Bs+(((wn*{tiles_n}+jn)*16+row)*{bk}+ki*16)));']
  lines += ['      }', '      #pragma unroll', f'      for (int im=0; im<{tiles_m}; im++) {{', '        #pragma unroll',
            f'        for (int jn=0; jn<{tiles_n}; jn++) c[im][jn]=WMMA(av[im],bv[jn],c[im][jn]);', '      }', '    }']
  lines += ['    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); '
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");', '  }']
  lines += ['  #pragma unroll', f'  for (int im=0; im<{tiles_m}; im++) {{', '    #pragma unroll',
            f'    for (int jn=0; jn<{tiles_n}; jn++) {{', '      #pragma unroll', '      for (int e=0; e<8; e++) {',
            f'        long oi=((long)(bm*{bm}+(wm*{tiles_m}+im)*16+e*2+halfrow)*{g.n})+'
            f'bn*{bn_size}+(wn*{tiles_n}+jn)*16+row;']
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
  lines.append(f'  for (int kt=0; kt<{g.k//bk}; kt++) {{')
  aseg_count = bm*bk//16
  for q in range((aseg_count+threads-1)//threads):
    prefix, suffix = (f'    if (tid+{q*threads}<{aseg_count}) {{ ', ' }') if aseg_count % threads else ('    ', '')
    lines += [f'{prefix}int aseg{q}=tid+{q*threads}, ak{q}=aseg{q}/{bm//16}, am{q}=(aseg{q}%{bm//16})*16;',
              f'    *((half16*)(As+ak{q}*{bm}+am{q}))=*((half16*)(p{aslot}+'
              f'((long)(batch*{g.k}+kt*{bk}+ak{q})*{g.m})+bm*{bm}+am{q}));{suffix}']
  bsegs = bn_size//16
  for q in range(bn_size*bk//(threads*16)):
    lines += [f'    int bseg{q}=tid+{q*threads}, bk{q}=bseg{q}/{bsegs}, bn{q}=(bseg{q}%{bsegs})*16;',
              f'    *((half16*)(Bs+bk{q}*{bn_size}+bn{q}))=' + ('' if exact_n else f'bn*{bn_size}+bn{q}<{g.n} ? ') +
              f'*((half16*)(p{bslot}+((long)(batch*{g.k}+kt*{bk}+bk{q})*{g.n})+bn*{bn_size}+bn{q}))' +
              (';' if exact_n else ' : (half16){0};')]
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
            ('      {' if exact_n else f'      if (on<{g.n}) {{'), '        #pragma unroll',
            f'        for (int e=0; e<8; e++) p{cslot}[((long)(om+e*2)*{g.n}+on)*{g.batch}+batch]=c[im][jn][e];',
            '      }', '    }', '  }']
  lines += ['}']
  return '\n'.join(lines), bm

def cooperative_gemm_program(ast:UOp, renderer:Renderer, compile_binary:bool) -> UOp|None:
  if (g:=_match_gemm(ast, renderer.target.device, renderer.target.arch)) is not None:
    name = f"coop_gemm_{g.m}_{g.n}_{g.k}{'_n64' if _gemm_block_n(g) == 64 else ''}{'_kxm' if g.a_kxm else ''}" \
           f"{'_kxn' if g.b_kxn else ''}{'_acc' if g.old is not None else ''}"
    source = _render_gemm(g, name)
    global_size = (g.n//_gemm_block_n(g), g.m//_gemm_block_m(g), 1)
    local_size = (_gemm_threads(g), 1, 1)
    estimates = Estimates(2*g.m*g.n*g.k, 2*(g.m*g.k+g.n*g.k+g.m*g.n), 2*(g.m*g.k+g.n*g.k+g.m*g.n))
    slots = tuple(sorted((g.c.arg.slot, g.a.arg.slot, g.b.arg.slot) + ((g.old.arg.slot,) if g.old is not None else ())))
    out_slot = g.c.arg.slot
  elif (bg:=_match_batched_gemm(ast, renderer.target.device, renderer.target.arch)) is not None:
    name = f"coop_bgemm_{bg.batch}_{bg.m}_{bg.n}_{bg.k}"
    source, bm = _render_batched_gemm(bg, name)
    global_size = ((bg.n+_batched_block_n(bg)-1)//_batched_block_n(bg), bg.m//bm, bg.batch)
    local_size = (_batched_threads(bg), 1, 1)
    estimates = Estimates(2*bg.batch*bg.m*bg.n*bg.k, 2*bg.batch*(bg.m*bg.k+bg.n*bg.k+2*bg.m*bg.n),
                          2*bg.batch*(bg.m*bg.k+bg.n*bg.k+2*bg.m*bg.n))
    slots, out_slot = tuple(sorted((bg.c.arg.slot, bg.a.arg.slot, bg.b.arg.slot))), bg.c.arg.slot
  else: return None
  sink = ast.replace(arg=replace(ast.arg, name=name, estimates=estimates))
  info = ProgramInfo(name=name, global_size=global_size, local_size=local_size, globals=slots,
                     outs=(out_slot,), ins=tuple(x for x in slots if x != out_slot))
  src:tuple[UOp, ...] = (sink, UOp(Ops.LINEAR), UOp(Ops.SOURCE, arg=source))
  if compile_binary: src += (UOp(Ops.BINARY, arg=renderer.compiler.compile_cached(source)),)
  return UOp(Ops.PROGRAM, src=src, arg=info)
