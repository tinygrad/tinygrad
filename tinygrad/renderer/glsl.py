import re
from tinygrad.renderer.cstyle import OpenCLRenderer, CStyleLanguage
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import Ops, UPat, PatternMatcher
from tinygrad.helpers import is_image_shape
from tinygrad.renderer import Target

STORAGE_TYPES = {dtypes.bool:'uint8_t', dtypes.i8:'int8_t', dtypes.u8:'uint8_t', dtypes.i16:'int16_t', dtypes.u16:'uint16_t', dtypes.bf16:'uint16_t'}
VECTOR_TYPES = (dtypes.float, dtypes.half, dtypes.int, dtypes.uint)
def narrow_storage(bidx):
  return bidx.op in (Ops.INDEX, Ops.SHRINK) and bidx.src[0].addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL) and bidx.src[0].dtype in STORAGE_TYPES

def render_narrow_load(ctx, bidx, var=None, gate=None):
  if not narrow_storage(bidx): return None
  dt, expr = bidx.src[0].dtype, ctx.render_access(bidx)
  if bidx.max_numel() == 1: expr = f'({expr} != 0)' if dt == dtypes.bool else f'{ctx.render_dtype(dt)}({expr})'
  return f'({ctx[gate]}?{expr}:{ctx[var]})' if gate is not None else expr

def render_narrow_store(ctx, bidx, var):
  if not narrow_storage(bidx): return None
  dt, typ = bidx.src[0].dtype, STORAGE_TYPES[bidx.src[0].dtype]
  def value(i):
    v = f'{ctx[var]}.{"xyzw"[i]}' if i is not None else ctx[var]
    return f'{typ}({v} ? 1 : 0)' if dt == dtypes.bool else f'{typ}({v})'
  if var.max_numel() == 1: return f'{ctx[bidx.src[0]]}[{ctx[bidx.src[1]]}] = {value(None)};'
  return '\n'.join(f'{ctx[bidx.src[0]]}[{ctx[bidx.src[1]]}+{i}] = {value(i)};' for i in range(var.max_numel()))

def render_cmp_eq(a, b, dtype):
  return f'(!isnan({a}) && !isnan({b}) && ({a})==({b}))' if dtype in dtypes.floats else f'(({a})==({b}))'
def render_cmp_ne(a, b, dtype):
  return f'(isnan({a}) || isnan({b}) || ({a})!=({b}))' if dtype in dtypes.floats else f'(({a})!=({b}))'

def render_bitcast(ctx, x):
  if x.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL): return None
  fn = {(dtypes.float, dtypes.int):'intBitsToFloat', (dtypes.float, dtypes.uint):'uintBitsToFloat',
    (dtypes.int, dtypes.float):'floatBitsToInt', (dtypes.uint, dtypes.float):'floatBitsToUint',
    (dtypes.half, dtypes.int16):'int16BitsToFloat16', (dtypes.half, dtypes.uint16):'uint16BitsToFloat16',
    (dtypes.int16, dtypes.half):'float16BitsToInt16', (dtypes.uint16, dtypes.half):'float16BitsToUint16',
    (dtypes.uint, dtypes.int):'uint', (dtypes.int, dtypes.uint):'int',
    (dtypes.u64, dtypes.i64):'uint64_t', (dtypes.i64, dtypes.u64):'int64_t',
    (dtypes.u16, dtypes.i16):'uint16_t', (dtypes.i16, dtypes.u16):'int16_t',
    (dtypes.u8, dtypes.i8):'uint8_t', (dtypes.i8, dtypes.u8):'int8_t'}.get((x.dtype, x.src[0].dtype))
  return f'{fn}({ctx[x.src[0]]})' if fn is not None else None

class MALIRenderer(OpenCLRenderer):
  has_local, has_shared = True, True
  global_max, local_max = (65535, 65535, 65535), (256, 256, 256)
  shared_max = 32768
  smem_align = ''
  float4 = 'vec4'
  infinity, nan = 'uintBitsToFloat(0x7f800000u)', 'uintBitsToFloat(0x7fffffffu)'
  code_for_workitem = {'g': lambda x: f'int(gl_WorkGroupID[{x}])', 'l': lambda x: f'int(gl_LocalInvocationID[{x}])'}
  type_map = {**CStyleLanguage.type_map, dtypes.i8:'int', dtypes.u8:'uint', dtypes.i16:'int', dtypes.u16:'uint',
              dtypes.i32:'int', dtypes.u32:'uint', dtypes.i64:'int64_t', dtypes.u64:'uint64_t', dtypes.f16:'float16_t', dtypes.bf16:'uint'}
  code_for_op = {**{k:v for k,v in CStyleLanguage.code_for_op.items() if k != Ops.SIN},
    Ops.CMPEQ: render_cmp_eq, Ops.CMPNE: render_cmp_ne,
    Ops.AND: lambda a,b,dtype: f'({a}{"&&" if dtype == dtypes.bool else "&"}{b})',
    Ops.OR: lambda a,b,dtype: f'({a}{"||" if dtype == dtypes.bool else "|"}{b})',
    Ops.XOR: lambda a,b,dtype: f'({a}{"!=" if dtype == dtypes.bool else "^"}{b})'}
  string_rewrite = PatternMatcher([
    (UPat.cvar('c').cast(dtypes.bool), lambda ctx,c: 'true' if c.val else 'false'),
    (UPat(Ops.RECIP, name='x'), lambda ctx,x: f'({ctx.render_dtype(x.dtype)}(1)/{ctx[x.src[0]]})'),
    (UPat(Ops.TRUNCMOD, name='x'), lambda ctx,x: f'(({ctx[x.src[0]]})-({ctx[x.src[0]]})/({ctx[x.src[1]]})*({ctx[x.src[1]]}))'
      if x.src[0].dtype in dtypes.sints else None),
    (UPat(Ops.CMPLT, name='x'), lambda ctx,x: f'({ctx[x.src[0]]}<{ctx[x.src[1]]})' if x.src[0].dtype != dtypes.bool
      else f'(int({ctx[x.src[0]]})<int({ctx[x.src[1]]}))'),
    (UPat(Ops.CMPEQ, name='x'), lambda ctx,x: render_cmp_eq(ctx[x.src[0]], ctx[x.src[1]], x.src[0].dtype)),
    (UPat(Ops.CMPNE, name='x'), lambda ctx,x: render_cmp_ne(ctx[x.src[0]], ctx[x.src[1]], x.src[0].dtype)),
    (UPat(Ops.LOAD, src=(UPat.var('bidx'),)), lambda ctx,bidx: render_narrow_load(ctx, bidx)),
    (UPat(Ops.LOAD, src=(UPat.var('bidx'), UPat.var('var'), UPat.var('gate'))), lambda ctx,bidx,var,gate: render_narrow_load(ctx, bidx, var, gate)),
    (UPat(Ops.STORE, src=(UPat.var('bidx'), UPat.var('var'))), render_narrow_store),
    (UPat(Ops.STACK, name='x'), lambda ctx,x: f"{ctx.render_type(x)}(" + ','.join(ctx[y] for y in x.src) + ')'),
    (UPat(Ops.BITCAST, name='x'), lambda ctx,x: ctx[x.src[0]] if x.dtype == x.src[0].dtype else None),
    (UPat(Ops.BITCAST, name='x'), render_bitcast),
    (UPat(Ops.STORE, name='s'), lambda ctx,s: '\n'.join(
      f'{ctx[s.src[0].src[0]]}[{ctx[s.src[0].src[1]]}+{i}] = {ctx[s.src[1]]}.{"xyzw"[i]};' for i in range(s.src[1].max_numel()))
      if 1 < s.src[1].max_numel() <= 4 and s.src[0].op in (Ops.INDEX, Ops.SHRINK) and s.src[0].src[0].dtype in VECTOR_TYPES else None),
    (UPat.var('buf').index(UPat.var('idx_y'), UPat.var('idx_x')), lambda ctx,buf,idx_y,idx_x: f'IMAGE<{ctx[buf]}, {ctx[idx_y]}, {ctx[idx_x]}>'),
    (UPat(Ops.LOAD, dtype=dtypes.float, src=(UPat.var('buf').index(UPat.var('idx_y'), UPat.var('idx_x')), UPat.var('var'), UPat.var('gate'))),
      lambda ctx,buf,idx_y,idx_x,var,gate: f'({ctx[gate]}?imageLoad({ctx[buf]}, ivec2({ctx[idx_x]},{ctx[idx_y]})):{ctx[var]})'),
    (UPat(Ops.LOAD, dtype=dtypes.float, src=(UPat.var('buf').index(UPat.var('idx_y'), UPat.var('idx_x')),)),
      lambda ctx,buf,idx_y,idx_x: f'imageLoad({ctx[buf]}, ivec2({ctx[idx_x]},{ctx[idx_y]}))'),
    (UPat(Ops.STORE, src=(UPat.var('buf').index(UPat.var('idx_y'), UPat.var('idx_x')), UPat.var('var', dtypes.float))),
      lambda ctx,buf,idx_y,idx_x,var: f'imageStore({ctx[buf]}, ivec2({ctx[idx_x]},{ctx[idx_y]}), {ctx[var]});'),
  ]) + OpenCLRenderer.string_rewrite

  def __init__(self, target:Target):
    super().__init__(target)
    from tinygrad.runtime.support.compiler_mali import MALICompiler
    self.compiler = MALICompiler()

  def supported_dtypes(self): return {d for d in CStyleLanguage.supported_dtypes(self) if d != dtypes.double}

  def _render_dtype(self, dtype, sz=1, addrspace=AddrSpace.ALU, mutable=True, override_ptr=False, shape=None):
    if sz > 1 and not is_image_shape(shape):
      if dtype == dtypes.bool: return f'bvec{sz}'
      if dtype == dtypes.f32: return f'vec{sz}'
      if dtype == dtypes.f16: return f'f16vec{sz}'
      if dtype in dtypes.sints: return f'ivec{sz}'
      if dtype in dtypes.uints: return f'uvec{sz}'
    return super()._render_dtype(dtype, sz, addrspace, mutable, override_ptr, shape)

  def render_buffer(self, x):
    if x.addrspace == AddrSpace.LOCAL:
      typ = STORAGE_TYPES.get(x.dtype, self._render_dtype(x.dtype))
      return f'{self.smem_prefix}{typ} {self[x]}[{x.max_numel()}];'
    return super().render_buffer(x)

  def render_index(self, x, buf, idx):
    if buf.addrspace == AddrSpace.ALU: return super().render_index(x, buf, idx)
    return f'{self[buf]}[{self[idx]}]'

  def render_ptr(self, u):
    ptr = u
    while ptr.op is Ops.BITCAST: ptr = ptr.src[0]
    if ptr.op in (Ops.INDEX, Ops.SHRINK) and len(ptr.src) >= 2:
      buf, idx = ptr.src[:2]
      if buf.dtype in STORAGE_TYPES and 1 < u.max_numel() <= 4:
        typ = f'bvec{u.max_numel()}' if buf.dtype == dtypes.bool else self._render_dtype(buf.dtype, u.max_numel())
        vals = (f'({self[buf]}[({self[idx]})+{i}] != 0)' if buf.dtype == dtypes.bool
                else f'{self.render_dtype(buf.dtype)}({self[buf]}[({self[idx]})+{i}])' for i in range(u.max_numel()))
        return f'{typ}(' + ','.join(vals) + ')'
      if buf.dtype in (dtypes.float, dtypes.half, dtypes.int, dtypes.uint) and 1 < u.max_numel() <= 4:
        typ = self._render_dtype(buf.dtype, u.max_numel())
        return f'{typ}(' + ','.join(f'{self[buf]}[({self[idx]})+{i}]' for i in range(u.max_numel())) + ')'
      return f'{self[buf]}[{self[idx]}]'
    return super().render_ptr(u)
  def render_access(self, u): return self.render_ptr(u)

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    local_size = [u.src[0].ssimplify() for u in sorted([u for u in uops if u.op is Ops.SPECIAL and u.arg[0] == 'l'], key=lambda u: u.arg)]
    if not local_size: local_size = [1]
    while len(local_size) < 3: local_size.append(1)
    declarations = []
    for i, (name, (u, mutable)) in enumerate(bufs):
      if is_image_shape(u._shape):
        fmt, access = ('rgba16f' if u.dtype.itemsize == 2 else 'rgba32f'), ('writeonly' if mutable else 'readonly')
        declarations.append(f'layout({fmt}, set={i // 16}, binding={i % 16}) {access} uniform image2D {name};')
      else:
        storage_type = STORAGE_TYPES.get(u.dtype, self._render_dtype(u.dtype))
        declarations.append(f'layout(std430, set={i // 16}, binding={i % 16}) buffer B{i} {{ {storage_type} {name}[]; }};')
    body = '\n'.join(kernel)
    body = body.replace('__local ', 'shared ').replace('__global ', '').replace('barrier(CLK_LOCAL_MEM_FENCE);', 'barrier();')
    body = re.sub(r'const sampler_t smp = [^;]+;\n', '', body)
    body = body.replace('half4', 'f16vec4').replace('half(', 'float16_t(')
    body = re.sub(r'\(\(vec4\)\((data\w+\[[^\]]+\])' + r'\)' * 2, r'\1', body)
    shared = re.findall(r'^\s*(shared\s+[^;]+;)\s*$', body, flags=re.M)
    body = re.sub(r'^\s*shared\s+[^;]+;\s*$', '', body, flags=re.M)
    header = '#version 450\n#extension GL_NV_explicit_typecast : require\n'
    if any(u.dtype in (dtypes.bool, dtypes.i8, dtypes.u8) and u.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL) for u in uops):
      header += '#extension GL_EXT_shader_8bit_storage : require\n#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n'
    if any(u.dtype in (dtypes.i16, dtypes.u16, dtypes.bf16, dtypes.half) and u.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL) for u in uops):
      header += '#extension GL_EXT_shader_16bit_storage : require\n'
    if any(u.dtype in (dtypes.i16, dtypes.u16) and u.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL) for u in uops):
      header += '#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n'
    if any(u.dtype == dtypes.half for u in uops): header += '#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n'
    if any(u.dtype in (dtypes.i64, dtypes.u64) for u in uops): header += '#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n'
    header += f'layout(local_size_x={local_size[0]}, local_size_y={local_size[1]}, local_size_z={local_size[2]}) in;\n'
    return header + '\n'.join(declarations + shared) + '\nvoid main() {\n' + body + '\n}\n'
