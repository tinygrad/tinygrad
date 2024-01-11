from __future__ import annotations
from typing import List, Tuple, Any, Optional, cast, DefaultDict, Dict, Union, Sequence, Final, Set, Iterator
import itertools, math, functools
from collections import defaultdict
from enum import Enum, auto
from dataclasses import dataclass

from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType
from tinygrad.helpers import colored, DEBUG, prod, getenv, all_same, to_function_name, flatten
from tinygrad.ops import LazyOp, UnaryOps, BinaryOps, TernaryOps, ReduceOps, ConstBuffer, MemBuffer, BufferOps, get_lazyop_info
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable, NumNode, Node, SumNode, MulNode, DivNode, ModNode, LtNode, AndNode
from tinygrad.codegen.kernel import LocalBuffer, Kernel
from tinygrad.features.image import to_image_idx

# bottom ones are asm only
class UOps(Enum):
  LOOP = auto(); IF = auto(); END = auto(); SPECIAL = auto() # loops can be global, local, or other # noqa: E702
  DEFINE_GLOBAL = auto(); DEFINE_LOCAL = auto(); DEFINE_ACC = auto() # this defines buffers # noqa: E702
  LOAD = auto(); STORE = auto(); CONST = auto(); BARRIER = auto(); PHI = auto() # noqa: E702
  ALU = auto(); WMMA = auto(); CAST = auto(); GEP = auto() # noqa: E702

@dataclass(eq=False)
class UOp:
  uop: UOps
  dtype: Optional[DType]
  vin: Tuple[UOp, ...]
  arg: Any
  def __repr__(self):
    return f"{str(self.uop):20s}: {str(self.dtype) if self.dtype is not None else '':25s} {str([x.uop for x in self.vin]):32s} {self.arg}"

def get_grouped_dims(prefix, start_dim, local_dims, maxdim:int=0):
  local_idxs = loop_local_idxs = [Variable(f"{prefix}{start_dim+i}", 0, s-1) for i,s in enumerate(local_dims[0:maxdim-1] + (prod(local_dims[maxdim-1:]),) if len(local_dims) > maxdim else local_dims)]  # noqa: E501
  if maxdim != 0 and len(local_dims) > maxdim:
    dd = local_idxs[maxdim-1]
    nli = []
    for s in local_dims[maxdim-1:][::-1]:
      nli.append(dd % s)
      dd //= s
    local_idxs = local_idxs[0:maxdim-1] + nli[::-1]
  return local_idxs, [x for x in loop_local_idxs if not isinstance(x, NumNode)]

def expand_idx(node:Node) -> Union[Variable, NumNode]: return next((v for v in node.vars() if v.expr is None), NumNode(0))
def iter_idxs(idxs:Tuple[Union[Variable, NumNode], ...]) -> Iterator[Tuple[int,...]]:
  yield from (x[::-1] for x in itertools.product(*[[x for x in range(v.min, v.max + 1)] for v in idxs[::-1]]))

# expand a Node into List[Node] that enumerates the underlying Variables from min to max
# expand increments earlier variables faster than later variables (as specified in the argument)
@functools.lru_cache(maxsize=None)
def expand_node(node:Node, idxs:Optional[Tuple[Union[Variable, NumNode], ...]]=None) -> List[Node]:
  if idxs is None: idxs = (expand_idx(node),)
  return [node.substitute({k:v for k,v in zip(idxs, (NumNode(x) for x in rep)) if isinstance(k, Variable)}) for rep in iter_idxs(idxs)]

class Linearizer(Kernel):
  def uop_alu_idx(self, a:UOp, b, ops, ctx:Linearizer, op, dtype=dtypes.int32):
    render_b:UOp = cast(UOp, (NumNode(b) if not isinstance(b, Node) else b).render(ops, ctx))
    return self.uop(UOps.ALU, dtype, (a, render_b), op)

  # NOTE: the consts have to be cached for deduping of downstream uops to work
  def const(self, b:Union[int,float], dtype=dtypes.int32, insert_before=None) -> UOp:
    return self.uop(UOps.CONST, dtype, tuple(), b, insert_before=insert_before)

  def cast(self, val: UOp, dtype) -> UOp: return self.uop(UOps.CAST, dtype, (val,)) if val.dtype != dtype else val

  def get_reduce_acc(self, reduceop:LazyOp):
    dtype = get_lazyop_info(reduceop).dtype
    if reduceop.op == ReduceOps.SUM: return 0.0 if dtypes.is_float(dtype) else 0
    elif reduceop.op == ReduceOps.MAX:
      if dtypes.is_int(dtype): return 0 if dtypes.is_unsigned(dtype) else -2**(dtype.itemsize*8-1)
      return -math.inf if dtypes.is_float(dtype) else False

  # NOTE: once images are loaded, we uop them as their base float
  def get_base_dtype(self, dt:DType): return dt.base if isinstance(dt, ImageDType) else dt

  render_ops: Any = { Variable: lambda self, ops, ctx: ctx.loop_uops[self.expr], NumNode: lambda self, ops, ctx: ctx.const(self.b),
                MulNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.MUL),
                DivNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.DIV),
                ModNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.MOD),
                LtNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.CMPLT, dtype=dtypes.bool),
    SumNode: lambda self,ops,ctx:
      functools.reduce(lambda a,b: ctx.uop_alu_idx(a, b, ops, ctx, BinaryOps.ADD), self.nodes[1:], self.nodes[0].render(ops,ctx)),
    AndNode: lambda self,ops,ctx:
      functools.reduce(lambda a,b: ctx.uop_alu_idx(a, b, ops, ctx, BinaryOps.MUL, dtype=dtypes.bool), self.nodes[1:], self.nodes[0].render(ops,ctx)) }

  def global_load(self, i:int, idxs:Sequence[Node], acc=None, barrier:Optional[UOp]=None) -> List[UOp]:
    buf = self.bufs[i]
    localtype = self.get_base_dtype(buf.dtype if acc is None else get_lazyop_info(self.reduceop).dtype)
    const = buf.val if isinstance(buf, ConstBuffer) else acc

    def rename_var(v: Union[Variable, NumNode], expr: str): return v if isinstance(v, NumNode) else Variable(expr, v.min, v.max)
    expand_vars = tuple([rename_var(expand_idx(idx), f"_uidx{j}") for j, idx in enumerate(idxs)])
    fake_idxs = [idx.substitute({eidx: ev}) if isinstance(eidx:=expand_idx(idx), Variable) else idx for idx, ev in zip(idxs, expand_vars)]

    dim, amt = None, 1
    # float 4 grouping
    if len(upcast_dim := self.get_float4_upcast_dim(i)) == 1 and len(float4_expand := expand_node(idxs[upcast_dim[0]])) in [4,2]:
      dim, amt = upcast_dim[0], len(float4_expand)
      g_idx, g_valid = self.sts[i].expr_idxs(fake_idxs[:dim] + [float4_expand[0]] + fake_idxs[dim+1:])
      # do not use float4 if idx is not aligned
      if g_idx != (g_idx//amt*amt): dim, amt = None, 1
    if dim is None:
      g_idx, g_valid = self.sts[i].expr_idxs(fake_idxs)

    if amt > 1: localtype = localtype.vec(amt)
    e_idxs, e_valids = expand_node(g_idx, expand_vars), expand_node(g_valid, expand_vars)

    ret = []
    invalid_value = 0 if dtypes.is_int(buf.dtype) else 0.0
    for idx, valid, rep_idx in zip(e_idxs, e_valids, iter_idxs(expand_vars)):
      this_const, idx, valid = (invalid_value, NumNode(0), NumNode(1)) if valid.max == 0 else (const, idx, valid)
      key = f"{acc}{localtype}{this_const if this_const is not None and acc is None else (buf.idx if isinstance(buf, MemBuffer) else cast(LocalBuffer, buf).name)}{idx.render()}{valid.render()}"  # noqa: E501
      if key not in self.load_cache:
        if acc is not None:
          self.load_cache[key] = self.uop(UOps.DEFINE_ACC, localtype, (), this_const, cachable=False)
        elif this_const is not None:
          self.load_cache[key] = self.const(this_const, localtype)
          if valid.min == 0 and valid.max == 1:
            valid_rendered = valid.render(self.render_ops, self)
            self.load_cache[key] = self.uop(UOps.ALU, localtype, (valid_rendered, self.load_cache[key], self.const(invalid_value, localtype)), TernaryOps.WHERE)  # noqa: E501
        elif isinstance(buf.dtype, ImageDType):
          buf_uop = self.buf_uops[i]
          assert buf_uop is not None, f"buffer {i} wasn't UOped"
          image_idx, valid = to_image_idx(buf.dtype.shape, idx, valid)
          rendered_idx = self.uop(UOps.CAST, dtypes.int.vec(2), tuple(x.render(self.render_ops, self) for x in image_idx))
          valid_tuple = (valid.render(self.render_ops, self), self.const(invalid_value, buf.dtype.base.vec(4))) if valid.min == 0 else tuple()
          self.load_cache[key] = self.uop(UOps.LOAD, buf.dtype.base.vec(4), (buf_uop, rendered_idx) + valid_tuple + ((barrier,) if barrier else ()))
          if localtype == localtype.scalar():
            idx_small = idx%4
            res = idx_small.render(self.render_ops, self)
            out = self.uop(UOps.GEP, localtype, (self.load_cache[key],), idx_small.max)
            for ix in range(idx_small.max, idx_small.min, -1):
              rvv = self.uop(UOps.GEP, localtype, (self.load_cache[key],), ix-1)
              sel = self.uop(UOps.ALU, dtypes.bool, (res, self.const(ix)), BinaryOps.CMPLT)
              out = self.uop(UOps.ALU, localtype, (sel, rvv, out), TernaryOps.WHERE)
            self.load_cache[key] = out
        else:
          buf_uop = self.buf_uops[i]
          assert buf_uop is not None, f"buffer {i} wasn't UOped"
          rendered_idx = idx.render(self.render_ops, self)
          valid_tuple = (valid.render(self.render_ops, self), self.const(invalid_value, localtype)) if valid.min == 0 else tuple()
          self.load_cache[key] = self.uop(UOps.LOAD, localtype, (buf_uop, rendered_idx) + valid_tuple + ((barrier,) if barrier else ()))
      ret.append(self.uop(UOps.GEP, localtype.scalar(), (self.load_cache[key],), rep_idx[dim]) if dim is not None else self.load_cache[key])
    return ret

  def global_store(self, i:int, idxs:List[Node], store:List[UOp]) -> List[UOp]:
    buf = self.bufs[i]
    buf_uop = self.buf_uops[i]
    assert buf_uop is not None, f"buffer {i} wasn't UOped"

    expanded_nodes = [expand_node(idx) for idx in idxs]
    _idxs = [x[::-1] for x in itertools.product(*expanded_nodes[::-1])]
    store_offset = dict(zip(_idxs, store))

    # float4 grouping
    if len(upcast_dim := self.get_float4_upcast_dim(i)) == 1 and len(float4_expand := expanded_nodes[upcast_dim[0]]) in [2,4]:
      grouped_store_offset = defaultdict(list)
      for k in store_offset:
        _idx = k[:upcast_dim[0]] + (float4_expand[0],) + k[upcast_dim[0]+1:]
        grouped_store_offset[_idx].append(store_offset[k])
      store_offset_new = {}
      for k,grouped in grouped_store_offset.items():
        amt = len(grouped)
        idx, valid = self.sts[i].expr_idxs(k)
        assert idx == ((idx//amt)*amt), "float4 stores are always aligned"
        store_offset_new[k] = self.uop(UOps.CAST, buf.dtype.vec(amt), tuple(grouped))
      store_offset = store_offset_new

    stores = []
    for idx, var in store_offset.items():
      idx, valid = self.sts[i].expr_idxs(idx)
      if isinstance(buf.dtype, ImageDType):
        idx, valid = to_image_idx(buf.dtype.shape, idx, valid)
        rendered_idx = self.uop(UOps.CAST, dtypes.int.vec(2), tuple(x.render(self.render_ops, self) for x in idx))
      else:
        rendered_idx = idx.render(self.render_ops, self)
      if valid.min == 1: stores.append(self.uop(UOps.STORE, None, (buf_uop, rendered_idx, var)))
      else: stores.append(self.uop(UOps.STORE, None, (buf_uop, rendered_idx, var, valid.render(self.render_ops, self))))
    return stores

  kernel_cnt: Final[DefaultDict[str, int]] = defaultdict(int)
  def linearize(self):
    # no new opts and we already ran? skip relinearizing
    if self.applied_opts == self.applied_opts_cache: return self

    # save backups
    sts_backup, gfr_backup, upc_backup = self.sts[:], self.group_for_reduce[:], self.upcasted

    # global uop cache
    self.saved_exprs: Dict[Tuple, UOp] = dict()

    # limit dims if we need to
    if self.opts.global_max and self.opts.local_max: self.limit_dims_to_max(self.opts.global_max, self.opts.local_max)

    # uops
    self.uops: List[UOp] = []
    self.buf_uops: List[Optional[UOp]] = [None]*len(self.bufs)
    self.loop_uops: Dict[str, UOp] = {}

    # add global buffers
    for i,buf in enumerate(self.bufs):
      if isinstance(buf, MemBuffer):
        self.buf_uops[i] = self.uop(UOps.DEFINE_GLOBAL, buf.dtype if isinstance(buf.dtype, ImageDType) else PtrDType(buf.dtype), (), f"data{buf.idx}")
    # add var vals
    for var in self.ast.vars():
      assert var.expr is not None
      self.loop_uops[var.expr] = self.uop(UOps.DEFINE_GLOBAL, dtypes.int32, (), var.expr)
    # define local buffers
    for lb in self.local_alias.values():
      self.buf_uops[self.bufs.index(lb)] = self.uop(UOps.DEFINE_LOCAL, PtrDType(dtypes.float32), (), (lb.name, self.sts[self.bufs.index(lb)].size))
    # add a local buffer for multistage reduce. # TODO: use local alias
    if self.group_for_reduce:
      # TODO: the strides of this can be controlled
      self.sts.append(ShapeTracker.from_shape(tuple([1] * self.global_dims + list(self.full_shape[self.global_dims:self.global_dims+self.local_dims+len(self.group_for_reduce)]) + [1] * (self.shape_len - self.upcasted - len(self.group_for_reduce) - self.first_reduce) + [x[0] for x in self.upcasted_axis(0)])))  # noqa: E501
      temp_dtype = self.get_base_dtype(get_lazyop_info(self.reduceop).dtype)
      self.bufs.append(LocalBuffer("temp", self.sts[-1].size, temp_dtype))
      self.buf_uops.append(self.uop(UOps.DEFINE_LOCAL, PtrDType(temp_dtype), (), ("temp", self.sts[-1].size)))

    # kernel name (before late upcast)
    self.name = ("r_" if self.reduceop else "E_") + colored('_', 'BLACK').join([colored(str(x), c) for x,c in zip(self.full_shape, self.colors())])

    # name the function something unique
    Linearizer.kernel_cnt[(function_name := to_function_name(self.name))] += 1
    suffix = f"{'n'+str(Linearizer.kernel_cnt[function_name]-1)}" if Linearizer.kernel_cnt[function_name] > 1 else ""
    self.name = self.name+colored(suffix, 'BLACK')

    # define indexes
    global_idxs, loop_global_idxs = get_grouped_dims("gidx", 0, self.full_shape[:self.global_dims], 3 if self.opts.has_local else 0)
    local_idxs, loop_local_idxs = get_grouped_dims("lidx", self.global_dims, self.full_shape[self.global_dims:self.first_reduce+len(self.group_for_reduce)], 3 if self.opts.has_local else 0)  # noqa: E501
    full_upcast_idxs = [Variable(None, 0, s-1) for s in self.full_shape[self.shape_len-self.upcasted:]]
    upcast_idxs = [Variable(None, 0, s-1) for s in self.output_shape[self.shape_len-self.upcasted:]]

    # global and local loops
    def render_loop(xx:List[Variable]) -> Tuple[UOp, ...]:
      new_loops = {x.expr:self.uop(UOps.LOOP, dtypes.int32, (
        self.const(x.min) if isinstance(x.min, int) else cast(Node, x.min).render(self.render_ops, self),
        self.const(x.max+1) if isinstance(x.max, int) else cast(Node, x.max+1).render(self.render_ops, self)), cachable=False) for x in xx if not isinstance(x, NumNode) and x.expr is not None}  # noqa: E501
      self.loop_uops.update(new_loops)
      return tuple(new_loops.values())

    # set global/local size
    self.global_size: Optional[List[int]] = None
    self.local_size: Optional[List[int]] = None
    if self.dont_use_locals:
      self.global_size = [x.max+1 for x in loop_global_idxs][::-1]
      self.loop_uops.update({x.expr:self.uop(UOps.SPECIAL, dtypes.int32, (), (len(loop_global_idxs)-1-i, x.expr.replace("gidx", "idx"), x.max+1)) for i,x in enumerate(loop_global_idxs)})  # noqa: E501
    elif self.opts.has_local:
      self.global_size, self.local_size = [x.max+1 for x in loop_global_idxs][::-1], [x.max+1 for x in loop_local_idxs][::-1]
      self.loop_uops.update({x.expr:self.uop(UOps.SPECIAL, dtypes.int32, (), (len(loop_global_idxs)-1-i, x.expr, x.max+1)) for i,x in enumerate(loop_global_idxs)})  # noqa: E501
      self.loop_uops.update({x.expr:self.uop(UOps.SPECIAL, dtypes.int32, (), (len(loop_local_idxs)-1-i, x.expr, x.max+1)) for i,x in enumerate(loop_local_idxs)})  # noqa: E501
    else:
      render_loop(loop_global_idxs+loop_local_idxs)

    # parse AST
    loaded_buffers = {}
    acc: List[UOp] = []
    self.load_cache: Dict[str, UOp] = {}

    # reduce op
    fake_reduce_idxs: List[Variable] = []
    if self.reduceop is not None:
      # define indexes
      reduce_idxs = [Variable(f"ridx{i}", 0, self.full_shape[i]-1) for i in range(self.first_reduce+len(self.group_for_reduce), self.shape_len-self.upcasted)]  # noqa: E501
      fake_reduce_idxs = [x*0 for x in reduce_idxs]

      # define accumulator
      acc = self.global_load(0, global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, self.get_reduce_acc(self.reduceop))

      if self.tensor_core:
        def calc_tc_idxs(local_size: int, aliases: List[List[int]]):
          replace_idxs = []
          for alias in aliases:
            full_var, full_var_sz = NumNode(0), 1
            if alias[0] != 0:
              for i in alias:
                next_var = local_idxs[-i] if i > 0 else Variable(None, 0, local_size-1)
                full_var += next_var * full_var_sz
                full_var_sz *= next_var.max+1
            replace_idxs.append(full_var)
          return replace_idxs
        replace_acc_idxs = calc_tc_idxs(self.tensor_core.thread_local_sizes[2], self.tensor_core.thread_local_aliases[2])
        for n in range(len(self.tensor_core.threads)):
          local_idxs[self.local_dims-len(self.tensor_core.threads)+n] = replace_acc_idxs[n] # replace locals
        for n in range(len(replace_acc_idxs)-len(self.tensor_core.threads)):
          upcast_idxs[n] = replace_acc_idxs[len(self.tensor_core.threads)+n] # replace upcasts

      # reduce loop
      loop_ctx = render_loop(reduce_idxs)

      # barrier for fast GEMM
      if self.tensor_core: self.uop(UOps.BARRIER, None, (), cachable=False)

      # compute local aliases
      locals_to_store = []
      for i in self.local_alias:
        localbuf_idx = self.bufs.index(self.local_alias[i])
        buf_idxs = [idx*0 if s == 0 else idx for idx,s in zip(global_idxs+local_idxs+reduce_idxs+full_upcast_idxs,self.sts[i].real_strides())]
        if self.tensor_core:
          min_alias_idx = min(self.local_alias.keys())
          replace_input_idxs = calc_tc_idxs(self.tensor_core.thread_local_sizes[i-min_alias_idx], self.tensor_core.thread_local_aliases[i-min_alias_idx])  # noqa: E501
          for n in range(len(self.tensor_core.threads)):
            buf_idxs[self.first_reduce-len(self.tensor_core.threads)+n] = replace_input_idxs[n] # replace locals
          for n in range(len(replace_input_idxs)-len(self.tensor_core.threads)):
            buf_idxs[self.shape_len-self.upcasted+n] = replace_input_idxs[len(self.tensor_core.threads)+n] # replace upcasts
        if DEBUG >= 3: print(f"{localbuf_idx} alias {i}: idxs=", buf_idxs)
        ll = self.global_load(i, buf_idxs)
        locals_to_store.append((localbuf_idx, buf_idxs, ll))

      # copy in any global buffers
      if self.tensor_core:
        wmma_sz, dtype_in, dtype_out = self.tensor_core.thread_local_sizes, self.tensor_core.dtype_in, self.tensor_core.dtype_out
        # calculate the number of local accumulator reduces and render WMMAs: this is bad... this needs to come from someplace else
        nx, ny, nacc = (len(locals_to_store[0][2])//wmma_sz[0]), (len(locals_to_store[1][2])//wmma_sz[1]), (len(acc)//wmma_sz[2])
        acc_reds = math.isqrt((nx*ny)//nacc)
        i, bx, by = 0, nx//acc_reds, ny//acc_reds
        for y in range(by):
          for x in range(bx):
            for j in range(acc_reds):
              ops = (self.uop(UOps.CAST, dtype_in.vec(wmma_sz[0]), tuple(locals_to_store[0][2][(x+(j*bx))*wmma_sz[0]:(x+(j*bx)+1)*wmma_sz[0]])),
                     self.uop(UOps.CAST, dtype_in.vec(wmma_sz[1]), tuple(locals_to_store[1][2][(y+(j*by))*wmma_sz[1]:(y+(j*by)+1)*wmma_sz[1]])),
                     self.uop(UOps.CAST, dtype_out.vec(wmma_sz[2]), tuple(op3:=acc[i:i+wmma_sz[2]])))
              ret = self.uop(UOps.WMMA, dtype_out.vec(wmma_sz[2]), ops, (self.opts.device, self.tensor_core.dtype_in, self.tensor_core.dtype_out,))
              for z in range(cast(DType, ret.dtype).sz):
                acc[i+z] = self.uop(UOps.PHI, dtypes.float, (op3[z], self.uop(UOps.GEP, dtypes.float, (ret,), z)) + loop_ctx)
            i += wmma_sz[2]
      else:
        if locals_to_store:
          self.uop(UOps.BARRIER, None, (), cachable=False)
          for i, idxs, ll in locals_to_store: self.global_store(i, idxs, ll)
          self.uop(UOps.BARRIER, None, (), cachable=False)

        # load earlybufs
        loaded_buffers.update({b:self.global_load(self.bufs.index(self.local_alias[i]) if i in self.local_alias else i, global_idxs+local_idxs+reduce_idxs+full_upcast_idxs) for i,b in enumerate(self.bufs[1:], start=1) if b in self.earlybufs})  # noqa: E501

        # run early AST (with reduce)
        self.ast_parse(self.reduceop, acc, self.acc_offsets(self.full_buf_index), loaded_buffers, do_reduce=True, loop_ctx=loop_ctx)

      # end the reduce loop
      self.load_cache.clear()

      # end the local loop, do the local reduce
      if self.group_for_reduce:
        fake_global_idxs = [x*0 for x in global_idxs]
        stores = self.global_store(-1, fake_global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, acc)  # store accumulators
        barrier = self.uop(UOps.BARRIER, None, tuple(stores), cachable=False)
        if self.opts.has_local:
          fake_idxs = [NumNode(0)]*len(self.sts[-1].shape)
          fake_idxs[self.global_dims+self.local_dims:self.global_dims+len(local_idxs)] = local_idxs[self.local_dims:]
          if_cond: UOp = (self.sts[-1].expr_idxs(fake_idxs)[0]<1).render(self.render_ops, self)
          barrier = self.uop(UOps.IF, None, (if_cond, barrier), cachable=False)

        # create new late reduce local loops and replace local_idxs that have been used
        end_local_idxs = [Variable(f"tidx{i}", 0, self.full_shape[i]-1 if i >= self.first_reduce and i not in self.upcast_in_mid_reduce_axes else 0) for i in range(0, self.first_reduce+len(self.group_for_reduce))]  # noqa: E501
        local_idxs = local_idxs[:self.local_dims] + end_local_idxs[self.global_dims + self.local_dims:]

        # if any group_for_reduce items aren't reduces, upcast them here
        for j in self.upcast_in_mid_reduce_axes:
          self.reshape_and_permute(None, [i for i in range(self.shape_len) if i != j] + [j])
          self.upcast()
          self.group_for_reduce.pop()
          local_idxs = local_idxs[:-1]
          end_local_idxs = end_local_idxs[:-1]
          # regenerate upcast_idxs
          upcast_idxs = [Variable(None, 0, s-1) for s in self.output_shape[self.shape_len-self.upcasted:]]

        # NOTE: this structure is the same as the reduce op above

        # define late accumulator
        acc = self.global_load(-1, fake_global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, self.get_reduce_acc(self.reduceop))

        # late reduce loop
        loop_ctx = render_loop(end_local_idxs)

        # load localbufs
        loaded_buffers[self.bufs[-1]] = self.global_load(-1, fake_global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, barrier=barrier)

        # there's no AST here (and there's no shape for the reduce LazyOp)
        self.ast_parse(LazyOp(self.reduceop.op, (LazyOp(BufferOps.LOAD, (), self.bufs[-1]),)), acc, self.acc_offsets(-1), loaded_buffers, do_reduce=True, loop_ctx=loop_ctx)  # noqa: E501

        # end the late reduce loop
        self.load_cache.clear()

    # load latebufs
    loaded_buffers.update({b:self.global_load(i, global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs) for i,b in enumerate(self.bufs) if b not in self.earlybufs and i != 0 and b.__class__ is not LocalBuffer})  # noqa: E501

    # run late AST (without the store)
    val = self.ast_parse(self.ast.src[0], acc, None, loaded_buffers)

    # store
    self.global_store(0, global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, val)

    # get PHI node loop scope, link anything using a DEFINE_ACC to the loop as a "parent"
    acc_scope: DefaultDict[UOp, List[UOp]] = defaultdict(list)
    for u in self.uops:
      if u.uop == UOps.PHI:
        acc_scope[u.vin[0]] += u.vin[2:]

    # graph helper functions
    @functools.lru_cache(None)
    def get_recursive_parents(x:UOp, with_phi=False) -> Set[UOp]:
      return set.union(set(x.vin), *[get_recursive_parents(p, with_phi) for p in x.vin], set(acc_scope[x]) if with_phi else set())

    def get_recursive_children(x:UOp) -> Set[UOp]:
      deps = set([x])
      ssize = 0
      while ssize != len(deps):
        ssize = len(deps)
        for u in self.uops:
          if len(deps.intersection([x for x in u.vin if x.uop != UOps.PHI])):
            deps.add(u)
      return deps

    def replace_op(old:UOp, new:UOp):
      for u in self.uops:
        u.vin = tuple(new if x is old else x for x in u.vin)
      self.uops.remove(old)

    # fix loop scope, push uops upward out of loop if it does not depend on the loop
    loop_stack: List[List[UOp]] = [[]]
    for u in self.uops:
      if not loop_stack[-1]: loop_stack[-1].append(u)
      elif u.uop == UOps.LOOP: loop_stack.append([u])
      elif u.uop not in [UOps.CONST, UOps.ALU, UOps.CAST, UOps.LOAD]: loop_stack[-1].append(u)
      else:
        parents = get_recursive_parents(u, with_phi=True)
        # don't push any local buffer because there might have STORE and BARRIER (not considered as parent) between DEFINE_LOCAL and here
        if any(u.uop == UOps.DEFINE_LOCAL for u in parents): loop_stack[-1].append(u)
        else:
          for i in reversed(range(len(loop_stack))):
            # check backwards and put the uop in the first encounter with some dependency
            if any(x in parents for x in loop_stack[i]) or i == 0:
              loop_stack[i].append(u)
              break
    self.uops = flatten(loop_stack)

    # uops optimization
    changed_something = True
    while changed_something:
      changed_something = False
      for u in self.uops:
        if u.uop == UOps.PHI and len(u.vin) == 3:
          # if the parents of the PHI node don't have the LOOP in their parents, it can be folded
          # TODO: ADD becomes a MUL, MAX can just become nothing
          if all(x.uop != UOps.LOOP for x in get_recursive_parents(UOp(u.uop, u.dtype, u.vin[0:2], u.arg))) and u.vin[1].arg == BinaryOps.ADD:
            if DEBUG >= 4: print(f"removing PHI node {u}")
            del self.saved_exprs[(u.uop, u.dtype, u.vin, u.arg)]
            # NOTE: assuming u.vin[2].vin[1] and u.vin[2].vin[0] have the same dtype
            loop_len = self.uop(UOps.ALU, u.vin[2].vin[1].dtype, (u.vin[2].vin[1], u.vin[2].vin[0]), BinaryOps.SUB, insert_before=self.uops.index(u))
            if loop_len.dtype != u.dtype: loop_len = self.uop(UOps.CAST, u.dtype, (loop_len,), insert_before=self.uops.index(u))
            replace_op(u, self.uop(UOps.ALU, u.dtype, (u.vin[1], loop_len,), BinaryOps.MUL, insert_before=self.uops.index(u)))
            changed_something = True

    # (recursively) remove childless uops
    # NOTE: DEFINE_GLOBAL should be removable, but we'd have to propagate that
    UOPS_W_SIDE_EFFECTS = {UOps.STORE, UOps.BARRIER, UOps.DEFINE_GLOBAL}
    while 1:
      has_child: Set[UOp] = set()
      for ru in self.uops:
        for vu in ru.vin:
          has_child.add(vu)
      nu: List[UOp] = [x for x in self.uops if x in has_child or x.uop in UOPS_W_SIDE_EFFECTS]
      if len(nu) == len(self.uops): break
      if DEBUG >= 4: print(f"reduced UOp count from {len(self.uops)} to {len(nu)}")
      self.uops = nu
      del nu

    # add UOps.END
    for u in self.uops:
      if u.uop == UOps.LOOP:
        # add END of loops after the last thing that (recursively) depends on them
        self.uop(UOps.END, None, (u,), cachable=False, insert_before=self.uops.index(sorted(list(get_recursive_children(u)), key=self.uops.index)[-1])+1)  # noqa: E501
      elif u.uop == UOps.IF:
        # END any if statements at the end of the uops
        self.uop(UOps.END, None, (u,), cachable=False)

    # maybe graph the uops
    if DEBUG >= 5:
      for u in self.uops:
        print(f"{self.uops.index(u):4d} {str(u.uop):20s}: {str(u.dtype) if u.dtype is not None else '':25s} {str([self.uops.index(x) for x in u.vin]):32s} {u.arg}")  # noqa: E501
    if getenv("GRAPHUOPS"):
      from tinygrad.graph import graph_uops
      graph_uops(self.uops)

    # restore backups
    self.sts, self.group_for_reduce, self.upcasted = sts_backup, gfr_backup, upc_backup

    # set cache and return
    self.applied_opts_cache = self.applied_opts[:]
    return self

  def uop(self, uop:UOps, dtype:Optional[DType]=None, vin:Tuple[UOp, ...]=tuple(), arg:Any=None, cachable=True, insert_before=None, simplify=True) -> UOp:  # noqa: E501
    if uop == UOps.ALU:
      if arg in UnaryOps:
        assert dtype == vin[0].dtype, f"{arg} dtype mismatch {dtype=} != {vin[0].dtype=}"
      elif arg in (BinaryOps.CMPLT, BinaryOps.CMPEQ):
        assert dtype == dtypes.bool, f"{arg} output dtype mismatch {dtype=} != {dtypes.bool}"
        assert vin[0].dtype == vin[1].dtype, f"{arg} dtype mismatch {dtype=} != {vin[0].dtype=} != {vin[1].dtype=}"
      elif arg in BinaryOps:
        assert dtype == vin[0].dtype == vin[1].dtype, f"{arg} dtype mismatch {dtype=} != {vin[0].dtype=} != {vin[1].dtype=}"
      elif arg == TernaryOps.WHERE:
        assert vin[0].dtype == dtypes.bool, f"{arg} selector dtype mismatch {vin[0].dtype=} != {dtypes.bool}"
        assert dtype == vin[1].dtype == vin[2].dtype, f"{arg} choice dtype mismatch {dtype=} != {vin[1].dtype=} != {vin[2].dtype=}"

    if simplify:
      if uop == UOps.PHI and len(vin) == 2: return vin[1]   # a phi without loops is a noop
      if uop == UOps.GEP and vin[0].uop == UOps.CONST: return self.const(vin[0].arg, dtype, insert_before)
      if uop == UOps.CAST and all(x.uop == UOps.CONST for x in vin) and all_same([x.arg for x in vin]):
        return self.const(vin[0].arg, dtype, insert_before)
      if uop == UOps.ALU:
        # rewrites. NOTE: the rewritten NEG op is still around...
        if arg == BinaryOps.ADD and vin[1].uop == UOps.ALU and vin[1].arg == UnaryOps.NEG:
          return self.uop(UOps.ALU, dtype, (vin[0], vin[1].vin[0]), BinaryOps.SUB, cachable, insert_before)
        # constant folding
        if arg == UnaryOps.NEG and vin[0].uop == UOps.CONST: return self.const(-vin[0].arg, dtype, insert_before)
        if arg == TernaryOps.WHERE and vin[1] == vin[2]: return vin[1] # a conditional with the same results either way is a noop
        # zero folding
        for x in [0,1]:
          if arg == BinaryOps.ADD and vin[x].uop == UOps.CONST and vin[x].arg == 0.0: return vin[1-x]
          if arg == BinaryOps.MUL and vin[x].uop == UOps.CONST and vin[x].arg == 1.0: return vin[1-x]
          if arg == BinaryOps.MUL and vin[x].uop == UOps.CONST and vin[x].arg == 0.0: return vin[x]
        if arg == BinaryOps.SUB and vin[1].uop == UOps.CONST and vin[1].arg == 0.0: return vin[0]
        if arg == BinaryOps.DIV and vin[1].uop == UOps.CONST and vin[1].arg == 1.0: return vin[0]

    key = (uop, dtype, vin, arg)
    if insert_before is None: insert_before = len(self.uops)
    # check if the cached expr is valid with the given insert place.
    if cachable and (expr:=self.saved_exprs.get(key, None)) is not None and self.uops.index(expr) <= insert_before: return expr
    ret = UOp(uop, dtype, vin, arg)
    self.uops.insert(insert_before, ret)
    if cachable: self.saved_exprs[key] = ret
    return ret

  def ast_parse(self, x:LazyOp, acc: List[UOp], offs:Optional[List[int]], loaded_buffers:Dict[Union[MemBuffer, ConstBuffer, LocalBuffer], List[UOp]], do_reduce=False, loop_ctx=tuple(), cache=None) -> List[UOp]:  # noqa: E501
    if cache is None: cache = {}
    if x in cache: return cache[x]
    if x.op in BufferOps: return loaded_buffers[x.arg]
    if x.op == UnaryOps.CAST:
      return [self.uop(UOps.CAST, self.get_base_dtype(x.arg[0]), (u,), x.arg) for u in self.ast_parse(x.src[0], acc, offs, loaded_buffers)]
    if x.op in ReduceOps and not do_reduce:
      assert offs is None, "not available if we aren't doing reduce"
      return acc
    # MULACC fusion. TODO: this is copied from Interpreted
    if x.op == ReduceOps.SUM:
      if x.src[0].op == BinaryOps.MUL: x = LazyOp(TernaryOps.MULACC, x.src[0].src, x.arg)
      if (castop:=x.src[0]).op == UnaryOps.CAST and (mulop:=castop.src[0]).op == BinaryOps.MUL:
        # MULACC with acc cast rewrite: MUL -> CAST -> SUM => CAST -> MULACC
        x = LazyOp(TernaryOps.MULACC, tuple(LazyOp(UnaryOps.CAST, (s, ), castop.arg) for s in mulop.src), x.arg)

    values = [self.ast_parse(v, acc, offs, loaded_buffers, loop_ctx=loop_ctx, cache=cache) for v in x.src]
    ops = {ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX, TernaryOps.MULACC:TernaryOps.MULACC}
    if x.op in ops:
      ret: List[UOp] = []
      input_acc = acc[:]
      for val, off in zip(zip(*values), cast(List[int], offs)):
        acc[off] = self.uop(UOps.ALU, acc[off].dtype, vin=val+(acc[off],), arg=ops[x.op])
        ret.append(acc[off])
      for off in range(len(acc)):
        if input_acc[off] != acc[off]:
          acc[off] = self.uop(UOps.PHI, input_acc[off].dtype, (input_acc[off], acc[off]) + tuple(loop_ctx))
    else:
      ret = [self.uop(UOps.ALU, dtypes.bool if x.op in (BinaryOps.CMPLT, BinaryOps.CMPEQ) else val[-1].dtype, val, x.op) for val in zip(*values)]
    cache[x] = ret
    return ret
