from __future__ import annotations
from typing import List, Tuple, Any, Optional, cast, DefaultDict, Dict, Union, Final, Iterator, Sequence
import itertools, math, functools
from collections import defaultdict

from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType
from tinygrad.helpers import colored, DEBUG, dedup, diskcache_put, prod, getenv, to_function_name, flatten
from tinygrad.ops import LazyOp, UnaryOps, BinaryOps, TernaryOps, ReduceOps, ConstBuffer, MemBuffer, BufferOps, get_lazyop_info
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable, NumNode, Node, SumNode, MulNode, DivNode, ModNode, LtNode, AndNode, create_lt_node, sint
from tinygrad.codegen.kernel import LocalBuffer, Kernel
from tinygrad.renderer import Program

from tinygrad.codegen.uops import UOps, UOp, UOpGraph

def get_grouped_dims(prefix:str, off:int, dims:Tuple[sint, ...], max_sizes:Optional[Tuple[int, ...]], reverse_dims:bool=False):
  """ Maps all global/local dims onto global/local sizes and returns the idxs, loop_idxs and sizes.

  * If there are fewer dims than size, size will be padded with 1s to the length of max_sizes.
  * If there are more dims than size, dims will be collapsed onto size starting from left-most (i.e. onto x, then y, then z).
  * If the dim is too large for the size, the dim will be split between adjacent size axes space permitting, otherwise assert

  Keyword arguments:
  prefix -- the prefix to use for the size Variable names.
  dims -- the global or local dims of the full shape.
  max_sizes -- the maximum values for each size in (x, y, z) order.
  reverse_dims -- reverse the order of the dims as they are mapped into size, i.e. if True, the right dim will go to the left size (.x).
  """

  # check the edge cases on max_sizes
  if max_sizes is None: max_sizes = tuple([0xFFFFFFFFFFFFFFFF] * len(dims))
  assert len(max_sizes) > 0 or len(dims) == 0, f"{prefix} dims should be empty because no size axes available"
  if len(max_sizes) == 0: return [], [], None

  # initialize the map of dims to size with a single dim in each size axis
  # TODO: support sint properly
  size_dims:List[List[Tuple[int, sint]]] = [[(dim_idx, dim)] for dim_idx, dim in enumerate(dims)]

  # reverse the order of the dims to size map, if desired (currently for globals where smallest stride is on the right)
  # TODO: remove reverse_dims, the mapping of dims to size for globals should be cosearched with memory layouts for optimal peformance
  if reverse_dims: size_dims = size_dims[::-1]

  # ensure that the initial dims initially fit the valid size axes
  for size_idx, max_sz in [(i, sz) for i, sz in enumerate(max_sizes[:len(size_dims)]) if size_dims[i][0][1] > sz]:
    # if the initial dim is too large, split the dim to separate size axes, if possible
    dim_idx, dim_max = size_dims[size_idx][0]
    assert isinstance(dim_max, int), "variable shape too large for size"
    for factor in range(2, int(dim_max**0.5)+1):
      if dim_max % factor == 0 and dim_max // factor <= max_sz:
        size_dims = size_dims[:size_idx] + [[(dim_idx, dim_max//factor)], [(dim_idx, factor)]] + size_dims[size_idx+1:]
        break
    assert size_dims[size_idx][0][1] <= max_sz, f"dim at {size_idx} too large and non-factorable: {dim_max} > {max_sz}"

  # compress the extra dims, collapsing them onto the left-most valid size axis
  # for run_process_replay, collapse onto the right-most dim to compare the outputs.  TODO: remove
  if reverse_dims: size_dims, max_sizes = size_dims[::-1], max_sizes[::-1]
  cur_size_idx = 0
  while len(size_dims) > len(max_sizes):
    if prod([dim_max for (_, dim_max) in size_dims[cur_size_idx]])*size_dims[cur_size_idx+1][0][1] < max_sizes[cur_size_idx]:
      size_dims = size_dims[:cur_size_idx] + [size_dims[cur_size_idx] + size_dims[cur_size_idx+1]] + size_dims[cur_size_idx+2:]
    elif cur_size_idx < len(max_sizes)-1: cur_size_idx += 1
    else: raise AssertionError(f"cannot fit dims in size: {dims=} {max_sizes=}")
  if reverse_dims: size_dims, max_sizes = size_dims[::-1], max_sizes[::-1]

  # construct the final dim idx variables from the the portions of the size variables
  sizes, idxs = [prod([dim_max for (_, dim_max) in size_dim]) for size_dim in size_dims], [NumNode(0)] * len(dims)
  size_vars = loop_idxs = [Variable(f"{prefix}{len(sizes)-1-(i+off) if reverse_dims else i+off}", 0, s-1) for i,s in enumerate(sizes)]
  for size_idx, size_var in enumerate(size_vars):
    for dim_idx, dim_max in size_dims[size_idx]:
      idxs[dim_idx] += (size_var % dim_max) * (idxs[dim_idx].max+1)
      size_var //= dim_max

  # pad the final sizes array to the proper length if necessary
  return idxs, [x for x in loop_idxs if not isinstance(x, NumNode)], sizes + [1]*(len(max_sizes)-len(sizes))

def expand_idx(node:Node) -> Union[Variable, NumNode]: return next((v for v in node.vars() if v.expr.startswith("_uidx")), NumNode(0))
def expand_idxs(nodes:Sequence[Node]) -> Tuple[Union[Variable, NumNode], ...]:
  eidxs = [expand_idx(node) for node in nodes]
  return tuple([v if v not in eidxs[:j] else NumNode(0) for j, v in enumerate(eidxs)])  # take only first occurrence of expand variable
def iter_idxs(idxs:Tuple[Union[Variable, NumNode], ...]) -> Iterator[Tuple[int,...]]:
  yield from (x[::-1] for x in itertools.product(*[[x for x in range(v.min, v.max + 1)] for v in idxs[::-1]]))

def to_image_idx(base_shape:Tuple[int, ...], idxy:Node, valid:Node) -> Tuple[Tuple[Node, Node], Node]:
  idx, idy = (idxy // 4) % base_shape[1], (idxy // (4 * base_shape[1]))
  # TODO: bring back the valid removal logic (correct!)
  if DEBUG>=5: print("to_image_idx", base_shape, idx.min, idx.max, idy.min, idy.max, idx, idy, valid)
  return (idx, idy), valid

# expand a Node into List[Node] that enumerates the underlying Variables from min to max
# expand increments earlier variables faster than later variables (as specified in the argument)
@functools.lru_cache(maxsize=None)
def expand_node(node:Node, idxs:Optional[Tuple[Union[Variable, NumNode], ...]]=None) -> List[Node]:
  if idxs is None: idxs = (expand_idx(node),)
  return [node.substitute({k:v for k,v in zip(idxs, (NumNode(x) for x in rep)) if isinstance(k, Variable)}) for rep in iter_idxs(idxs)]

class Linearizer(Kernel):
  def uop_alu_idx(self, a:UOp, b, ops, ctx:Linearizer, op): return UOp.alu(op, a, (NumNode(b) if not isinstance(b, Node) else b).render(ops, ctx))

  def get_reduce_acc(self, reduceop:LazyOp):
    if reduceop.op is ReduceOps.SUM: return 0.0 if dtypes.is_float(reduceop.dtype) else 0
    if reduceop.op is ReduceOps.MAX:
      if dtypes.is_int(reduceop.dtype): return 0 if dtypes.is_unsigned(reduceop.dtype) else -2**(reduceop.dtype.itemsize*8-1)
      return -math.inf if dtypes.is_float(reduceop.dtype) else False

  # NOTE: once images are loaded, we uop them as their base float
  def get_base_dtype(self, dt:DType) -> DType: return dt.base if isinstance(dt, ImageDType) else dt

  render_ops: Any = { Variable: lambda self, ops, ctx: ctx.loop_uops[self.expr], NumNode: lambda self, ops, ctx: UOp.const(dtypes.int, self.b),
                MulNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.MUL),
                DivNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.IDIV),
                ModNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.MOD),
                LtNode: lambda self, ops, ctx: ctx.uop_alu_idx(self.a.render(ops, ctx), self.b, ops, ctx, BinaryOps.CMPLT),
    SumNode: lambda self,ops,ctx:
      functools.reduce(lambda a,b: ctx.uop_alu_idx(a, b, ops, ctx, BinaryOps.ADD), self.nodes[1:], self.nodes[0].render(ops,ctx)),
    AndNode: lambda self,ops,ctx:
      functools.reduce(lambda a,b: ctx.uop_alu_idx(a, b, ops, ctx, BinaryOps.MUL), self.nodes[1:], self.nodes[0].render(ops,ctx)) }

  def global_load(self, i:int, idxs:List[Node], acc:Optional[LazyOp]=None, barrier:Optional[UOp]=None, loop_ctx:Tuple[UOp, ...]=()) -> List[UOp]:
    buf = self.bufs[i]
    localtype = self.get_base_dtype(buf.dtype if acc is None else acc.dtype)
    const = buf.val if isinstance(buf, ConstBuffer) else None

    expand_vars = expand_idxs(idxs)

    dim, amt = None, 1
    # float 4 grouping
    if len(upcast_dim := self.get_float4_upcast_dim(i)) == 1 and len(float4_expand := expand_node(idxs[upcast_dim[0]])) in [4,2]:
      dim, amt = upcast_dim[0], len(float4_expand)
      g_idx, g_valid = self.sts[i].expr_idxs(idxs[:dim] + [float4_expand[0]] + idxs[dim+1:])
      # do not use float4 if idx is not aligned
      if g_idx != (g_idx//amt*amt): dim, amt = None, 1
    if dim is None:
      g_idx, g_valid = self.sts[i].expr_idxs(idxs)
    # todo: multioutput test with different output valids to add if acc is None: g_valid = NumNode(1)

    if amt > 1: localtype = localtype.vec(amt)
    e_idxs, e_valids = expand_node(g_idx, expand_vars), expand_node(g_valid, expand_vars)  # pylint: disable=possibly-used-before-assignment

    ret = []
    invalid_value = 0
    acc_count = 0
    for idx, valid, rep_idx in zip(e_idxs, e_valids, iter_idxs(expand_vars)):
      this_const, idx, valid = (invalid_value, NumNode(0), NumNode(1)) if valid.max == 0 else (const, idx, valid)
      key = f"{'' if acc is None else self.reduceops.index(acc)}{localtype}{'CONST'+str(this_const) if this_const is not None and acc is None else (buf.idx if isinstance(buf, MemBuffer) else cast(LocalBuffer, buf).name)}{idx.render()}{valid.render()}"  # noqa: E501
      if key not in self.load_cache:
        if acc is not None:
          self.load_cache[key] = UOp(UOps.DEFINE_ACC, localtype, (UOp.const(localtype, self.get_reduce_acc(acc)), *loop_ctx), (i, acc_count))
          acc_count += 1
        elif this_const is not None:
          self.load_cache[key] = UOp.const(localtype, this_const)
          if valid.min == 0 and valid.max == 1:
            valid_rendered = valid.render(self.render_ops, self)
            self.load_cache[key] = UOp.alu(TernaryOps.WHERE, valid_rendered, self.load_cache[key], UOp.const(localtype, invalid_value))
        elif isinstance(buf.dtype, ImageDType):
          buf_uop = self.buf_uops[i]
          assert buf_uop is not None, f"buffer {i} wasn't UOped"
          image_idx, valid = to_image_idx(buf.dtype.shape, idx, valid)
          rendered_idx = UOp(UOps.CAST, dtypes.int.vec(2), tuple(x.render(self.render_ops, self) for x in image_idx))
          valid_tuple = (valid.render(self.render_ops, self), UOp.const(buf.dtype.base.vec(4), invalid_value)) if valid.min == 0 else tuple()
          self.load_cache[key] = UOp(UOps.LOAD, buf.dtype.base.vec(4),
                                               (buf_uop, rendered_idx) + valid_tuple + ((barrier,) if barrier else ()))
          if localtype == localtype.scalar():
            idx_small = idx%4
            res = idx_small.render(self.render_ops, self)
            out = UOp(UOps.GEP, localtype, (self.load_cache[key],), idx_small.max)
            for ix in range(idx_small.max, idx_small.min, -1):
              rvv = UOp(UOps.GEP, localtype, (self.load_cache[key],), ix-1)
              sel = UOp.alu(BinaryOps.CMPLT, res, UOp.const(dtypes.int, ix))
              out = UOp.alu(TernaryOps.WHERE, sel, rvv, out)
            self.load_cache[key] = out
        else:
          buf_uop = self.buf_uops[i]
          assert buf_uop is not None, f"buffer {i} wasn't UOped"
          rendered_idx = idx.render(self.render_ops, self)
          valid_tuple = (valid.render(self.render_ops, self), UOp.const(localtype, invalid_value)) if valid.min == 0 else tuple()
          self.load_cache[key] = UOp(UOps.LOAD, localtype, (buf_uop, rendered_idx) + valid_tuple + ((barrier,) if barrier else ()))
      ret.append(UOp(UOps.GEP, localtype.scalar(), (self.load_cache[key],), rep_idx[dim]) if dim is not None else self.load_cache[key])
    return ret

  def global_store(self, i:int, idxs:List[Node], store:List[UOp]) -> List[UOp]:
    buf = self.bufs[i]
    buf_uop = self.buf_uops[i]
    assert buf_uop is not None, f"buffer {i} wasn't UOped"

    expand_vars = expand_idxs(idxs)
    _idxs = zip(*[expand_node(idx, expand_vars) for idx in idxs]) if idxs else [tuple()]  # transpose
    store_offset = dict(zip(_idxs, store))

    # float4 grouping
    if len(upcast_dim := self.get_float4_upcast_dim(i)) == 1 and len(float4_expand := expand_node(idxs[upcast_dim[0]])) in [2,4]:
      grouped_store_offset = defaultdict(list)
      for k in store_offset:
        _idx = k[:upcast_dim[0]] + (float4_expand[0],) + k[upcast_dim[0]+1:]
        grouped_store_offset[_idx].append(store_offset[k])
      store_offset_new = {}
      for k,grouped in grouped_store_offset.items():
        amt = len(grouped)
        idx, valid = self.sts[i].expr_idxs(k)
        assert idx == ((idx//amt)*amt), "float4 stores are always aligned"
        store_offset_new[k] = UOp(UOps.CAST, buf.dtype.vec(amt), tuple(grouped))
      store_offset = store_offset_new

    stores = []
    for _idx, var in store_offset.items():
      idx, valid = self.sts[i].expr_idxs(_idx)
      if isinstance(buf.dtype, ImageDType):
        image_idx, valid = to_image_idx(buf.dtype.shape, idx, valid)
        rendered_idx = UOp(UOps.CAST, dtypes.int.vec(2), \
                      tuple(x.render(self.render_ops, self) for x in image_idx))
      else:
        rendered_idx = idx.render(self.render_ops, self)
      # TODO: let UPat check this once it's fast
      if valid.min == 1: stores.append(UOp(UOps.STORE, None, (buf_uop, rendered_idx, var)))
      else: stores.append(UOp(UOps.STORE, None, (buf_uop, rendered_idx, var, valid.render(self.render_ops, self))))
    return stores

  # render loop
  def render_loop(self, xx:List[Variable], depth:int) -> Tuple[UOp, ...]:
    new_loops = {x.expr:UOp(UOps.RANGE, dtypes.int32, (
      UOp.const(dtypes.int, x.min) if isinstance(x.min, int) else cast(Node, x.min).render(self.render_ops, self),
      UOp.const(dtypes.int, x.max+1) if isinstance(x.max, int) else cast(Node, x.max+1).render(self.render_ops, self)), arg=(depth,i)) for i,x in enumerate(xx) if not isinstance(x, NumNode) and x.expr is not None}  # noqa: E501
    self.loop_uops.update(new_loops)
    return tuple(new_loops.values())

  def index_local_aliases(self, global_idxs, local_idxs, reduce_idxs, upcast_idxs, full_upcast_idxs):
    def calc_tc_idxs(local_sizes: List[int], aliases: List[List[int]]):
      replace_idxs, thread_idxs, thread_idx = [], [], Variable("_uidx_tc", 0, prod(local_sizes)-1)
      for s in local_sizes:
        thread_idxs.append(thread_idx % s)
        thread_idx //= s
      for alias in aliases:
        full_var, full_var_sz = NumNode(0), 1
        if alias[0] != 0:
          for i in alias:
            next_var = local_idxs[i-1] if i > 0 else thread_idxs[-i-1]
            full_var += next_var * full_var_sz
            full_var_sz *= next_var.max+1
        replace_idxs.append(full_var)
      return replace_idxs

    # compute local aliases
    alias_buf_idxs: DefaultDict[LazyOp, List[Tuple[int, int, List]]] = defaultdict(list)
    for op, local_alias in self.local_alias.items():
      for i in local_alias:
        localbuf_idx = self.bufs.index(local_alias[i])
        buf_idxs = [idx*0 if s == 0 else idx for idx,s in zip(global_idxs+local_idxs+reduce_idxs+full_upcast_idxs,self.sts[i].real_strides())]
        if (tc:=self.tensor_core):
          min_alias_idx = min(local_alias.keys())
          replace_input_idxs = calc_tc_idxs(tc.thread_local_sizes[i-min_alias_idx], tc.thread_local_aliases[i-min_alias_idx])
          for n in range(len(tc.threads)):
            buf_idxs[self.global_dims+n] = replace_input_idxs[n] # replace locals
          for n in range(tc.num_upcasts()):
            buf_idxs[self.shape_len-self.upcasted+n] = replace_input_idxs[len(tc.threads)+n] # replace upcasts
        if DEBUG >= 3: print(f"{localbuf_idx} alias {i}: sts={self.sts[i]} idxs={buf_idxs}")
        alias_buf_idxs[op].append((i, localbuf_idx, buf_idxs))
    # modify idxs if necessary for TC
    if (tc:=self.tensor_core):
      replace_acc_idxs = calc_tc_idxs(tc.thread_local_sizes[2], tc.thread_local_aliases[2])
      for n in range(len(tc.threads)):
        local_idxs[n] = replace_acc_idxs[n] # replace locals
      for n in range(len(replace_acc_idxs)-len(tc.threads)):
        upcast_idxs[n] = replace_acc_idxs[len(tc.threads)+n] # replace upcasts
      if DEBUG >= 3: print(f"store alias: sts={self.sts[0]} idxs={global_idxs+local_idxs+upcast_idxs}")
    return alias_buf_idxs

  def render_reduceop(self, reduceop:LazyOp, accs:Dict[LazyOp, List[UOp]], loaded_buffers:Dict[Union[MemBuffer, ConstBuffer, LocalBuffer], List[UOp]],
                      global_idxs, local_idxs, upcast_idxs, full_upcast_idxs, reduce_idxs, fake_reduce_idxs,
                      alias_buf_idxs:List[Tuple[int, int, List]]) -> Tuple[List[NumNode|Variable], List[NumNode|Variable]]:
    # reduce loop
    loop_ctx = self.render_loop(reduce_idxs, (i:=self.reduceops.index(reduceop))*2+2)

    # define accumulator - modify idxs if necessary for TC
    out_buf = -len(self.reduceops)+i if self.group_for_reduces else 0
    accs[reduceop] = self.global_load(out_buf, global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, acc=reduceop, loop_ctx=loop_ctx)

    # store local aliases
    locals_to_store = [(localbuf_idx, buf_idxs, self.global_load(i, buf_idxs)) for i, localbuf_idx, buf_idxs in alias_buf_idxs]

    if (tc:=self.tensor_core):
      # run tensor cores AST
      wmma_sz = [prod(l) for l in tc.thread_local_sizes]
      def upcast_strides(buf:int):
        strides, next_ = [], 1
        for (sz, stride, _) in self.upcasted_axis(buf)[tc.num_upcasts():]:
          strides.append((0 if stride == 0 else next_, sz))
          next_ *= 1 if stride == 0 else sz
        return strides
      upcasts, dev = [upcast_strides(x) for x in [locals_to_store[0][0], locals_to_store[1][0], 0]], self.opts.device
      # cast initial accs
      wmmas = [UOp(UOps.CAST, (dt3:=tc.dtype_out.vec(wmma_sz[2])), tuple(accs[reduceop][x:x+wmma_sz[2]]))
               for x in range(0, len(accs[reduceop]), wmma_sz[2])]
      for it in [x[::-1] for x in itertools.product(*[x for x in [range(sz) for _,sz in upcasts[0]][::-1]])]:
        offs = [x*y for (x,y) in zip([sum([prod(x) for x in zip(it, [stride for stride,_ in y])]) for y in upcasts], wmma_sz)]
        ops = (UOp(UOps.CAST, tc.dtype_in.vec(wmma_sz[0]), tuple(locals_to_store[0][2][offs[0]:offs[0]+wmma_sz[0]])),
                UOp(UOps.CAST, tc.dtype_in.vec(wmma_sz[1]), tuple(locals_to_store[1][2][offs[1]:offs[1]+wmma_sz[1]])),
                wmmas[(wmma_idx:=offs[2]//wmma_sz[2])])
        # TODO: don't need to DEFINE_ACC, pass to WMMA in op3, or PHI accs that are not valid
        wmmas[wmma_idx] = UOp(UOps.WMMA, dt3, ops, (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, tuple(wmma_sz), dev))
      # phi the last wmmas back to accs
      accs[reduceop] = [UOp(UOps.PHI, tc.dtype_out, (acc, UOp(UOps.GEP, tc.dtype_out, (wmmas[z//wmma_sz[2]],), z%wmma_sz[2])))
                        for z, acc in enumerate(accs[reduceop])]
    else:
      assert not locals_to_store, "storing locals isn't supported here"

      # load earlybufs
      loaded_buffers.update({b:self.global_load(self.bufs.index(self.local_alias[reduceop][i]) if i in self.local_alias else i,
        global_idxs+local_idxs+reduce_idxs+full_upcast_idxs) for i,b in enumerate(self.bufs) if b in self.earlybufs})

      def gate_acc(r, idxs): return [
        UOp.alu(TernaryOps.WHERE, valid.render(self.render_ops, self), acc, UOp.const(r.dtype, 0)) if valid.min == 0 and valid.max == 1 else acc
        for valid, acc in zip(expand_node(self.sts[self.full_buf_index].expr_idxs(idxs)[1], expand_idxs(idxs)), accs[r])]
      local_accs = {r: gate_acc(r,global_idxs+local_idxs+reduce_idxs+full_upcast_idxs) for r in accs}

      # run early AST (with reduce)
      self.ast_parse(reduceop, local_accs, self.acc_offsets(self.full_buf_index), loaded_buffers, reduce_acc=accs[reduceop])

    # end the reduce loop
    self.load_cache.clear()

    # end the local loop, do the local reduce
    if self.group_for_reduces:
      fake_global_idxs = [x*0 for x in global_idxs]
      stores = self.global_store(out_buf, fake_global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, accs[reduceop])  # store accumulators
      barrier = UOp(UOps.BARRIER, None, tuple(stores))
      if self.opts.has_local:
        fake_idxs = [NumNode(0)]*len(self.sts[-1].shape)
        fake_idxs[self.global_dims+self.local_dims:self.global_dims+len(local_idxs)] = local_idxs[self.local_dims:]
        if_cond: UOp = create_lt_node(self.sts[-1].expr_idxs(fake_idxs)[0], 1).render(self.render_ops, self)
        barrier = UOp(UOps.IF, None, (if_cond, barrier))

      # create new late reduce local loops and replace local_idxs that have been used
      end_local_idxs = [Variable(f"tidx{i}", 0, self.full_shape[i]-1 if i >= self.first_reduce and i not in self.upcast_in_mid_reduce_axes else 0) for i in range(0, self.first_reduce+self.group_for_reduces)]  # noqa: E501
      local_idxs = local_idxs[:self.local_dims] + end_local_idxs[self.global_dims + self.local_dims:]

      # if any group_for_reduce items aren't reduces, upcast them here
      for j in self.upcast_in_mid_reduce_axes:
        self.reshape_and_permute(None, [i for i in range(self.shape_len) if i != j] + [j])
        self.upcast()
        self.group_for_reduces -= 1
        local_idxs = local_idxs[:-1]
        end_local_idxs = end_local_idxs[:-1]
        # regenerate upcast_idxs
        upcast_idxs = [Variable(f"_uidx{i}", 0, s-1) for i, s in enumerate(self.output_shape[self.shape_len-self.upcasted:])]

      # NOTE: this structure is the same as the reduce op above

      # late reduce loop
      loop_ctx = self.render_loop(end_local_idxs, i*2+3)

      # define late accumulator
      accs[reduceop] = self.global_load(0, fake_global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, acc=reduceop, loop_ctx=loop_ctx)

      # load localbufs
      loaded_buffers[self.bufs[out_buf]] = self.global_load(out_buf, fake_global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, barrier=barrier)

      # there's no AST here (and there's no shape for the reduce LazyOp)
      self.ast_parse(LazyOp(reduceop.op, (LazyOp(BufferOps.LOAD, (), self.bufs[out_buf]),)),\
                     accs, self.acc_offsets(-1), loaded_buffers, reduce_acc=accs[reduceop])

      # end the late reduce loop
      self.load_cache.clear()

      if reduceop is not self.reduceops[-1]:
        for j in self.upcast_in_mid_reduce_axes:
          self.upcasted -= 1
          self.group_for_reduces += 1
        assert self.buf_uops[out_buf] is not None, "Local reduce buf must have been uoped at this point"
        fake_local_idxs = local_idxs[:self.local_dims] + [x*0 for x in local_idxs[self.local_dims:]]
        stores = self.global_store(out_buf, fake_global_idxs+fake_local_idxs+fake_reduce_idxs+upcast_idxs, accs[reduceop])
        barrier = UOp(UOps.BARRIER, None, tuple(stores))
        accs[reduceop] = self.global_load(out_buf, fake_global_idxs+fake_local_idxs+fake_reduce_idxs+upcast_idxs, barrier=barrier)
    return local_idxs[:self.local_dims] + [NumNode(0) for _ in range(self.group_for_reduces)], upcast_idxs

  kernel_cnt: Final[DefaultDict[str, int]] = defaultdict(int)
  def linearize(self) -> Linearizer:
    # no new opts and we already ran? skip relinearizing
    if self.applied_opts == self.applied_opts_cache: return self

    # late alias the tensor core buffers
    if (tc:=self.tensor_core) and self.tensor_core_opts is not None:
      alias_pattern = [0]*(self.global_dims) + [2]*(len(tc.threads)) + [0]*(self.local_dims-len(tc.threads)) + [0]*(self.shape_len-self.upcasted-self.first_reduce) + [1,1] + [3]*(self.upcasted-2)  # noqa: E501
      for op, tc_bufs in self.bufs_for_tensor_core.items():
        for tc_buf in tc_bufs: self.alias_buffer(op, tc_buf, alias_pattern)

    # save backups
    sts_backup, gfr_backup, upc_backup = self.sts[:], self.group_for_reduces, self.upcasted

    # uops
    self.buf_uops: List[Optional[UOp]] = [None]*len(self.bufs)
    self.loop_uops: Dict[str, UOp] = {}

    # add global buffers
    for i,buf in enumerate(self.bufs):
      if isinstance(buf, MemBuffer):
        self.buf_uops[i] = UOp(UOps.DEFINE_GLOBAL,
                                         buf.dtype if isinstance(buf.dtype, ImageDType) else PtrDType(buf.dtype), (),
                                         (buf.idx, any(buf.idx == x.idx for x in self.outbufs)))
    # add var vals
    for i,var in enumerate(self.vars):
      assert var.expr is not None
      self.loop_uops[var.expr] = UOp(UOps.DEFINE_VAR, dtypes.int32, (), var)
    # define local buffers
    for aliases in self.local_alias.values():
      for lb in aliases.values(): self.buf_uops[self.bufs.index(lb)] = UOp(UOps.DEFINE_LOCAL, PtrDType(lb.dtype),
                                                                                     (), (lb.name, self.sts[self.bufs.index(lb)].size))
    # add a local buffer for multistage reduce. # TODO: use local alias
    if self.group_for_reduces:
      for i in range(len(self.reduceops)):
        # TODO: the strides of this can be controlled
        self.sts.append(ShapeTracker.from_shape(tuple([1] * self.global_dims + list(self.full_shape[self.global_dims:self.global_dims+self.local_dims+self.group_for_reduces]) + [1] * (self.shape_len - self.upcasted - self.group_for_reduces - self.first_reduce) + [x[0] for x in self.upcasted_axis(0)])))  # noqa: E501
        temp_dtype = self.get_base_dtype(cast(LazyOp, self.reduceop).dtype)
        self.bufs.append(LocalBuffer(name:=f"temp{i if len(self.reduceops) > 1 else ''}", buf_size:=self.sts[-1].size, temp_dtype))
        self.buf_uops.append(UOp(UOps.DEFINE_LOCAL, PtrDType(temp_dtype), (), (name, buf_size)))

    # kernel name (before late upcast)
    self.name = ("r" if self.reduceop else ("C" if all(x.op in BufferOps for x in self.lazyops) else "E")) + \
                 (f"{len(self.outbufs)}_" if len(self.outbufs) > 1 else "_") + \
                 colored('_', 'BLACK').join([colored(str(x), c) for x,c in zip(self.full_shape, self.colors())])

    # name the function something unique
    Linearizer.kernel_cnt[(function_name := to_function_name(self.name))] += 1
    suffix = f"{'n'+str(Linearizer.kernel_cnt[function_name]-1)}" if Linearizer.kernel_cnt[function_name] > 1 else ""
    self.name = self.name+colored(suffix, 'BLACK')

    # define indexes
    gl_dims = self.full_shape[:self.first_reduce+self.group_for_reduces]
    global_idxs, loop_global_idxs, self.global_size = get_grouped_dims("idx" if self.dont_use_locals else "gidx", 0, gl_dims[:self.global_dims],
                                                                       self.opts.global_max, self.opts.has_local)
    local_idxs, loop_local_idxs, self.local_size = get_grouped_dims("lidx", self.global_dims, gl_dims[self.global_dims:],
                                                                    self.opts.local_max if self.opts.has_local else (), False)
    upcast_idxs = [Variable(f"_uidx{i}", 0, s-1) for i, s in enumerate(self.output_shape[self.shape_len-self.upcasted:])]
    full_upcast_idxs = [Variable(f"_uidx{i}", 0, s-1) for i, s in enumerate(self.full_shape[self.shape_len-self.upcasted:])]

    # render global and local as specials or a loop
    if self.opts.has_local:
      self.loop_uops.update({x.expr:UOp(UOps.SPECIAL, dtypes.int32, (), (i, x.expr, x.max+1)) for i,x in enumerate(loop_global_idxs)})
      if not self.dont_use_locals:
        self.loop_uops.update({x.expr:UOp(UOps.SPECIAL, dtypes.int32, (), (i, x.expr, x.max+1)) for i,x in enumerate(loop_local_idxs)})
    else:
      self.global_size, self.local_size = None, None
      self.render_loop(loop_global_idxs+loop_local_idxs, 1)

    # define idxs for aliased buffers TODO: this doesn't belong in Kernel, but it can't exist in Block either (because of multireduce tensor cores)
    reduce_idxs = [Variable(f"ridx{i}", 0, self.full_shape[i]-1) for i in range(self.first_reduce+self.group_for_reduces, self.shape_len-self.upcasted)]  # noqa: E501
    alias_buf_idxs = self.index_local_aliases(global_idxs,local_idxs,reduce_idxs,upcast_idxs,full_upcast_idxs)

    # parse AST
    self.load_cache: Dict[str, UOp] = {}
    loaded_buffers:Dict[Union[MemBuffer, ConstBuffer, LocalBuffer], List[UOp]] = {}
    accs: Dict[LazyOp, List[UOp]] = {}

    # render reduceops by depth
    for reduceop in self.reduceops:
      self.render_block((reduceop, ), global_idxs, local_idxs, upcast_idxs, full_upcast_idxs, alias_buf_idxs, loaded_buffers, accs)
    stores = self.render_block(self.ast, global_idxs, local_idxs, upcast_idxs, full_upcast_idxs, alias_buf_idxs, loaded_buffers, accs)

    # only the final stores are needed to define the full UOps graph
    self.uops:UOpGraph = UOpGraph(flatten(stores))

    # maybe graph the uops
    if DEBUG >= 5: self.uops.print()
    if getenv("GRAPHUOPS"): self.uops.graph()

    # restore backups
    self.sts, self.group_for_reduces, self.upcasted = sts_backup, gfr_backup, upc_backup

    # set cache and return
    self.applied_opts_cache = self.applied_opts[:]
    return self

  def render_block(self, outputs:Tuple[LazyOp, ...], global_idxs, local_idxs, upcast_idxs, full_upcast_idxs,
                   alias_buf_idxs:DefaultDict[LazyOp,List[Tuple[int,int,List[NumNode|Variable]]]],
                   loaded_buffers:Dict[Union[MemBuffer, ConstBuffer, LocalBuffer], List[UOp]], accs:Dict[LazyOp,List[UOp]]) -> List[List[UOp]]:
    reduceops = dedup(x for x in outputs if x.op in ReduceOps)
    assert len(reduceops) <= 1, "max one reduceop per block"
    reduce_idxs = [Variable(f"ridx{i}", 0, self.full_shape[i]-1) for i in range(self.first_reduce+self.group_for_reduces, self.shape_len-self.upcasted)]  # noqa: E501
    fake_reduce_idxs = [x*0 for x in reduce_idxs]

    if len(reduceops) != 0:
      # TODO: delete render_reduceop and move the logic for group_for_reduces to Block
      nlidx, nuidx = self.render_reduceop((r:=reduceops[0]),accs,loaded_buffers,\
                                          global_idxs,local_idxs,upcast_idxs,full_upcast_idxs,reduce_idxs,fake_reduce_idxs,alias_buf_idxs[r])

      # all local indices which were used for group_for_reduce are not valid any more and should be replaced with fake NumNode(0), since they have
      # been rewritten with fake end_local_idxs.
      if r is self.reduceops[-1]: local_idxs[:], upcast_idxs[:] = nlidx, nuidx
      return [accs[r]]

    # load latebufs
    loaded_buffers.update({b:self.global_load(i, global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs) \
                           for i,b in enumerate(self.bufs) if b not in self.earlybufs and b.__class__ is not LocalBuffer})
    # run late AST (without the store)
    store_vals = {op.arg.idx:self.ast_parse(op.src[0], accs, None, loaded_buffers) for op in self.ast}
    return [self.global_store(i, global_idxs+local_idxs+fake_reduce_idxs+upcast_idxs, val) for i, val in store_vals.items()]

  def ast_parse(self, x:LazyOp, accs:Dict[LazyOp, List[UOp]], offs:Optional[List[int]], loaded_buffers:Dict[Union[MemBuffer, ConstBuffer, LocalBuffer], List[UOp]], reduce_acc:Optional[List[UOp]]=None, cache=None) -> List[UOp]: # noqa: E501
    if cache is None: cache = {}
    if x in cache: return cache[x]
    if x.op in BufferOps: return loaded_buffers[x.arg]
    if x.op in [UnaryOps.CAST, UnaryOps.BITCAST]:
      return [UOp(UOps.BITCAST if x.op is UnaryOps.BITCAST else UOps.CAST,
                            self.get_base_dtype(x.arg), (u,)) for u in self.ast_parse(x.src[0], accs, offs, loaded_buffers)]
    if x.op in ReduceOps and reduce_acc is None:
      return [accs[x][i] for i in offs] if offs else accs[x]

    values = [self.ast_parse(v, accs, offs, loaded_buffers, cache=cache) for v in x.src]
    ops = {ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX}
    if x.op in ops:
      assert reduce_acc is not None
      ret: List[UOp] = []
      acc, input_acc = reduce_acc, reduce_acc[:]
      for val, off in zip(zip(*values), cast(List[int], offs)):
        acc[off] = UOp.alu(ops[cast(ReduceOps, x.op)], *(val+(acc[off], )))
        ret.append(acc[off])
      for off in range(len(acc)):
        if input_acc[off] != acc[off]:
          acc[off] = UOp(UOps.PHI, input_acc[off].dtype, (input_acc[off], acc[off]))
    else: ret = [UOp.alu(x.op, *src) for src in zip(*values)]
    cache[x] = ret
    return ret

  def to_program(self) -> Program:
    self.linearize()
    info = get_lazyop_info(self.ast[0])
    src = self.opts.render(to_function_name(self.name), self.uops)
    if getenv("RUN_PROCESS_REPLAY"): diskcache_put("process_replay", id(self), (self, src))
    ops, mem = self.uops.flops_mem()
    run_count = prod((self.global_size if self.global_size else []) + (self.local_size if self.local_size else []))
    # NOTE: we use min here to ignore the indexing FLOPS
    return Program(self.name, src, self.opts.device, self.global_size, self.local_size,
                   self.uops, min(info.flops, ops * run_count), min(info.mem_estimate, mem * run_count))
