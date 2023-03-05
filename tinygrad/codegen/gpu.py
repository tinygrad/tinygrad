import math
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Set, Final, NamedTuple
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, FusedOps, LazyOp, Op, ASTRunner
from tinygrad.codegen.ast import ASTKernel, Token, Types
from tinygrad.shape.symbolic import Node, MulNode, DivNode, SumNode, Variable, render_python
from tinygrad.shape import ShapeTracker, View
from tinygrad.helpers import getenv, DEBUG, dedup, prod, partition, mnum, all_same

# div is different in cl than python
render_cl = render_python.copy()
render_cl[DivNode] = lambda self,ops,ctx: f"({self.a.render(ops)}/{self.b})"

VALIDHACKS = getenv("VALIDHACKS", 0)    # TODO: remove the need for this
NATIVE_EXPLOG = getenv("NATIVE_EXPLOG", 0)  # this is needed as a switch for the tests to pass

class GPULanguage(NamedTuple):
  kernel_prefix : str = ""
  buffer_prefix : str = ""
  buffer_suffix : str = ""
  smem_prefix : str = ""
  barrier : str = ""
  gid : List[str] = []
  lid : List[str] = []
  extra_args : List[str] = []
  float4 : Optional[str] = None

def to_image_idx(base_shape:Tuple[int, ...], idxy:Node, valid:Node, validhacks=False) -> Tuple[Node, Node]:
  idy = (idxy//(4*base_shape[1]))
  if validhacks and valid.min == 0:
    idx = (idxy//4) + (idy*-base_shape[1])
    # find the ones in idx that didn't factorize and remove them (TODO: this is not universal)
    if isinstance(idx, SumNode):
      unfactored, idx_nodes = partition(idx.nodes, lambda x: isinstance(x, MulNode) and x.b == -base_shape[1])
      assert len(unfactored) <= 1
      idx = Variable.sum(idx_nodes)
      unfactored = (Variable.sum(unfactored) // base_shape[1])
      idy += unfactored
      # ugh really...handtuned garbage
      if idx.min >= (base_shape[1]*3)//4:
        idx -= base_shape[1]
        idy += 1
  else:
    idx = (idxy//4)%base_shape[1]
  #print(base_shape, idx.min, idx.max, idy.min, idy.max, idx, idy)
  return idx, idy

def float4_can_group(grp):
  return all(g.tok.endswith(e) for g,e in zip(grp, [".x", ".y", ".z", ".w"])) and all_same([g.tok.split(".")[0] for g in grp])

def float4_group(grp):
  assert float4_can_group(grp)
  return Token(grp[0].tok.split(".")[0], Types.FLOAT4)

def float4_factorize(values):
  # swizzle
  new_values = []
  for v in values:
    nv = []
    for _ in range(0, len(v), 16):
      nv += v[:16:4]
      nv += v[1:16:4]
      nv += v[2:16:4]
      nv += v[3:16:4]
      v = v[16:]
    new_values.append(nv)
  a,b,c = new_values[0], new_values[1], new_values[2]
  na,nb,nc = [], [], []
  for i in range(0, len(a), 4):
    if float4_can_group(a[i:i+4]):
      na.append(float4_group(a[i:i+4]))
      if all_same(b[i:i+4]):
        nb.append(b[i])
      elif float4_can_group(b[i:i+4]):
        nb.append(float4_group(b[i:i+4]))
      else:
        return new_values
      if all_same(c[i:i+4]):
        nc.append(c[i])
      elif float4_can_group(c[i:i+4]):
        nc.append(float4_group(c[i:i+4]))
      else:
        return new_values
    else:
      return new_values
  return (na, nb, nc)

class GPUCodegen(ASTKernel):
  lang : GPULanguage = GPULanguage()

  # for renaming
  kernel_cnt : Final[Dict[str, int]] = defaultdict(lambda: -1)
  kernel_name_cache : Final[Dict[str, str]] = {}

  code_for_op : Final[Dict[Op, str]] = {
    UnaryOps.NOOP: "(A)", UnaryOps.NEG: "(-(A))", UnaryOps.NOT: "((float)1.0-A)",
    UnaryOps.EXP: "native_exp(A)" if NATIVE_EXPLOG else "exp(A)",
    UnaryOps.LOG: "native_log(A)" if NATIVE_EXPLOG else "log(A)",
    BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)", BinaryOps.MUL: "(A*B)",
    BinaryOps.DIV: "(A/B)", BinaryOps.POW: "pow(A,B)", BinaryOps.CMPEQ: "(A==B)",
    BinaryOps.MAX: "max(A,B)", ReduceOps.SUM: "A+=B", ReduceOps.MAX: "A=max(A,B)",
    FusedOps.MULACC: "A=mad(B,C,A)"
  }
  start_for_op : Final[Dict[Op, str]] = {ReduceOps.SUM: "0.0", ReduceOps.MAX: "-INFINITY"}

  def store(self, buf_index:int, value:List[Token]) -> None:
    assert len(value) == self.buftokens[buf_index].size(), f"size mismatch {len(value)} != {self.buftokens[buf_index].size()}"
    assert len(self.sts[buf_index].views) == 1, "store has more than one view"

    # all stores can merge, since they have one view and are valid
    should_upcast = self.lang.float4 and self.buftokens[buf_index].can_float4()

    to_store = {o:v for o,v in zip(self.buftokens[buf_index].offsets(), value)}
    did_store = set()
    for o,v in to_store.items():
      if o in did_store: continue
      idxy, valid = self.sts[buf_index].expr_idxs(o)
      assert valid.min == 1, "store must always be valid"
      if should_upcast:
        for j in range(4): did_store.add(o+j)
        # is float4
        if float4_can_group([to_store[o+j] for j in range(4)]):
          v = float4_group([to_store[o+j] for j in range(4)])
        else:
          v = Token(f"{self.lang.float4}({','.join([to_store[o+j].tok for j in range(4)])})", Types.FLOAT4)
      if self.bufs[buf_index] is not None and hasattr(self.bufs[buf_index]._buf, "IMAGE"):
        assert v.typ == Types.FLOAT4, "Image requires upcasting to FLOAT4"
        idx, idy = to_image_idx(self.bufs[buf_index]._base_shape, idxy, valid)
        self.kernel.append(f"write_imagef({self.buftokens[buf_index].tok}, (int2)({idx.render(render_cl)}, {idy.render(render_cl)}), {v.tok});  /* {self.bufs[buf_index]._base_shape} */\n")
      elif v.typ == Types.FLOAT4:
        self.kernel.append(f"(({self.lang.buffer_prefix if self.bufs[buf_index] is not None else self.lang.smem_prefix}float4*){self.buftokens[buf_index].tok})[{(idxy//4).render(render_cl)}] = {v.tok};\n")
      else:
        self.kernel.append(f"{self.buftokens[buf_index].tok}[{(idxy//(4 if v.typ == Types.FLOAT4 else 1)).render(render_cl)}] = {v.tok};\n")

  def load(self, buf_index:int, idx_override:Optional[str]=None) -> List[Token]:
    # constant folding
    const = None
    if self.bufs[buf_index] is not None and self.bufs[buf_index]._base_shape == (1,) and self.bufs[buf_index]._backing is not None:
      if buf_index != 0: self.bufs_to_delete.add(buf_index)
      val = self.bufs[buf_index]._backing[0]
      assert not math.isnan(val)
      const = Token(f"({val}f)", Types.FLOAT)
    should_upcast = self.lang.float4 and const is None and self.buftokens[buf_index].can_float4()
    tokens = []
    test_idy = []
    for o in self.buftokens[buf_index].offsets():
      key = f"val{mnum(buf_index)}_{mnum(o)}"
      if (buf_index, o) not in self.loaded_keys:
        idxy, valid = self.sts[buf_index].expr_idxs(o) if idx_override is None else self.sts[buf_index].expr_node(idx_override, o)
        if should_upcast:
          can_merge = True
          for j in range(1,4):
            idxy_test, valid_test = self.sts[buf_index].expr_idxs(o+j) if idx_override is None else self.sts[buf_index].expr_node(idx_override, o+j)
            can_merge = can_merge and valid.render() == valid_test.render()
            can_merge = can_merge and (idxy+j).render() == idxy_test.render()
            #print((idxy+j).render(), idxy_test.render(), valid.render(), valid_test.render(), can_merge)
        if const is not None:
          ldr = const
        elif self.bufs[buf_index] is not None and hasattr(self.bufs[buf_index]._buf, "IMAGE"):
          assert should_upcast and can_merge, f"Image requires upcasting to FLOAT4 {self.buftokens[buf_index]}"
          idx, idy = to_image_idx(self.bufs[buf_index]._base_shape, idxy, valid, VALIDHACKS)
          ldr = Token(f"read_imagef({self.buftokens[buf_index].tok}, smp, (int2)({idx.render(render_cl)}, {idy.render(render_cl)})) /* {self.bufs[buf_index]._base_shape} */", Types.FLOAT4)
          test_idy.append(idy.render(render_cl))
        elif should_upcast and can_merge:
          ldr = Token(f"(({self.lang.buffer_prefix if self.bufs[buf_index] is not None else self.lang.smem_prefix}float4*){self.buftokens[buf_index].tok})[{(idxy//4).render(render_cl)}]", Types.FLOAT4)
        else:
          ldr = Token(f"{self.buftokens[buf_index].tok}[{idxy.render(render_cl)}]", Types.FLOAT)
        ldr = ldr if valid.min == 1 or (VALIDHACKS and hasattr(self.bufs[buf_index]._buf, "IMAGE")) else (Token(f"({valid.render(render_cl)} ? {ldr.tok} : 0.0f)", ldr.typ) if valid.max == 1 else Token("0.0f", ldr.typ))
        if const is not None:
          self.loaded_keys[(buf_index,o)] = ldr
        else:
          self.kernel.append(f"{ldr.decltype()} {key} = {ldr.tok};\n")
          if should_upcast and can_merge:
            for j in range(4):
              self.loaded_keys[(buf_index,o+j)] = Token(key+f'.{"xyzw"[j]}', Types.FLOAT)
          else:
            self.loaded_keys[(buf_index,o)] = Token(key, Types.FLOAT)
      tokens.append(self.loaded_keys[(buf_index,o)])
    assert not VALIDHACKS or all_same(test_idy), f"idy changed! {test_idy}"
    return tokens

  def ast_parse(self, x, acc:List[Token], do_reduce=False) -> List[Token]:
    if not isinstance(x, LazyOp): return self.load(self.bufs.index(x), "mid" if x is None else None)  # hack for local
    if isinstance(x.op, ReduceOps) and not do_reduce: return acc
    values : List[List[Token]] = ([acc] if isinstance(x.op, (ReduceOps, FusedOps)) else []) + [self.ast_parse(v, acc, do_reduce) for v in x.src]
    code = GPUCodegen.code_for_op[x.op]  # TODO: replace this with a function
    if len(values) == 3:
      values = float4_factorize(values)
      return [Token(code.replace("A", a.tok).replace("B", b.tok).replace("C", c.tok), a.typ) for a,b,c in zip(values[0], values[1], values[2])]
    elif len(values) == 2:
      assert len(values[0]) == len(values[1]) and values[0][0].typ == values[1][0].typ, f"values mismatch {values}"
      return [Token(code.replace("A", a.tok).replace("B", b.tok), a.typ) for a,b in zip(values[0], values[1])]
    else:
      return [Token(code.replace("A", a.tok), a.typ) for a in values[0]]

  def required_optimizations(self, early_only=False):
    for buf_index,buf in enumerate(self.bufs):
      upcast_strides = [self.sts[buf_index].strides[i] for i in self.upcast_in_mid_reduce_axes]
      if (not early_only or buf in self.earlybufs) and hasattr(buf._buf, "IMAGE") and not (self.buftokens[buf_index].can_float4() or (buf not in self.earlybufs and (1 in upcast_strides))):
        axes = [i for i,x in enumerate(self.sts[buf_index].strides) if x == 1]
        assert len(axes) == 1, f"wrong number of stride 1 axis : {axes}"
        self.shift_to(axes[0], 4)
        self.upcast()
        assert self.buftokens[buf_index].can_float4()

  def hand_coded_optimizations(self):
    # if there's images in the earlybufs, we have to make an axis the 4 loading one
    self.required_optimizations(early_only=True)

    # simplify (sets first_reduce)
    self.simplify_ones()

    # are we grouping? what does this have to do with float4?
    if self.lang.float4 and not self.buftokens[0].can_float4() and self.first_reduce <= 2 and self.first_reduce + 1 <= self.shape_len and prod(self.sts[0].shape[:self.first_reduce]) <= 2048:
      # TODO: use 1024 if it's allowed in a smarter way
      for sz in (([256, 16]) if prod(self.sts[0].shape[:self.first_reduce]) <= 32 else [16]):
        if all([st.shape[self.first_reduce] % sz == 0 or st.shape[self.first_reduce] == 1 for st in self.sts]):
          self.shift_to(self.first_reduce, sz, top=True, insert_before=self.first_reduce)
          self.group_for_reduce.append(sz)
          break

    # are we upcasting in mid reduce?
    if hasattr(self.bufs[0]._buf, "IMAGE") and not self.buftokens[0].can_float4() and self.group_for_reduce and self.first_reduce <= 2:
      axes = [i for i,x in enumerate(self.sts[0].strides) if x == 1]
      assert len(axes) == 1, f"wrong number of stride 1 axis : {axes}"
      self.shift_to(axes[0], 4, insert_before=self.first_reduce + len(self.group_for_reduce))   # insert at the end of the grouped axis
      self.group_for_reduce.append(4)

    # now do everything required
    self.required_optimizations()

    # simplify (sets first_reduce)
    self.simplify_ones()

    # use more opencl indexing if the output buffer is an image and we have room
    if hasattr(self.bufs[0]._buf, "IMAGE") and self.first_reduce+len(self.group_for_reduce) < 3:
      base_shape = self.bufs[0]._base_shape
      if (base_shape[0]*base_shape[1]) % self.sts[0].shape[0] == 0 and self.sts[0].shape[0]//base_shape[0] != 0:
        if DEBUG >= 4: print("split opencl", base_shape, self.sts[0].shape)
        self.reshape_and_permute(lambda x: [base_shape[0], x[0]//base_shape[0]]+list(x[1:]), None)
        self.simplify_ones()

    # **** below this line need to be optional and benchmarked ****

    # split to 4 float4s based on a heuristic
    if self.buftokens[0].can_float4() and any(hasattr(buf._buf, "IMAGE") for buf in self.earlybufs) and prod(self.sts[0].shape[:self.first_reduce]) >= 2048 and not self.group_for_reduce:
      xb_choices = []
      for i in range(self.first_reduce):
        if all(st.shape[i]%4 == 0 for st in self.sts):
          xb_choices.append((sum(st.views[-1].strides[i]>0 for st in self.sts), sum(st.views[-1].strides[i] for st in self.sts), i))

      if len(xb_choices):
        xb_choice = sorted(xb_choices)[0][2]
        if DEBUG >= 4: print(f"float4 merging axis {xb_choice} : {xb_choices}")

        # this leaves the last axis in place
        self.shift_to(xb_choice, 4)

        # drop the last dimension
        self.upcast()

        # re-simplify
        self.simplify_ones()

    # if last dim <= 3 and it's a reduce dim, upcast (loop unrolling). no simplify needed since it's just an upcast
    # NOTE: careful, this can break VALIDHACKS
    if not self.group_for_reduce and self.first_reduce < self.shape_len and self.full_shape[-1] > 1 and self.full_shape[-1] <= 3 and (max([x.size() for i,x in enumerate(self.buftokens) if self.bufs[i] in self.earlybufs]) <= 4 or not any(r for _,_,r in self.buftokens[self.full_buf_index].axis)):
      self.upcast()

  # STOP WASTING TIME WITH DOING THE RESHAPES AND PERMUTES BY HAND. KERNEL SEARCH IS THE ONLY WAY IT WILL EVER BE GOOD
  # group_for_reduce will have to be better first
  def codegen(self) -> ASTRunner:
    self.process()
    if DEBUG >= 4: self.printbufs("old:", DEBUG>=5)

    self.hand_coded_optimizations()

    # fancy colored shape printer
    if DEBUG >= 3: print(self.colorshape(), end="")

    # add a local buffer for multistage reduce
    if len(self.group_for_reduce):
      self.bufs.append(None)
      # TODO: the strides of this can be controlled
      st = ShapeTracker(tuple([1] * self.first_reduce + self.group_for_reduce + [1] * (self.shape_len - len(self.group_for_reduce) - self.first_reduce) + [x[0] for x in self.buftokens[0].axis]))
      buftoken = Token("temp", Types.FLOAT, ptr=True)
      # manual upcast of the local
      for _,_,r in self.buftokens[0].axis[::-1]:
        buftoken.array(st.shape[-1], st.views[-1].strides[-1], r)
        st.views[-1] = View(st.shape[0:-1], st.views[-1].strides[0:-1], st.views[-1].offset)
      self.sts.append(st)
      self.buftokens.append(buftoken)

    self.output_shape : Tuple[int, ...] = self.sts[0].shape[:self.first_reduce] + tuple(self.group_for_reduce)
    assert self.full_shape[:len(self.output_shape)] == self.output_shape, f"output shape mismatch : {self.full_shape[:len(self.output_shape)]} != {self.output_shape}"
    if DEBUG >= 4:
      print("output shape", self.output_shape)
      self.printbufs("new:", DEBUG>=5)

    self.bufs_to_delete : Set[int] = set()
    self.loaded_keys : Dict[Tuple[int,int], Token] = {}
    self.prekernel : Set[str] = set()
    self.kernel : List[str] = ["const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"] if any(hasattr(buf._buf, "IMAGE") for buf in self.bufs if buf is not None) else []

    if len(self.lang.gid) == 0:
      self.kernel += [f"for (int idx{i} = 0; idx{i} < {self.output_shape[i]}; idx{i}++) {{\n" for i in range(0, len(self.output_shape))]
    else:
      # output_shape[-1] is get_global_id(0)
      self.kernel += [f"int idx{len(self.output_shape)-1-i} = {self.lang.gid[i]}; /* {self.output_shape[-1-i]} */\n" for i in range(min(len(self.lang.gid), len(self.output_shape))) if self.output_shape[-1-i] != 1]
      if len(self.output_shape) > len(self.lang.gid):
        # sometimes, there's more dimensions. compact all the dimensions into the first one
        # TODO: these compactions should be searchable (they sort of are with reshapes and permutes)
        final_dimension = len(self.output_shape)-len(self.lang.gid)
        for i in range(final_dimension-1, -1, -1):
          self.kernel += [f"int idx{i} = idx{final_dimension} % {self.output_shape[i]};", f"idx{final_dimension} = idx{final_dimension} / {self.output_shape[i]};\n"]
        self.output_shape = (prod(self.output_shape[0:final_dimension+1]), ) + self.output_shape[final_dimension+1:]
        if DEBUG >= 3: print(f"replaced output shape with {self.output_shape}")

    # early ast
    accumulators : List[Token] = []
    if self.reduceop is not None:
      if self.buftokens[0].can_float4():
        accumulators = [Token("acc%d.%s" % (i//4, "xyzw"[i%4]), self.buftokens[0].typ) for i in self.buftokens[0].offsets()]
        self.kernel += [f"float4 {tok} = {GPUCodegen.start_for_op[self.reduceop.op]};\n" for tok in dedup([x.tok.split('.')[0] for x in accumulators])]
      else:
        accumulators = [Token("acc%d" % i, self.buftokens[0].typ) for i in range(self.buftokens[0].size())]
        self.kernel += [f"{accumulator.decltype()} {accumulator.tok} = {GPUCodegen.start_for_op[self.reduceop.op]};\n" for accumulator in accumulators]
      acc_offsets = self.buftokens[self.bufs.index(self.earlybufs[0])].acc_offsets()
      self.kernel += [f"for (int idx{i} = 0; idx{i} < {self.full_shape[i]}; idx{i}++) {{\n" for i in range(self.first_reduce+len(self.group_for_reduce), self.shape_len)]
      fused_reduceop = LazyOp(FusedOps.MULACC, self.reduceop.src[0].src, self.reduceop.arg) if self.reduceop.op == ReduceOps.SUM and isinstance(self.reduceop.src[0], LazyOp) and self.reduceop.src[0].op == BinaryOps.MUL else self.reduceop
      self.kernel += [f"{x.tok};\n" for x in self.ast_parse(fused_reduceop, [accumulators[off] for off in acc_offsets], do_reduce=True)]
      self.kernel += ["}\n"] * (self.shape_len - (self.first_reduce + len(self.group_for_reduce)))

    # middle
    if self.group_for_reduce and self.reduceop is not None:
      self.kernel.append(self.lang.smem_prefix + f"float {self.buftokens[-1].tok}[{self.sts[-1].size()*self.buftokens[-1].size()}];\n")
      self.store(-1, accumulators)  # TODO: this is assuming the local size = global size. should use lidxs
      self.kernel.append(self.lang.barrier+"\n")

      # this is used to identify the thread doing the reducing (lidx == 0) and is repeated from store
      # must happen before the upcast
      lidx, lvalid = self.sts[-1].expr_idxs()
      assert lvalid.min == 1, "local buffer must always be valid"

      # if any group_for_reduce items aren't reduces, upcast them here
      for j in self.upcast_in_mid_reduce_axes:
        self.reshape_and_permute(None, [i for i in range(self.shape_len) if i != j] + [j])
        self.upcast()

      self.kernel.append(f"if ({lidx.render(render_cl)} == 0) {{\n")

      # second stage reduce with a new set of accumulators. TODO: this is very similar to above
      accumulators = [Token(f"output{i}", self.buftokens[0].typ) for i in range(self.buftokens[0].size())]
      # TODO: do we need acc_offsets here?
      self.kernel += [f"{accumulator.decltype()} {accumulator.tok} = {GPUCodegen.start_for_op[self.reduceop.op]};\n" for accumulator in accumulators]
      self.kernel.append(f"for (int mid = 0; mid < {self.sts[-1].size()}; mid++) {{\n")
      self.kernel += [f"{x.tok};\n" for x in self.ast_parse(LazyOp(self.reduceop.op, (None,), self.sts[0].shape), accumulators, do_reduce=True)]

      self.kernel.append("}\n")

    # late ast
    self.store(0, self.ast_parse(self.ast, accumulators))
    if self.group_for_reduce: self.kernel.append("}")
    if len(self.lang.gid) == 0: self.kernel += ["}"] * len(self.output_shape)
    self.kernel.append("\n}")

    # concat kernel into prg
    buftypes = [f"{'read_only' if i > 0 else 'write_only'} image2d_t" if hasattr(x._buf, "IMAGE") else self.lang.buffer_prefix+self.buftokens[i].decltype()+self.lang.buffer_suffix for i,x in enumerate(self.bufs) if x is not None]
    prg = ' '.join(list(self.prekernel) + [f"{self.lang.kernel_prefix} void KERNEL_NAME_PLACEHOLDER(",] +
      [', '.join([f'{t} data{i}' for i,t in enumerate(buftypes) if i not in self.bufs_to_delete] + self.lang.extra_args)] +
      [") {\n"] + self.kernel)

    # kernel function definition
    function_name = ("re_S" if self.reduceop else "ew_S") + '_'.join([str(x) for x in self.full_shape])

    # painfully name the function
    if prg in GPUCodegen.kernel_name_cache:
      function_name = GPUCodegen.kernel_name_cache[prg]
    else:
      GPUCodegen.kernel_cnt[function_name] += 1
      if GPUCodegen.kernel_cnt[function_name]:
        function_name = f"{function_name}{'_N'+str(GPUCodegen.kernel_cnt[function_name])}"
      GPUCodegen.kernel_name_cache[prg] = function_name

    return ASTRunner(function_name, prg.replace("KERNEL_NAME_PLACEHOLDER", function_name), self.bufs_to_delete,
      list(self.output_shape[::-1]) if len(self.output_shape) > 0 else [1],
      (self.group_for_reduce[::-1] + [1]*(len(self.output_shape)-len(self.group_for_reduce))) if self.group_for_reduce else None,
      op_estimate=self.info.flops, mem_estimate=sum(prod(x._base_shape) for x in self.bufs if x is not None))
