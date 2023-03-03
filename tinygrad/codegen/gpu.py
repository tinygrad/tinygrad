import math
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Set, Final, NamedTuple
from tinygrad.ops import UnaryOps, BinaryOps, ReduceOps, LazyOp, Op, ASTRunner
from tinygrad.codegen.ast import ASTKernel, Token, Types
from tinygrad.shape.symbolic import Node, ModNode, DivNode, render_python
from tinygrad.shape import ShapeTracker
from tinygrad.helpers import getenv, DEBUG, prod

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

def to_image_idx(base_shape:Tuple[int, ...], idxy:Node, validhacks=False):
  idx = (idxy//4)%base_shape[1]
  idy = (idxy//(4*base_shape[1]))%base_shape[0]
  if validhacks: idx, idy = [x.a if isinstance(x, ModNode) and x.a.max < x.b*2 else x for x in (idx, idy)]
  return f"(int2)({idx.render(render_cl)}, {idy.render(render_cl)})"

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
    BinaryOps.MAX: "max(A,B)", ReduceOps.SUM: "A=(A+B)", ReduceOps.MAX: "A=max(A,B)"
  }
  start_for_op : Final[Dict[Op, str]] = {ReduceOps.SUM: "0.0", ReduceOps.MAX: "-INFINITY"}

  def store(self, buf_index:int, value:List[Token]) -> None:
    assert len(value) == self.buftokens[buf_index].size(), f"size mismatch {len(value)} != {self.buftokens[buf_index].size()}"
    assert len(self.bufs[buf_index].st.views) == 1, "store has more than one view"

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
        v = Token(f"{self.lang.float4}({','.join([to_store[o+j].tok for j in range(4)])})", Types.FLOAT4) 
      if hasattr(self.bufs[buf_index]._buf, "IMAGE"):
        assert v.typ == Types.FLOAT4, "Image requires upcasting to FLOAT4"
        self.kernel.append(f"write_imagef(data{buf_index}, {to_image_idx(self.bufs[buf_index]._base_shape, idxy)}, {v.tok});  /* {self.bufs[buf_index]._base_shape} */\n")
      elif v.typ == Types.FLOAT4:
        self.kernel.append(f"(({self.lang.buffer_prefix}float4*)data{buf_index})[{(idxy//4).render(render_cl)}] = {v.tok};\n")
      else:
        self.kernel.append(f"data{buf_index}[{(idxy//(4 if v.typ == Types.FLOAT4 else 1)).render(render_cl)}] = {v.tok};\n")

  def load(self, buf_index:int) -> List[Token]:
    # constant folding
    const = None
    if self.bufs[buf_index]._base_shape == (1,) and self.bufs[buf_index]._backing is not None:
      if buf_index != 0: self.bufs_to_delete.add(buf_index)
      val = self.bufs[buf_index]._backing[0]
      assert not math.isnan(val)
      const = Token(f"({val}f)", Types.FLOAT)
    should_upcast = self.lang.float4 and const is None and self.buftokens[buf_index].can_float4()
    tokens = []
    for o in self.buftokens[buf_index].offsets():
      key = f"val{buf_index}_{o}" if o >= 0 else f"val{buf_index}_m{-o}"
      if (buf_index, o) not in self.loaded_keys:
        idxy, valid = self.sts[buf_index].expr_idxs(o)
        if should_upcast:
          can_merge = True
          for j in range(1,4):
            idxy_test, valid_test = self.sts[buf_index].expr_idxs(o+j)
            can_merge = can_merge and valid.render() == valid_test.render()
            can_merge = can_merge and (idxy+j).render() == idxy_test.render()
            #print((idxy+j).render(), idxy_test.render(), valid.render(), valid_test.render(), can_merge)
        if const is not None:
          ldr = const
        elif hasattr(self.bufs[buf_index]._buf, "IMAGE"):
          assert should_upcast and can_merge, f"Image requires upcasting to FLOAT4 {self.buftokens[buf_index]}"
          ldr = Token(f"read_imagef({self.buftokens[buf_index].tok}, smp, {to_image_idx(self.bufs[buf_index]._base_shape, idxy, VALIDHACKS)}) /* {self.bufs[buf_index]._base_shape} */", Types.FLOAT4)
        elif should_upcast and can_merge:
          ldr = Token(f"(({self.lang.buffer_prefix}float4*){self.buftokens[buf_index].tok})[{(idxy//4).render(render_cl)}]", Types.FLOAT4)
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
    return tokens

  def ast_parse(self, x, acc:List[Token], do_reduce=False) -> List[Token]:
    if not isinstance(x, LazyOp): return self.load(self.bufs.index(x))
    if isinstance(x.op, ReduceOps) and not do_reduce: return acc
    values : List[List[Token]] = ([acc] if isinstance(x.op, ReduceOps) else []) + [self.ast_parse(v, acc, do_reduce) for v in x.src]
    code = GPUCodegen.code_for_op[x.op]  # TODO: replace this with a function
    if len(values) == 2:
      assert len(values[0]) == len(values[1]) and values[0][0].typ == values[1][0].typ, f"values mismatch {values}"
      return [Token(code.replace("A", a.tok).replace("B", b.tok), a.typ) for a,b in zip(values[0], values[1])]
    else:
      return [Token(code.replace("A", a.tok), a.typ) for a in values[0]]

  def hand_coded_optimizations(self):
    # if there's images in the earlybufs, we have to make an axis the 4 loading one
    # shove the axis to the end and remove 
    if any(hasattr(buf._buf, "IMAGE") for buf in self.earlybufs):
      eb_valids = [True] * self.shape_len
      for i in range(len(self.bufs)):
        if hasattr(self.bufs[i]._buf, "IMAGE") and self.bufs[i] in self.earlybufs:
          valids = [self.sts[i].shape[j]%4 == 0 and self.sts[i].views[-1].strides[j] == 1 for j in range(self.shape_len)]
          eb_valids = [x and y for x,y in zip(eb_valids, valids)]
      assert any(eb_valids), f"invalid op with images {eb_valids}"
      eb_valid = eb_valids.index(True)
      if DEBUG >= 3: print(f"early merging axis {eb_valid} from {eb_valids}")

      # no change, we added a dimension
      self.reshape_and_permute(
        lambda x: list(x[0:eb_valid]) + ([x[eb_valid]//4, 4] if x[eb_valid] > 1 else [1,1]) + list(x[eb_valid+1:]),
        [i for i in range(self.shape_len+1) if i != eb_valid+1] + [eb_valid+1])

      # drop the last dimension
      self.upcast()

    # simplify (sets first_reduce)
    self.simplify_ones()

    # are we grouping?
    if self.lang.float4 and not self.buftokens[0].can_float4() and self.first_reduce <= 2 and self.first_reduce + 1 <= self.shape_len and prod(self.sts[0].shape[:self.first_reduce]) <= 2048:
      # TODO: use 1024 if it's allowed in a smarter way
      for sz in (([256, 16]) if prod(self.sts[0].shape[:self.first_reduce]) <= 32 else [16]):
        if all([st.shape[self.first_reduce] % sz == 0 or st.shape[self.first_reduce] == 1 for st in self.sts]):
          self.group_for_reduce.append(sz)
          break

    # if there's images in the latebufs, we have to make an axis the 4 storing one. this affects the kernel shape
    if any(hasattr(buf._buf, "IMAGE") for buf in self.bufs if buf not in self.earlybufs) and not self.buftokens[0].can_float4():
      lb_valids = [True] * self.shape_len
      for i in range(len(self.bufs)):
        valids = [self.sts[i].shape[j]%4 == 0 and (self.sts[i].views[-1].strides[j] == 1 or not hasattr(self.bufs[i]._buf, "IMAGE") or self.bufs[i] in self.earlybufs) for j in range(self.shape_len)]
        lb_valids = [x and y for x,y in zip(lb_valids, valids)]
      assert any(lb_valids), f"invalid op with images {lb_valids}"
      lb_valid = lb_valids.index(True)
      assert lb_valid < self.first_reduce, f"can't be in the reduce {lb_valid}"
      if DEBUG >= 3: print(f"late merging axis {lb_valid} from {lb_valids}")

      # no change, we added a dimension
      self.reshape_and_permute(
        lambda x: list(x[0:lb_valid]) + [x[lb_valid]//4, 4] + list(x[lb_valid+1:]),
        [i for i in range(self.shape_len+1) if i != lb_valid+1] + [lb_valid+1])

      if self.group_for_reduce and self.first_reduce <= 2:
        self.upcast_in_mid_reduce = True
        self.group_for_reduce.append(4)
      else:
        # drop the last dimension
        self.upcast()

    # simplify (sets first_reduce)
    self.simplify_ones()

    # split to 4 float4s
    if self.buftokens[0].can_float4() and any(hasattr(buf._buf, "IMAGE") for buf in self.earlybufs) and prod(self.sts[0].shape[:self.first_reduce]) >= 2048 and not self.group_for_reduce:
      xb_choices = []
      for i in range(self.first_reduce):
        if all(st.shape[i]%4 == 0 for st in self.sts):
          xb_choices.append((sum(st.views[-1].strides[i]>0 for st in self.sts), sum(st.views[-1].strides[i] for st in self.sts), i))

      if len(xb_choices):
        xb_choice = sorted(xb_choices)[0][2]
        if DEBUG >= 3: print(f"float4 merging axis {xb_choice} : {xb_choices}")

        # this leaves the last axis in place
        self.reshape_and_permute(
          lambda x: list(x[0:xb_choice]) + [x[xb_choice]//4, 4] + list(x[xb_choice+1:]),
          [i for i in range(self.shape_len+1) if i != xb_choice+1] + [xb_choice+1])

        # drop the last dimension
        self.upcast()

        # re-simplify
        self.simplify_ones()

    # use more opencl indexing
    if self.first_reduce == 2 and hasattr(self.bufs[0]._buf, "IMAGE"):
      base_shape = self.bufs[0]._base_shape
      if all([(base_shape[0]*base_shape[1])%st.shape[0] == 0 and st.shape[0]//base_shape[0] != 0 for st in self.sts]):
        if DEBUG >= 3: print("split opencl", base_shape, self.sts[0].shape)
        self.reshape_and_permute(lambda x: [base_shape[0], x[0]//base_shape[0]]+list(x[1:]), None)
        self.simplify_ones()

    # group for reduce
    if len(self.group_for_reduce):
      # with permute for memory coalesing
      if len(self.group_for_reduce) == 2:
        permute_axis = list(range(0, self.first_reduce)) + [self.first_reduce+1, self.shape_len, self.first_reduce] + list(range(self.first_reduce+2, self.shape_len))
      else:
        permute_axis = list(range(0, self.first_reduce)) + [self.first_reduce+1, self.first_reduce] + list(range(self.first_reduce+2, self.shape_len+1))
      self.reshape_and_permute(lambda x: list(x[0:self.first_reduce]) + [max(1, x[self.first_reduce]//self.group_for_reduce[0]), min(x[self.first_reduce], self.group_for_reduce[0])] + list(x[self.first_reduce+1:]), permute_axis)

    # if last dim <= 3 and it's a reduce dim, upcast (loop unrolling)
    end_dimension = max([st.shape[-1] for st in self.sts])
    if self.first_reduce < self.shape_len and end_dimension > 1 and end_dimension <= 3 and max([x.size() for i,x in enumerate(self.buftokens) if self.bufs[i] in self.earlybufs]) <= 4:
      self.upcast()

  # STOP WASTING TIME WITH DOING THE RESHAPES AND PERMUTES BY HAND. KERNEL SEARCH IS THE ONLY WAY IT WILL EVER BE GOOD
  # group_for_reduce will have to be better first
  def codegen(self) -> ASTRunner:
    self.process()
    self.upcast_in_mid_reduce = False
    self.hand_coded_optimizations()

    # add a local buffer for multistage reduce
    if len(self.group_for_reduce):
      self.sts.append(ShapeTracker(tuple([1] * self.first_reduce + self.group_for_reduce + [1] * (self.shape_len - len(self.group_for_reduce) - self.first_reduce))))
      self.buftokens.append(Token("temp", Types.FLOAT, ptr=True))

    self.output_shape = list(self.sts[0].shape[:self.first_reduce]) + self.group_for_reduce
    if DEBUG >= 3:
      print("output shape", self.output_shape)
      self.printbufs("new:", DEBUG>=4)

    self.bufs_to_delete : Set[int] = set()
    self.loaded_keys : Dict[Tuple[int,int], Token] = {}

    self.prekernel : Set[str] = set()
    self.kernel : List[str] = ["const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"] if any(hasattr(buf._buf, "IMAGE") for buf in self.bufs) else []

    # output_shape[-1] is get_global_id(0)
    if len(self.lang.gid) == 0:
      self.kernel += [f"for (int idx{i} = 0; idx{i} < {self.output_shape[i]}; idx{i}++) {{\n" for i in range(0, len(self.output_shape))]
    else:
      self.kernel += [f"int idx{len(self.output_shape)-1-i} = {self.lang.gid[i]}; /* {self.output_shape[-1-i]} */\n" for i in range(min(len(self.lang.gid), len(self.output_shape))) if self.output_shape[-1-i] != 1]
      if len(self.output_shape) > len(self.lang.gid):
        # sometimes, there's more dimensions. compact all the dimensions into the first one
        # TODO: these compactions should be searchable
        final_dimension = len(self.output_shape)-len(self.lang.gid)
        for i in range(final_dimension-1, -1, -1):
          self.kernel += [f"int idx{i} = idx{final_dimension} % {self.output_shape[i]};", f"idx{final_dimension} = idx{final_dimension} / {self.output_shape[i]};\n"]
        self.output_shape = [prod(self.output_shape[0:final_dimension+1])] + list(self.output_shape[final_dimension+1:])
        if DEBUG >= 3: print(f"replaced output shape with {self.output_shape}")

    # early ast
    accumulators : List[Token] = [Token("acc%d" % i, self.buftokens[0].typ) for i in range(self.buftokens[0].size())]
    if self.reduceop is not None:
      full_shape_candidates = [x.shape for x in self.sts if x.shape != self.sts[0].shape]
      full_shape : Tuple[int, ...] = self.sts[0].shape if len(full_shape_candidates) == 0 else full_shape_candidates[0]

      acc_offsets = self.buftokens[self.bufs.index(self.earlybufs[0])].acc_offsets()
      assert self.reduceopop is not None
      self.kernel += [f"{accumulator.decltype()} {accumulator.tok} = {GPUCodegen.start_for_op[self.reduceopop]};\n" for accumulator in accumulators]
      self.kernel += [f"for (int idx{i} = 0; idx{i} < {full_shape[i]}; idx{i}++) {{\n" for i in range(self.first_reduce+len(self.group_for_reduce), self.shape_len)]
      self.kernel += [f"{x.tok};\n" for x in self.ast_parse(self.reduceop, [accumulators[off] for off in acc_offsets], do_reduce=True)] + ["}\n"] * (self.shape_len - (self.first_reduce + len(self.group_for_reduce)))
    
    # middle
    if self.group_for_reduce:
      lidx, lvalid = self.sts[-1].expr_idxs()
      assert lvalid.min == 1, "local buffer must always be valid"
      self.kernel.append(f"int mid_idx = {lidx.render(render_cl)};\n")
      for i,acc in enumerate(accumulators):
        self.kernel.append(self.lang.smem_prefix + f"{acc.decltype()} {self.buftokens[-1].tok}{i}[{prod(self.group_for_reduce)}];")
        self.kernel.append(f"{self.buftokens[-1].tok}{i}[mid_idx] = {acc.tok};\n")
      self.kernel.append(self.lang.barrier+"\n")

      if self.upcast_in_mid_reduce:
        assert len(self.group_for_reduce) == 2
        # it should be the last dimension
        self.reshape_and_permute(None, [i for i in range(self.shape_len) if i != self.first_reduce+1] + [self.first_reduce+1])
        self.upcast()

      assert self.reduceopop is not None
      self.kernel.append("if (mid_idx == 0) {\n")
      new_accumulators = [Token(f"output{i}", self.buftokens[0].typ) for i in range(self.buftokens[0].size())]
      for acc in new_accumulators: self.kernel.append(f"{acc.decltype()} {acc.tok} = 0.0;\n")
      self.kernel.append(f"for (int mid = 0; mid < {prod(self.group_for_reduce)//(4 if self.upcast_in_mid_reduce else 1)}; mid++) {{\n")
      for i in range(len(new_accumulators)//(4 if self.upcast_in_mid_reduce else 1)):
        if self.upcast_in_mid_reduce:
          self.kernel.append(f'float4 ld = vload4(0, &temp{i}[mid*4]);\n')
          for j in range(4):
            self.kernel.append(GPUCodegen.code_for_op[self.reduceopop].replace('A', new_accumulators[i*4+j].tok).replace('B', f'ld.{"xyzw"[j]}')+";\n")
        else:
          self.kernel.append(GPUCodegen.code_for_op[self.reduceopop].replace('A', new_accumulators[i].tok).replace('B', f'temp{i}[mid]')+";\n")
      self.kernel.append("}\n")
      accumulators = new_accumulators
    
    # late ast
    self.store(0, self.ast_parse(self.ast, accumulators))
    if self.group_for_reduce: self.kernel.append("}")
    if len(self.lang.gid) == 0: self.kernel += ["}"] * len(self.output_shape)
    self.kernel.append("\n}")

    # concat kernel into prg
    buftypes = [f"{'read_only' if i > 0 else 'write_only'} image2d_t" if hasattr(x._buf, "IMAGE") else self.lang.buffer_prefix+self.buftokens[i].decltype()+self.lang.buffer_suffix for i,x in enumerate(self.bufs)]
    prg = ' '.join(list(self.prekernel) + [f"{self.lang.kernel_prefix} void KERNEL_NAME_PLACEHOLDER(",] +
      [', '.join([f'{t} data{i}' for i,t in enumerate(buftypes) if i not in self.bufs_to_delete] + self.lang.extra_args)] +
      [") {\n"] + self.kernel)

    # kernel function definition
    function_name = ("re_S" if self.reduceop else "ew_S") + '_'.join([str(x) for x in self.bufs[0].shape if x != 1])

    # painfully name the function
    if prg in GPUCodegen.kernel_name_cache:
      function_name = GPUCodegen.kernel_name_cache[prg]
    else:
      GPUCodegen.kernel_cnt[function_name] += 1
      if GPUCodegen.kernel_cnt[function_name]:
        function_name = f"{function_name}{'_N'+str(GPUCodegen.kernel_cnt[function_name])}"
      GPUCodegen.kernel_name_cache[prg] = function_name

    if DEBUG >= 3 and len(self.bufs_to_delete): print(f"deleting buffers {self.bufs_to_delete}")
    return ASTRunner(function_name, prg.replace("KERNEL_NAME_PLACEHOLDER", function_name), self.bufs_to_delete,
      self.output_shape[::-1] if len(self.output_shape) > 0 else [1],
      (self.group_for_reduce[::-1] + [1]*(len(self.output_shape)-len(self.group_for_reduce))) if self.group_for_reduce else None,
      op_estimate=self.info.flops, mem_estimate=sum(prod(x._base_shape) for x in self.bufs))
