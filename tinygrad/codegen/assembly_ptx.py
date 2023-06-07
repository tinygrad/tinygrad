from typing import Dict
from tinygrad.codegen.assembly import AssemblyCodegen
from tinygrad.ops import BinaryOps, UnaryOps, FusedOps
from tinygrad.codegen.linearizer import UOps, Token
from tinygrad.shape.symbolic import Variable, NumNode, MulNode, DivNode, ModNode, GeNode, LtNode, SumNode, AndNode
from tinygrad.helpers import dtypes
import functools, struct

dtype_to_nvtype = {dtypes.float32: "f32", dtypes.float16: "u16"}
def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])

# https://docs.nvidia.com/cuda/parallel-thread-execution/#
class PTXCodegen(AssemblyCodegen):
  #supports_constant_folding: bool = True

  def generate(self):
    ins = [".version 7.8", ".target sm_86", ".address_size 64", f".visible .entry test({', '.join(f'.param .u64 buf_{i}' for i in range(len(self.bufs)))}) {{"]

    # load buffers, is the cvta.to needed?
    ins += [f"ld.param.u64 %rd{i}, [buf_{i}];" for i in range(len(self.bufs))]
    ins += [f"cvta.to.global.u64 %rd{i}, %rd{i};" for i in range(len(self.bufs))]

    # register allocation
    # TODO: work out non overlapping regs
    reg: Dict[Token, str] = {}
    for _,newvar,_,_ in self.uops:
      if newvar is not None and newvar not in reg:
        reg[newvar] = f"%f{len(reg)}"

    reduce_vars: Dict[str, str] = {}
    def new_reduce_var(vv):
      nonlocal reduce_vars
      assert vv.expr not in reduce_vars
      reduce_vars[vv.expr] = f"%v{len(reduce_vars)}"
      ins.append(f"mov.u32 {reduce_vars[vv.expr]}, {vv.min};")

    v, p = len(reduce_vars), 0
    max_v, max_p = v, 1
    def new_var():
      nonlocal v, max_v
      v += 1
      max_v = max(v, max_v)
      return f"%v{v-1}"

    def new_pred_var():
      nonlocal p, max_p
      p += 1
      max_p = max(p, max_p)
      return f"%p{p-1}"

    def render_xnode(node, pred=False):
      def _render_xnode(self, ops, ctx):
        v = new_pred_var() if pred else new_var()
        ins.append(f"{node} {v}, {self.a.render(ops, ctx)}, {self.b};")
        return v
      return _render_xnode

    def idx_to_t(idx):
      nonlocal v, p
      # reset this
      v, p = len(reduce_vars), 0

      def render_variable(self, ops, ctx):
        if self.expr.startswith("gidx"):
          v = new_var()
          ins.append(f"mov.u32 {v}, {global_regs[int(self.expr[4:])]};")
          return v
        elif self.expr.startswith("lidx"):
          v = new_var()
          ins.append(f"mov.u32 {v}, {local_regs[int(self.expr[4:])]};")
          return v
        elif self.expr in reduce_vars:
          return reduce_vars[self.expr]
        else:
          raise RuntimeError(f"unknown variable {self.expr}")

      def render_numnode(self, ops, ctx):
        v = new_var()
        ins.append(f"mov.u32 {v}, {self.b};")
        return v

      def render_add(a, b):
        v = new_var()
        ins.append(f"add.u32 {v}, {a}, {b};")
        return v

      def render_and(a, b):
        p = new_pred_var()
        ins.append(f"and.pred {p}, {a}, {b};")
        return p

      return idx.render({ Variable: render_variable, NumNode: render_numnode, MulNode: render_xnode("mul.lo.u32"),
                          GeNode: render_xnode("setp.ge.s32", True), LtNode: render_xnode("setp.lt.s32", True),
                          DivNode: render_xnode("div.u32"), ModNode: render_xnode("rem.u32"),
        SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: render_add(a, b.render(ops,ctx)), self.nodes[1:], self.nodes[0].render(ops,ctx)),
        AndNode: lambda self,ops,ctx: functools.reduce(lambda a,b: render_and(a, b.render(ops,ctx)), self.nodes[1:], self.nodes[0].render(ops,ctx)),}, ins)

    local_size = []  # TODO: make this work
    local_regs = []
    global_size = []
    global_regs = []
    shared_name = None
    skipload_branch = 0
    for uop,newvar,vin,args in self.uops:
      if uop == UOps.DEFINE_LOCAL:
        ins.append(f".shared .align 4 .b8 {args[0]}[{args[1]*4}];")
        shared_name = args[0]
      elif uop == UOps.CONST and newvar is not None:
        ins.append(f"mov.f32 {reg[newvar]}, 0f{float_to_hex(args)};")
      elif uop == UOps.LOOP:
        if args[1] == "global":
          for i,var in enumerate(args[0]):
            global_size.append(var.max+1)
            global_regs.append(f"%global{i}")
            # is this needed?
            ins.append(f"mov.u32 %temp0, %ctaid.{'xyz'[len(args[0])-1-i]};")
            ins.append(f"mov.u32 %temp1, %ntid.{'xyz'[len(args[0])-1-i]};")
            ins.append(f"mov.u32 %temp2, %tid.{'xyz'[len(args[0])-1-i]};")
            ins.append(f"mad.lo.s32 {global_regs[-1]}, %temp0, %temp1, %temp2;")
        elif args[1] == "local":
          for i,var in enumerate(args[0]):
            local_size.append(var.max+1)
            global_size[i] *= local_size[i]
            local_regs.append(f"%local{i}")
            ins.append(f"mov.u32 {local_regs[-1]}, %tid.{'xyz'[len(args[0])-1-i]};")
        else:
          for var in args[0]:
            if not isinstance(var, NumNode):  # TODO: why is this coming through?
              ins.append(f"// LOOP {var}")
              new_reduce_var(var)
              ins.append(f"$loop_{var.expr}:")
      elif uop == UOps.ENDLOOP:
        if args[1] not in ["global", "local"]:
          for var in args[0][::-1]:
            if not isinstance(var, NumNode):  # TODO: why is this coming through?
              ins.append(f"// ENDLOOP {var} as {reduce_vars[var.expr]}")
              ins.append(f"setp.ne.s32 %p0, {reduce_vars[var.expr]}, {var.max};")
              ins.append(f"add.u32 {reduce_vars[var.expr]}, {reduce_vars[var.expr]}, 1;")
              ins.append(f"@%p0 bra $loop_{var.expr};")
      elif uop == UOps.LOAD and newvar is not None:
        ins.append(f"// LOAD {args}")
        if args.valid.min != 1:
          ins.append(f"mov.f32 {reg[newvar]}, 0f00000000;")  # 0.0 is the alt value
          if args.valid.max == 1:
            ins.append(f"@!{idx_to_t(args.valid)} bra $skipload_{skipload_branch};")
        if args.valid.max == 1:
          ins.append(f"cvt.u64.u32 %bt0, {idx_to_t(args.idx*self.bufs[args.i].dtype.itemsize)};")
          # doing a real load
          sreg = reg[newvar]
          stype = dtype_to_nvtype[self.bufs[args.i].dtype]
          if self.bufs[args.i].dtype == dtypes.float16:
            sreg = "%h"
          if args.i == -1:
            ins.append(f"mov.u64 %bt1, {shared_name};")
            ins.append("add.u64 %bt0, %bt1, %bt0;")
            ins.append(f"ld.shared.{stype} {sreg}, [%bt0];")
          else:
            ins.append(f"add.u64 %bt0, %rd{args.i}, %bt0;")
            ins.append(f"ld.global.{stype} {sreg}, [%bt0];")
          if self.bufs[args.i].dtype == dtypes.float16:
            ins.append(f"cvt.f32.f16 {reg[newvar]}, %h;")
          if args.valid.min != 1:
            ins.append(f"$skipload_{skipload_branch}:")
            skipload_branch += 1
      elif uop == UOps.ALU and newvar is not None:
        if args == BinaryOps.CMPEQ:
          ins.append(f"setp.eq.f32 %p0, {reg[vin[0]]}, {reg[vin[1]]};")
          ins.append(f"selp.f32 {reg[newvar]}, 0f3F800000, 0f00000000, %p0;")
        elif args == UnaryOps.LOG:
          # TODO: push this up the stack? seems like all GPUs have this
          ins.append(f"lg2.approx.f32 {reg[newvar]}, {reg[vin[0]]};")
          ins.append(f"mul.f32 {reg[newvar]}, {reg[newvar]}, 0f3f317218;") # log(2)/log(e)
        elif args == UnaryOps.EXP:
          ins.append(f"mul.f32 {reg[newvar]}, {reg[vin[0]]}, 0f3fb8aa3b;") # log(e)/log(2)
          ins.append(f"ex2.approx.ftz.f32 {reg[newvar]}, {reg[newvar]};")
        #elif args == BinaryOps.POW:
          # pow(a,b) = exp(a*log(b))
          # actually...we might want to write it this way in tinygrad
          #ins.append(f"lg2.approx.f32 {reg[newvar]}, {reg[vin[1]]};")
          #ins.append(f"mul.f32 {reg[newvar]}, {reg[newvar]}, 0f3f317218;") # log(2)/log(e)
        else:
          alu = {BinaryOps.ADD: "add.f32", BinaryOps.SUB: "sub.f32",
                 BinaryOps.MUL: "mul.f32", BinaryOps.DIV: "div.rn.f32",
                 BinaryOps.MAX: "max.f32", UnaryOps.SIN: "sin.approx.f32",
                 FusedOps.MULACC: "fma.rn.f32"}
          if args == FusedOps.MULACC: vin = [vin[1], vin[2], vin[0]]  # TODO: reorder MULACC everywhere
          ins.append(f"{alu[args]} {reg[newvar]}, {', '.join([reg[x] for x in vin])};")
      elif uop == UOps.STORE:
        ins.append(f"// STORE {args}")
        ins.append(f"cvt.u64.u32 %bt0, {idx_to_t(args.idx*self.bufs[args.i].dtype.itemsize)};")
        sreg = reg[vin[0]]
        stype = dtype_to_nvtype[self.bufs[args.i].dtype]
        if self.bufs[args.i].dtype == dtypes.float16:
          ins.append(f"cvt.rn.f16.f32 %h, {sreg};")
          sreg = "%h"
        if args.i == -1:
          ins.append(f"mov.u64 %bt1, {shared_name};")
          ins.append("add.u64 %bt0, %bt1, %bt0;")
          ins.append(f"st.shared.{stype} [%bt0], {sreg};")
        else:
          ins.append(f"add.u64 %bt0, %rd{args.i}, %bt0;")
          ins.append(f"st.global.{stype} [%bt0], {sreg};")

    ins = ins[0:4] + [f".reg .b64 %rd<{len(self.bufs)}>;",
                      f".reg .f32 %f<{len(reg)}>;",
                      ".reg .b16 %h;",
                      ".reg .b64 %bt<2>;",
                      ".reg .b32 %temp<3>;",
                      f".reg .b32 %v<{max_v}>;",  # TODO: make this dynamic, does it matter?
                      f".reg .pred %p<{max_p}>;",
                      f".reg .b32 %local<{len(local_regs)}>;",
                      f".reg .b32 %global<{len(global_regs)}>;"] + ins[4:]
    ins += ["ret;", "}"]
    return "test", '\n'.join(ins), global_size, local_size
