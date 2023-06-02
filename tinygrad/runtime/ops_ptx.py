from typing import Any
from tinygrad.ops import Compiled
from tinygrad.codegen.assembly import AssemblyCodegen
from tinygrad.runtime.ops_cuda import RawCUDABuffer, CUDAProgram, cuda
from tinygrad.ops import BinaryOps, UnaryOps
from tinygrad.codegen.linearizer import UOps
import functools

# https://docs.nvidia.com/cuda/parallel-thread-execution/#
class PTXCodegen(AssemblyCodegen):
  def generate(self):
    ins = [".version 7.8", ".target sm_86", ".address_size 64", f".visible .entry test({', '.join(f'.param .u64 buf_{i}' for i in range(len(self.bufs)))}) {{"]

    # load buffers
    ins += [f"ld.param.u64 %rd{i}, [buf_{i}];" for i in range(len(self.bufs))]

    # is this needed?
    #ins += [f"cvta.to.global.u64 %rd{i}, %rd{i};" for i in range(len(self.bufs))]

    # register allocation
    # TODO: work out non overlapping regs
    reg = {}
    for _,newvar,_,_ in self.uops:
      if newvar is not None:
        reg[newvar] = f"%f{len(reg)}"

    def idx_to_t(idx):
      from tinygrad.shape.symbolic import Variable, NumNode, MulNode, DivNode, ModNode, GeNode, LtNode, SumNode, AndNode
      v = 0
      def new_var():
        nonlocal v
        v += 1
        return f"%t{v-1}"

      def render_variable(self, ops, ctx):
        if self.expr.startswith("gidx"):
          v = new_var()
          ins.append(f"mov.u32 {v}, {global_regs[int(self.expr[4:])]};")
          return v
        else:
          raise RuntimeError(f"unknown variable {self.expr}")
      def render_numnode(self, ops, ctx):
        v = new_var()
        ins.append(f"mov.u32 {v}, {self.b};")
        return v

      def render_mulnode(self, ops, ctx):
        v = new_var()
        ins.append(f"mul.lo.u32 {v}, {self.a.render(ops, ctx)}, {self.b};")
        return v

      def render_add(a, b):
        v = new_var()
        ins.append(f"add.u32 {v}, {a}, {b};")
        return v

      return idx.render({ Variable: render_variable, NumNode: render_numnode, MulNode: render_mulnode,
        SumNode: lambda self,ops,ctx: functools.reduce(lambda a,b: render_add(a, b.render(ops,ctx)), self.nodes[1:], self.nodes[0].render(ops,ctx)),}, ins)

    local_size = [1]
    global_size = []
    global_regs = []
    for uop,newvar,vin,args in self.uops:
      if uop == UOps.LOOP:
        if args[1] == "global":
          for i,var in enumerate(args[0]):
            global_size.append(var.max+1)
            global_regs.append(f"%global{i}")
            ins.append(f"mov.u32 {global_regs[-1]}, %ctaid.{'xyz'[i]};")
            # and %ntid.x
            #ins.append(f"mov.u32 %r{i}, %tid.{'xyz'[i]};")
            #ins.append(f"cvt.u64.u32 {global_regs[-1]}, %lt{i};")
            #ins.append(f"mul.wide.u32 {global_regs[-1]}, %r{i}, 4;")
      elif uop == UOps.LOAD:
        ins.append(f"// LOAD {args}")
        ins.append(f"cvt.u64.u32 %bt0, {idx_to_t(args.idx*4)};")
        ins.append(f"add.u64 %bt0, %rd{args.i}, %bt0;")
        ins.append(f"ld.global.f32 {reg[newvar]}, [%bt0];")
      elif uop == UOps.ALU:
        alu = {BinaryOps.ADD: "add.f32", BinaryOps.SUB: "sub.f32", BinaryOps.MUL: "mul.f32"}
        ins.append(f"{alu[args]} {reg[newvar]}, {', '.join([reg[x] for x in vin])};")
      elif uop == UOps.STORE:
        ins.append(f"// STORE {args}")
        ins.append(f"cvt.u64.u32 %bt0, {idx_to_t(args.idx*4)};")
        ins.append(f"add.u64 %bt0, %rd{args.i}, %bt0;")
        ins.append(f"st.global.f32 [%bt0], {reg[vin[0]]};")

    ins = ins[0:4] + [f".reg .b64 %rd<{len(self.bufs)}>;",
                      f".reg .f32 %f<{len(reg)}>;",
                      f".reg .b64 %bt<1>;",
                      f".reg .b32 %t<8>;",
                      f".reg .b32 %global<{len(global_regs)}>;"] + ins[4:]
    ins += ["ret;", "}"]
    return "test", '\n'.join(ins), global_size, local_size

PTXBuffer = Compiled(RawCUDABuffer, PTXCodegen, CUDAProgram, cuda.Context.synchronize)
