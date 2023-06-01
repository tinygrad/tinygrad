from tinygrad.ops import Compiled
from tinygrad.codegen.assembly import AssemblyCodegen
from tinygrad.runtime.ops_cuda import RawCUDABuffer, CUDAProgram, cuda
from tinygrad.ops import BinaryOps, UnaryOps
from tinygrad.codegen.linearizer import UOps

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

    local_size = [1]
    global_size = []
    global_regs = []
    for uop,newvar,vin,args in self.uops:
      if uop == UOps.LOOP:
        if args[1] == "global":
          for i,var in enumerate(args[0]):
            global_size.append(var.max+1)
            global_regs.append(f"%rw{i}")
            # and %ntid.x
            #ins.append(f"mov.u32 %r{i}, %tid.{'xyz'[i]};")
            ins.append(f"mov.u32 %r{i}, %ctaid.{'xyz'[i]};")
            ins.append(f"mul.wide.u32 {global_regs[-1]}, %r{i}, 4;")
            #ins.append(f"cvt.u64.u32 {global_regs[-1]}, %r{i};")
      elif uop == UOps.LOAD:
        ins.append(f"add.u64 %t0, %rd{args.i}, {global_regs[0]};")
        ins.append(f"ld.global.f32 {reg[newvar]}, [%t0];")
      elif uop == UOps.ALU:
        if args == BinaryOps.ADD:
          ins.append(f"add.f32 {reg[newvar]}, {reg[vin[0]]}, {reg[vin[1]]};")
      elif uop == UOps.STORE:
        ins.append(f"add.u64 %t0, %rd{args.i}, {global_regs[0]};")
        ins.append(f"st.global.f32 [%t0], {reg[vin[0]]};")

    ins = ins[0:4] + [f".reg .b64 %rd<{len(self.bufs)}>;",
                      f".reg .f32 %f<{len(reg)}>;",
                      f".reg .b64 %t<1>;",
                      f".reg .b32 %r<{len(global_regs)}>;",
                      f".reg .b64 %rw<{len(global_regs)}>;"] + ins[4:]
    ins += ["ret;", "}"]
    return "test", '\n'.join(ins), global_size, local_size

PTXBuffer = Compiled(RawCUDABuffer, PTXCodegen, CUDAProgram, cuda.Context.synchronize)
