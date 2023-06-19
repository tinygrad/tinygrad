from tinygrad.codegen.assembly import AssemblyCodegen, AssemblyInstruction, float_to_hex
from tinygrad.ops import BinaryOps, UnaryOps
from tinygrad.codegen.linearizer import UOps
from tinygrad.helpers import DEBUG, dtypes

class X86Codegen(AssemblyCodegen):
  #supports_constant_folding: bool = True

  def specialize(self, asm):
    type_to_op_suffix = {dtypes.float32: 'l', dtypes.bool: 'b', dtypes.int32: 'l', dtypes.int64: 'q', dtypes.uint32: 'l', dtypes.uint64: 'q'}
    def inst(instr, reg, fp=False, simd=False): return f"{instr}{'s' if fp else ''}{'p' if simd else ''}{type_to_op_suffix[reg_type[reg.nm]]}"
    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.CMPLT: "cmp",  BinaryOps.CMPEQ: "cmp", BinaryOps.MAX: "max",
         UnaryOps.NOOP: "mov", UnaryOps.SIN: "call sinf@PLT", UnaryOps.LOG2: "call log2f@PLT", UnaryOps.EXP2: "call exp2f@PLT"}

    ins = []
    reg_map = {}
    reg_type = {}
    local_var_size = 0
    for uop, out, vin, arg in filter(lambda op: op.op == UOps.DEFINE_REGISTER, asm):
      for i in range(arg[2]):
        local_var_size += arg[0].itemsize
        reg_map[f"%{arg[1]}{i}"] = f"-{local_var_size}(%rsp)"
        reg_type[f"%{arg[1]}{i}"] = arg[0]

    print(reg_map)
    print(reg_type)
    print(local_var_size)

    for uop, out, vin, arg in asm:
      if DEBUG >= 5: ins.append(f"# {AssemblyInstruction(uop, out, vin, arg)}")
      if uop == UOps.DEFINE_REGISTER: pass
      elif uop == UOps.DEFINE_LOCAL: pass
      elif uop == UOps.SPECIAL:
        # TODO depending on call convention, get the parms from different places (windows stores everything on the stack)
        # https://en.wikipedia.org/wiki/X86_calling_conventions
        unix_call_conv = {'buf0': '%rdi', 'buf1': '%rsi', 'buf2': '%rdx', 'buf3': '%rcx', 'buf4': '%r8', 'buf5': '%r9'}
        if arg.startswith('buf'):
          ins.append(f"{inst('mov', out)} {unix_call_conv[arg]}, {reg_map[out.nm]}")
          # TODO pop remaining args from stack
      elif uop == UOps.ALU:
        if dtypes.is_float(out.dtype):
          ins.append(f"movsd {reg_map[vin[0].nm]}, %xmm0")
          ins.append(f"{alu[arg]}ss {reg_map[vin[1].nm]}, %xmm0")
          ins.append(f"movsd %xmm0, {reg_map[out.nm]}")
        # TODO non fp add stuff
        else:
          acc_reg_a, acc_reg_b = "%rax" if out.dtype.itemsize == 8 else "%eax", "%rbx" if out.dtype.itemsize == 8 else "%ebx"
          ins.append(f"{inst('mov', vin[0])} {reg_map[vin[0].nm]}, {acc_reg_a}")
          ins.append(f"{inst('mov', vin[0])} {reg_map[vin[1].nm] if not isinstance(vin[1], int) else f'${vin[1]}'}, {acc_reg_b}")
          if arg == BinaryOps.MUL:
            ins.append(f"{inst(alu[arg], vin[0])} {acc_reg_b}")
          else:
            ins.append(f"{inst(alu[arg], vin[0])} {acc_reg_b}, {acc_reg_a}")
            if arg in [BinaryOps.CMPLT, BinaryOps.CMPEQ]: 
              cmp_map = {BinaryOps.CMPLT: "setae", BinaryOps.CMPEQ: "sete"}
              ins.append(f"{cmp_map[arg]} %al")
          ins.append(f"{inst('mov', vin[0])} {acc_reg_a}, {reg_map[out.nm]}")
      elif uop == UOps.LOAD:
        # TODO can we use the shortform for indirect memory access?
        acc_reg_a = "%rax" if out.dtype.itemsize == 8 else "%eax"        
        ins.append(f"movq {reg_map[vin[0].nm]}, %rbx")
        ins.append(f"{inst('mov', out)} {arg[0]}(%rbx), {acc_reg_a}")
        ins.append(f"{inst('mov', out)} {acc_reg_a}, {reg_map[out.nm]}")
      elif uop == UOps.STORE:
        # TODO is there a better way to save in the address in address
        acc_reg_a = "%rax" if vin[1].dtype.itemsize == 8 else "%eax"        
        ins.append(f"{inst('mov', vin[1])} {reg_map[vin[1].nm]}, {acc_reg_a}")
        ins.append(f"movq {reg_map[vin[0].nm]}, %rbx")
        ins.append(f"{inst('mov', vin[1])} {acc_reg_a}, {arg[0]}(%rbx)")
      elif uop == UOps.CAST:     
        # singed extend
        ins.append(f"movslq {reg_map[vin[0].nm]}, %rax")
        ins.append(f"movq %rax, {reg_map[out.nm]}")
      elif uop == UOps.CONST:
        ins.append(f"{inst('mov', out)} $0x{float_to_hex(arg) if dtypes.is_float(out.dtype) else arg}, {reg_map[out.nm]}")
      elif uop == UOps.LABEL:
        ins.append(f"{arg.replace('$', '.')}:")
      elif uop == UOps.COND_BRANCH:
        # TODO arg1 is not?
        acc_reg_a = "%rax" if vin[0].dtype.itemsize == 8 else "%eax"    
        ins.append(f"movb {reg_map[vin[0].nm]}, %al")
        ins.append(f"test %al, %al")
        ins.append(f"{'je' if arg[1] else 'jne'} {arg[0].replace('$', '.')}")

    return "_kernel", '\n'.join([".section .text", ".globl _kernel", "_kernel:", f"enter ${local_var_size}, $0"] + ins + ["leave", "ret", ""])
