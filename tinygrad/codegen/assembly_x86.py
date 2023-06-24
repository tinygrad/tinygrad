from tinygrad.codegen.assembly import AssemblyCodegen, AssemblyInstruction, float_to_hex
from tinygrad.ops import BinaryOps, UnaryOps
from tinygrad.codegen.linearizer import UOps
from tinygrad.helpers import DEBUG, dtypes

class X86Codegen(AssemblyCodegen):
  #supports_constant_folding: bool = True

  def specialize(self, asm):
    type_to_op_suffix = {dtypes.float32: 'l', dtypes.bool: 'b', dtypes.int32: 'l', dtypes.int64: 'q', dtypes.uint32: 'l', dtypes.uint64: 'q'}
    def inst(instr, reg, fp=False, simd=False): return f"{instr}{'s' if fp else ''}{'p' if simd else ''}{type_to_op_suffix[reg.dtype]}"
    def reg(name, reg): return {1: f"%{name}l", 4: f"%e{name}x", 8: f"%r{name}x"}[reg.dtype.itemsize]

    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div",
           BinaryOps.CMPLT: "cmp",  BinaryOps.CMPEQ: "cmp", BinaryOps.MAX: "max",
         UnaryOps.NOOP: "mov", UnaryOps.SIN: "call sinf@PLT", UnaryOps.LOG2: "call log2f@PLT", UnaryOps.EXP2: "call exp2f@PLT"}
    alu_fp = {**alu, BinaryOps.CMPLT: "cmplt",  BinaryOps.CMPEQ: "cmpeq"}

    ins, reg_map, local_var_size = [], {}, 0
    for uop, out, vin, arg in asm:
      if DEBUG >= 5: ins.append(f"# {AssemblyInstruction(uop, out, vin, arg)}")
      if uop == UOps.DEFINE_REGISTER:
        for i in range(arg[2]):
          local_var_size += arg[0].itemsize
          # TODO use float registers
          reg_map[f"%{arg[1]}{i}"] = f"-{local_var_size}(%rbp)"
      elif uop == UOps.DEFINE_LOCAL:
        local_var_size += arg * 4
        ins.append(f"movq %rbp, %rax")
        ins.append(f"subq ${local_var_size}, %rax")
        ins.append(f"movq %rax, {reg_map[out.nm]}")
      elif uop == UOps.SPECIAL:
        # TODO depending on call convention, get the parms from different places (windows stores everything on the stack)
        # https://en.wikipedia.org/wiki/X86_calling_conventions
        unix_call_conv = ['%rdi', '%rsi', '%rdx', '%rcx', '%r8', '%r9']
        if arg.startswith('buf'):
          buf_id = int(arg.replace('buf', ''))
          if buf_id < len(unix_call_conv):
            ins.append(f"movq {unix_call_conv[buf_id]}, {reg_map[out.nm]}")
          else:
            raise Exception()
          # TODO pop remaining args from stack
      elif uop == UOps.ALU:
        if dtypes.is_float(out.dtype) or dtypes.is_float(vin[0].dtype):      
          ins.append(f"movd {reg_map[vin[0].nm]}, %xmm0")
          if arg not in [UnaryOps.SIN, UnaryOps.LOG2, UnaryOps.EXP2]: 
            ins.append(f"{alu_fp[arg]}ss {reg_map[vin[1].nm]}, %xmm0")
          else:
            ins.append(f"{alu_fp[arg]}")
          if arg in [BinaryOps.CMPLT, BinaryOps.CMPEQ]:
              cmp_map = {BinaryOps.CMPLT: "setae", BinaryOps.CMPEQ: "sete"}
              ins.append(f"{cmp_map[arg]} %al")
          ins.append(f"movd %xmm0, {reg_map[out.nm]}")
          # TODO non fp add stuff
        else:
          reg_a, reg_b = reg('a', vin[0]), reg('b', vin[0])
          ins.append(f"{inst('mov', vin[0])} {reg_map[vin[0].nm]}, {reg_a}")
          ins.append(f"{inst('mov', vin[0])} {reg_map[vin[1].nm] if not isinstance(vin[1], int) else f'${vin[1]}'}, {reg_b}")
          if arg == BinaryOps.MUL:
            ins.append(f"{inst(alu[arg], vin[0])} {reg_b}")
          elif arg == BinaryOps.MOD:
            ins.append(f"cltd")
            ins.append(f"{inst('idiv', vin[0])} {reg_b}")
            ins.append(f"{inst('mov', vin[0])} {reg('d', out)}, {reg_a}")
          else:
            ins.append(f"{inst(alu[arg], vin[0])} {reg_b}, {reg_a}")
            if arg in [BinaryOps.CMPLT, BinaryOps.CMPEQ]: 
              cmp_map = {BinaryOps.CMPLT: "setae", BinaryOps.CMPEQ: "sete"}
              ins.append(f"{cmp_map[arg]} %al")
          ins.append(f"{inst('mov', vin[0])} {reg_a}, {reg_map[out.nm]}")
      elif uop == UOps.LOAD:
        # TODO can we use the shortform for indirect memory access?
        ins.append(f"movq {reg_map[vin[0].nm]}, %rbx")
        ins.append(f"{inst('mov', out)} {arg[0]}(%rbx), {reg('a', out)}")
        ins.append(f"{inst('mov', out)} {reg('a', out)}, {reg_map[out.nm]}")
      elif uop == UOps.STORE:
        # TODO is there a better way to save in the address in address
        ins.append(f"{inst('mov', vin[1])} {reg_map[vin[1].nm]}, {reg('a', vin[1])}")
        ins.append(f"movq {reg_map[vin[0].nm]}, %rbx")
        ins.append(f"{inst('mov', vin[1])} {reg('a', vin[1])}, {arg[0]}(%rbx)")
      elif uop == UOps.CAST:
        if out.dtype == dtypes.float32:
          ins.append(f"{inst('mov', vin[0])} {reg_map[vin[0].nm]}, {reg('b', vin[0])}")
          ins.append(f"movzbl	%bl, %eax")
          ins.append(f"andl	$1, %eax") # TODO why is this mask needed?
          ins.append(f"cvtsi2ssl %eax, %xmm0")
          ins.append(f"movd %xmm0, {reg_map[out.nm]}")
        else:
          ins.append(f"movslq {reg_map[vin[0].nm]}, %rax")
          ins.append(f"movq %rax, {reg_map[out.nm]}")
      elif uop == UOps.CONST:
        ins.append(f"{inst('mov', out)} $0x{float_to_hex(arg) if dtypes.is_float(out.dtype) else arg}, {reg_map[out.nm]}")
      elif uop == UOps.LABEL:
        ins.append(f"{arg.replace('$', '.')}:")
      elif uop == UOps.COND_BRANCH:
        # TODO arg1 is not?
        ins.append(f"movb {reg_map[vin[0].nm]}, %al")
        ins.append(f"test %al, %al")
        ins.append(f"{'je' if arg[1] else 'jne'} {arg[0].replace('$', '.')}")

    print(reg_map)
    print(local_var_size)

    return "_kernel", '\n'.join([".section .text", ".globl _kernel", "_kernel:", "pushq	%rbp", "movq	%rsp, %rbp", f"subq	${local_var_size}, %rsp"] + ins + ["leave", "ret", ""])
