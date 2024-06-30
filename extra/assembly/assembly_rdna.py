from typing import List
import struct
from tinygrad.codegen.assembly import uops_to_asmstyle, AssemblyLanguage
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad import dtypes
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps
from tinygrad.runtime.ops_gpu import ROCM_LLVM_PATH

dtype_to_rdnatype = {dtypes.float32: "f32", dtypes.float16: "f16", dtypes.int64: "s64", dtypes.int32: "s32", dtypes.int8: "s8", dtypes.bool: "b32", dtypes.uint64: "u64", dtypes.uint32: "u32", dtypes.uint16: "u16", dtypes.uint8: "u8", "bits16": "b16", dtypes.float64: "f64"}
def float_to_hex(x): return "%02X%02X%02X%02X" % tuple(struct.pack("f",x)[::-1])

def rdna_needs_cast(dest_dtype, src_dtype):
    return dtypes.is_float(dest_dtype) and dtypes.is_int(src_dtype) or \
           dtypes.is_int(dest_dtype) and dtypes.is_float(src_dtype) or \
           (dtypes.is_float(src_dtype) and dtypes.is_float(dest_dtype) and dest_dtype.itemsize != src_dtype.itemsize)

def render_cast(ins, inp, out):
    if inp.dtype == dtypes.bool and (dtypes.is_float(out.dtype) or dtypes.is_int(out.dtype)):
        ins.append(f"v_cndmask_b32 {out}, 0, {'1.0' if dtypes.is_float(out.dtype) else '1'}, {inp}")
    elif out.dtype == dtypes.bool:
        if inp.dtype == dtypes.bool:
            ins.append(f"v_mov_b32 {out}, {inp}")
        else:
            ins.append(f"v_cmp_ne_u32 vcc, {'0.0' if dtypes.is_float(inp.dtype) else '0'}, {inp}")
            ins.append(f"v_cndmask_b32 {out}, 0, 1, vcc")
    else:
        round_mod = "_rtz" if dtypes.is_int(out.dtype) and dtypes.is_float(inp.dtype) else '' if dtypes.is_float(out.dtype) and (dtypes.is_int(inp.dtype) or dtypes.is_float(inp.dtype) and inp.dtype.itemsize > out.dtype.itemsize) else ''
        ins.append(f"v_cvt{round_mod}_{dtype_to_rdnatype[out.dtype]}_{dtype_to_rdnatype[inp.dtype]} {out}, {inp}")

class RDNALanguage(AssemblyLanguage):
    supports_constant_folding: bool = True

def specialize_to_rdna(lang, function_name):
    param_cnt = 0
    ins = []
    alu = {BinaryOps.ADD: "add", BinaryOps.SUB: "sub", BinaryOps.MUL: "mul", BinaryOps.DIV: "div", BinaryOps.MAX: "max",
           BinaryOps.MOD: "mod", BinaryOps.CMPLT: "cmp_lt", UnaryOps.SQRT: "sqrt",
           UnaryOps.NOOP: "mov", UnaryOps.NEG: "neg",
           UnaryOps.SIN: "sin", UnaryOps.LOG2: "log", UnaryOps.EXP2: "exp",
           TernaryOps.MULACC: "fma", TernaryOps.WHERE: "cndmask"}
    for uop, out, vin, arg in lang.ins:
        if uop == UOps.ENDLOOP:
            ins.append("s_endpgm")
        elif uop == UOps.DEFINE_LOCAL:
            ins.append(f".shared .align 4 .b8 {arg[0]}[{arg[1]*4}];")
        elif uop == UOps.SPECIAL:
            if arg.startswith('data'):
                param_cnt += 1
                ins.append(f"s_load_dwordx2 {out}, s[0:1], {param_cnt*8}")
            elif arg.startswith('gid'):
                ins.append(f"v_mov_b32 {out}, s[{2+int(arg[3:])}]")
            elif arg.startswith('lid'):
                ins.append(f"v_mov_b32 {out}, v[{int(arg[3:])}]")
        elif uop == UOps.ALU:
            if arg == BinaryOps.MUL and out.dtype == dtypes.bool:
                ins.append(f"v_and_b32 {out}, {', '.join(str(x) for x in vin)}")
            else:
                otype = vin[0].dtype if arg in [BinaryOps.CMPLT] else out.dtype
                if arg == TernaryOps.WHERE:
                    if vin[0].dtype == dtypes.bool:
                        reg = vin[0]
                    else:
                        reg = lang.newreg((vin[0], 'bool'), dtypes.bool)
                        ins.append(f"v_cmp_ne_u32 vcc, {'0.0' if dtypes.is_float(vin[0].dtype) else '0'}, {vin[0]}")
                        ins.append(f"v_cndmask_b32 {reg}, 0, 1, vcc")
                    vin = vin[1:] + [reg]
                ins.append(f"v_{alu[arg]}_{dtype_to_rdnatype[otype]} {out}, {', '.join(str(x) for x in vin)}")
        elif uop == UOps.LOAD:
            if arg.__class__ in (int, float):
                ins.append(f"v_mov_b32 {out}, {'0x'+float_to_hex(arg) if dtypes.is_float(out.dtype) else int(arg)}")
            elif arg[2] is not None and (arg[2] == dtypes.bool or arg[2] != out.dtype):
                dt = ('u16', dtypes.uint16) if arg[2] == dtypes.bool == out.dtype else ('u8', dtypes.uint8) if arg[2] == dtypes.bool else ('b16', dtypes.float16) if arg[2] == dtypes.half else (dtype_to_rdnatype[arg[2]], arg[2])
                reg = lang.newreg((out, dt[0]), dtype=dt[1])
                ins.append(f"global_load_{dt[0]} {reg}, {vin[0]}, {arg[0] if arg[0] is not None else 'off'}")
                render_cast(ins, reg, out)
            else:
                ins.append(f"global_load_{dtype_to_rdnatype[dtypes.float if arg[2] is None else arg[2]]} {out}, {vin[0]}, {arg[0] if arg[0] is not None else 'off'}")
        elif uop == UOps.STORE:
            if rdna_needs_cast(dtypes.float if arg[2] is None else arg[2], vin[1].dtype) or arg[2] == dtypes.bool:
                if arg[2] == dtypes.bool != vin[1].dtype:
                    prereg = lang.newreg((vin[1],'bool'), dtype=dtypes.bool)
                    render_cast(ins, vin[1], prereg)
                else:
                    prereg = vin[1]
                reg = lang.newreg((prereg, dtypes.uint16 if arg[2] == dtypes.bool else arg[2]), dtype=dtypes.uint16 if arg[2] == dtypes.bool else dtypes.float if arg[2] is None else arg[2])
                render_cast(ins, prereg, reg)
                ins.append(f"global_store_{dtype_to_rdnatype['bits16' if arg[2] == dtypes.float16 else dtypes.uint8 if arg[2] == dtypes.bool else dtypes.float if arg[2] is None else arg[2]]} {vin[0]}, {reg}, {arg[0] if arg[0] is not None else 'off'}")
            else:
                ins.append(f"global_store_{dtype_to_rdnatype[dtypes.float if arg[2] is None else arg[2]]} {vin[0]}, {vin[1]}, {arg[0] if arg[0] is not None else 'off'}")
        elif uop == UOps.CAST:
            render_cast(ins, vin[0], out)
        elif uop == UOps.LABEL:
            ins.append(f"{arg}:")
        elif uop == UOps.COND_BRANCH:
            ins.append(f"s_cbranch_vccnz {arg[0]}" if arg[1] else f"s_cbranch_vccz {arg[0]}")

    ins_prefix = [".amdgcn_target \"amdgcn-amd-amdhsa--gfx1100\"",
                  ".text",
                  f".globl {function_name}",
                  f".p2align 8",
                  f".type {function_name},@function",
                  f"{function_name}:"]
    for arg in [(dtype, lang.type_to_letter(dtype), c) for dtype,c in lang.cnts.items()]:
        ins_prefix.append(f".reg .{dtype_to_rdnatype[arg[0][0]]} %{arg[1]}<{arg[2]}>;")
    ins = ins_prefix + ins
    ins += ["s_endpgm"]
    return '\n'.join(ins)

def uops_to_rdna_asm(function_name:str, uops:List[UOp]):
    lang = RDNALanguage()
    global_size, local_size = uops_to_asmstyle(lang, function_name, uops)
    return specialize_to_rdna(lang, function_name), global_size[::-1], local_size[::-1], True