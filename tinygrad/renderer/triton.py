from typing import Dict, List, Final, Callable
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, Op
from tinygrad.helpers import dtypes, ImageDType, DEBUG, getenv
from tinygrad.codegen.linearizer import  UOp, UOps, ConstOp
from tinygrad.shape.symbolic import NumNode
from triton.compiler import compile as triton_compile
import hashlib
import math
def next_power_of_2(x):
  return 1 << (x - 1).bit_length()

def render_valid(node):
  return '(' * (len(node.nodes) -1) + ') and '.join([f'{n.render()}<{n.max+1}' for n in node.nodes]) if hasattr(node, "nodes") else f"{node.render()}<{node.max+1}"

#NOTE triton requires matching dimensions for load/store, disable this and see TestOps::test_output_padded_conv_transpose2d fail to compile
def fill_dims_for_idx(idx, dims):
  return "(" + idx.render() + "+ (" + (f"0 * ({'+'.join(d for d in dims)})))")

def uops_to_triton(function_name:str, uops:List[UOp]):
  kernel = []
  global_size: List[int] = []
  local_size: List[int] = []
  depth = 1
  bufs = []
  signatures = []
  dims = []
  def kk(s): kernel.append("  "*depth+s)  
  gid = [f"tl.program_id({i})" for i in range(3)]
  code_for_op: Final[Dict[Op, Callable]] = {
    UnaryOps.EXP2: lambda x: f"tl.math.exp2({x})",
    UnaryOps.LOG2: lambda x: f"tl.math.log2({x})", # TODO: is fast_log2f ok?
    UnaryOps.SIN: lambda x: f"tl.sin({x})",
    UnaryOps.SQRT: lambda x: f"tl.sqrt({x})",
    BinaryOps.ADD: lambda x,y: f"({x}+{y})", BinaryOps.SUB: lambda x,y: f"({x}-{y})",
    BinaryOps.MUL: lambda x,y: f"({x}*{y})", BinaryOps.DIV: lambda x,y: f"({x}/{y})",
    BinaryOps.MAX: lambda x,y: f"tl.maximum({x},{y})", # axis?
    BinaryOps.CMPLT: lambda x,y: f"({x}<{y})",
    BinaryOps.MOD: lambda x,y: f"({x}%{y})",
    TernaryOps.MULACC: lambda x,y,z: f"(({x}*{y})+{z})",
    TernaryOps.WHERE: lambda x,y,z: f"tl.where({x},{y},{z})",
  }
  triton_dtypes = {dtypes.double: "tl.float64", dtypes.float32: "tl.float32", dtypes.float16: "tl.float16", dtypes.bool: "tl.int8", dtypes.int8: "tl.int8", dtypes.uint8: "tl.uint8", dtypes.int32: "tl.int32", dtypes.int64: "tl.int64"}
  signature_dtypes = {dtypes.double: "*fp64",dtypes.float32: "*fp32", dtypes.float16: "*fp16", dtypes.bool: "*i8", dtypes.int8: "*i8", dtypes.uint8: "*u8", dtypes._arg_int32: "*i32", dtypes.int32: "*i32", dtypes.int64: "*i64"}
  for uop,newvar,vin,args in uops:
    if uop == UOps.LOOP:
      for i,var in enumerate(args[0]):
        if isinstance(var, NumNode): continue # python doesnt have block scope
        else:
          dims.append(var.expr)
          if args[1] == "global":
            global_size.append(var.max+1)
            kk(f"{var.expr} = {gid[i]} # {var.max+1}")
          elif args[1] == "local":
            assert var.min == 0, "local loop must start at 0"
            kk(f"{var.expr} = tl.arange({0}, {next_power_of_2(var.max+1)})[{', '.join([':' if i == j else 'None' for j in range(len(args[0]))])}]")
            local_size.append(var.max+1)
          else:
            kk(f"for {var.expr} in range({var.min}, {var.max+1}):")
            depth += 1
    elif uop == UOps.ENDLOOP:
      if args[1] not in ["global", "local", "global+local"] and len(args[0]):
        depth -= len(args[0])
        kk(f"# end {args[1]}")
    elif uop == UOps.ALU:
      assert newvar is not None
      kk(f"{newvar.render()} = {code_for_op[args](*[x.render() for x in vin])}")
    elif uop == UOps.LOAD:
      assert newvar is not None
      if isinstance(args, ConstOp):
        val = ('-' if math.isinf(args.value) and args.value<0 else'') + ('float("inf")' if math.isinf(args.value) else str(args.value))
        if math.isnan(args.value): val = "float('nan')"
        if len(local_size) > 0:
          kk(f"{newvar.render()} = tl.where({args.valid.render()},tl.full(({','.join([str(next_power_of_2(x)) for x in local_size])},),{val}, dtype={triton_dtypes[newvar.dtype]}),tl.full(({','.join([str(next_power_of_2(x)) for x in local_size])},),{args.invalid_value}, dtype={triton_dtypes[newvar.dtype]}))") 
        else:
          kk(f"{newvar.render()} = tl.where({args.valid.render()},{val},{args.invalid_value})")
      elif args.valid.min == 1: kk(f"{newvar.render()} = tl.load({args.name} + {args.idx.render()}, mask = {render_valid(args.idx)}).to({triton_dtypes[args.memory_dtype]})")
      else: kk(f"{newvar.render()} = tl.where({args.valid.render()}, tl.load({args.name}+{fill_dims_for_idx(args.idx,dims)} , mask={args.valid.render()}), 0.0).to({triton_dtypes[args.memory_dtype]})")
    elif uop == UOps.STORE:
      assert vin[0].dtype == dtypes.float, "unimplemented: float4 store"
      assert not isinstance(args.memory_dtype, ImageDType), "unimplemented: image store"
      assert args.valid.min == 1, "store must be valid"
      kk(f"tl.store({args.name} + {args.idx.render()}, {vin[0].render()}, mask = {render_valid(args.idx)}) ")
    elif uop == UOps.DEFINE_GLOBAL:
      bufs.append(args)
      signatures.append(signature_dtypes[args[1]])
    elif uop == UOps.CAST: raise NotImplementedError("unimplemented: cast")
    else:
      raise NotImplementedError(f"unimplemented: {uop}")  
  
  prg = f"@triton.jit\ndef {function_name}("+','.join(f"{buf[0]}" for buf in bufs)+"):\n"
  prg += '\n'.join(kernel)
  acc_local_size = 1
  for x in local_size: acc_local_size *= next_power_of_2(x)
  local_size = [acc_local_size] + [1] * (len(local_size) - 1)  

  prg = "import triton\nimport triton.language as tl\ntl.core.TRITON_MAX_TENSOR_NUMEL = float('inf')\n" + prg
  if DEBUG >=4: print(prg)
  hsh = hashlib.md5(prg.encode('utf-8')).hexdigest()
  fn = f"/tmp/{hsh}.py"
  with open(fn, "w") as f: f.write(prg)
  codeObject = compile(prg, fn, "exec")
  exec(codeObject, globals()) # pylint: disable=W0122
  prg = triton_compile(globals()[function_name], signature=",".join(signatures), device_type="cuda", debug=True, cc=(35 if getenv("CUDACPU", 0) else None)).asm["ptx"]
  return prg, global_size, [int(x) for x in prg.split(".maxntid ")[1].split("\n")[0].split(", ")], True