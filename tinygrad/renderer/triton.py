from typing import Dict, List, Final, Callable, DefaultDict
from collections import defaultdict
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, Op
from tinygrad.helpers import dtypes, ImageDType, DEBUG, getenv
from tinygrad.codegen.linearizer import  UOp, UOps
from tinygrad.shape.symbolic import NumNode
from triton.compiler import compile as triton_compile
import hashlib
import math
import re
def next_power_of_2(x):
  return 1 << (x - 1).bit_length()

def render_valid(node):
  return '(' * (len(node.nodes) -1) + ') and '.join([f'{n.render()}<{get_max(n+1)}' for n in node.nodes]) if hasattr(node, "nodes") else f"{node.render()}<{get_max(node+1)}"

#NOTE triton requires matching dimensions for load/store, disable this and see TestOps::test_output_padded_conv_transpose2d fail to compile
def fill_dims_for_idx(idx, dims):
  return "(" + idx.render() + "+ (" + (f"0 * ({'+'.join(d for d in dims)})))")

def get_max(var):
  if isinstance(var.max, int): return var.max
  return re.sub(r'\[(.*?)\]', '', str(var.max))[1:-1]

#NOTE can be removed after https://github.com/gpuocelot/gpuocelot/issues/8 gets resolved
def remove_single_scalar_curly_braces(ptx_code):
  return '\n'.join([re.sub(r'\{\s*(%\w+)\s*\}', r'\1', line) for line in ptx_code.split('\n')])

def uops_to_triton(function_name:str, uops:List[UOp]):
  kernel = []
  global_size: List[int] = []
  local_size: List[int] = []
  depth = 1
  bufs = []
  signatures = []
  dims = []
  c: DefaultDict[str, int] = defaultdict(int)
  def ssa(prefix="t"):
    nonlocal c
    c[prefix] += 1
    return f"{prefix}{c[prefix]-1}"
  r: Dict[UOp, str] = {}
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
  signature_dtypes = {dtypes.double: "*fp64",dtypes.float32: "*fp32", dtypes.float16: "*fp16", dtypes.bool: "*i8", dtypes.int8: "*i8", dtypes.uint8: "*u8", dtypes._arg_int32: "i32", dtypes.int32: "*i32", dtypes.int64: "*i64"}

  for u in uops:
    uop,newvar,vin,args,_ = u
    if uop == UOps.LOOP:
      for i,var in enumerate(args[0]):
        if isinstance(var, NumNode): continue # python doesnt have block scope
        else:
          dims.append(var.expr)
          if args[1] == "global":
            global_size.append(var.max+1)
            kk(f"{var.expr} = {gid[i]} # {get_max(var+1)}")
          elif args[1] == "local":
            assert var.min == 0, "local loop must start at 0"
            kk(f"{var.expr} = tl.arange({0}, {next_power_of_2(var.max+1)})[{', '.join([':' if i == j else 'None' for j in range(len(args[0]))])}]")
            local_size.append(var.max+1)
          else:
            kk(f"for {var.expr} in range({var.min}, {get_max(var+1)}):")
            depth += 1
    elif uop == UOps.ENDLOOP:
      if args[1] not in ["global", "local", "global+local"] and len(args[0]):
        depth -= len(args[0])
        kk(f"# end {args[1]}")
    elif uop == UOps.ALU:
      assert newvar is not None
      r[u] = ssa("alu")
      kk(f"{r[u]} = ({code_for_op[args](*[r[x] for x in vin])}).to({triton_dtypes[newvar]})")
    elif uop == UOps.LOAD:
      assert newvar is not None
      r[u] = ssa("val")
      if args.valid.min == 1: kk(f"{r[u]} = tl.load({args.name} + {args.idx.render()}, mask = {render_valid(args.idx)}).to({triton_dtypes[args.memory_dtype]})")
      else: kk(f"{r[u]} = tl.where({args.valid.render()}, tl.load({args.name}+{fill_dims_for_idx(args.idx,dims)} , mask={args.valid.render()}), 0.0).to({triton_dtypes[args.memory_dtype]})")
    elif uop == UOps.CONST or uop == UOps.DEFINE_ACC:
      r[u] = ssa("const" if uop == UOps.CONST else "acc")
      val = (('-' if args<0 else '') + 'float("inf")') if math.isinf(args) else ('float("nan")' if math.isnan(args) else str(args))
      if len(local_size) > 0:
        kk(f"{r[u]} = tl.full(({','.join([str(next_power_of_2(x)) for x in local_size])},),{val}, dtype={triton_dtypes[newvar]})") 
      else:
        kk(f"{r[u]} = tl.where(1, {val}, {val}).to({triton_dtypes[newvar]})")
    elif uop == UOps.STORE:
      assert not isinstance(newvar, ImageDType), "unimplemented: image store"
      if args is None:
        kk(f"{r[vin[0]]} =  {r[vin[1]]}")
      else:
        kk(f"tl.store({args.name} + {args.idx.render()}, {r[vin[0]]}, mask = {render_valid(args.idx)}) ")
    elif uop == UOps.DEFINE_GLOBAL:
      bufs.append(args)
      signatures.append(signature_dtypes[args[1]])
    elif uop == UOps.SPECIAL:
      r[u] = args.expr
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
  prg = triton_compile(globals()[function_name], signature=",".join(signatures), device_type="cuda", debug=False, cc=(35 if getenv("CUDACPU", 0) else None)).asm["ptx"]
  if getenv("CUDACPU"):
    prg = remove_single_scalar_curly_braces(prg.split(".file")[0].split(".visible .func")[0])
  max_local_size =  [int(x) for x in prg.split(".maxntid ")[1].split("\n")[0].split(", ")]
  for i in range(len(local_size)): local_size[i] = min(local_size[i], max_local_size[i])
  return prg, global_size, local_size, True
