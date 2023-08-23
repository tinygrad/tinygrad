from typing import Dict, List, Final, Callable
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, Op
from tinygrad.helpers import dtypes, ImageDType
from tinygrad.codegen.linearizer import  UOp, UOps, ConstOp
from tinygrad.shape.symbolic import NumNode

def next_power_of_2(x):
  return 1 << (x - 1).bit_length()

def uops_to_triton(function_name:str, uops:List[UOp]):
  kernel = []
  global_size: List[int] = []
  local_size: List[int] = []
  depth = 1
  bufs = []
  signatures = []
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
    TernaryOps.MULACC: lambda x,y,z: f"(({x}*{y})+{z})",
    TernaryOps.WHERE: lambda x,y,z: f"tl.where({x},{y},{z})",
  }
  triton_dtypes = {dtypes.double: "tl.float64", dtypes.float32: "tl.float32", dtypes.float16: "tl.float16", dtypes.int8: "tl.int8", dtypes.uint8: "tl.uint8", dtypes.int32: "tl.int32", dtypes.int64: "tl.int64"}
  signature_dtypes = {dtypes.double: "*fp64",dtypes.float32: "*fp32", dtypes.float16: "*fp16", dtypes.int8: "*i8", dtypes.uint8: "*u8", dtypes.int32: "*i32", dtypes.int64: "*i64"}
  inf = 'float("inf")'
  for uop,newvar,vin,args in uops:
    if uop == UOps.LOOP:
      for i,var in enumerate(args[0]):
        if isinstance(var, NumNode): continue # python doesnt have block scope
        else:
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
      if args[1] not in ["global", "local"] and len(args[0]):
        depth -= 1
        kk(f"# end {args[1]}")
    elif uop == UOps.ALU:
      assert newvar is not None
      kk(f"{newvar.render()} = {code_for_op[args](*[x.render() for x in vin])}")
    elif uop == UOps.LOAD:
      assert newvar is not None
      if isinstance(args, ConstOp):
        if len(local_size) > 0:
          kk(f"{newvar.render()} = tl.full(({','.join([str(next_power_of_2(x)) for x in local_size])},), {inf if args.value == float('inf') else args.value}, dtype={triton_dtypes[newvar.dtype]})") 
        else:
          kk(f"{newvar.render()} = ({inf if args.value == float('inf') else args.value})")
      elif args.valid.min == 1: kk(f"{newvar.render()} = tl.load({args.name} + {args.idx.render()}, mask = {args.idx.render()}<{args.idx.max+1}).to({triton_dtypes[args.memory_dtype]})")
      else: kk(f"{newvar.render()} = tl.where({args.valid.render()}, tl.load({args.name}), 0.0).to({triton_dtypes[args.memory_dtype]})")
    elif uop == UOps.STORE:
      assert vin[0].dtype == dtypes.float, "unimplemented: float4 store"
      assert not isinstance(args.memory_dtype, ImageDType), "unimplemented: image store"
      assert args.valid.min == 1, "store must be valid"
      kk(f"tl.store({args.name} + {args.idx.render()}, {vin[0].render()}, mask = {args.idx.render()}<{args.idx.max+1}) ")
    elif uop == UOps.DEFINE_GLOBAL:
      bufs.append(args)
      signatures.append(signature_dtypes[args[1]])
    elif uop == UOps.CAST: raise NotImplementedError("unimplemented: cast")
    else:
      raise NotImplementedError(f"unimplemented: {uop}")  
  
  prg = f"@triton.jit\ndef {function_name}("+','.join(f"{buf[0]}" for buf in bufs)+"):\n"
  prg += '\n'.join(kernel)
  prg += "#SIGNATURE:" + ",".join(signatures)
  acc_local_size = 1
  for x in local_size: acc_local_size *= next_power_of_2(x)
  local_size = [acc_local_size] + [1] * (len(local_size) - 1)  
  return prg, global_size, local_size