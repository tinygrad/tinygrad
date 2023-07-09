from typing import Any, Callable, Dict, Final, Tuple
from triton.compiler import compile as triton_compile # type: ignore
import hashlib
import math

from tinygrad.ops import BinaryOps, ASTRunner, Op, ReduceOps, UnaryOps
from tinygrad.codegen.linearizer import Linearizer, LocalBuffer, UOps
from tinygrad.helpers import DEBUG, ImageDType, dtypes
from tinygrad.shape.symbolic import NumNode

class TritonCodegen(Linearizer):
  has_mulacc: bool = False
  triton_style_group: bool = True

  def hand_coded_optimizations(self):
    if len(self.full_shape) == 1: return

    # if nothing at all is upcasted and it's easy to, do an upcast
    for splits in [32]:
      if self.upcasted == 0 and len(self.full_unupcasted_shape) > 0 and self.full_unupcasted_shape[-1] % splits == 0:
        self.shift_to(len(self.full_unupcasted_shape)-1, splits, insert_before=len(self.full_unupcasted_shape)-1)
        self.group_for_reduce.append(splits)
        break
        #self.upcast()

    # only locals
    for axis in range(self.first_reduce - self.local_dims - 1, -1, -1):
      if self.full_shape[axis] == 1: continue
      last_try = self.local_dims == 0 and axis == 0
      if any(self.sts[buf_index].views[-1].strides[axis] == 0 for buf_index in range(len(self.sts))) or last_try:
        for sz in [x for x in [64,32,16,8,4] if self.full_shape[axis] % x == 0]:
          self.shift_to(axis, sz, insert_before=self.first_reduce-self.local_dims)
          self.local_dims += 1
          break
      if self.local_dims >= 3: break

  def codegen(self):
    self.process()
    self.hand_coded_optimizations()
    self.simplify_ones()
    self.limit_global_dims(3)
    self.linearize()

    kernel = []
    global_size = []
    depth = 0
    def kk(s): kernel.append("  "*depth+s)

    kk("@triton.jit")
    kk("def fxn("+','.join(f"data{i}" for i in range(len(self.bufs)))+"):")
    depth += 1

    full_local_shape: Tuple[Any, ...] = ()
    acc_local_shape = 1

    gid = [f"tl.program_id({i})" for i in range(3)]
    code_for_op: Final[Dict[Op, Callable]] = {
      UnaryOps.EXP2: lambda x: f"tl.math.exp2({x})",
      UnaryOps.LOG2: lambda x: f"tl.math.log2({x})", # TODO: is fast_log2f ok?
      UnaryOps.SIN: lambda x: f"tl.sin({x})",
      UnaryOps.SQRT: lambda x: f"tl.sqrt({x})",
      BinaryOps.ADD: lambda x,y: f"({x}+{y})", BinaryOps.SUB: lambda x,y: f"({x}-{y})",
      BinaryOps.MUL: lambda x,y: f"({x}*{y})", BinaryOps.DIV: lambda x,y: f"({x}/{y})",
      BinaryOps.MAX: lambda x,y: f"tl.maximum({x},{y})", # axis?
      BinaryOps.CMPEQ: lambda x,y: f"({x}=={y})",
      ReduceOps.SUM: lambda x: f"tl.expand_dims(tl.sum({x}, axis={len(full_local_shape)-len(self.group_for_reduce)}), axis={len(full_local_shape)-len(self.group_for_reduce)})" if len(self.group_for_reduce) != len(full_local_shape) else f"tl.sum({x}, axis={len(full_local_shape)-len(self.group_for_reduce)})",
    }
    bufnames = ["temp" if isinstance(b, LocalBuffer) else f"data{i}" for i,b in enumerate(self.bufs)]


    for uop,newvar,vin,args in self.uops:
      if uop == UOps.LOOP:
        for i,var in enumerate(args[0]):
          if isinstance(var, NumNode): continue # python doesnt have block scope
          else:
            if args[1] == "global":
              global_size.append(var.max+1)
              kk(f"{var.expr} = {gid[i]} # {var.max+1}")
            elif args[1] == "local":
              full_local_shape = tuple([var.max+1 for var in args[0]])
              assert var.min == 0, "local loop must start at 0"
              kk(f"{var.expr} = tl.arange({0}, {var.max+1})[{', '.join([':' if i == j else 'None' for j in range(len(args[0]))])}]")
              acc_local_shape *= var.max+1
            else:
              kk(f"for {var.expr} in range({var.min}, {var.max+1}):")
              depth += 1
      elif uop == UOps.ENDLOOP:
        if args[1] not in ["global", "local"] and len(args[0]):
          depth -= 1
          kk(f"# end {args[1]}")
      elif uop == UOps.CONST:
        assert newvar is not None
        if args == -math.inf: ld = "-math.inf"
        else: ld = args
        if full_local_shape and len(self.group_for_reduce) != len(full_local_shape): ld = f"tl.full({full_local_shape[:-len(self.group_for_reduce)] + (1,)*len(self.group_for_reduce) if len(self.group_for_reduce) else full_local_shape}, {ld}, tl.float32)"
        kk(f"{newvar.render()} = {ld}")
      elif uop == UOps.ALU:
        assert newvar is not None
        kk(f"{newvar.render()} = {code_for_op[args](*[x.render() for x in vin])}")
      elif uop == UOps.LOAD:
        assert newvar is not None
        val = f"{bufnames[args.i]} + {args.idx.render()}" # defaults to render_python
        triton_dtype = {dtypes.float32: "tl.float32", dtypes.float16: "tl.float16", dtypes.int8: "tl.int8", dtypes.uint8: "tl.uint8", dtypes.int32: "tl.int32", dtypes.int64: "tl.int64"}[newvar.dtype]
        if args.valid.min == 1: kk(f"{newvar.render()} = tl.load({val}).to({triton_dtype})")
        else: kk(f"{newvar.render()} = tl.where({args.valid.render()}, tl.load({val}, mask={args.valid.render()}), 0.0).to({triton_dtype})")
      elif uop == UOps.STORE:
        assert vin[0].dtype == dtypes.float, "unimplemented: float4 store"
        assert not isinstance(self.bufs[args.i].dtype, ImageDType), "unimplemented: image store"
        assert args.valid.min == 1, "store must be valid"
        kk(f"tl.store({bufnames[args.i]} + {args.idx.render()}, {vin[0].render()})")
      elif uop == UOps.CAST: raise NotImplementedError("unimplemented: cast")
      else:
        raise NotImplementedError(f"unimplemented: {uop}")

    prg = '\n'.join(kernel)
    if DEBUG >= 4: print(prg)

    # write out python to compile
    prg = "import triton\nimport triton.language as tl\ntl.core.TRITON_MAX_TENSOR_NUMEL = float('inf')\n" + prg
    fn = f"/tmp/{hashlib.md5(prg.encode('utf-8')).hexdigest()}.py"
    with open(fn, "w") as f: f.write(prg)
    codeobj = compile(prg, fn, "exec")
    exec(codeobj, globals()) # pylint: disable=W0122
    triton_prg = triton_compile(globals()["fxn"], signature=','.join([{dtypes.float32: "*fp32", dtypes.float16: "*fp16", dtypes.int8: "*i8", dtypes.uint8: "*u8", dtypes.int32: "*i32", dtypes.int64: "*i64"}[buf.dtype] for buf in self.bufs]), device_type="cuda", debug=True)
    asm = triton_prg.asm['ptx']
    name = asm.split(".visible .entry ")[1].split("(")[0]
    local_size = [int(x) for x in asm.split(".maxntid ")[1].split("\n")[0].split(", ")]  # [128, 1, 1] is num_warps=4

    return ASTRunner(name, asm,
      global_size, local_size,
      op_estimate=self.info.flops, mem_estimate=self.mem_estimate, display_name=self.display_name, runtime_args={"binary": True, "shared": triton_prg.metadata['shared']})
