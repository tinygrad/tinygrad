from wgpu.utils._device import get_default_device
from tinygrad.runtime.lib import RawBufferMapped, RawConst
from tinygrad.codegen.linearizer import Linearizer, LocalBuffer, UOps
from tinygrad.helpers import DType, dtypes, DEBUG
from tinygrad.ops import Compiled, Op, UnaryOps, BinaryOps, ASTRunner, FusedOps
from tinygrad.shape.symbolic import NumNode
import math
import wgpu
device = get_default_device()


class WebGPUProgram:
  def __init__(self, name: str, prg: str):
    if DEBUG >= 5: print(prg)
    self.name,self.prg = name,device.create_shader_module(code=prg)

  def __call__(self, global_size, local_size, *bufs, wait=False):
    binding_layouts = [{"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}} for i in range(len(bufs))]
    bindings = [{"binding": i, "resource": {"buffer": x._buf, "offset": 0, "size": x._buf.size}} for i, x in enumerate(bufs)]
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)
    compute_pipeline = device.create_compute_pipeline(layout=pipeline_layout,compute={"module": self.prg, "entry_point": self.name},)
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    # last 2 elements not used
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
    compute_pass.dispatch_workgroups(*global_size)  # x y z
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])


type_map = {dtypes.float: "f32", dtypes.half: "f16", dtypes.int32: "i32", dtypes.uint32: "u32", dtypes.bool: "bool"}
code_for_op = {
  UnaryOps.EXP2: lambda x: f"exp2({x})", UnaryOps.LOG2: lambda x: f"log2({x})", UnaryOps.SIN: lambda x: f"sin({x})",
  BinaryOps.ADD: lambda x,y: f"({x}+{y})", BinaryOps.SUB: lambda x,y: f"({x}-{y})", BinaryOps.MUL: lambda x,y: f"({x}*{y})", BinaryOps.DIV: lambda x,y: f"({x}/{y})",
  BinaryOps.POW: lambda x,y: f"pow({x}, {y})", BinaryOps.MAX: lambda x,y: f"max({x}, {y})", BinaryOps.CMPEQ: lambda x,y: f"f32({x}=={y})",
  FusedOps.MULACC: lambda x,y,z: f"({x}+{y}*{z})",
}

class WebGpuCodegen(Linearizer):
  supports_float4 = False

  def codegen(self):
    self.process()
    self.hand_coded_optimizations()
    self.limit_global_dims(3)
    self.linearize()

    kernel = []
    global_size = []
    local_size = []
    depth = 0
    def kk(s): kernel.append(" "*depth+s)
    bufnames = ["temp" if isinstance(b, LocalBuffer) else f"data{i}" for i,b in enumerate(self.bufs)]
    depth += 1
    gid = [f"gindex.{'xyz'[x]}" for x in range(3)]
    for uop,newvar,vin,args in self.uops:
      if uop == UOps.LOOP:
        for i,var in enumerate(args[0]):
          if isinstance(var, NumNode): 
            print("NUMNODE", var)
            continue
          else:
            if args[1] == "global":
              global_size.append(var.max+1)
              kk(f"{{let {var.expr} = i32({gid[len(args[0])-1-i]}); // {var.max+1}")
            elif args[1] == "local":
              kk(f"for(var {var.expr}: i32 = 0; {var.expr} < {var.max+1}; {var.expr}++) {{")
              depth += 1
              local_size.append(var.max+1)
            else:
              kk(f"for(var {var.expr}: i32 = 0; {var.expr} < {var.max+1}; {var.expr}++) {{")
              depth += 1
      elif uop == UOps.ENDLOOP:
        if args[1] == "local":
          kk("workgroupBarrier();")
          depth -= 1
        print("ENDING LOOP")
        kk("}"*len(args[0])  + f" /* {args[1]} */")
      elif uop == UOps.LOAD and newvar is not None:
        if self.bufs[args.i] is not None and isinstance(self.bufs[args.i].realized, RawConst):
          assert newvar.dtype == dtypes.float, "only floats"
          print("LOADING")
        else:
          val = f"{bufnames[args.i]}[{args.idx.render()}]"
          print(val)
        if args.valid.min == 1: kk(f"let {newvar.render()}: {type_map[newvar.dtype]} = {val};")
        else: kk(f"var {newvar.render()} = select(0.0f, {val}, {args.valid.render()});")
      elif uop == UOps.ALU:
        assert newvar is not None
        if newvar in vin:
          kk(f"{newvar.render()} = {code_for_op[args](*[x.render() for x in vin])};")
        else:
          kk(f"let {newvar.render()} = {code_for_op[args](*[x.render() for x in vin])};")
      elif uop == UOps.STORE:
        kk(f"{bufnames[args.i]}[{args.idx.render()}] = {vin[0].render()};")
      elif uop == UOps.CONST:
        assert newvar is not None
        kk(f"var {newvar.render()}: {type_map[newvar.dtype]} = {args};")
      else:
        raise RuntimeError(f"failed to render {uop}")

    prg = "\n".join([f"@group(0) @binding({i}) var<storage,read_write> data{i}: array<{type_map[x.dtype]}>;" for i,x in enumerate(self.bufs)])
    # TODO: revert local_size {','.join([str(x) for x in local_size])} once bug is fixed
    prg += f"\n@compute @workgroup_size(1) fn {self.function_name}(@builtin(global_invocation_id) gindex: vec3<u32>, @builtin(local_invocation_id) lindex: vec3<u32>) {{\n" + "\n".join(kernel) + "\n}"
    return ASTRunner(self.function_name, prg, global_size[::-1] if len(global_size) else [1], local_size[::-1] if len(local_size) else [1])


class RawWebGPUBuffer(RawBufferMapped):
    def __init__(self, size, dtype: DType): super().__init__(size, dtype, device.create_buffer(
        size=size*dtype.itemsize, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC))

    def _copyin(self, x) -> None: device.queue.write_buffer(self._buf, 0, x)

    def _buffer(
        self) -> memoryview: return device.queue.read_buffer(self._buf, 0)


WebGpuBuffer = Compiled(RawWebGPUBuffer, WebGpuCodegen, WebGPUProgram)
