import tempfile, subprocess, os
from dataclasses import replace
import numpy as np
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.engine.realize import lower_schedule_item, CompiledRunner, ExecItem
from tinygrad.helpers import diskcache, getenv, TracingKey

# ** constants

M = N = K = 4096
BLOCK_SIZE = 256
WARPS_M = 2
WARPS_N = 4
NUM_WARPS = WARPS_M * WARPS_N
WARP_THREADS = 64
NUM_THREADS = WARP_THREADS * NUM_WARPS

G = (N // BLOCK_SIZE) * (M // BLOCK_SIZE)
L = NUM_THREADS

Tensor.manual_seed(0)

# ** kittens compiler

#@diskcache
def compile_kitten(src:str) -> bytes:
  root = getenv("THUNDERKITTENS_ROOT", "")
  with tempfile.NamedTemporaryFile(suffix=".cpp") as fp, tempfile.NamedTemporaryFile(suffix=".co") as co, \
      tempfile.NamedTemporaryFile(suffix=".o") as o:
    fp.write(src.encode())
    fp.flush()
    subprocess.run(["hipcc", fp.name, "--genco", "--offload-arch=gfx950", "-std=c++20", "-DKITTENS_CDNA4",
                    f"-I{root}/include", "-I/opt/rocm/include/hip", "-o", co.name], check=True)
    subprocess.run(["/opt/rocm/llvm/bin/clang-offload-bundler", "--unbundle", f"--input={co.name}", f"--output={o.name}",
                    "--targets=hipv4-amdgcn-amd-amdhsa--gfx950", "--type=o"], check=True)
    lib = o.read()
  return lib
with open(os.path.dirname(__file__)+"/matmul.cpp", "r") as f: lib = compile_kitten(f.read())

# ** inputs

a = Tensor.randn((N, N), dtype=dtypes.float16, device="CPU").to(Device.DEFAULT)
b = Tensor.randn((N, N), dtype=dtypes.float16, device="CPU").to(Device.DEFAULT)
c = Tensor.zeros((N, N), dtype=dtypes.float32, device="CPU").contiguous().to(Device.DEFAULT)
with Context(DEBUG=0):
  Tensor.realize(a, b, c)

# ** reference gemm

ref = a.matmul(b, dtype=dtypes.float32)
eis = [lower_schedule_item(ref.schedule()[0])]

# ** kittens gemm

prg = CompiledRunner(precompiled=lib, p=replace(eis[0].prg.p, src=lib, name="matmul", global_size=(G, 1, 1), local_size=(L, 1, 1)))
#Device[Device.DEFAULT].compiler.disassemble(lib)
eis.append(ExecItem(prg, [c.uop.buffer, a.uop.buffer, b.uop.buffer]))

for ei in eis: ei.run()
np.testing.assert_allclose(c.numpy(), ref.numpy(), atol=1e-6, rtol=1e-3)
