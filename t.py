from tinygrad.helpers import system, temp, colored, getenv
if getenv("MOCKGPU"): system("cargo build --release --manifest-path ./extra/remu/Cargo.toml")

from dataclasses import replace
from tinygrad import Tensor, Device, Context
from tinygrad.engine.realize import lower_schedule_item, run_schedule, CompiledRunner, ExecItem, get_program
from tinygrad.runtime.support.compiler_amd import compile_hip

Device.DEFAULT = "AMD"
dev = Device[Device.DEFAULT]

lib = None
if getenv("ASM", 1):
  system(f"clang -x assembler -target amdgcn-amd-amdhsa -mcpu={dev.arch} -mcode-object-version=5 -c test_{dev.arch}.s -o {temp('test.o')}")
  system(f"ld.lld -shared -o {temp('test.hsaco')} {temp('test.o')}")
  with open(temp('test.hsaco'), 'rb') as f: lib = f.read()

x = Tensor.randn((1,1,16,16,16), device="CPU").tolist()
a = Tensor(x).avg_pool2d(kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False)

sched = a.schedule()
it = [(si, lower_schedule_item(si)) for si in sched[:-1]]

si = sched[-1]
if lib is not None:
  ei = ExecItem(CompiledRunner(get_program(si.ast), lib), si.bufs)
else:
  ei = lower_schedule_item(si)
  #dev.compiler.disassemble(ei.prg.lib)
it.append((si, ei))

with Context(VALIDATE_WITH_CPU=1):
  run_schedule(sched, it=it)
  print(a.uop.buffer.numpy())
print(colored("** asm kernel passed", "green"))
