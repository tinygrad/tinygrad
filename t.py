from dataclasses import replace
from tinygrad import Tensor, Device
from tinygrad.helpers import diskcache, system, temp
from tinygrad.engine.realize import lower_schedule_item, run_schedule, CompiledRunner
from tinygrad.runtime.support.compiler_amd import compile_hip

Device.DEFAULT = "AMD"
dev = Device[Device.DEFAULT]

system(f"clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1200 -mcode-object-version=5 -c test.s -o {temp('test.o')}")
system(f"ld.lld -shared -o {temp('test.hsaco')} {temp('test.o')}")
with open(temp('test.hsaco'), 'rb') as f: lib = f.read()

a = Tensor([2.])+1
sched = a.schedule()
run_schedule(sched[:-1])

ei = lower_schedule_item(sched[-1])
dev.compiler.disassemble(lib)
ei2 = replace(ei, prg=CompiledRunner(ei.prg.p, lib))
ei2.run()

print(a.uop.buffer.numpy())
