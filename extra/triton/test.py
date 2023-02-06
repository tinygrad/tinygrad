import torch
import triton
import triton.language as tl
from triton.compiler import compile
from triton.runtime import JITFunction

def program(b0, b1, b2):
  idx = tl.program_id(0)
  x = tl.load(b1 + idx)
  y = tl.load(b2 + idx)
  tl.store(b0 + idx, x+y)

program_jit = JITFunction(program)

# JITFunction(__main__:program) {'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'constants': {}, 'num_warps': 4, 'num_stages': 3, 'extern_libs': None, 'configs': (instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=()),)}
# ast -> ttir -> ttgir -> llir -> ptx -> cubin
compiled = compile(program_jit, signature={0: '*fp32', 1: '*fp32', 2: '*fp32'})
print(compiled.asm['ast'])
print(compiled.asm['ttir'])
#print(compiled.asm['ttgir'])
print(eval(compiled.asm['llir']).decode('utf-8'))
#print(compiled.asm['ptx'])

print("running")
size = 4
x = torch.ones(size, device='cuda')
y = torch.ones(size, device='cuda')
output = torch.empty_like(x)
out = compiled[(output.numel(),1,1)](output, x, y)
print(output)
