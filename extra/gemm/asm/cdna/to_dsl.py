import pathlib
from tinygrad.viz.serve import _op2dsl
from extra.assembly.amd.dsl import s, v, Reg, VCC_LO, VCC_HI, VCC, EXEC_LO, EXEC_HI, EXEC, SCC, M0, NULL, OFF, Inst
from extra.assembly.amd.autogen.cdna.ins import *

lines = (pathlib.Path(__file__).parent/"gemm.s").read_text().split("\n")
dsl:list[Inst] = []
for line in lines:
  asm = line.split("//")[0].lstrip()
  if not asm: continue
  inst, *rest = asm.split(" ")
  print(eval(inst))
