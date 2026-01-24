import pathlib
from tinygrad.viz.serve import _op2dsl
from extra.assembly.amd.dsl import s, v, Reg, VCC_LO, VCC_HI, VCC, EXEC_LO, EXEC_HI, EXEC, SCC, M0, NULL, OFF, Inst
from extra.assembly.amd.autogen.cdna.ins import *

lines = (pathlib.Path(__file__).parent/"gemm.s").read_text().split("\n")
dsl:list[Inst|str] = []
for line in lines:
  asm = line.split("//")[0].lstrip()
  if not asm: continue
  # label
  if asm.endswith(":"):
    dsl.append(asm)
    continue
  inst, *rest = asm.split(" ")
  if inst == "v_accvgpr_write_b32": inst = "v_accvgpr_write"
  if inst == "v_accvgpr_read_b32": inst = "v_accvgpr_read"
  try:
    inst = eval(inst)
  except NameError:
    if "e64" in inst: inst = eval(inst.replace("_e64", ""))
    else: inst = eval(inst+"_e64")
  print(inst)
