from extra.assembly.amd.autogen.cdna import *
from extra.assembly.amd.asm import asm
from extra.assembly.amd.dsl import Inst
from extra.assembly.amd.test.test_roundtrip import compile_asm

gemm = [
  s_load_dwordx2(s[28:29], s[0:1], 0x0),
]


# verify a single instruction against llvm

def verify_inst(inst:Inst) -> None:
  b = inst.to_bytes()
  st = inst.disasm()
  reasm = asm(st)
  desc = f"{st:25s} {inst} {b!r} {reasm}"
  ref = compile_asm(st, mcpu='gfx950', mattr='+wavefrontsize64')
  assert b == ref, f"Bytes mismatch {b} != {ref} for {st}"

if __name__ == "__main__":
  import pathlib
  asm_lines:list[str] = []
  for l in open(pathlib.Path(__file__).parent/"gemm.s"):
    if l.strip().startswith("//"): continue
    if not l.strip(): continue
    asm_lines.append(l.strip())

  code:list[str] = []
  on = False
  for l in open(__file__):
    if on and l.strip() == "]": break
    if on: code.append(l.strip())
    elif "[" in l: on = True

  #assert len(code) == len(asm_lines)
  for c,a in zip(code, asm_lines):
    print(c)
    print(a)

  for a in asm_lines:
    if a.endswith(":"): continue
    print(a)
    print(compile_asm(a, mcpu='gfx950', mattr='+wavefrontsize64'))
  """
  for inst in gemm:
    print(inst._lineno)
  """
