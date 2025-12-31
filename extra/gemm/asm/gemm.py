import functools
from extra.assembly.amd.autogen.cdna import *
from extra.assembly.amd.asm import asm
from extra.assembly.amd.dsl import Inst, RawImm
from extra.assembly.amd.test.test_roundtrip import compile_asm

compile_asm = functools.partial(compile_asm, mcpu='gfx950', mattr='+wavefrontsize64')

gemm = [
  s_load_dwordx2(sdata=s[28:29], sbase=s[0:1], offset=0x0, soffset=RawImm(0), imm=1),
]


# verify a single instruction against llvm

def verify_inst(inst:Inst) -> None:
  b = inst.to_bytes()
  st = inst.disasm()
  reasm = asm(st)
  desc = f"{st:25s} {inst} {b!r} {reasm}"
  ref = compile_asm(st, mcpu='gfx950', mattr='+wavefrontsize64')
  assert b == ref, f"Bytes mismatch {b} != {ref} for {st}"

def prepare_verification():
  import pathlib
  raw_txt_lines:list[str] = []
  for l in open(pathlib.Path(__file__).parent/"gemm.s"):
    if l.strip().startswith("//"): continue
    if not l.strip(): continue
    raw_txt_lines.append(l.strip())
  code:list[str] = []
  on = False
  for l in open(__file__):
    if on and l.strip() == "]": break
    if on: code.append(l.strip())
    elif "[" in l: on = True
  return raw_txt_lines, code

if __name__ == "__main__":
  raw_txt_lines, python_code = prepare_verification()

  assert len(python_code) == len(gemm)
  for py_txt,asm_txt,inst in zip(python_code, raw_txt_lines, gemm):
    ref = compile_asm(asm_txt)
    print(asm_txt)
    b = inst.to_bytes()
    st = inst.disasm()
    reasm = asm(st, arch='cdna')
    desc = f"{st:25s} {inst} {b!r} {reasm}"
    ref = compile_asm(st)
    assert b == ref == reasm.to_bytes(), f"Bytes mismatch {b} != {ref} for {st}"
