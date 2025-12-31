from extra.assembly.amd.autogen.rdna3 import *
from extra.assembly.amd.asm import asm
from extra.assembly.amd.dsl import Inst
from extra.assembly.amd.test.test_roundtrip import compile_asm

gemm = [
  s_load_b128(s[4:7], s[0:1], NULL, 0),
]


# verify a single instruction against llvm

def verify_inst(inst:Inst) -> None:
  b = inst.to_bytes()
  st = inst.disasm()
  reasm = asm(st)
  desc = f"{st:25s} {inst} {b!r} {reasm}"
  assert b == compile_asm(st), f"Bytes mismatch {b} != {compile_asm(st)} for {st}"

if __name__ == "__main__":
  for inst in gemm:
    verify_inst(inst)
