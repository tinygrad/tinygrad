#!/usr/bin/env python3
import llvmlite.binding as llvm
from tinygrad.runtime.support.elf import elf_loader
from hexdump import hexdump

src = """
; ModuleID = "/Users/diane/tinygrad/tinygrad/renderer/llvmir.py"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"E_3"(float* noalias %".1", float* noalias %".2", float* noalias %".3")
{
entry:
  br label %"loop_body_0"
loop_body_0:
  %"loop0" = phi  i32 [0, %"entry"], [%".13", %"loop_body_0"]
  %".6" = getelementptr inbounds float, float* %".2", i32 %"loop0"
  %".7" = load float, float* %".6"
  %".8" = getelementptr inbounds float, float* %".3", i32 %"loop0"
  %".9" = load float, float* %".8"
  %".10" = fadd nsz arcp contract afn reassoc float %".7", %".9"
  %".11" = getelementptr inbounds float, float* %".1", i32 %"loop0"
  store float %".10", float* %".11"
  %".13" = add i32 %"loop0", 1
  %".14" = icmp ult i32 %".13", 3
  br i1 %".14", label %"loop_body_0", label %"loop_exit_0"
loop_exit_0:
  ret void
}
"""

if __name__ == "__main__":
  llvm.initialize()
  llvm.initialize_all_targets()
  llvm.initialize_all_asmprinters()
  target_machine: llvm.targets.TargetMachine = llvm.Target.from_triple("hexagon").create_target_machine(opt=2)
  mod = llvm.parse_assembly(src)
  mod.verify()
  print(target_machine.emit_assembly(mod))
  obj = target_machine.emit_object(mod)
  img, sections, relocs = elf_loader(obj, elf32=True)
  hexdump(img)
