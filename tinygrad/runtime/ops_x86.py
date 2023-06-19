from tinygrad.codegen.assembly_x86 import X86Codegen
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawMallocBuffer
from tinygrad.runtime.ops_clang import ClangProgram

X86Buffer = Compiled(RawMallocBuffer, X86Codegen, ClangProgram)
