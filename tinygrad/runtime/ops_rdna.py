from tinygrad.ops import Compiled
from tinygrad.codegen.assembly import AssemblyCodegen
from tinygrad.runtime.ops_gpu import CLBuffer, CLProgram, CL

RDNABuffer = Compiled(CLBuffer, AssemblyCodegen, CLProgram, CL.synchronize)
