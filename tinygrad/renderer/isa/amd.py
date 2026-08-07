from tinygrad.runtime.autogen.amd.rdna3.amd_lib import AMDRenderer, AMDOps, install_amdllvm_tc, expand_wmma_lds_tiles
from tinygrad.renderer.llvmir import AMDLLVMRenderer
import tinygrad.codegen as cg
cg.expand_wmma_lds_hook = expand_wmma_lds_tiles
install_amdllvm_tc(AMDLLVMRenderer)
__all__ = ["AMDRenderer", "AMDOps"]
