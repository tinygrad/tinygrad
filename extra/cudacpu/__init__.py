from typing import List, Tuple

from tinygrad.runtime.lib import RawBuffer
import cffi

from enum import Enum

__all__ = ["run_ptx", "DEBUGCUCPU"]

ffi = cffi.FFI()
ffi.cdef("""
int run_ptx(const char* source, int n_args, void* args[],
    int blck_x, int blck_y, int blck_z,
    int grid_x, int grid_y, int grid_z,
    int debug_lvl);
""")
lib = ffi.dlopen("./extra/cudacpu/libtinygrad.so")

DEBUGCUCPU = getenv("DEBUGCL", 0)

class PTX_ERR(Enum):
	SUCCESS = 0
	LOAD_FAILED = 1
	KERNEL_NOT_FOUND = 2
	ARGS_MISMATCH = 3

def run_ptx(source:str, args:List[RawBuffer], block:Tuple[int, int, int], grid:Tuple[int, int, int]) -> PTX_ERR:
    return PTX_ERR(lib.run_ptx(source.encode(),
        [ffi.cast("void*", x._buf) for x in args],
        *block, *grid, DEBUGCPU
    ))


# kernel = r"""
# .version 7.5
# .target sm_35
# .address_size 64

#         // .globl       _Z4E_16Pf

# .visible .entry _Z4E_16Pf(
#         .param .u64 _Z4E_16Pf_param_0
# )
# {
#         .reg .b32       %r<6>;
#         .reg .b64       %rd<5>;


#         ld.param.u64    %rd1, [_Z4E_16Pf_param_0];
#         cvta.to.global.u64      %rd2, %rd1;
#         mov.u32         %r1, %ntid.x;
#         mov.u32         %r2, %ctaid.x;
#         mov.u32         %r3, %tid.x;
#         mad.lo.s32      %r4, %r1, %r2, %r3;
#         mul.wide.s32    %rd3, %r4, 4;
#         add.s64         %rd4, %rd2, %rd3;
#         mov.u32         %r5, 1065353216;
#         st.global.u32   [%rd4], %r5;
#         ret;

# }
# """

# def run_ptx(kernel: str, )

# import numpy as np

# buff = np.zeros(16, dtype=np.float32)
# buffp = ffi.cast("void*", buff.ctypes.data)

# lib.run_ptx(kernel.encode(), 1, [buffp], 16, 1, 1, 1, 1, 1)

# print(buff)
