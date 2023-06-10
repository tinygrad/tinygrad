from typing import List, Tuple, Any
from enum import Enum
from tinygrad.helpers import getenv
import cffi

__all__ = ["run_ptx", "PTX_ERR", "DEBUGCUDACPU"]

ffi = cffi.FFI()
ffi.cdef("""
int run_ptx(const char* source, int n_args, void* args[],
    int blck_x, int blck_y, int blck_z,
    int grid_x, int grid_y, int grid_z,
    int debug_lvl);
""")
lib = ffi.dlopen("./extra/cudacpu/libcudacpu.so")

DEBUGCUDACPU = getenv("DEBUGCUDACPU", 0)

class PTX_ERR(Enum):
	SUCCESS = 0
	LOAD_FAILED = 1
	KERNEL_NOT_FOUND = 2
	ARGS_MISMATCH = 3

def run_ptx(source:str, args:Tuple[Any,...], block:Tuple[int, ...], grid:Tuple[int, ...]) -> PTX_ERR:
    # print(args)
    return PTX_ERR(lib.run_ptx(source.encode(),
        len(args),
        [ffi.cast("void*", ffi.from_buffer(x._buffer())) for x in args],
        *block, *grid, DEBUGCUDACPU
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
