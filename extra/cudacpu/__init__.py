from tinygrad.helpers import getenv
from ctypes import CDLL, c_char_p, c_void_p, c_int, POINTER, cast

lib = CDLL("./extra/cudacpu/libcudacpu.so")
DEBUGCUDACPU = getenv("DEBUGCUDACPU", 0)

lib.ptx_kernel_create.argtypes = [c_char_p]
lib.ptx_kernel_create.restype = c_void_p
lib.ptx_kernel_destroy.argtypes = [c_void_p]
lib.ptx_call.argtypes = [c_void_p,  c_int, POINTER(c_void_p), c_int, c_int, c_int, c_int, c_int, c_int]

class PTXKernel:
    def __init__(self, source: bytes):
        self.kernel = lib.ptx_kernel_create(c_char_p(source))
    def __call__(self, *args, block, grid):
        lib.ptx_call(self.kernel, len(args), (c_void_p * len(args))(*[cast(x, c_void_p) for x in args]), *block, *grid)
    def __del__(self):
        lib.ptx_kernel_destroy(self.kernel)
