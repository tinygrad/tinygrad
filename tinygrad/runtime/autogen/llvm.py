# mypy: ignore-errors
import ctypes
from tinygrad.helpers import unwrap
from tinygrad.runtime.support.c import Struct, CEnum, _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support.llvm import LLVM_PATH
def dll():
  try: return ctypes.CDLL(unwrap(LLVM_PATH))
  except: pass
  return None
dll = dll()

intmax_t = ctypes.c_int64
# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

class imaxdiv_t(Struct): pass
imaxdiv_t._fields_ = [
  ('quot', ctypes.c_int64),
  ('rem', ctypes.c_int64),
]
# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

uintmax_t = ctypes.c_uint64
# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

__gwchar_t = ctypes.c_int32
# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

class fd_set(Struct): pass
__fd_mask = ctypes.c_int64
fd_set._fields_ = [
  ('fds_bits', (ctypes.c_int64 * 16)),
]
class struct_timeval(Struct): pass
__time_t = ctypes.c_int64
__suseconds_t = ctypes.c_int64
struct_timeval._fields_ = [
  ('tv_sec', ctypes.c_int64),
  ('tv_usec', ctypes.c_int64),
]
# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

class struct_timespec(Struct): pass
__syscall_slong_t = ctypes.c_int64
struct_timespec._fields_ = [
  ('tv_sec', ctypes.c_int64),
  ('tv_nsec', ctypes.c_int64),
]
class __sigset_t(Struct): pass
__sigset_t._fields_ = [
  ('__val', (ctypes.c_uint64 * 16)),
]
# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

LLVMVerifierFailureAction = CEnum(ctypes.c_uint32)
LLVMAbortProcessAction = LLVMVerifierFailureAction.define('LLVMAbortProcessAction', 0)
LLVMPrintMessageAction = LLVMVerifierFailureAction.define('LLVMPrintMessageAction', 1)
LLVMReturnStatusAction = LLVMVerifierFailureAction.define('LLVMReturnStatusAction', 2)

LLVMBool = ctypes.c_int32
class struct_LLVMOpaqueModule(Struct): pass
LLVMModuleRef = ctypes.POINTER(struct_LLVMOpaqueModule)
# LLVMBool LLVMVerifyModule(LLVMModuleRef M, LLVMVerifierFailureAction Action, char **OutMessage)
try: (LLVMVerifyModule:=dll.LLVMVerifyModule).restype, LLVMVerifyModule.argtypes = LLVMBool, [LLVMModuleRef, LLVMVerifierFailureAction, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

class struct_LLVMOpaqueValue(Struct): pass
LLVMValueRef = ctypes.POINTER(struct_LLVMOpaqueValue)
# LLVMBool LLVMVerifyFunction(LLVMValueRef Fn, LLVMVerifierFailureAction Action)
try: (LLVMVerifyFunction:=dll.LLVMVerifyFunction).restype, LLVMVerifyFunction.argtypes = LLVMBool, [LLVMValueRef, LLVMVerifierFailureAction]
except AttributeError: pass

# void LLVMViewFunctionCFG(LLVMValueRef Fn)
try: (LLVMViewFunctionCFG:=dll.LLVMViewFunctionCFG).restype, LLVMViewFunctionCFG.argtypes = None, [LLVMValueRef]
except AttributeError: pass

# void LLVMViewFunctionCFGOnly(LLVMValueRef Fn)
try: (LLVMViewFunctionCFGOnly:=dll.LLVMViewFunctionCFGOnly).restype, LLVMViewFunctionCFGOnly.argtypes = None, [LLVMValueRef]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

class struct_LLVMOpaqueMemoryBuffer(Struct): pass
LLVMMemoryBufferRef = ctypes.POINTER(struct_LLVMOpaqueMemoryBuffer)
# LLVMBool LLVMParseBitcode(LLVMMemoryBufferRef MemBuf, LLVMModuleRef *OutModule, char **OutMessage)
try: (LLVMParseBitcode:=dll.LLVMParseBitcode).restype, LLVMParseBitcode.argtypes = LLVMBool, [LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMParseBitcode2(LLVMMemoryBufferRef MemBuf, LLVMModuleRef *OutModule)
try: (LLVMParseBitcode2:=dll.LLVMParseBitcode2).restype, LLVMParseBitcode2.argtypes = LLVMBool, [LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef)]
except AttributeError: pass

class struct_LLVMOpaqueContext(Struct): pass
LLVMContextRef = ctypes.POINTER(struct_LLVMOpaqueContext)
# LLVMBool LLVMParseBitcodeInContext(LLVMContextRef ContextRef, LLVMMemoryBufferRef MemBuf, LLVMModuleRef *OutModule, char **OutMessage)
try: (LLVMParseBitcodeInContext:=dll.LLVMParseBitcodeInContext).restype, LLVMParseBitcodeInContext.argtypes = LLVMBool, [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMParseBitcodeInContext2(LLVMContextRef ContextRef, LLVMMemoryBufferRef MemBuf, LLVMModuleRef *OutModule)
try: (LLVMParseBitcodeInContext2:=dll.LLVMParseBitcodeInContext2).restype, LLVMParseBitcodeInContext2.argtypes = LLVMBool, [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef)]
except AttributeError: pass

# LLVMBool LLVMGetBitcodeModuleInContext(LLVMContextRef ContextRef, LLVMMemoryBufferRef MemBuf, LLVMModuleRef *OutM, char **OutMessage)
try: (LLVMGetBitcodeModuleInContext:=dll.LLVMGetBitcodeModuleInContext).restype, LLVMGetBitcodeModuleInContext.argtypes = LLVMBool, [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMGetBitcodeModuleInContext2(LLVMContextRef ContextRef, LLVMMemoryBufferRef MemBuf, LLVMModuleRef *OutM)
try: (LLVMGetBitcodeModuleInContext2:=dll.LLVMGetBitcodeModuleInContext2).restype, LLVMGetBitcodeModuleInContext2.argtypes = LLVMBool, [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef)]
except AttributeError: pass

# LLVMBool LLVMGetBitcodeModule(LLVMMemoryBufferRef MemBuf, LLVMModuleRef *OutM, char **OutMessage)
try: (LLVMGetBitcodeModule:=dll.LLVMGetBitcodeModule).restype, LLVMGetBitcodeModule.argtypes = LLVMBool, [LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMGetBitcodeModule2(LLVMMemoryBufferRef MemBuf, LLVMModuleRef *OutM)
try: (LLVMGetBitcodeModule2:=dll.LLVMGetBitcodeModule2).restype, LLVMGetBitcodeModule2.argtypes = LLVMBool, [LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef)]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

# int LLVMWriteBitcodeToFile(LLVMModuleRef M, const char *Path)
try: (LLVMWriteBitcodeToFile:=dll.LLVMWriteBitcodeToFile).restype, LLVMWriteBitcodeToFile.argtypes = ctypes.c_int32, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int LLVMWriteBitcodeToFD(LLVMModuleRef M, int FD, int ShouldClose, int Unbuffered)
try: (LLVMWriteBitcodeToFD:=dll.LLVMWriteBitcodeToFD).restype, LLVMWriteBitcodeToFD.argtypes = ctypes.c_int32, [LLVMModuleRef, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
except AttributeError: pass

# int LLVMWriteBitcodeToFileHandle(LLVMModuleRef M, int Handle)
try: (LLVMWriteBitcodeToFileHandle:=dll.LLVMWriteBitcodeToFileHandle).restype, LLVMWriteBitcodeToFileHandle.argtypes = ctypes.c_int32, [LLVMModuleRef, ctypes.c_int32]
except AttributeError: pass

# LLVMMemoryBufferRef LLVMWriteBitcodeToMemoryBuffer(LLVMModuleRef M)
try: (LLVMWriteBitcodeToMemoryBuffer:=dll.LLVMWriteBitcodeToMemoryBuffer).restype, LLVMWriteBitcodeToMemoryBuffer.argtypes = LLVMMemoryBufferRef, [LLVMModuleRef]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

LLVMComdatSelectionKind = CEnum(ctypes.c_uint32)
LLVMAnyComdatSelectionKind = LLVMComdatSelectionKind.define('LLVMAnyComdatSelectionKind', 0)
LLVMExactMatchComdatSelectionKind = LLVMComdatSelectionKind.define('LLVMExactMatchComdatSelectionKind', 1)
LLVMLargestComdatSelectionKind = LLVMComdatSelectionKind.define('LLVMLargestComdatSelectionKind', 2)
LLVMNoDeduplicateComdatSelectionKind = LLVMComdatSelectionKind.define('LLVMNoDeduplicateComdatSelectionKind', 3)
LLVMSameSizeComdatSelectionKind = LLVMComdatSelectionKind.define('LLVMSameSizeComdatSelectionKind', 4)

class struct_LLVMComdat(Struct): pass
LLVMComdatRef = ctypes.POINTER(struct_LLVMComdat)
# LLVMComdatRef LLVMGetOrInsertComdat(LLVMModuleRef M, const char *Name)
try: (LLVMGetOrInsertComdat:=dll.LLVMGetOrInsertComdat).restype, LLVMGetOrInsertComdat.argtypes = LLVMComdatRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMComdatRef LLVMGetComdat(LLVMValueRef V)
try: (LLVMGetComdat:=dll.LLVMGetComdat).restype, LLVMGetComdat.argtypes = LLVMComdatRef, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetComdat(LLVMValueRef V, LLVMComdatRef C)
try: (LLVMSetComdat:=dll.LLVMSetComdat).restype, LLVMSetComdat.argtypes = None, [LLVMValueRef, LLVMComdatRef]
except AttributeError: pass

# LLVMComdatSelectionKind LLVMGetComdatSelectionKind(LLVMComdatRef C)
try: (LLVMGetComdatSelectionKind:=dll.LLVMGetComdatSelectionKind).restype, LLVMGetComdatSelectionKind.argtypes = LLVMComdatSelectionKind, [LLVMComdatRef]
except AttributeError: pass

# void LLVMSetComdatSelectionKind(LLVMComdatRef C, LLVMComdatSelectionKind Kind)
try: (LLVMSetComdatSelectionKind:=dll.LLVMSetComdatSelectionKind).restype, LLVMSetComdatSelectionKind.argtypes = None, [LLVMComdatRef, LLVMComdatSelectionKind]
except AttributeError: pass

LLVMFatalErrorHandler = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char))
# void LLVMInstallFatalErrorHandler(LLVMFatalErrorHandler Handler)
try: (LLVMInstallFatalErrorHandler:=dll.LLVMInstallFatalErrorHandler).restype, LLVMInstallFatalErrorHandler.argtypes = None, [LLVMFatalErrorHandler]
except AttributeError: pass

# void LLVMResetFatalErrorHandler(void)
try: (LLVMResetFatalErrorHandler:=dll.LLVMResetFatalErrorHandler).restype, LLVMResetFatalErrorHandler.argtypes = None, []
except AttributeError: pass

# void LLVMEnablePrettyStackTrace(void)
try: (LLVMEnablePrettyStackTrace:=dll.LLVMEnablePrettyStackTrace).restype, LLVMEnablePrettyStackTrace.argtypes = None, []
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

LLVMOpcode = CEnum(ctypes.c_uint32)
LLVMRet = LLVMOpcode.define('LLVMRet', 1)
LLVMBr = LLVMOpcode.define('LLVMBr', 2)
LLVMSwitch = LLVMOpcode.define('LLVMSwitch', 3)
LLVMIndirectBr = LLVMOpcode.define('LLVMIndirectBr', 4)
LLVMInvoke = LLVMOpcode.define('LLVMInvoke', 5)
LLVMUnreachable = LLVMOpcode.define('LLVMUnreachable', 7)
LLVMCallBr = LLVMOpcode.define('LLVMCallBr', 67)
LLVMFNeg = LLVMOpcode.define('LLVMFNeg', 66)
LLVMAdd = LLVMOpcode.define('LLVMAdd', 8)
LLVMFAdd = LLVMOpcode.define('LLVMFAdd', 9)
LLVMSub = LLVMOpcode.define('LLVMSub', 10)
LLVMFSub = LLVMOpcode.define('LLVMFSub', 11)
LLVMMul = LLVMOpcode.define('LLVMMul', 12)
LLVMFMul = LLVMOpcode.define('LLVMFMul', 13)
LLVMUDiv = LLVMOpcode.define('LLVMUDiv', 14)
LLVMSDiv = LLVMOpcode.define('LLVMSDiv', 15)
LLVMFDiv = LLVMOpcode.define('LLVMFDiv', 16)
LLVMURem = LLVMOpcode.define('LLVMURem', 17)
LLVMSRem = LLVMOpcode.define('LLVMSRem', 18)
LLVMFRem = LLVMOpcode.define('LLVMFRem', 19)
LLVMShl = LLVMOpcode.define('LLVMShl', 20)
LLVMLShr = LLVMOpcode.define('LLVMLShr', 21)
LLVMAShr = LLVMOpcode.define('LLVMAShr', 22)
LLVMAnd = LLVMOpcode.define('LLVMAnd', 23)
LLVMOr = LLVMOpcode.define('LLVMOr', 24)
LLVMXor = LLVMOpcode.define('LLVMXor', 25)
LLVMAlloca = LLVMOpcode.define('LLVMAlloca', 26)
LLVMLoad = LLVMOpcode.define('LLVMLoad', 27)
LLVMStore = LLVMOpcode.define('LLVMStore', 28)
LLVMGetElementPtr = LLVMOpcode.define('LLVMGetElementPtr', 29)
LLVMTrunc = LLVMOpcode.define('LLVMTrunc', 30)
LLVMZExt = LLVMOpcode.define('LLVMZExt', 31)
LLVMSExt = LLVMOpcode.define('LLVMSExt', 32)
LLVMFPToUI = LLVMOpcode.define('LLVMFPToUI', 33)
LLVMFPToSI = LLVMOpcode.define('LLVMFPToSI', 34)
LLVMUIToFP = LLVMOpcode.define('LLVMUIToFP', 35)
LLVMSIToFP = LLVMOpcode.define('LLVMSIToFP', 36)
LLVMFPTrunc = LLVMOpcode.define('LLVMFPTrunc', 37)
LLVMFPExt = LLVMOpcode.define('LLVMFPExt', 38)
LLVMPtrToInt = LLVMOpcode.define('LLVMPtrToInt', 39)
LLVMIntToPtr = LLVMOpcode.define('LLVMIntToPtr', 40)
LLVMBitCast = LLVMOpcode.define('LLVMBitCast', 41)
LLVMAddrSpaceCast = LLVMOpcode.define('LLVMAddrSpaceCast', 60)
LLVMICmp = LLVMOpcode.define('LLVMICmp', 42)
LLVMFCmp = LLVMOpcode.define('LLVMFCmp', 43)
LLVMPHI = LLVMOpcode.define('LLVMPHI', 44)
LLVMCall = LLVMOpcode.define('LLVMCall', 45)
LLVMSelect = LLVMOpcode.define('LLVMSelect', 46)
LLVMUserOp1 = LLVMOpcode.define('LLVMUserOp1', 47)
LLVMUserOp2 = LLVMOpcode.define('LLVMUserOp2', 48)
LLVMVAArg = LLVMOpcode.define('LLVMVAArg', 49)
LLVMExtractElement = LLVMOpcode.define('LLVMExtractElement', 50)
LLVMInsertElement = LLVMOpcode.define('LLVMInsertElement', 51)
LLVMShuffleVector = LLVMOpcode.define('LLVMShuffleVector', 52)
LLVMExtractValue = LLVMOpcode.define('LLVMExtractValue', 53)
LLVMInsertValue = LLVMOpcode.define('LLVMInsertValue', 54)
LLVMFreeze = LLVMOpcode.define('LLVMFreeze', 68)
LLVMFence = LLVMOpcode.define('LLVMFence', 55)
LLVMAtomicCmpXchg = LLVMOpcode.define('LLVMAtomicCmpXchg', 56)
LLVMAtomicRMW = LLVMOpcode.define('LLVMAtomicRMW', 57)
LLVMResume = LLVMOpcode.define('LLVMResume', 58)
LLVMLandingPad = LLVMOpcode.define('LLVMLandingPad', 59)
LLVMCleanupRet = LLVMOpcode.define('LLVMCleanupRet', 61)
LLVMCatchRet = LLVMOpcode.define('LLVMCatchRet', 62)
LLVMCatchPad = LLVMOpcode.define('LLVMCatchPad', 63)
LLVMCleanupPad = LLVMOpcode.define('LLVMCleanupPad', 64)
LLVMCatchSwitch = LLVMOpcode.define('LLVMCatchSwitch', 65)

LLVMTypeKind = CEnum(ctypes.c_uint32)
LLVMVoidTypeKind = LLVMTypeKind.define('LLVMVoidTypeKind', 0)
LLVMHalfTypeKind = LLVMTypeKind.define('LLVMHalfTypeKind', 1)
LLVMFloatTypeKind = LLVMTypeKind.define('LLVMFloatTypeKind', 2)
LLVMDoubleTypeKind = LLVMTypeKind.define('LLVMDoubleTypeKind', 3)
LLVMX86_FP80TypeKind = LLVMTypeKind.define('LLVMX86_FP80TypeKind', 4)
LLVMFP128TypeKind = LLVMTypeKind.define('LLVMFP128TypeKind', 5)
LLVMPPC_FP128TypeKind = LLVMTypeKind.define('LLVMPPC_FP128TypeKind', 6)
LLVMLabelTypeKind = LLVMTypeKind.define('LLVMLabelTypeKind', 7)
LLVMIntegerTypeKind = LLVMTypeKind.define('LLVMIntegerTypeKind', 8)
LLVMFunctionTypeKind = LLVMTypeKind.define('LLVMFunctionTypeKind', 9)
LLVMStructTypeKind = LLVMTypeKind.define('LLVMStructTypeKind', 10)
LLVMArrayTypeKind = LLVMTypeKind.define('LLVMArrayTypeKind', 11)
LLVMPointerTypeKind = LLVMTypeKind.define('LLVMPointerTypeKind', 12)
LLVMVectorTypeKind = LLVMTypeKind.define('LLVMVectorTypeKind', 13)
LLVMMetadataTypeKind = LLVMTypeKind.define('LLVMMetadataTypeKind', 14)
LLVMTokenTypeKind = LLVMTypeKind.define('LLVMTokenTypeKind', 16)
LLVMScalableVectorTypeKind = LLVMTypeKind.define('LLVMScalableVectorTypeKind', 17)
LLVMBFloatTypeKind = LLVMTypeKind.define('LLVMBFloatTypeKind', 18)
LLVMX86_AMXTypeKind = LLVMTypeKind.define('LLVMX86_AMXTypeKind', 19)
LLVMTargetExtTypeKind = LLVMTypeKind.define('LLVMTargetExtTypeKind', 20)

LLVMLinkage = CEnum(ctypes.c_uint32)
LLVMExternalLinkage = LLVMLinkage.define('LLVMExternalLinkage', 0)
LLVMAvailableExternallyLinkage = LLVMLinkage.define('LLVMAvailableExternallyLinkage', 1)
LLVMLinkOnceAnyLinkage = LLVMLinkage.define('LLVMLinkOnceAnyLinkage', 2)
LLVMLinkOnceODRLinkage = LLVMLinkage.define('LLVMLinkOnceODRLinkage', 3)
LLVMLinkOnceODRAutoHideLinkage = LLVMLinkage.define('LLVMLinkOnceODRAutoHideLinkage', 4)
LLVMWeakAnyLinkage = LLVMLinkage.define('LLVMWeakAnyLinkage', 5)
LLVMWeakODRLinkage = LLVMLinkage.define('LLVMWeakODRLinkage', 6)
LLVMAppendingLinkage = LLVMLinkage.define('LLVMAppendingLinkage', 7)
LLVMInternalLinkage = LLVMLinkage.define('LLVMInternalLinkage', 8)
LLVMPrivateLinkage = LLVMLinkage.define('LLVMPrivateLinkage', 9)
LLVMDLLImportLinkage = LLVMLinkage.define('LLVMDLLImportLinkage', 10)
LLVMDLLExportLinkage = LLVMLinkage.define('LLVMDLLExportLinkage', 11)
LLVMExternalWeakLinkage = LLVMLinkage.define('LLVMExternalWeakLinkage', 12)
LLVMGhostLinkage = LLVMLinkage.define('LLVMGhostLinkage', 13)
LLVMCommonLinkage = LLVMLinkage.define('LLVMCommonLinkage', 14)
LLVMLinkerPrivateLinkage = LLVMLinkage.define('LLVMLinkerPrivateLinkage', 15)
LLVMLinkerPrivateWeakLinkage = LLVMLinkage.define('LLVMLinkerPrivateWeakLinkage', 16)

LLVMVisibility = CEnum(ctypes.c_uint32)
LLVMDefaultVisibility = LLVMVisibility.define('LLVMDefaultVisibility', 0)
LLVMHiddenVisibility = LLVMVisibility.define('LLVMHiddenVisibility', 1)
LLVMProtectedVisibility = LLVMVisibility.define('LLVMProtectedVisibility', 2)

LLVMUnnamedAddr = CEnum(ctypes.c_uint32)
LLVMNoUnnamedAddr = LLVMUnnamedAddr.define('LLVMNoUnnamedAddr', 0)
LLVMLocalUnnamedAddr = LLVMUnnamedAddr.define('LLVMLocalUnnamedAddr', 1)
LLVMGlobalUnnamedAddr = LLVMUnnamedAddr.define('LLVMGlobalUnnamedAddr', 2)

LLVMDLLStorageClass = CEnum(ctypes.c_uint32)
LLVMDefaultStorageClass = LLVMDLLStorageClass.define('LLVMDefaultStorageClass', 0)
LLVMDLLImportStorageClass = LLVMDLLStorageClass.define('LLVMDLLImportStorageClass', 1)
LLVMDLLExportStorageClass = LLVMDLLStorageClass.define('LLVMDLLExportStorageClass', 2)

LLVMCallConv = CEnum(ctypes.c_uint32)
LLVMCCallConv = LLVMCallConv.define('LLVMCCallConv', 0)
LLVMFastCallConv = LLVMCallConv.define('LLVMFastCallConv', 8)
LLVMColdCallConv = LLVMCallConv.define('LLVMColdCallConv', 9)
LLVMGHCCallConv = LLVMCallConv.define('LLVMGHCCallConv', 10)
LLVMHiPECallConv = LLVMCallConv.define('LLVMHiPECallConv', 11)
LLVMAnyRegCallConv = LLVMCallConv.define('LLVMAnyRegCallConv', 13)
LLVMPreserveMostCallConv = LLVMCallConv.define('LLVMPreserveMostCallConv', 14)
LLVMPreserveAllCallConv = LLVMCallConv.define('LLVMPreserveAllCallConv', 15)
LLVMSwiftCallConv = LLVMCallConv.define('LLVMSwiftCallConv', 16)
LLVMCXXFASTTLSCallConv = LLVMCallConv.define('LLVMCXXFASTTLSCallConv', 17)
LLVMX86StdcallCallConv = LLVMCallConv.define('LLVMX86StdcallCallConv', 64)
LLVMX86FastcallCallConv = LLVMCallConv.define('LLVMX86FastcallCallConv', 65)
LLVMARMAPCSCallConv = LLVMCallConv.define('LLVMARMAPCSCallConv', 66)
LLVMARMAAPCSCallConv = LLVMCallConv.define('LLVMARMAAPCSCallConv', 67)
LLVMARMAAPCSVFPCallConv = LLVMCallConv.define('LLVMARMAAPCSVFPCallConv', 68)
LLVMMSP430INTRCallConv = LLVMCallConv.define('LLVMMSP430INTRCallConv', 69)
LLVMX86ThisCallCallConv = LLVMCallConv.define('LLVMX86ThisCallCallConv', 70)
LLVMPTXKernelCallConv = LLVMCallConv.define('LLVMPTXKernelCallConv', 71)
LLVMPTXDeviceCallConv = LLVMCallConv.define('LLVMPTXDeviceCallConv', 72)
LLVMSPIRFUNCCallConv = LLVMCallConv.define('LLVMSPIRFUNCCallConv', 75)
LLVMSPIRKERNELCallConv = LLVMCallConv.define('LLVMSPIRKERNELCallConv', 76)
LLVMIntelOCLBICallConv = LLVMCallConv.define('LLVMIntelOCLBICallConv', 77)
LLVMX8664SysVCallConv = LLVMCallConv.define('LLVMX8664SysVCallConv', 78)
LLVMWin64CallConv = LLVMCallConv.define('LLVMWin64CallConv', 79)
LLVMX86VectorCallCallConv = LLVMCallConv.define('LLVMX86VectorCallCallConv', 80)
LLVMHHVMCallConv = LLVMCallConv.define('LLVMHHVMCallConv', 81)
LLVMHHVMCCallConv = LLVMCallConv.define('LLVMHHVMCCallConv', 82)
LLVMX86INTRCallConv = LLVMCallConv.define('LLVMX86INTRCallConv', 83)
LLVMAVRINTRCallConv = LLVMCallConv.define('LLVMAVRINTRCallConv', 84)
LLVMAVRSIGNALCallConv = LLVMCallConv.define('LLVMAVRSIGNALCallConv', 85)
LLVMAVRBUILTINCallConv = LLVMCallConv.define('LLVMAVRBUILTINCallConv', 86)
LLVMAMDGPUVSCallConv = LLVMCallConv.define('LLVMAMDGPUVSCallConv', 87)
LLVMAMDGPUGSCallConv = LLVMCallConv.define('LLVMAMDGPUGSCallConv', 88)
LLVMAMDGPUPSCallConv = LLVMCallConv.define('LLVMAMDGPUPSCallConv', 89)
LLVMAMDGPUCSCallConv = LLVMCallConv.define('LLVMAMDGPUCSCallConv', 90)
LLVMAMDGPUKERNELCallConv = LLVMCallConv.define('LLVMAMDGPUKERNELCallConv', 91)
LLVMX86RegCallCallConv = LLVMCallConv.define('LLVMX86RegCallCallConv', 92)
LLVMAMDGPUHSCallConv = LLVMCallConv.define('LLVMAMDGPUHSCallConv', 93)
LLVMMSP430BUILTINCallConv = LLVMCallConv.define('LLVMMSP430BUILTINCallConv', 94)
LLVMAMDGPULSCallConv = LLVMCallConv.define('LLVMAMDGPULSCallConv', 95)
LLVMAMDGPUESCallConv = LLVMCallConv.define('LLVMAMDGPUESCallConv', 96)

LLVMValueKind = CEnum(ctypes.c_uint32)
LLVMArgumentValueKind = LLVMValueKind.define('LLVMArgumentValueKind', 0)
LLVMBasicBlockValueKind = LLVMValueKind.define('LLVMBasicBlockValueKind', 1)
LLVMMemoryUseValueKind = LLVMValueKind.define('LLVMMemoryUseValueKind', 2)
LLVMMemoryDefValueKind = LLVMValueKind.define('LLVMMemoryDefValueKind', 3)
LLVMMemoryPhiValueKind = LLVMValueKind.define('LLVMMemoryPhiValueKind', 4)
LLVMFunctionValueKind = LLVMValueKind.define('LLVMFunctionValueKind', 5)
LLVMGlobalAliasValueKind = LLVMValueKind.define('LLVMGlobalAliasValueKind', 6)
LLVMGlobalIFuncValueKind = LLVMValueKind.define('LLVMGlobalIFuncValueKind', 7)
LLVMGlobalVariableValueKind = LLVMValueKind.define('LLVMGlobalVariableValueKind', 8)
LLVMBlockAddressValueKind = LLVMValueKind.define('LLVMBlockAddressValueKind', 9)
LLVMConstantExprValueKind = LLVMValueKind.define('LLVMConstantExprValueKind', 10)
LLVMConstantArrayValueKind = LLVMValueKind.define('LLVMConstantArrayValueKind', 11)
LLVMConstantStructValueKind = LLVMValueKind.define('LLVMConstantStructValueKind', 12)
LLVMConstantVectorValueKind = LLVMValueKind.define('LLVMConstantVectorValueKind', 13)
LLVMUndefValueValueKind = LLVMValueKind.define('LLVMUndefValueValueKind', 14)
LLVMConstantAggregateZeroValueKind = LLVMValueKind.define('LLVMConstantAggregateZeroValueKind', 15)
LLVMConstantDataArrayValueKind = LLVMValueKind.define('LLVMConstantDataArrayValueKind', 16)
LLVMConstantDataVectorValueKind = LLVMValueKind.define('LLVMConstantDataVectorValueKind', 17)
LLVMConstantIntValueKind = LLVMValueKind.define('LLVMConstantIntValueKind', 18)
LLVMConstantFPValueKind = LLVMValueKind.define('LLVMConstantFPValueKind', 19)
LLVMConstantPointerNullValueKind = LLVMValueKind.define('LLVMConstantPointerNullValueKind', 20)
LLVMConstantTokenNoneValueKind = LLVMValueKind.define('LLVMConstantTokenNoneValueKind', 21)
LLVMMetadataAsValueValueKind = LLVMValueKind.define('LLVMMetadataAsValueValueKind', 22)
LLVMInlineAsmValueKind = LLVMValueKind.define('LLVMInlineAsmValueKind', 23)
LLVMInstructionValueKind = LLVMValueKind.define('LLVMInstructionValueKind', 24)
LLVMPoisonValueValueKind = LLVMValueKind.define('LLVMPoisonValueValueKind', 25)
LLVMConstantTargetNoneValueKind = LLVMValueKind.define('LLVMConstantTargetNoneValueKind', 26)
LLVMConstantPtrAuthValueKind = LLVMValueKind.define('LLVMConstantPtrAuthValueKind', 27)

LLVMIntPredicate = CEnum(ctypes.c_uint32)
LLVMIntEQ = LLVMIntPredicate.define('LLVMIntEQ', 32)
LLVMIntNE = LLVMIntPredicate.define('LLVMIntNE', 33)
LLVMIntUGT = LLVMIntPredicate.define('LLVMIntUGT', 34)
LLVMIntUGE = LLVMIntPredicate.define('LLVMIntUGE', 35)
LLVMIntULT = LLVMIntPredicate.define('LLVMIntULT', 36)
LLVMIntULE = LLVMIntPredicate.define('LLVMIntULE', 37)
LLVMIntSGT = LLVMIntPredicate.define('LLVMIntSGT', 38)
LLVMIntSGE = LLVMIntPredicate.define('LLVMIntSGE', 39)
LLVMIntSLT = LLVMIntPredicate.define('LLVMIntSLT', 40)
LLVMIntSLE = LLVMIntPredicate.define('LLVMIntSLE', 41)

LLVMRealPredicate = CEnum(ctypes.c_uint32)
LLVMRealPredicateFalse = LLVMRealPredicate.define('LLVMRealPredicateFalse', 0)
LLVMRealOEQ = LLVMRealPredicate.define('LLVMRealOEQ', 1)
LLVMRealOGT = LLVMRealPredicate.define('LLVMRealOGT', 2)
LLVMRealOGE = LLVMRealPredicate.define('LLVMRealOGE', 3)
LLVMRealOLT = LLVMRealPredicate.define('LLVMRealOLT', 4)
LLVMRealOLE = LLVMRealPredicate.define('LLVMRealOLE', 5)
LLVMRealONE = LLVMRealPredicate.define('LLVMRealONE', 6)
LLVMRealORD = LLVMRealPredicate.define('LLVMRealORD', 7)
LLVMRealUNO = LLVMRealPredicate.define('LLVMRealUNO', 8)
LLVMRealUEQ = LLVMRealPredicate.define('LLVMRealUEQ', 9)
LLVMRealUGT = LLVMRealPredicate.define('LLVMRealUGT', 10)
LLVMRealUGE = LLVMRealPredicate.define('LLVMRealUGE', 11)
LLVMRealULT = LLVMRealPredicate.define('LLVMRealULT', 12)
LLVMRealULE = LLVMRealPredicate.define('LLVMRealULE', 13)
LLVMRealUNE = LLVMRealPredicate.define('LLVMRealUNE', 14)
LLVMRealPredicateTrue = LLVMRealPredicate.define('LLVMRealPredicateTrue', 15)

LLVMLandingPadClauseTy = CEnum(ctypes.c_uint32)
LLVMLandingPadCatch = LLVMLandingPadClauseTy.define('LLVMLandingPadCatch', 0)
LLVMLandingPadFilter = LLVMLandingPadClauseTy.define('LLVMLandingPadFilter', 1)

LLVMThreadLocalMode = CEnum(ctypes.c_uint32)
LLVMNotThreadLocal = LLVMThreadLocalMode.define('LLVMNotThreadLocal', 0)
LLVMGeneralDynamicTLSModel = LLVMThreadLocalMode.define('LLVMGeneralDynamicTLSModel', 1)
LLVMLocalDynamicTLSModel = LLVMThreadLocalMode.define('LLVMLocalDynamicTLSModel', 2)
LLVMInitialExecTLSModel = LLVMThreadLocalMode.define('LLVMInitialExecTLSModel', 3)
LLVMLocalExecTLSModel = LLVMThreadLocalMode.define('LLVMLocalExecTLSModel', 4)

LLVMAtomicOrdering = CEnum(ctypes.c_uint32)
LLVMAtomicOrderingNotAtomic = LLVMAtomicOrdering.define('LLVMAtomicOrderingNotAtomic', 0)
LLVMAtomicOrderingUnordered = LLVMAtomicOrdering.define('LLVMAtomicOrderingUnordered', 1)
LLVMAtomicOrderingMonotonic = LLVMAtomicOrdering.define('LLVMAtomicOrderingMonotonic', 2)
LLVMAtomicOrderingAcquire = LLVMAtomicOrdering.define('LLVMAtomicOrderingAcquire', 4)
LLVMAtomicOrderingRelease = LLVMAtomicOrdering.define('LLVMAtomicOrderingRelease', 5)
LLVMAtomicOrderingAcquireRelease = LLVMAtomicOrdering.define('LLVMAtomicOrderingAcquireRelease', 6)
LLVMAtomicOrderingSequentiallyConsistent = LLVMAtomicOrdering.define('LLVMAtomicOrderingSequentiallyConsistent', 7)

LLVMAtomicRMWBinOp = CEnum(ctypes.c_uint32)
LLVMAtomicRMWBinOpXchg = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpXchg', 0)
LLVMAtomicRMWBinOpAdd = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpAdd', 1)
LLVMAtomicRMWBinOpSub = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpSub', 2)
LLVMAtomicRMWBinOpAnd = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpAnd', 3)
LLVMAtomicRMWBinOpNand = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpNand', 4)
LLVMAtomicRMWBinOpOr = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpOr', 5)
LLVMAtomicRMWBinOpXor = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpXor', 6)
LLVMAtomicRMWBinOpMax = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpMax', 7)
LLVMAtomicRMWBinOpMin = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpMin', 8)
LLVMAtomicRMWBinOpUMax = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUMax', 9)
LLVMAtomicRMWBinOpUMin = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUMin', 10)
LLVMAtomicRMWBinOpFAdd = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFAdd', 11)
LLVMAtomicRMWBinOpFSub = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFSub', 12)
LLVMAtomicRMWBinOpFMax = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFMax', 13)
LLVMAtomicRMWBinOpFMin = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFMin', 14)
LLVMAtomicRMWBinOpUIncWrap = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUIncWrap', 15)
LLVMAtomicRMWBinOpUDecWrap = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUDecWrap', 16)
LLVMAtomicRMWBinOpUSubCond = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUSubCond', 17)
LLVMAtomicRMWBinOpUSubSat = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUSubSat', 18)

LLVMDiagnosticSeverity = CEnum(ctypes.c_uint32)
LLVMDSError = LLVMDiagnosticSeverity.define('LLVMDSError', 0)
LLVMDSWarning = LLVMDiagnosticSeverity.define('LLVMDSWarning', 1)
LLVMDSRemark = LLVMDiagnosticSeverity.define('LLVMDSRemark', 2)
LLVMDSNote = LLVMDiagnosticSeverity.define('LLVMDSNote', 3)

LLVMInlineAsmDialect = CEnum(ctypes.c_uint32)
LLVMInlineAsmDialectATT = LLVMInlineAsmDialect.define('LLVMInlineAsmDialectATT', 0)
LLVMInlineAsmDialectIntel = LLVMInlineAsmDialect.define('LLVMInlineAsmDialectIntel', 1)

LLVMModuleFlagBehavior = CEnum(ctypes.c_uint32)
LLVMModuleFlagBehaviorError = LLVMModuleFlagBehavior.define('LLVMModuleFlagBehaviorError', 0)
LLVMModuleFlagBehaviorWarning = LLVMModuleFlagBehavior.define('LLVMModuleFlagBehaviorWarning', 1)
LLVMModuleFlagBehaviorRequire = LLVMModuleFlagBehavior.define('LLVMModuleFlagBehaviorRequire', 2)
LLVMModuleFlagBehaviorOverride = LLVMModuleFlagBehavior.define('LLVMModuleFlagBehaviorOverride', 3)
LLVMModuleFlagBehaviorAppend = LLVMModuleFlagBehavior.define('LLVMModuleFlagBehaviorAppend', 4)
LLVMModuleFlagBehaviorAppendUnique = LLVMModuleFlagBehavior.define('LLVMModuleFlagBehaviorAppendUnique', 5)

_anonenum0 = CEnum(ctypes.c_int32)
LLVMAttributeReturnIndex = _anonenum0.define('LLVMAttributeReturnIndex', 0)
LLVMAttributeFunctionIndex = _anonenum0.define('LLVMAttributeFunctionIndex', -1)

LLVMAttributeIndex = ctypes.c_uint32
LLVMTailCallKind = CEnum(ctypes.c_uint32)
LLVMTailCallKindNone = LLVMTailCallKind.define('LLVMTailCallKindNone', 0)
LLVMTailCallKindTail = LLVMTailCallKind.define('LLVMTailCallKindTail', 1)
LLVMTailCallKindMustTail = LLVMTailCallKind.define('LLVMTailCallKindMustTail', 2)
LLVMTailCallKindNoTail = LLVMTailCallKind.define('LLVMTailCallKindNoTail', 3)

_anonenum1 = CEnum(ctypes.c_uint32)
LLVMFastMathAllowReassoc = _anonenum1.define('LLVMFastMathAllowReassoc', 1)
LLVMFastMathNoNaNs = _anonenum1.define('LLVMFastMathNoNaNs', 2)
LLVMFastMathNoInfs = _anonenum1.define('LLVMFastMathNoInfs', 4)
LLVMFastMathNoSignedZeros = _anonenum1.define('LLVMFastMathNoSignedZeros', 8)
LLVMFastMathAllowReciprocal = _anonenum1.define('LLVMFastMathAllowReciprocal', 16)
LLVMFastMathAllowContract = _anonenum1.define('LLVMFastMathAllowContract', 32)
LLVMFastMathApproxFunc = _anonenum1.define('LLVMFastMathApproxFunc', 64)
LLVMFastMathNone = _anonenum1.define('LLVMFastMathNone', 0)
LLVMFastMathAll = _anonenum1.define('LLVMFastMathAll', 127)

LLVMFastMathFlags = ctypes.c_uint32
_anonenum2 = CEnum(ctypes.c_uint32)
LLVMGEPFlagInBounds = _anonenum2.define('LLVMGEPFlagInBounds', 1)
LLVMGEPFlagNUSW = _anonenum2.define('LLVMGEPFlagNUSW', 2)
LLVMGEPFlagNUW = _anonenum2.define('LLVMGEPFlagNUW', 4)

LLVMGEPNoWrapFlags = ctypes.c_uint32
# void LLVMShutdown(void)
try: (LLVMShutdown:=dll.LLVMShutdown).restype, LLVMShutdown.argtypes = None, []
except AttributeError: pass

# void LLVMGetVersion(unsigned int *Major, unsigned int *Minor, unsigned int *Patch)
try: (LLVMGetVersion:=dll.LLVMGetVersion).restype, LLVMGetVersion.argtypes = None, [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

# char *LLVMCreateMessage(const char *Message)
try: (LLVMCreateMessage:=dll.LLVMCreateMessage).restype, LLVMCreateMessage.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMDisposeMessage(char *Message)
try: (LLVMDisposeMessage:=dll.LLVMDisposeMessage).restype, LLVMDisposeMessage.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class struct_LLVMOpaqueDiagnosticInfo(Struct): pass
LLVMDiagnosticHandler = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_LLVMOpaqueDiagnosticInfo), ctypes.c_void_p)
LLVMYieldCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_LLVMOpaqueContext), ctypes.c_void_p)
# LLVMContextRef LLVMContextCreate(void)
try: (LLVMContextCreate:=dll.LLVMContextCreate).restype, LLVMContextCreate.argtypes = LLVMContextRef, []
except AttributeError: pass

# LLVMContextRef LLVMGetGlobalContext(void)
try: (LLVMGetGlobalContext:=dll.LLVMGetGlobalContext).restype, LLVMGetGlobalContext.argtypes = LLVMContextRef, []
except AttributeError: pass

# void LLVMContextSetDiagnosticHandler(LLVMContextRef C, LLVMDiagnosticHandler Handler, void *DiagnosticContext)
try: (LLVMContextSetDiagnosticHandler:=dll.LLVMContextSetDiagnosticHandler).restype, LLVMContextSetDiagnosticHandler.argtypes = None, [LLVMContextRef, LLVMDiagnosticHandler, ctypes.c_void_p]
except AttributeError: pass

# LLVMDiagnosticHandler LLVMContextGetDiagnosticHandler(LLVMContextRef C)
try: (LLVMContextGetDiagnosticHandler:=dll.LLVMContextGetDiagnosticHandler).restype, LLVMContextGetDiagnosticHandler.argtypes = LLVMDiagnosticHandler, [LLVMContextRef]
except AttributeError: pass

# void *LLVMContextGetDiagnosticContext(LLVMContextRef C)
try: (LLVMContextGetDiagnosticContext:=dll.LLVMContextGetDiagnosticContext).restype, LLVMContextGetDiagnosticContext.argtypes = ctypes.c_void_p, [LLVMContextRef]
except AttributeError: pass

# void LLVMContextSetYieldCallback(LLVMContextRef C, LLVMYieldCallback Callback, void *OpaqueHandle)
try: (LLVMContextSetYieldCallback:=dll.LLVMContextSetYieldCallback).restype, LLVMContextSetYieldCallback.argtypes = None, [LLVMContextRef, LLVMYieldCallback, ctypes.c_void_p]
except AttributeError: pass

# LLVMBool LLVMContextShouldDiscardValueNames(LLVMContextRef C)
try: (LLVMContextShouldDiscardValueNames:=dll.LLVMContextShouldDiscardValueNames).restype, LLVMContextShouldDiscardValueNames.argtypes = LLVMBool, [LLVMContextRef]
except AttributeError: pass

# void LLVMContextSetDiscardValueNames(LLVMContextRef C, LLVMBool Discard)
try: (LLVMContextSetDiscardValueNames:=dll.LLVMContextSetDiscardValueNames).restype, LLVMContextSetDiscardValueNames.argtypes = None, [LLVMContextRef, LLVMBool]
except AttributeError: pass

# void LLVMContextDispose(LLVMContextRef C)
try: (LLVMContextDispose:=dll.LLVMContextDispose).restype, LLVMContextDispose.argtypes = None, [LLVMContextRef]
except AttributeError: pass

LLVMDiagnosticInfoRef = ctypes.POINTER(struct_LLVMOpaqueDiagnosticInfo)
# char *LLVMGetDiagInfoDescription(LLVMDiagnosticInfoRef DI)
try: (LLVMGetDiagInfoDescription:=dll.LLVMGetDiagInfoDescription).restype, LLVMGetDiagInfoDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMDiagnosticInfoRef]
except AttributeError: pass

# LLVMDiagnosticSeverity LLVMGetDiagInfoSeverity(LLVMDiagnosticInfoRef DI)
try: (LLVMGetDiagInfoSeverity:=dll.LLVMGetDiagInfoSeverity).restype, LLVMGetDiagInfoSeverity.argtypes = LLVMDiagnosticSeverity, [LLVMDiagnosticInfoRef]
except AttributeError: pass

# unsigned int LLVMGetMDKindIDInContext(LLVMContextRef C, const char *Name, unsigned int SLen)
try: (LLVMGetMDKindIDInContext:=dll.LLVMGetMDKindIDInContext).restype, LLVMGetMDKindIDInContext.argtypes = ctypes.c_uint32, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

# unsigned int LLVMGetMDKindID(const char *Name, unsigned int SLen)
try: (LLVMGetMDKindID:=dll.LLVMGetMDKindID).restype, LLVMGetMDKindID.argtypes = ctypes.c_uint32, [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

size_t = ctypes.c_uint64
# unsigned int LLVMGetSyncScopeID(LLVMContextRef C, const char *Name, size_t SLen)
try: (LLVMGetSyncScopeID:=dll.LLVMGetSyncScopeID).restype, LLVMGetSyncScopeID.argtypes = ctypes.c_uint32, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# unsigned int LLVMGetEnumAttributeKindForName(const char *Name, size_t SLen)
try: (LLVMGetEnumAttributeKindForName:=dll.LLVMGetEnumAttributeKindForName).restype, LLVMGetEnumAttributeKindForName.argtypes = ctypes.c_uint32, [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# unsigned int LLVMGetLastEnumAttributeKind(void)
try: (LLVMGetLastEnumAttributeKind:=dll.LLVMGetLastEnumAttributeKind).restype, LLVMGetLastEnumAttributeKind.argtypes = ctypes.c_uint32, []
except AttributeError: pass

class struct_LLVMOpaqueAttributeRef(Struct): pass
LLVMAttributeRef = ctypes.POINTER(struct_LLVMOpaqueAttributeRef)
uint64_t = ctypes.c_uint64
# LLVMAttributeRef LLVMCreateEnumAttribute(LLVMContextRef C, unsigned int KindID, uint64_t Val)
try: (LLVMCreateEnumAttribute:=dll.LLVMCreateEnumAttribute).restype, LLVMCreateEnumAttribute.argtypes = LLVMAttributeRef, [LLVMContextRef, ctypes.c_uint32, uint64_t]
except AttributeError: pass

# unsigned int LLVMGetEnumAttributeKind(LLVMAttributeRef A)
try: (LLVMGetEnumAttributeKind:=dll.LLVMGetEnumAttributeKind).restype, LLVMGetEnumAttributeKind.argtypes = ctypes.c_uint32, [LLVMAttributeRef]
except AttributeError: pass

# uint64_t LLVMGetEnumAttributeValue(LLVMAttributeRef A)
try: (LLVMGetEnumAttributeValue:=dll.LLVMGetEnumAttributeValue).restype, LLVMGetEnumAttributeValue.argtypes = uint64_t, [LLVMAttributeRef]
except AttributeError: pass

class struct_LLVMOpaqueType(Struct): pass
LLVMTypeRef = ctypes.POINTER(struct_LLVMOpaqueType)
# LLVMAttributeRef LLVMCreateTypeAttribute(LLVMContextRef C, unsigned int KindID, LLVMTypeRef type_ref)
try: (LLVMCreateTypeAttribute:=dll.LLVMCreateTypeAttribute).restype, LLVMCreateTypeAttribute.argtypes = LLVMAttributeRef, [LLVMContextRef, ctypes.c_uint32, LLVMTypeRef]
except AttributeError: pass

# LLVMTypeRef LLVMGetTypeAttributeValue(LLVMAttributeRef A)
try: (LLVMGetTypeAttributeValue:=dll.LLVMGetTypeAttributeValue).restype, LLVMGetTypeAttributeValue.argtypes = LLVMTypeRef, [LLVMAttributeRef]
except AttributeError: pass

# LLVMAttributeRef LLVMCreateConstantRangeAttribute(LLVMContextRef C, unsigned int KindID, unsigned int NumBits, const uint64_t LowerWords[], const uint64_t UpperWords[])
try: (LLVMCreateConstantRangeAttribute:=dll.LLVMCreateConstantRangeAttribute).restype, LLVMCreateConstantRangeAttribute.argtypes = LLVMAttributeRef, [LLVMContextRef, ctypes.c_uint32, ctypes.c_uint32, (uint64_t * 0), (uint64_t * 0)]
except AttributeError: pass

# LLVMAttributeRef LLVMCreateStringAttribute(LLVMContextRef C, const char *K, unsigned int KLength, const char *V, unsigned int VLength)
try: (LLVMCreateStringAttribute:=dll.LLVMCreateStringAttribute).restype, LLVMCreateStringAttribute.argtypes = LLVMAttributeRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

# const char *LLVMGetStringAttributeKind(LLVMAttributeRef A, unsigned int *Length)
try: (LLVMGetStringAttributeKind:=dll.LLVMGetStringAttributeKind).restype, LLVMGetStringAttributeKind.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMAttributeRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

# const char *LLVMGetStringAttributeValue(LLVMAttributeRef A, unsigned int *Length)
try: (LLVMGetStringAttributeValue:=dll.LLVMGetStringAttributeValue).restype, LLVMGetStringAttributeValue.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMAttributeRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

# LLVMBool LLVMIsEnumAttribute(LLVMAttributeRef A)
try: (LLVMIsEnumAttribute:=dll.LLVMIsEnumAttribute).restype, LLVMIsEnumAttribute.argtypes = LLVMBool, [LLVMAttributeRef]
except AttributeError: pass

# LLVMBool LLVMIsStringAttribute(LLVMAttributeRef A)
try: (LLVMIsStringAttribute:=dll.LLVMIsStringAttribute).restype, LLVMIsStringAttribute.argtypes = LLVMBool, [LLVMAttributeRef]
except AttributeError: pass

# LLVMBool LLVMIsTypeAttribute(LLVMAttributeRef A)
try: (LLVMIsTypeAttribute:=dll.LLVMIsTypeAttribute).restype, LLVMIsTypeAttribute.argtypes = LLVMBool, [LLVMAttributeRef]
except AttributeError: pass

# LLVMTypeRef LLVMGetTypeByName2(LLVMContextRef C, const char *Name)
try: (LLVMGetTypeByName2:=dll.LLVMGetTypeByName2).restype, LLVMGetTypeByName2.argtypes = LLVMTypeRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMModuleRef LLVMModuleCreateWithName(const char *ModuleID)
try: (LLVMModuleCreateWithName:=dll.LLVMModuleCreateWithName).restype, LLVMModuleCreateWithName.argtypes = LLVMModuleRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMModuleRef LLVMModuleCreateWithNameInContext(const char *ModuleID, LLVMContextRef C)
try: (LLVMModuleCreateWithNameInContext:=dll.LLVMModuleCreateWithNameInContext).restype, LLVMModuleCreateWithNameInContext.argtypes = LLVMModuleRef, [ctypes.POINTER(ctypes.c_char), LLVMContextRef]
except AttributeError: pass

# LLVMModuleRef LLVMCloneModule(LLVMModuleRef M)
try: (LLVMCloneModule:=dll.LLVMCloneModule).restype, LLVMCloneModule.argtypes = LLVMModuleRef, [LLVMModuleRef]
except AttributeError: pass

# void LLVMDisposeModule(LLVMModuleRef M)
try: (LLVMDisposeModule:=dll.LLVMDisposeModule).restype, LLVMDisposeModule.argtypes = None, [LLVMModuleRef]
except AttributeError: pass

# LLVMBool LLVMIsNewDbgInfoFormat(LLVMModuleRef M)
try: (LLVMIsNewDbgInfoFormat:=dll.LLVMIsNewDbgInfoFormat).restype, LLVMIsNewDbgInfoFormat.argtypes = LLVMBool, [LLVMModuleRef]
except AttributeError: pass

# void LLVMSetIsNewDbgInfoFormat(LLVMModuleRef M, LLVMBool UseNewFormat)
try: (LLVMSetIsNewDbgInfoFormat:=dll.LLVMSetIsNewDbgInfoFormat).restype, LLVMSetIsNewDbgInfoFormat.argtypes = None, [LLVMModuleRef, LLVMBool]
except AttributeError: pass

# const char *LLVMGetModuleIdentifier(LLVMModuleRef M, size_t *Len)
try: (LLVMGetModuleIdentifier:=dll.LLVMGetModuleIdentifier).restype, LLVMGetModuleIdentifier.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# void LLVMSetModuleIdentifier(LLVMModuleRef M, const char *Ident, size_t Len)
try: (LLVMSetModuleIdentifier:=dll.LLVMSetModuleIdentifier).restype, LLVMSetModuleIdentifier.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# const char *LLVMGetSourceFileName(LLVMModuleRef M, size_t *Len)
try: (LLVMGetSourceFileName:=dll.LLVMGetSourceFileName).restype, LLVMGetSourceFileName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# void LLVMSetSourceFileName(LLVMModuleRef M, const char *Name, size_t Len)
try: (LLVMSetSourceFileName:=dll.LLVMSetSourceFileName).restype, LLVMSetSourceFileName.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# const char *LLVMGetDataLayoutStr(LLVMModuleRef M)
try: (LLVMGetDataLayoutStr:=dll.LLVMGetDataLayoutStr).restype, LLVMGetDataLayoutStr.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef]
except AttributeError: pass

# const char *LLVMGetDataLayout(LLVMModuleRef M)
try: (LLVMGetDataLayout:=dll.LLVMGetDataLayout).restype, LLVMGetDataLayout.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef]
except AttributeError: pass

# void LLVMSetDataLayout(LLVMModuleRef M, const char *DataLayoutStr)
try: (LLVMSetDataLayout:=dll.LLVMSetDataLayout).restype, LLVMSetDataLayout.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# const char *LLVMGetTarget(LLVMModuleRef M)
try: (LLVMGetTarget:=dll.LLVMGetTarget).restype, LLVMGetTarget.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef]
except AttributeError: pass

# void LLVMSetTarget(LLVMModuleRef M, const char *Triple)
try: (LLVMSetTarget:=dll.LLVMSetTarget).restype, LLVMSetTarget.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class struct_LLVMOpaqueModuleFlagEntry(Struct): pass
LLVMModuleFlagEntry = struct_LLVMOpaqueModuleFlagEntry
# LLVMModuleFlagEntry *LLVMCopyModuleFlagsMetadata(LLVMModuleRef M, size_t *Len)
try: (LLVMCopyModuleFlagsMetadata:=dll.LLVMCopyModuleFlagsMetadata).restype, LLVMCopyModuleFlagsMetadata.argtypes = ctypes.POINTER(LLVMModuleFlagEntry), [LLVMModuleRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# void LLVMDisposeModuleFlagsMetadata(LLVMModuleFlagEntry *Entries)
try: (LLVMDisposeModuleFlagsMetadata:=dll.LLVMDisposeModuleFlagsMetadata).restype, LLVMDisposeModuleFlagsMetadata.argtypes = None, [ctypes.POINTER(LLVMModuleFlagEntry)]
except AttributeError: pass

# LLVMModuleFlagBehavior LLVMModuleFlagEntriesGetFlagBehavior(LLVMModuleFlagEntry *Entries, unsigned int Index)
try: (LLVMModuleFlagEntriesGetFlagBehavior:=dll.LLVMModuleFlagEntriesGetFlagBehavior).restype, LLVMModuleFlagEntriesGetFlagBehavior.argtypes = LLVMModuleFlagBehavior, [ctypes.POINTER(LLVMModuleFlagEntry), ctypes.c_uint32]
except AttributeError: pass

# const char *LLVMModuleFlagEntriesGetKey(LLVMModuleFlagEntry *Entries, unsigned int Index, size_t *Len)
try: (LLVMModuleFlagEntriesGetKey:=dll.LLVMModuleFlagEntriesGetKey).restype, LLVMModuleFlagEntriesGetKey.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(LLVMModuleFlagEntry), ctypes.c_uint32, ctypes.POINTER(size_t)]
except AttributeError: pass

class struct_LLVMOpaqueMetadata(Struct): pass
LLVMMetadataRef = ctypes.POINTER(struct_LLVMOpaqueMetadata)
# LLVMMetadataRef LLVMModuleFlagEntriesGetMetadata(LLVMModuleFlagEntry *Entries, unsigned int Index)
try: (LLVMModuleFlagEntriesGetMetadata:=dll.LLVMModuleFlagEntriesGetMetadata).restype, LLVMModuleFlagEntriesGetMetadata.argtypes = LLVMMetadataRef, [ctypes.POINTER(LLVMModuleFlagEntry), ctypes.c_uint32]
except AttributeError: pass

# LLVMMetadataRef LLVMGetModuleFlag(LLVMModuleRef M, const char *Key, size_t KeyLen)
try: (LLVMGetModuleFlag:=dll.LLVMGetModuleFlag).restype, LLVMGetModuleFlag.argtypes = LLVMMetadataRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# void LLVMAddModuleFlag(LLVMModuleRef M, LLVMModuleFlagBehavior Behavior, const char *Key, size_t KeyLen, LLVMMetadataRef Val)
try: (LLVMAddModuleFlag:=dll.LLVMAddModuleFlag).restype, LLVMAddModuleFlag.argtypes = None, [LLVMModuleRef, LLVMModuleFlagBehavior, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef]
except AttributeError: pass

# void LLVMDumpModule(LLVMModuleRef M)
try: (LLVMDumpModule:=dll.LLVMDumpModule).restype, LLVMDumpModule.argtypes = None, [LLVMModuleRef]
except AttributeError: pass

# LLVMBool LLVMPrintModuleToFile(LLVMModuleRef M, const char *Filename, char **ErrorMessage)
try: (LLVMPrintModuleToFile:=dll.LLVMPrintModuleToFile).restype, LLVMPrintModuleToFile.argtypes = LLVMBool, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# char *LLVMPrintModuleToString(LLVMModuleRef M)
try: (LLVMPrintModuleToString:=dll.LLVMPrintModuleToString).restype, LLVMPrintModuleToString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef]
except AttributeError: pass

# const char *LLVMGetModuleInlineAsm(LLVMModuleRef M, size_t *Len)
try: (LLVMGetModuleInlineAsm:=dll.LLVMGetModuleInlineAsm).restype, LLVMGetModuleInlineAsm.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# void LLVMSetModuleInlineAsm2(LLVMModuleRef M, const char *Asm, size_t Len)
try: (LLVMSetModuleInlineAsm2:=dll.LLVMSetModuleInlineAsm2).restype, LLVMSetModuleInlineAsm2.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# void LLVMAppendModuleInlineAsm(LLVMModuleRef M, const char *Asm, size_t Len)
try: (LLVMAppendModuleInlineAsm:=dll.LLVMAppendModuleInlineAsm).restype, LLVMAppendModuleInlineAsm.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMValueRef LLVMGetInlineAsm(LLVMTypeRef Ty, const char *AsmString, size_t AsmStringSize, const char *Constraints, size_t ConstraintsSize, LLVMBool HasSideEffects, LLVMBool IsAlignStack, LLVMInlineAsmDialect Dialect, LLVMBool CanThrow)
try: (LLVMGetInlineAsm:=dll.LLVMGetInlineAsm).restype, LLVMGetInlineAsm.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, LLVMBool, LLVMBool, LLVMInlineAsmDialect, LLVMBool]
except AttributeError: pass

# const char *LLVMGetInlineAsmAsmString(LLVMValueRef InlineAsmVal, size_t *Len)
try: (LLVMGetInlineAsmAsmString:=dll.LLVMGetInlineAsmAsmString).restype, LLVMGetInlineAsmAsmString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# const char *LLVMGetInlineAsmConstraintString(LLVMValueRef InlineAsmVal, size_t *Len)
try: (LLVMGetInlineAsmConstraintString:=dll.LLVMGetInlineAsmConstraintString).restype, LLVMGetInlineAsmConstraintString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# LLVMInlineAsmDialect LLVMGetInlineAsmDialect(LLVMValueRef InlineAsmVal)
try: (LLVMGetInlineAsmDialect:=dll.LLVMGetInlineAsmDialect).restype, LLVMGetInlineAsmDialect.argtypes = LLVMInlineAsmDialect, [LLVMValueRef]
except AttributeError: pass

# LLVMTypeRef LLVMGetInlineAsmFunctionType(LLVMValueRef InlineAsmVal)
try: (LLVMGetInlineAsmFunctionType:=dll.LLVMGetInlineAsmFunctionType).restype, LLVMGetInlineAsmFunctionType.argtypes = LLVMTypeRef, [LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMGetInlineAsmHasSideEffects(LLVMValueRef InlineAsmVal)
try: (LLVMGetInlineAsmHasSideEffects:=dll.LLVMGetInlineAsmHasSideEffects).restype, LLVMGetInlineAsmHasSideEffects.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMGetInlineAsmNeedsAlignedStack(LLVMValueRef InlineAsmVal)
try: (LLVMGetInlineAsmNeedsAlignedStack:=dll.LLVMGetInlineAsmNeedsAlignedStack).restype, LLVMGetInlineAsmNeedsAlignedStack.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMGetInlineAsmCanUnwind(LLVMValueRef InlineAsmVal)
try: (LLVMGetInlineAsmCanUnwind:=dll.LLVMGetInlineAsmCanUnwind).restype, LLVMGetInlineAsmCanUnwind.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# LLVMContextRef LLVMGetModuleContext(LLVMModuleRef M)
try: (LLVMGetModuleContext:=dll.LLVMGetModuleContext).restype, LLVMGetModuleContext.argtypes = LLVMContextRef, [LLVMModuleRef]
except AttributeError: pass

# LLVMTypeRef LLVMGetTypeByName(LLVMModuleRef M, const char *Name)
try: (LLVMGetTypeByName:=dll.LLVMGetTypeByName).restype, LLVMGetTypeByName.argtypes = LLVMTypeRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class struct_LLVMOpaqueNamedMDNode(Struct): pass
LLVMNamedMDNodeRef = ctypes.POINTER(struct_LLVMOpaqueNamedMDNode)
# LLVMNamedMDNodeRef LLVMGetFirstNamedMetadata(LLVMModuleRef M)
try: (LLVMGetFirstNamedMetadata:=dll.LLVMGetFirstNamedMetadata).restype, LLVMGetFirstNamedMetadata.argtypes = LLVMNamedMDNodeRef, [LLVMModuleRef]
except AttributeError: pass

# LLVMNamedMDNodeRef LLVMGetLastNamedMetadata(LLVMModuleRef M)
try: (LLVMGetLastNamedMetadata:=dll.LLVMGetLastNamedMetadata).restype, LLVMGetLastNamedMetadata.argtypes = LLVMNamedMDNodeRef, [LLVMModuleRef]
except AttributeError: pass

# LLVMNamedMDNodeRef LLVMGetNextNamedMetadata(LLVMNamedMDNodeRef NamedMDNode)
try: (LLVMGetNextNamedMetadata:=dll.LLVMGetNextNamedMetadata).restype, LLVMGetNextNamedMetadata.argtypes = LLVMNamedMDNodeRef, [LLVMNamedMDNodeRef]
except AttributeError: pass

# LLVMNamedMDNodeRef LLVMGetPreviousNamedMetadata(LLVMNamedMDNodeRef NamedMDNode)
try: (LLVMGetPreviousNamedMetadata:=dll.LLVMGetPreviousNamedMetadata).restype, LLVMGetPreviousNamedMetadata.argtypes = LLVMNamedMDNodeRef, [LLVMNamedMDNodeRef]
except AttributeError: pass

# LLVMNamedMDNodeRef LLVMGetNamedMetadata(LLVMModuleRef M, const char *Name, size_t NameLen)
try: (LLVMGetNamedMetadata:=dll.LLVMGetNamedMetadata).restype, LLVMGetNamedMetadata.argtypes = LLVMNamedMDNodeRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMNamedMDNodeRef LLVMGetOrInsertNamedMetadata(LLVMModuleRef M, const char *Name, size_t NameLen)
try: (LLVMGetOrInsertNamedMetadata:=dll.LLVMGetOrInsertNamedMetadata).restype, LLVMGetOrInsertNamedMetadata.argtypes = LLVMNamedMDNodeRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# const char *LLVMGetNamedMetadataName(LLVMNamedMDNodeRef NamedMD, size_t *NameLen)
try: (LLVMGetNamedMetadataName:=dll.LLVMGetNamedMetadataName).restype, LLVMGetNamedMetadataName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMNamedMDNodeRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# unsigned int LLVMGetNamedMetadataNumOperands(LLVMModuleRef M, const char *Name)
try: (LLVMGetNamedMetadataNumOperands:=dll.LLVMGetNamedMetadataNumOperands).restype, LLVMGetNamedMetadataNumOperands.argtypes = ctypes.c_uint32, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMGetNamedMetadataOperands(LLVMModuleRef M, const char *Name, LLVMValueRef *Dest)
try: (LLVMGetNamedMetadataOperands:=dll.LLVMGetNamedMetadataOperands).restype, LLVMGetNamedMetadataOperands.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMValueRef)]
except AttributeError: pass

# void LLVMAddNamedMetadataOperand(LLVMModuleRef M, const char *Name, LLVMValueRef Val)
try: (LLVMAddNamedMetadataOperand:=dll.LLVMAddNamedMetadataOperand).restype, LLVMAddNamedMetadataOperand.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMValueRef]
except AttributeError: pass

# const char *LLVMGetDebugLocDirectory(LLVMValueRef Val, unsigned int *Length)
try: (LLVMGetDebugLocDirectory:=dll.LLVMGetDebugLocDirectory).restype, LLVMGetDebugLocDirectory.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

# const char *LLVMGetDebugLocFilename(LLVMValueRef Val, unsigned int *Length)
try: (LLVMGetDebugLocFilename:=dll.LLVMGetDebugLocFilename).restype, LLVMGetDebugLocFilename.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

# unsigned int LLVMGetDebugLocLine(LLVMValueRef Val)
try: (LLVMGetDebugLocLine:=dll.LLVMGetDebugLocLine).restype, LLVMGetDebugLocLine.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# unsigned int LLVMGetDebugLocColumn(LLVMValueRef Val)
try: (LLVMGetDebugLocColumn:=dll.LLVMGetDebugLocColumn).restype, LLVMGetDebugLocColumn.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMAddFunction(LLVMModuleRef M, const char *Name, LLVMTypeRef FunctionTy)
try: (LLVMAddFunction:=dll.LLVMAddFunction).restype, LLVMAddFunction.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMGetNamedFunction(LLVMModuleRef M, const char *Name)
try: (LLVMGetNamedFunction:=dll.LLVMGetNamedFunction).restype, LLVMGetNamedFunction.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMGetNamedFunctionWithLength(LLVMModuleRef M, const char *Name, size_t Length)
try: (LLVMGetNamedFunctionWithLength:=dll.LLVMGetNamedFunctionWithLength).restype, LLVMGetNamedFunctionWithLength.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMValueRef LLVMGetFirstFunction(LLVMModuleRef M)
try: (LLVMGetFirstFunction:=dll.LLVMGetFirstFunction).restype, LLVMGetFirstFunction.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

# LLVMValueRef LLVMGetLastFunction(LLVMModuleRef M)
try: (LLVMGetLastFunction:=dll.LLVMGetLastFunction).restype, LLVMGetLastFunction.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

# LLVMValueRef LLVMGetNextFunction(LLVMValueRef Fn)
try: (LLVMGetNextFunction:=dll.LLVMGetNextFunction).restype, LLVMGetNextFunction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetPreviousFunction(LLVMValueRef Fn)
try: (LLVMGetPreviousFunction:=dll.LLVMGetPreviousFunction).restype, LLVMGetPreviousFunction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetModuleInlineAsm(LLVMModuleRef M, const char *Asm)
try: (LLVMSetModuleInlineAsm:=dll.LLVMSetModuleInlineAsm).restype, LLVMSetModuleInlineAsm.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMTypeKind LLVMGetTypeKind(LLVMTypeRef Ty)
try: (LLVMGetTypeKind:=dll.LLVMGetTypeKind).restype, LLVMGetTypeKind.argtypes = LLVMTypeKind, [LLVMTypeRef]
except AttributeError: pass

# LLVMBool LLVMTypeIsSized(LLVMTypeRef Ty)
try: (LLVMTypeIsSized:=dll.LLVMTypeIsSized).restype, LLVMTypeIsSized.argtypes = LLVMBool, [LLVMTypeRef]
except AttributeError: pass

# LLVMContextRef LLVMGetTypeContext(LLVMTypeRef Ty)
try: (LLVMGetTypeContext:=dll.LLVMGetTypeContext).restype, LLVMGetTypeContext.argtypes = LLVMContextRef, [LLVMTypeRef]
except AttributeError: pass

# void LLVMDumpType(LLVMTypeRef Val)
try: (LLVMDumpType:=dll.LLVMDumpType).restype, LLVMDumpType.argtypes = None, [LLVMTypeRef]
except AttributeError: pass

# char *LLVMPrintTypeToString(LLVMTypeRef Val)
try: (LLVMPrintTypeToString:=dll.LLVMPrintTypeToString).restype, LLVMPrintTypeToString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTypeRef]
except AttributeError: pass

# LLVMTypeRef LLVMInt1TypeInContext(LLVMContextRef C)
try: (LLVMInt1TypeInContext:=dll.LLVMInt1TypeInContext).restype, LLVMInt1TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMInt8TypeInContext(LLVMContextRef C)
try: (LLVMInt8TypeInContext:=dll.LLVMInt8TypeInContext).restype, LLVMInt8TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMInt16TypeInContext(LLVMContextRef C)
try: (LLVMInt16TypeInContext:=dll.LLVMInt16TypeInContext).restype, LLVMInt16TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMInt32TypeInContext(LLVMContextRef C)
try: (LLVMInt32TypeInContext:=dll.LLVMInt32TypeInContext).restype, LLVMInt32TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMInt64TypeInContext(LLVMContextRef C)
try: (LLVMInt64TypeInContext:=dll.LLVMInt64TypeInContext).restype, LLVMInt64TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMInt128TypeInContext(LLVMContextRef C)
try: (LLVMInt128TypeInContext:=dll.LLVMInt128TypeInContext).restype, LLVMInt128TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntTypeInContext(LLVMContextRef C, unsigned int NumBits)
try: (LLVMIntTypeInContext:=dll.LLVMIntTypeInContext).restype, LLVMIntTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMInt1Type(void)
try: (LLVMInt1Type:=dll.LLVMInt1Type).restype, LLVMInt1Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMInt8Type(void)
try: (LLVMInt8Type:=dll.LLVMInt8Type).restype, LLVMInt8Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMInt16Type(void)
try: (LLVMInt16Type:=dll.LLVMInt16Type).restype, LLVMInt16Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMInt32Type(void)
try: (LLVMInt32Type:=dll.LLVMInt32Type).restype, LLVMInt32Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMInt64Type(void)
try: (LLVMInt64Type:=dll.LLVMInt64Type).restype, LLVMInt64Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMInt128Type(void)
try: (LLVMInt128Type:=dll.LLVMInt128Type).restype, LLVMInt128Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMIntType(unsigned int NumBits)
try: (LLVMIntType:=dll.LLVMIntType).restype, LLVMIntType.argtypes = LLVMTypeRef, [ctypes.c_uint32]
except AttributeError: pass

# unsigned int LLVMGetIntTypeWidth(LLVMTypeRef IntegerTy)
try: (LLVMGetIntTypeWidth:=dll.LLVMGetIntTypeWidth).restype, LLVMGetIntTypeWidth.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

# LLVMTypeRef LLVMHalfTypeInContext(LLVMContextRef C)
try: (LLVMHalfTypeInContext:=dll.LLVMHalfTypeInContext).restype, LLVMHalfTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMBFloatTypeInContext(LLVMContextRef C)
try: (LLVMBFloatTypeInContext:=dll.LLVMBFloatTypeInContext).restype, LLVMBFloatTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMFloatTypeInContext(LLVMContextRef C)
try: (LLVMFloatTypeInContext:=dll.LLVMFloatTypeInContext).restype, LLVMFloatTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMDoubleTypeInContext(LLVMContextRef C)
try: (LLVMDoubleTypeInContext:=dll.LLVMDoubleTypeInContext).restype, LLVMDoubleTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMX86FP80TypeInContext(LLVMContextRef C)
try: (LLVMX86FP80TypeInContext:=dll.LLVMX86FP80TypeInContext).restype, LLVMX86FP80TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMFP128TypeInContext(LLVMContextRef C)
try: (LLVMFP128TypeInContext:=dll.LLVMFP128TypeInContext).restype, LLVMFP128TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMPPCFP128TypeInContext(LLVMContextRef C)
try: (LLVMPPCFP128TypeInContext:=dll.LLVMPPCFP128TypeInContext).restype, LLVMPPCFP128TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMHalfType(void)
try: (LLVMHalfType:=dll.LLVMHalfType).restype, LLVMHalfType.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMBFloatType(void)
try: (LLVMBFloatType:=dll.LLVMBFloatType).restype, LLVMBFloatType.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMFloatType(void)
try: (LLVMFloatType:=dll.LLVMFloatType).restype, LLVMFloatType.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMDoubleType(void)
try: (LLVMDoubleType:=dll.LLVMDoubleType).restype, LLVMDoubleType.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMX86FP80Type(void)
try: (LLVMX86FP80Type:=dll.LLVMX86FP80Type).restype, LLVMX86FP80Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMFP128Type(void)
try: (LLVMFP128Type:=dll.LLVMFP128Type).restype, LLVMFP128Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMPPCFP128Type(void)
try: (LLVMPPCFP128Type:=dll.LLVMPPCFP128Type).restype, LLVMPPCFP128Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMFunctionType(LLVMTypeRef ReturnType, LLVMTypeRef *ParamTypes, unsigned int ParamCount, LLVMBool IsVarArg)
try: (LLVMFunctionType:=dll.LLVMFunctionType).restype, LLVMFunctionType.argtypes = LLVMTypeRef, [LLVMTypeRef, ctypes.POINTER(LLVMTypeRef), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

# LLVMBool LLVMIsFunctionVarArg(LLVMTypeRef FunctionTy)
try: (LLVMIsFunctionVarArg:=dll.LLVMIsFunctionVarArg).restype, LLVMIsFunctionVarArg.argtypes = LLVMBool, [LLVMTypeRef]
except AttributeError: pass

# LLVMTypeRef LLVMGetReturnType(LLVMTypeRef FunctionTy)
try: (LLVMGetReturnType:=dll.LLVMGetReturnType).restype, LLVMGetReturnType.argtypes = LLVMTypeRef, [LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMCountParamTypes(LLVMTypeRef FunctionTy)
try: (LLVMCountParamTypes:=dll.LLVMCountParamTypes).restype, LLVMCountParamTypes.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

# void LLVMGetParamTypes(LLVMTypeRef FunctionTy, LLVMTypeRef *Dest)
try: (LLVMGetParamTypes:=dll.LLVMGetParamTypes).restype, LLVMGetParamTypes.argtypes = None, [LLVMTypeRef, ctypes.POINTER(LLVMTypeRef)]
except AttributeError: pass

# LLVMTypeRef LLVMStructTypeInContext(LLVMContextRef C, LLVMTypeRef *ElementTypes, unsigned int ElementCount, LLVMBool Packed)
try: (LLVMStructTypeInContext:=dll.LLVMStructTypeInContext).restype, LLVMStructTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, ctypes.POINTER(LLVMTypeRef), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

# LLVMTypeRef LLVMStructType(LLVMTypeRef *ElementTypes, unsigned int ElementCount, LLVMBool Packed)
try: (LLVMStructType:=dll.LLVMStructType).restype, LLVMStructType.argtypes = LLVMTypeRef, [ctypes.POINTER(LLVMTypeRef), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

# LLVMTypeRef LLVMStructCreateNamed(LLVMContextRef C, const char *Name)
try: (LLVMStructCreateNamed:=dll.LLVMStructCreateNamed).restype, LLVMStructCreateNamed.argtypes = LLVMTypeRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# const char *LLVMGetStructName(LLVMTypeRef Ty)
try: (LLVMGetStructName:=dll.LLVMGetStructName).restype, LLVMGetStructName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTypeRef]
except AttributeError: pass

# void LLVMStructSetBody(LLVMTypeRef StructTy, LLVMTypeRef *ElementTypes, unsigned int ElementCount, LLVMBool Packed)
try: (LLVMStructSetBody:=dll.LLVMStructSetBody).restype, LLVMStructSetBody.argtypes = None, [LLVMTypeRef, ctypes.POINTER(LLVMTypeRef), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

# unsigned int LLVMCountStructElementTypes(LLVMTypeRef StructTy)
try: (LLVMCountStructElementTypes:=dll.LLVMCountStructElementTypes).restype, LLVMCountStructElementTypes.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

# void LLVMGetStructElementTypes(LLVMTypeRef StructTy, LLVMTypeRef *Dest)
try: (LLVMGetStructElementTypes:=dll.LLVMGetStructElementTypes).restype, LLVMGetStructElementTypes.argtypes = None, [LLVMTypeRef, ctypes.POINTER(LLVMTypeRef)]
except AttributeError: pass

# LLVMTypeRef LLVMStructGetTypeAtIndex(LLVMTypeRef StructTy, unsigned int i)
try: (LLVMStructGetTypeAtIndex:=dll.LLVMStructGetTypeAtIndex).restype, LLVMStructGetTypeAtIndex.argtypes = LLVMTypeRef, [LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMBool LLVMIsPackedStruct(LLVMTypeRef StructTy)
try: (LLVMIsPackedStruct:=dll.LLVMIsPackedStruct).restype, LLVMIsPackedStruct.argtypes = LLVMBool, [LLVMTypeRef]
except AttributeError: pass

# LLVMBool LLVMIsOpaqueStruct(LLVMTypeRef StructTy)
try: (LLVMIsOpaqueStruct:=dll.LLVMIsOpaqueStruct).restype, LLVMIsOpaqueStruct.argtypes = LLVMBool, [LLVMTypeRef]
except AttributeError: pass

# LLVMBool LLVMIsLiteralStruct(LLVMTypeRef StructTy)
try: (LLVMIsLiteralStruct:=dll.LLVMIsLiteralStruct).restype, LLVMIsLiteralStruct.argtypes = LLVMBool, [LLVMTypeRef]
except AttributeError: pass

# LLVMTypeRef LLVMGetElementType(LLVMTypeRef Ty)
try: (LLVMGetElementType:=dll.LLVMGetElementType).restype, LLVMGetElementType.argtypes = LLVMTypeRef, [LLVMTypeRef]
except AttributeError: pass

# void LLVMGetSubtypes(LLVMTypeRef Tp, LLVMTypeRef *Arr)
try: (LLVMGetSubtypes:=dll.LLVMGetSubtypes).restype, LLVMGetSubtypes.argtypes = None, [LLVMTypeRef, ctypes.POINTER(LLVMTypeRef)]
except AttributeError: pass

# unsigned int LLVMGetNumContainedTypes(LLVMTypeRef Tp)
try: (LLVMGetNumContainedTypes:=dll.LLVMGetNumContainedTypes).restype, LLVMGetNumContainedTypes.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

# LLVMTypeRef LLVMArrayType(LLVMTypeRef ElementType, unsigned int ElementCount)
try: (LLVMArrayType:=dll.LLVMArrayType).restype, LLVMArrayType.argtypes = LLVMTypeRef, [LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMArrayType2(LLVMTypeRef ElementType, uint64_t ElementCount)
try: (LLVMArrayType2:=dll.LLVMArrayType2).restype, LLVMArrayType2.argtypes = LLVMTypeRef, [LLVMTypeRef, uint64_t]
except AttributeError: pass

# unsigned int LLVMGetArrayLength(LLVMTypeRef ArrayTy)
try: (LLVMGetArrayLength:=dll.LLVMGetArrayLength).restype, LLVMGetArrayLength.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

# uint64_t LLVMGetArrayLength2(LLVMTypeRef ArrayTy)
try: (LLVMGetArrayLength2:=dll.LLVMGetArrayLength2).restype, LLVMGetArrayLength2.argtypes = uint64_t, [LLVMTypeRef]
except AttributeError: pass

# LLVMTypeRef LLVMPointerType(LLVMTypeRef ElementType, unsigned int AddressSpace)
try: (LLVMPointerType:=dll.LLVMPointerType).restype, LLVMPointerType.argtypes = LLVMTypeRef, [LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMBool LLVMPointerTypeIsOpaque(LLVMTypeRef Ty)
try: (LLVMPointerTypeIsOpaque:=dll.LLVMPointerTypeIsOpaque).restype, LLVMPointerTypeIsOpaque.argtypes = LLVMBool, [LLVMTypeRef]
except AttributeError: pass

# LLVMTypeRef LLVMPointerTypeInContext(LLVMContextRef C, unsigned int AddressSpace)
try: (LLVMPointerTypeInContext:=dll.LLVMPointerTypeInContext).restype, LLVMPointerTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, ctypes.c_uint32]
except AttributeError: pass

# unsigned int LLVMGetPointerAddressSpace(LLVMTypeRef PointerTy)
try: (LLVMGetPointerAddressSpace:=dll.LLVMGetPointerAddressSpace).restype, LLVMGetPointerAddressSpace.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

# LLVMTypeRef LLVMVectorType(LLVMTypeRef ElementType, unsigned int ElementCount)
try: (LLVMVectorType:=dll.LLVMVectorType).restype, LLVMVectorType.argtypes = LLVMTypeRef, [LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMScalableVectorType(LLVMTypeRef ElementType, unsigned int ElementCount)
try: (LLVMScalableVectorType:=dll.LLVMScalableVectorType).restype, LLVMScalableVectorType.argtypes = LLVMTypeRef, [LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

# unsigned int LLVMGetVectorSize(LLVMTypeRef VectorTy)
try: (LLVMGetVectorSize:=dll.LLVMGetVectorSize).restype, LLVMGetVectorSize.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMGetConstantPtrAuthPointer(LLVMValueRef PtrAuth)
try: (LLVMGetConstantPtrAuthPointer:=dll.LLVMGetConstantPtrAuthPointer).restype, LLVMGetConstantPtrAuthPointer.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetConstantPtrAuthKey(LLVMValueRef PtrAuth)
try: (LLVMGetConstantPtrAuthKey:=dll.LLVMGetConstantPtrAuthKey).restype, LLVMGetConstantPtrAuthKey.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetConstantPtrAuthDiscriminator(LLVMValueRef PtrAuth)
try: (LLVMGetConstantPtrAuthDiscriminator:=dll.LLVMGetConstantPtrAuthDiscriminator).restype, LLVMGetConstantPtrAuthDiscriminator.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetConstantPtrAuthAddrDiscriminator(LLVMValueRef PtrAuth)
try: (LLVMGetConstantPtrAuthAddrDiscriminator:=dll.LLVMGetConstantPtrAuthAddrDiscriminator).restype, LLVMGetConstantPtrAuthAddrDiscriminator.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMTypeRef LLVMVoidTypeInContext(LLVMContextRef C)
try: (LLVMVoidTypeInContext:=dll.LLVMVoidTypeInContext).restype, LLVMVoidTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMLabelTypeInContext(LLVMContextRef C)
try: (LLVMLabelTypeInContext:=dll.LLVMLabelTypeInContext).restype, LLVMLabelTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMX86AMXTypeInContext(LLVMContextRef C)
try: (LLVMX86AMXTypeInContext:=dll.LLVMX86AMXTypeInContext).restype, LLVMX86AMXTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMTokenTypeInContext(LLVMContextRef C)
try: (LLVMTokenTypeInContext:=dll.LLVMTokenTypeInContext).restype, LLVMTokenTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMMetadataTypeInContext(LLVMContextRef C)
try: (LLVMMetadataTypeInContext:=dll.LLVMMetadataTypeInContext).restype, LLVMMetadataTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

# LLVMTypeRef LLVMVoidType(void)
try: (LLVMVoidType:=dll.LLVMVoidType).restype, LLVMVoidType.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMLabelType(void)
try: (LLVMLabelType:=dll.LLVMLabelType).restype, LLVMLabelType.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMX86AMXType(void)
try: (LLVMX86AMXType:=dll.LLVMX86AMXType).restype, LLVMX86AMXType.argtypes = LLVMTypeRef, []
except AttributeError: pass

# LLVMTypeRef LLVMTargetExtTypeInContext(LLVMContextRef C, const char *Name, LLVMTypeRef *TypeParams, unsigned int TypeParamCount, unsigned int *IntParams, unsigned int IntParamCount)
try: (LLVMTargetExtTypeInContext:=dll.LLVMTargetExtTypeInContext).restype, LLVMTargetExtTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTypeRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
except AttributeError: pass

# const char *LLVMGetTargetExtTypeName(LLVMTypeRef TargetExtTy)
try: (LLVMGetTargetExtTypeName:=dll.LLVMGetTargetExtTypeName).restype, LLVMGetTargetExtTypeName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMGetTargetExtTypeNumTypeParams(LLVMTypeRef TargetExtTy)
try: (LLVMGetTargetExtTypeNumTypeParams:=dll.LLVMGetTargetExtTypeNumTypeParams).restype, LLVMGetTargetExtTypeNumTypeParams.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

# LLVMTypeRef LLVMGetTargetExtTypeTypeParam(LLVMTypeRef TargetExtTy, unsigned int Idx)
try: (LLVMGetTargetExtTypeTypeParam:=dll.LLVMGetTargetExtTypeTypeParam).restype, LLVMGetTargetExtTypeTypeParam.argtypes = LLVMTypeRef, [LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

# unsigned int LLVMGetTargetExtTypeNumIntParams(LLVMTypeRef TargetExtTy)
try: (LLVMGetTargetExtTypeNumIntParams:=dll.LLVMGetTargetExtTypeNumIntParams).restype, LLVMGetTargetExtTypeNumIntParams.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMGetTargetExtTypeIntParam(LLVMTypeRef TargetExtTy, unsigned int Idx)
try: (LLVMGetTargetExtTypeIntParam:=dll.LLVMGetTargetExtTypeIntParam).restype, LLVMGetTargetExtTypeIntParam.argtypes = ctypes.c_uint32, [LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMTypeOf(LLVMValueRef Val)
try: (LLVMTypeOf:=dll.LLVMTypeOf).restype, LLVMTypeOf.argtypes = LLVMTypeRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueKind LLVMGetValueKind(LLVMValueRef Val)
try: (LLVMGetValueKind:=dll.LLVMGetValueKind).restype, LLVMGetValueKind.argtypes = LLVMValueKind, [LLVMValueRef]
except AttributeError: pass

# const char *LLVMGetValueName2(LLVMValueRef Val, size_t *Length)
try: (LLVMGetValueName2:=dll.LLVMGetValueName2).restype, LLVMGetValueName2.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# void LLVMSetValueName2(LLVMValueRef Val, const char *Name, size_t NameLen)
try: (LLVMSetValueName2:=dll.LLVMSetValueName2).restype, LLVMSetValueName2.argtypes = None, [LLVMValueRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# void LLVMDumpValue(LLVMValueRef Val)
try: (LLVMDumpValue:=dll.LLVMDumpValue).restype, LLVMDumpValue.argtypes = None, [LLVMValueRef]
except AttributeError: pass

# char *LLVMPrintValueToString(LLVMValueRef Val)
try: (LLVMPrintValueToString:=dll.LLVMPrintValueToString).restype, LLVMPrintValueToString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef]
except AttributeError: pass

# LLVMContextRef LLVMGetValueContext(LLVMValueRef Val)
try: (LLVMGetValueContext:=dll.LLVMGetValueContext).restype, LLVMGetValueContext.argtypes = LLVMContextRef, [LLVMValueRef]
except AttributeError: pass

class struct_LLVMOpaqueDbgRecord(Struct): pass
LLVMDbgRecordRef = ctypes.POINTER(struct_LLVMOpaqueDbgRecord)
# char *LLVMPrintDbgRecordToString(LLVMDbgRecordRef Record)
try: (LLVMPrintDbgRecordToString:=dll.LLVMPrintDbgRecordToString).restype, LLVMPrintDbgRecordToString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMDbgRecordRef]
except AttributeError: pass

# void LLVMReplaceAllUsesWith(LLVMValueRef OldVal, LLVMValueRef NewVal)
try: (LLVMReplaceAllUsesWith:=dll.LLVMReplaceAllUsesWith).restype, LLVMReplaceAllUsesWith.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMIsConstant(LLVMValueRef Val)
try: (LLVMIsConstant:=dll.LLVMIsConstant).restype, LLVMIsConstant.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMIsUndef(LLVMValueRef Val)
try: (LLVMIsUndef:=dll.LLVMIsUndef).restype, LLVMIsUndef.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMIsPoison(LLVMValueRef Val)
try: (LLVMIsPoison:=dll.LLVMIsPoison).restype, LLVMIsPoison.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAArgument(LLVMValueRef Val)
try: (LLVMIsAArgument:=dll.LLVMIsAArgument).restype, LLVMIsAArgument.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsABasicBlock(LLVMValueRef Val)
try: (LLVMIsABasicBlock:=dll.LLVMIsABasicBlock).restype, LLVMIsABasicBlock.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAInlineAsm(LLVMValueRef Val)
try: (LLVMIsAInlineAsm:=dll.LLVMIsAInlineAsm).restype, LLVMIsAInlineAsm.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAUser(LLVMValueRef Val)
try: (LLVMIsAUser:=dll.LLVMIsAUser).restype, LLVMIsAUser.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAConstant(LLVMValueRef Val)
try: (LLVMIsAConstant:=dll.LLVMIsAConstant).restype, LLVMIsAConstant.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsABlockAddress(LLVMValueRef Val)
try: (LLVMIsABlockAddress:=dll.LLVMIsABlockAddress).restype, LLVMIsABlockAddress.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAConstantAggregateZero(LLVMValueRef Val)
try: (LLVMIsAConstantAggregateZero:=dll.LLVMIsAConstantAggregateZero).restype, LLVMIsAConstantAggregateZero.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAConstantArray(LLVMValueRef Val)
try: (LLVMIsAConstantArray:=dll.LLVMIsAConstantArray).restype, LLVMIsAConstantArray.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAConstantDataSequential(LLVMValueRef Val)
try: (LLVMIsAConstantDataSequential:=dll.LLVMIsAConstantDataSequential).restype, LLVMIsAConstantDataSequential.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAConstantDataArray(LLVMValueRef Val)
try: (LLVMIsAConstantDataArray:=dll.LLVMIsAConstantDataArray).restype, LLVMIsAConstantDataArray.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAConstantDataVector(LLVMValueRef Val)
try: (LLVMIsAConstantDataVector:=dll.LLVMIsAConstantDataVector).restype, LLVMIsAConstantDataVector.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAConstantExpr(LLVMValueRef Val)
try: (LLVMIsAConstantExpr:=dll.LLVMIsAConstantExpr).restype, LLVMIsAConstantExpr.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAConstantFP(LLVMValueRef Val)
try: (LLVMIsAConstantFP:=dll.LLVMIsAConstantFP).restype, LLVMIsAConstantFP.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAConstantInt(LLVMValueRef Val)
try: (LLVMIsAConstantInt:=dll.LLVMIsAConstantInt).restype, LLVMIsAConstantInt.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAConstantPointerNull(LLVMValueRef Val)
try: (LLVMIsAConstantPointerNull:=dll.LLVMIsAConstantPointerNull).restype, LLVMIsAConstantPointerNull.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAConstantStruct(LLVMValueRef Val)
try: (LLVMIsAConstantStruct:=dll.LLVMIsAConstantStruct).restype, LLVMIsAConstantStruct.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAConstantTokenNone(LLVMValueRef Val)
try: (LLVMIsAConstantTokenNone:=dll.LLVMIsAConstantTokenNone).restype, LLVMIsAConstantTokenNone.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAConstantVector(LLVMValueRef Val)
try: (LLVMIsAConstantVector:=dll.LLVMIsAConstantVector).restype, LLVMIsAConstantVector.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAConstantPtrAuth(LLVMValueRef Val)
try: (LLVMIsAConstantPtrAuth:=dll.LLVMIsAConstantPtrAuth).restype, LLVMIsAConstantPtrAuth.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAGlobalValue(LLVMValueRef Val)
try: (LLVMIsAGlobalValue:=dll.LLVMIsAGlobalValue).restype, LLVMIsAGlobalValue.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAGlobalAlias(LLVMValueRef Val)
try: (LLVMIsAGlobalAlias:=dll.LLVMIsAGlobalAlias).restype, LLVMIsAGlobalAlias.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAGlobalObject(LLVMValueRef Val)
try: (LLVMIsAGlobalObject:=dll.LLVMIsAGlobalObject).restype, LLVMIsAGlobalObject.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAFunction(LLVMValueRef Val)
try: (LLVMIsAFunction:=dll.LLVMIsAFunction).restype, LLVMIsAFunction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAGlobalVariable(LLVMValueRef Val)
try: (LLVMIsAGlobalVariable:=dll.LLVMIsAGlobalVariable).restype, LLVMIsAGlobalVariable.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAGlobalIFunc(LLVMValueRef Val)
try: (LLVMIsAGlobalIFunc:=dll.LLVMIsAGlobalIFunc).restype, LLVMIsAGlobalIFunc.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAUndefValue(LLVMValueRef Val)
try: (LLVMIsAUndefValue:=dll.LLVMIsAUndefValue).restype, LLVMIsAUndefValue.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAPoisonValue(LLVMValueRef Val)
try: (LLVMIsAPoisonValue:=dll.LLVMIsAPoisonValue).restype, LLVMIsAPoisonValue.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAInstruction(LLVMValueRef Val)
try: (LLVMIsAInstruction:=dll.LLVMIsAInstruction).restype, LLVMIsAInstruction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAUnaryOperator(LLVMValueRef Val)
try: (LLVMIsAUnaryOperator:=dll.LLVMIsAUnaryOperator).restype, LLVMIsAUnaryOperator.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsABinaryOperator(LLVMValueRef Val)
try: (LLVMIsABinaryOperator:=dll.LLVMIsABinaryOperator).restype, LLVMIsABinaryOperator.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsACallInst(LLVMValueRef Val)
try: (LLVMIsACallInst:=dll.LLVMIsACallInst).restype, LLVMIsACallInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAIntrinsicInst(LLVMValueRef Val)
try: (LLVMIsAIntrinsicInst:=dll.LLVMIsAIntrinsicInst).restype, LLVMIsAIntrinsicInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsADbgInfoIntrinsic(LLVMValueRef Val)
try: (LLVMIsADbgInfoIntrinsic:=dll.LLVMIsADbgInfoIntrinsic).restype, LLVMIsADbgInfoIntrinsic.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsADbgVariableIntrinsic(LLVMValueRef Val)
try: (LLVMIsADbgVariableIntrinsic:=dll.LLVMIsADbgVariableIntrinsic).restype, LLVMIsADbgVariableIntrinsic.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsADbgDeclareInst(LLVMValueRef Val)
try: (LLVMIsADbgDeclareInst:=dll.LLVMIsADbgDeclareInst).restype, LLVMIsADbgDeclareInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsADbgLabelInst(LLVMValueRef Val)
try: (LLVMIsADbgLabelInst:=dll.LLVMIsADbgLabelInst).restype, LLVMIsADbgLabelInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAMemIntrinsic(LLVMValueRef Val)
try: (LLVMIsAMemIntrinsic:=dll.LLVMIsAMemIntrinsic).restype, LLVMIsAMemIntrinsic.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAMemCpyInst(LLVMValueRef Val)
try: (LLVMIsAMemCpyInst:=dll.LLVMIsAMemCpyInst).restype, LLVMIsAMemCpyInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAMemMoveInst(LLVMValueRef Val)
try: (LLVMIsAMemMoveInst:=dll.LLVMIsAMemMoveInst).restype, LLVMIsAMemMoveInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAMemSetInst(LLVMValueRef Val)
try: (LLVMIsAMemSetInst:=dll.LLVMIsAMemSetInst).restype, LLVMIsAMemSetInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsACmpInst(LLVMValueRef Val)
try: (LLVMIsACmpInst:=dll.LLVMIsACmpInst).restype, LLVMIsACmpInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAFCmpInst(LLVMValueRef Val)
try: (LLVMIsAFCmpInst:=dll.LLVMIsAFCmpInst).restype, LLVMIsAFCmpInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAICmpInst(LLVMValueRef Val)
try: (LLVMIsAICmpInst:=dll.LLVMIsAICmpInst).restype, LLVMIsAICmpInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAExtractElementInst(LLVMValueRef Val)
try: (LLVMIsAExtractElementInst:=dll.LLVMIsAExtractElementInst).restype, LLVMIsAExtractElementInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAGetElementPtrInst(LLVMValueRef Val)
try: (LLVMIsAGetElementPtrInst:=dll.LLVMIsAGetElementPtrInst).restype, LLVMIsAGetElementPtrInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAInsertElementInst(LLVMValueRef Val)
try: (LLVMIsAInsertElementInst:=dll.LLVMIsAInsertElementInst).restype, LLVMIsAInsertElementInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAInsertValueInst(LLVMValueRef Val)
try: (LLVMIsAInsertValueInst:=dll.LLVMIsAInsertValueInst).restype, LLVMIsAInsertValueInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsALandingPadInst(LLVMValueRef Val)
try: (LLVMIsALandingPadInst:=dll.LLVMIsALandingPadInst).restype, LLVMIsALandingPadInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAPHINode(LLVMValueRef Val)
try: (LLVMIsAPHINode:=dll.LLVMIsAPHINode).restype, LLVMIsAPHINode.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsASelectInst(LLVMValueRef Val)
try: (LLVMIsASelectInst:=dll.LLVMIsASelectInst).restype, LLVMIsASelectInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAShuffleVectorInst(LLVMValueRef Val)
try: (LLVMIsAShuffleVectorInst:=dll.LLVMIsAShuffleVectorInst).restype, LLVMIsAShuffleVectorInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAStoreInst(LLVMValueRef Val)
try: (LLVMIsAStoreInst:=dll.LLVMIsAStoreInst).restype, LLVMIsAStoreInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsABranchInst(LLVMValueRef Val)
try: (LLVMIsABranchInst:=dll.LLVMIsABranchInst).restype, LLVMIsABranchInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAIndirectBrInst(LLVMValueRef Val)
try: (LLVMIsAIndirectBrInst:=dll.LLVMIsAIndirectBrInst).restype, LLVMIsAIndirectBrInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAInvokeInst(LLVMValueRef Val)
try: (LLVMIsAInvokeInst:=dll.LLVMIsAInvokeInst).restype, LLVMIsAInvokeInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAReturnInst(LLVMValueRef Val)
try: (LLVMIsAReturnInst:=dll.LLVMIsAReturnInst).restype, LLVMIsAReturnInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsASwitchInst(LLVMValueRef Val)
try: (LLVMIsASwitchInst:=dll.LLVMIsASwitchInst).restype, LLVMIsASwitchInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAUnreachableInst(LLVMValueRef Val)
try: (LLVMIsAUnreachableInst:=dll.LLVMIsAUnreachableInst).restype, LLVMIsAUnreachableInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAResumeInst(LLVMValueRef Val)
try: (LLVMIsAResumeInst:=dll.LLVMIsAResumeInst).restype, LLVMIsAResumeInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsACleanupReturnInst(LLVMValueRef Val)
try: (LLVMIsACleanupReturnInst:=dll.LLVMIsACleanupReturnInst).restype, LLVMIsACleanupReturnInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsACatchReturnInst(LLVMValueRef Val)
try: (LLVMIsACatchReturnInst:=dll.LLVMIsACatchReturnInst).restype, LLVMIsACatchReturnInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsACatchSwitchInst(LLVMValueRef Val)
try: (LLVMIsACatchSwitchInst:=dll.LLVMIsACatchSwitchInst).restype, LLVMIsACatchSwitchInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsACallBrInst(LLVMValueRef Val)
try: (LLVMIsACallBrInst:=dll.LLVMIsACallBrInst).restype, LLVMIsACallBrInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAFuncletPadInst(LLVMValueRef Val)
try: (LLVMIsAFuncletPadInst:=dll.LLVMIsAFuncletPadInst).restype, LLVMIsAFuncletPadInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsACatchPadInst(LLVMValueRef Val)
try: (LLVMIsACatchPadInst:=dll.LLVMIsACatchPadInst).restype, LLVMIsACatchPadInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsACleanupPadInst(LLVMValueRef Val)
try: (LLVMIsACleanupPadInst:=dll.LLVMIsACleanupPadInst).restype, LLVMIsACleanupPadInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAUnaryInstruction(LLVMValueRef Val)
try: (LLVMIsAUnaryInstruction:=dll.LLVMIsAUnaryInstruction).restype, LLVMIsAUnaryInstruction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAAllocaInst(LLVMValueRef Val)
try: (LLVMIsAAllocaInst:=dll.LLVMIsAAllocaInst).restype, LLVMIsAAllocaInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsACastInst(LLVMValueRef Val)
try: (LLVMIsACastInst:=dll.LLVMIsACastInst).restype, LLVMIsACastInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAAddrSpaceCastInst(LLVMValueRef Val)
try: (LLVMIsAAddrSpaceCastInst:=dll.LLVMIsAAddrSpaceCastInst).restype, LLVMIsAAddrSpaceCastInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsABitCastInst(LLVMValueRef Val)
try: (LLVMIsABitCastInst:=dll.LLVMIsABitCastInst).restype, LLVMIsABitCastInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAFPExtInst(LLVMValueRef Val)
try: (LLVMIsAFPExtInst:=dll.LLVMIsAFPExtInst).restype, LLVMIsAFPExtInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAFPToSIInst(LLVMValueRef Val)
try: (LLVMIsAFPToSIInst:=dll.LLVMIsAFPToSIInst).restype, LLVMIsAFPToSIInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAFPToUIInst(LLVMValueRef Val)
try: (LLVMIsAFPToUIInst:=dll.LLVMIsAFPToUIInst).restype, LLVMIsAFPToUIInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAFPTruncInst(LLVMValueRef Val)
try: (LLVMIsAFPTruncInst:=dll.LLVMIsAFPTruncInst).restype, LLVMIsAFPTruncInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAIntToPtrInst(LLVMValueRef Val)
try: (LLVMIsAIntToPtrInst:=dll.LLVMIsAIntToPtrInst).restype, LLVMIsAIntToPtrInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAPtrToIntInst(LLVMValueRef Val)
try: (LLVMIsAPtrToIntInst:=dll.LLVMIsAPtrToIntInst).restype, LLVMIsAPtrToIntInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsASExtInst(LLVMValueRef Val)
try: (LLVMIsASExtInst:=dll.LLVMIsASExtInst).restype, LLVMIsASExtInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsASIToFPInst(LLVMValueRef Val)
try: (LLVMIsASIToFPInst:=dll.LLVMIsASIToFPInst).restype, LLVMIsASIToFPInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsATruncInst(LLVMValueRef Val)
try: (LLVMIsATruncInst:=dll.LLVMIsATruncInst).restype, LLVMIsATruncInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAUIToFPInst(LLVMValueRef Val)
try: (LLVMIsAUIToFPInst:=dll.LLVMIsAUIToFPInst).restype, LLVMIsAUIToFPInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAZExtInst(LLVMValueRef Val)
try: (LLVMIsAZExtInst:=dll.LLVMIsAZExtInst).restype, LLVMIsAZExtInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAExtractValueInst(LLVMValueRef Val)
try: (LLVMIsAExtractValueInst:=dll.LLVMIsAExtractValueInst).restype, LLVMIsAExtractValueInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsALoadInst(LLVMValueRef Val)
try: (LLVMIsALoadInst:=dll.LLVMIsALoadInst).restype, LLVMIsALoadInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAVAArgInst(LLVMValueRef Val)
try: (LLVMIsAVAArgInst:=dll.LLVMIsAVAArgInst).restype, LLVMIsAVAArgInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAFreezeInst(LLVMValueRef Val)
try: (LLVMIsAFreezeInst:=dll.LLVMIsAFreezeInst).restype, LLVMIsAFreezeInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAAtomicCmpXchgInst(LLVMValueRef Val)
try: (LLVMIsAAtomicCmpXchgInst:=dll.LLVMIsAAtomicCmpXchgInst).restype, LLVMIsAAtomicCmpXchgInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAAtomicRMWInst(LLVMValueRef Val)
try: (LLVMIsAAtomicRMWInst:=dll.LLVMIsAAtomicRMWInst).restype, LLVMIsAAtomicRMWInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAFenceInst(LLVMValueRef Val)
try: (LLVMIsAFenceInst:=dll.LLVMIsAFenceInst).restype, LLVMIsAFenceInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAMDNode(LLVMValueRef Val)
try: (LLVMIsAMDNode:=dll.LLVMIsAMDNode).restype, LLVMIsAMDNode.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAValueAsMetadata(LLVMValueRef Val)
try: (LLVMIsAValueAsMetadata:=dll.LLVMIsAValueAsMetadata).restype, LLVMIsAValueAsMetadata.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsAMDString(LLVMValueRef Val)
try: (LLVMIsAMDString:=dll.LLVMIsAMDString).restype, LLVMIsAMDString.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# const char *LLVMGetValueName(LLVMValueRef Val)
try: (LLVMGetValueName:=dll.LLVMGetValueName).restype, LLVMGetValueName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef]
except AttributeError: pass

# void LLVMSetValueName(LLVMValueRef Val, const char *Name)
try: (LLVMSetValueName:=dll.LLVMSetValueName).restype, LLVMSetValueName.argtypes = None, [LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class struct_LLVMOpaqueUse(Struct): pass
LLVMUseRef = ctypes.POINTER(struct_LLVMOpaqueUse)
# LLVMUseRef LLVMGetFirstUse(LLVMValueRef Val)
try: (LLVMGetFirstUse:=dll.LLVMGetFirstUse).restype, LLVMGetFirstUse.argtypes = LLVMUseRef, [LLVMValueRef]
except AttributeError: pass

# LLVMUseRef LLVMGetNextUse(LLVMUseRef U)
try: (LLVMGetNextUse:=dll.LLVMGetNextUse).restype, LLVMGetNextUse.argtypes = LLVMUseRef, [LLVMUseRef]
except AttributeError: pass

# LLVMValueRef LLVMGetUser(LLVMUseRef U)
try: (LLVMGetUser:=dll.LLVMGetUser).restype, LLVMGetUser.argtypes = LLVMValueRef, [LLVMUseRef]
except AttributeError: pass

# LLVMValueRef LLVMGetUsedValue(LLVMUseRef U)
try: (LLVMGetUsedValue:=dll.LLVMGetUsedValue).restype, LLVMGetUsedValue.argtypes = LLVMValueRef, [LLVMUseRef]
except AttributeError: pass

# LLVMValueRef LLVMGetOperand(LLVMValueRef Val, unsigned int Index)
try: (LLVMGetOperand:=dll.LLVMGetOperand).restype, LLVMGetOperand.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMUseRef LLVMGetOperandUse(LLVMValueRef Val, unsigned int Index)
try: (LLVMGetOperandUse:=dll.LLVMGetOperandUse).restype, LLVMGetOperandUse.argtypes = LLVMUseRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# void LLVMSetOperand(LLVMValueRef User, unsigned int Index, LLVMValueRef Val)
try: (LLVMSetOperand:=dll.LLVMSetOperand).restype, LLVMSetOperand.argtypes = None, [LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError: pass

# int LLVMGetNumOperands(LLVMValueRef Val)
try: (LLVMGetNumOperands:=dll.LLVMGetNumOperands).restype, LLVMGetNumOperands.argtypes = ctypes.c_int32, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstNull(LLVMTypeRef Ty)
try: (LLVMConstNull:=dll.LLVMConstNull).restype, LLVMConstNull.argtypes = LLVMValueRef, [LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMConstAllOnes(LLVMTypeRef Ty)
try: (LLVMConstAllOnes:=dll.LLVMConstAllOnes).restype, LLVMConstAllOnes.argtypes = LLVMValueRef, [LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMGetUndef(LLVMTypeRef Ty)
try: (LLVMGetUndef:=dll.LLVMGetUndef).restype, LLVMGetUndef.argtypes = LLVMValueRef, [LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMGetPoison(LLVMTypeRef Ty)
try: (LLVMGetPoison:=dll.LLVMGetPoison).restype, LLVMGetPoison.argtypes = LLVMValueRef, [LLVMTypeRef]
except AttributeError: pass

# LLVMBool LLVMIsNull(LLVMValueRef Val)
try: (LLVMIsNull:=dll.LLVMIsNull).restype, LLVMIsNull.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstPointerNull(LLVMTypeRef Ty)
try: (LLVMConstPointerNull:=dll.LLVMConstPointerNull).restype, LLVMConstPointerNull.argtypes = LLVMValueRef, [LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMConstInt(LLVMTypeRef IntTy, unsigned long long N, LLVMBool SignExtend)
try: (LLVMConstInt:=dll.LLVMConstInt).restype, LLVMConstInt.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.c_uint64, LLVMBool]
except AttributeError: pass

# LLVMValueRef LLVMConstIntOfArbitraryPrecision(LLVMTypeRef IntTy, unsigned int NumWords, const uint64_t Words[])
try: (LLVMConstIntOfArbitraryPrecision:=dll.LLVMConstIntOfArbitraryPrecision).restype, LLVMConstIntOfArbitraryPrecision.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.c_uint32, (uint64_t * 0)]
except AttributeError: pass

uint8_t = ctypes.c_ubyte
# LLVMValueRef LLVMConstIntOfString(LLVMTypeRef IntTy, const char *Text, uint8_t Radix)
try: (LLVMConstIntOfString:=dll.LLVMConstIntOfString).restype, LLVMConstIntOfString.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), uint8_t]
except AttributeError: pass

# LLVMValueRef LLVMConstIntOfStringAndSize(LLVMTypeRef IntTy, const char *Text, unsigned int SLen, uint8_t Radix)
try: (LLVMConstIntOfStringAndSize:=dll.LLVMConstIntOfStringAndSize).restype, LLVMConstIntOfStringAndSize.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, uint8_t]
except AttributeError: pass

# LLVMValueRef LLVMConstReal(LLVMTypeRef RealTy, double N)
try: (LLVMConstReal:=dll.LLVMConstReal).restype, LLVMConstReal.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.c_double]
except AttributeError: pass

# LLVMValueRef LLVMConstRealOfString(LLVMTypeRef RealTy, const char *Text)
try: (LLVMConstRealOfString:=dll.LLVMConstRealOfString).restype, LLVMConstRealOfString.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMConstRealOfStringAndSize(LLVMTypeRef RealTy, const char *Text, unsigned int SLen)
try: (LLVMConstRealOfStringAndSize:=dll.LLVMConstRealOfStringAndSize).restype, LLVMConstRealOfStringAndSize.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

# unsigned long long LLVMConstIntGetZExtValue(LLVMValueRef ConstantVal)
try: (LLVMConstIntGetZExtValue:=dll.LLVMConstIntGetZExtValue).restype, LLVMConstIntGetZExtValue.argtypes = ctypes.c_uint64, [LLVMValueRef]
except AttributeError: pass

# long long LLVMConstIntGetSExtValue(LLVMValueRef ConstantVal)
try: (LLVMConstIntGetSExtValue:=dll.LLVMConstIntGetSExtValue).restype, LLVMConstIntGetSExtValue.argtypes = ctypes.c_int64, [LLVMValueRef]
except AttributeError: pass

# double LLVMConstRealGetDouble(LLVMValueRef ConstantVal, LLVMBool *losesInfo)
try: (LLVMConstRealGetDouble:=dll.LLVMConstRealGetDouble).restype, LLVMConstRealGetDouble.argtypes = ctypes.c_double, [LLVMValueRef, ctypes.POINTER(LLVMBool)]
except AttributeError: pass

# LLVMValueRef LLVMConstStringInContext(LLVMContextRef C, const char *Str, unsigned int Length, LLVMBool DontNullTerminate)
try: (LLVMConstStringInContext:=dll.LLVMConstStringInContext).restype, LLVMConstStringInContext.argtypes = LLVMValueRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

# LLVMValueRef LLVMConstStringInContext2(LLVMContextRef C, const char *Str, size_t Length, LLVMBool DontNullTerminate)
try: (LLVMConstStringInContext2:=dll.LLVMConstStringInContext2).restype, LLVMConstStringInContext2.argtypes = LLVMValueRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMBool]
except AttributeError: pass

# LLVMValueRef LLVMConstString(const char *Str, unsigned int Length, LLVMBool DontNullTerminate)
try: (LLVMConstString:=dll.LLVMConstString).restype, LLVMConstString.argtypes = LLVMValueRef, [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

# LLVMBool LLVMIsConstantString(LLVMValueRef c)
try: (LLVMIsConstantString:=dll.LLVMIsConstantString).restype, LLVMIsConstantString.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# const char *LLVMGetAsString(LLVMValueRef c, size_t *Length)
try: (LLVMGetAsString:=dll.LLVMGetAsString).restype, LLVMGetAsString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# LLVMValueRef LLVMConstStructInContext(LLVMContextRef C, LLVMValueRef *ConstantVals, unsigned int Count, LLVMBool Packed)
try: (LLVMConstStructInContext:=dll.LLVMConstStructInContext).restype, LLVMConstStructInContext.argtypes = LLVMValueRef, [LLVMContextRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

# LLVMValueRef LLVMConstStruct(LLVMValueRef *ConstantVals, unsigned int Count, LLVMBool Packed)
try: (LLVMConstStruct:=dll.LLVMConstStruct).restype, LLVMConstStruct.argtypes = LLVMValueRef, [ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

# LLVMValueRef LLVMConstArray(LLVMTypeRef ElementTy, LLVMValueRef *ConstantVals, unsigned int Length)
try: (LLVMConstArray:=dll.LLVMConstArray).restype, LLVMConstArray.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMConstArray2(LLVMTypeRef ElementTy, LLVMValueRef *ConstantVals, uint64_t Length)
try: (LLVMConstArray2:=dll.LLVMConstArray2).restype, LLVMConstArray2.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(LLVMValueRef), uint64_t]
except AttributeError: pass

# LLVMValueRef LLVMConstNamedStruct(LLVMTypeRef StructTy, LLVMValueRef *ConstantVals, unsigned int Count)
try: (LLVMConstNamedStruct:=dll.LLVMConstNamedStruct).restype, LLVMConstNamedStruct.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMGetAggregateElement(LLVMValueRef C, unsigned int Idx)
try: (LLVMGetAggregateElement:=dll.LLVMGetAggregateElement).restype, LLVMGetAggregateElement.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMGetElementAsConstant(LLVMValueRef C, unsigned int idx) __attribute__((deprecated("Use LLVMGetAggregateElement instead")))
try: (LLVMGetElementAsConstant:=dll.LLVMGetElementAsConstant).restype, LLVMGetElementAsConstant.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMConstVector(LLVMValueRef *ScalarConstantVals, unsigned int Size)
try: (LLVMConstVector:=dll.LLVMConstVector).restype, LLVMConstVector.argtypes = LLVMValueRef, [ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMConstantPtrAuth(LLVMValueRef Ptr, LLVMValueRef Key, LLVMValueRef Disc, LLVMValueRef AddrDisc)
try: (LLVMConstantPtrAuth:=dll.LLVMConstantPtrAuth).restype, LLVMConstantPtrAuth.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMOpcode LLVMGetConstOpcode(LLVMValueRef ConstantVal)
try: (LLVMGetConstOpcode:=dll.LLVMGetConstOpcode).restype, LLVMGetConstOpcode.argtypes = LLVMOpcode, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMAlignOf(LLVMTypeRef Ty)
try: (LLVMAlignOf:=dll.LLVMAlignOf).restype, LLVMAlignOf.argtypes = LLVMValueRef, [LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMSizeOf(LLVMTypeRef Ty)
try: (LLVMSizeOf:=dll.LLVMSizeOf).restype, LLVMSizeOf.argtypes = LLVMValueRef, [LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMConstNeg(LLVMValueRef ConstantVal)
try: (LLVMConstNeg:=dll.LLVMConstNeg).restype, LLVMConstNeg.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstNSWNeg(LLVMValueRef ConstantVal)
try: (LLVMConstNSWNeg:=dll.LLVMConstNSWNeg).restype, LLVMConstNSWNeg.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstNUWNeg(LLVMValueRef ConstantVal) __attribute__((deprecated("Use LLVMConstNull instead.")))
try: (LLVMConstNUWNeg:=dll.LLVMConstNUWNeg).restype, LLVMConstNUWNeg.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstNot(LLVMValueRef ConstantVal)
try: (LLVMConstNot:=dll.LLVMConstNot).restype, LLVMConstNot.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstAdd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)
try: (LLVMConstAdd:=dll.LLVMConstAdd).restype, LLVMConstAdd.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstNSWAdd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)
try: (LLVMConstNSWAdd:=dll.LLVMConstNSWAdd).restype, LLVMConstNSWAdd.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstNUWAdd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)
try: (LLVMConstNUWAdd:=dll.LLVMConstNUWAdd).restype, LLVMConstNUWAdd.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstSub(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)
try: (LLVMConstSub:=dll.LLVMConstSub).restype, LLVMConstSub.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstNSWSub(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)
try: (LLVMConstNSWSub:=dll.LLVMConstNSWSub).restype, LLVMConstNSWSub.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstNUWSub(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)
try: (LLVMConstNUWSub:=dll.LLVMConstNUWSub).restype, LLVMConstNUWSub.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstMul(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)
try: (LLVMConstMul:=dll.LLVMConstMul).restype, LLVMConstMul.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstNSWMul(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)
try: (LLVMConstNSWMul:=dll.LLVMConstNSWMul).restype, LLVMConstNSWMul.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstNUWMul(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)
try: (LLVMConstNUWMul:=dll.LLVMConstNUWMul).restype, LLVMConstNUWMul.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstXor(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)
try: (LLVMConstXor:=dll.LLVMConstXor).restype, LLVMConstXor.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstGEP2(LLVMTypeRef Ty, LLVMValueRef ConstantVal, LLVMValueRef *ConstantIndices, unsigned int NumIndices)
try: (LLVMConstGEP2:=dll.LLVMConstGEP2).restype, LLVMConstGEP2.argtypes = LLVMValueRef, [LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMConstInBoundsGEP2(LLVMTypeRef Ty, LLVMValueRef ConstantVal, LLVMValueRef *ConstantIndices, unsigned int NumIndices)
try: (LLVMConstInBoundsGEP2:=dll.LLVMConstInBoundsGEP2).restype, LLVMConstInBoundsGEP2.argtypes = LLVMValueRef, [LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMConstGEPWithNoWrapFlags(LLVMTypeRef Ty, LLVMValueRef ConstantVal, LLVMValueRef *ConstantIndices, unsigned int NumIndices, LLVMGEPNoWrapFlags NoWrapFlags)
try: (LLVMConstGEPWithNoWrapFlags:=dll.LLVMConstGEPWithNoWrapFlags).restype, LLVMConstGEPWithNoWrapFlags.argtypes = LLVMValueRef, [LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, LLVMGEPNoWrapFlags]
except AttributeError: pass

# LLVMValueRef LLVMConstTrunc(LLVMValueRef ConstantVal, LLVMTypeRef ToType)
try: (LLVMConstTrunc:=dll.LLVMConstTrunc).restype, LLVMConstTrunc.argtypes = LLVMValueRef, [LLVMValueRef, LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMConstPtrToInt(LLVMValueRef ConstantVal, LLVMTypeRef ToType)
try: (LLVMConstPtrToInt:=dll.LLVMConstPtrToInt).restype, LLVMConstPtrToInt.argtypes = LLVMValueRef, [LLVMValueRef, LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMConstIntToPtr(LLVMValueRef ConstantVal, LLVMTypeRef ToType)
try: (LLVMConstIntToPtr:=dll.LLVMConstIntToPtr).restype, LLVMConstIntToPtr.argtypes = LLVMValueRef, [LLVMValueRef, LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMConstBitCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType)
try: (LLVMConstBitCast:=dll.LLVMConstBitCast).restype, LLVMConstBitCast.argtypes = LLVMValueRef, [LLVMValueRef, LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMConstAddrSpaceCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType)
try: (LLVMConstAddrSpaceCast:=dll.LLVMConstAddrSpaceCast).restype, LLVMConstAddrSpaceCast.argtypes = LLVMValueRef, [LLVMValueRef, LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMConstTruncOrBitCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType)
try: (LLVMConstTruncOrBitCast:=dll.LLVMConstTruncOrBitCast).restype, LLVMConstTruncOrBitCast.argtypes = LLVMValueRef, [LLVMValueRef, LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMConstPointerCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType)
try: (LLVMConstPointerCast:=dll.LLVMConstPointerCast).restype, LLVMConstPointerCast.argtypes = LLVMValueRef, [LLVMValueRef, LLVMTypeRef]
except AttributeError: pass

# LLVMValueRef LLVMConstExtractElement(LLVMValueRef VectorConstant, LLVMValueRef IndexConstant)
try: (LLVMConstExtractElement:=dll.LLVMConstExtractElement).restype, LLVMConstExtractElement.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstInsertElement(LLVMValueRef VectorConstant, LLVMValueRef ElementValueConstant, LLVMValueRef IndexConstant)
try: (LLVMConstInsertElement:=dll.LLVMConstInsertElement).restype, LLVMConstInsertElement.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstShuffleVector(LLVMValueRef VectorAConstant, LLVMValueRef VectorBConstant, LLVMValueRef MaskConstant)
try: (LLVMConstShuffleVector:=dll.LLVMConstShuffleVector).restype, LLVMConstShuffleVector.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

class struct_LLVMOpaqueBasicBlock(Struct): pass
LLVMBasicBlockRef = ctypes.POINTER(struct_LLVMOpaqueBasicBlock)
# LLVMValueRef LLVMBlockAddress(LLVMValueRef F, LLVMBasicBlockRef BB)
try: (LLVMBlockAddress:=dll.LLVMBlockAddress).restype, LLVMBlockAddress.argtypes = LLVMValueRef, [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

# LLVMValueRef LLVMGetBlockAddressFunction(LLVMValueRef BlockAddr)
try: (LLVMGetBlockAddressFunction:=dll.LLVMGetBlockAddressFunction).restype, LLVMGetBlockAddressFunction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetBlockAddressBasicBlock(LLVMValueRef BlockAddr)
try: (LLVMGetBlockAddressBasicBlock:=dll.LLVMGetBlockAddressBasicBlock).restype, LLVMGetBlockAddressBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMConstInlineAsm(LLVMTypeRef Ty, const char *AsmString, const char *Constraints, LLVMBool HasSideEffects, LLVMBool IsAlignStack)
try: (LLVMConstInlineAsm:=dll.LLVMConstInlineAsm).restype, LLVMConstInlineAsm.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMBool, LLVMBool]
except AttributeError: pass

# LLVMModuleRef LLVMGetGlobalParent(LLVMValueRef Global)
try: (LLVMGetGlobalParent:=dll.LLVMGetGlobalParent).restype, LLVMGetGlobalParent.argtypes = LLVMModuleRef, [LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMIsDeclaration(LLVMValueRef Global)
try: (LLVMIsDeclaration:=dll.LLVMIsDeclaration).restype, LLVMIsDeclaration.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# LLVMLinkage LLVMGetLinkage(LLVMValueRef Global)
try: (LLVMGetLinkage:=dll.LLVMGetLinkage).restype, LLVMGetLinkage.argtypes = LLVMLinkage, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetLinkage(LLVMValueRef Global, LLVMLinkage Linkage)
try: (LLVMSetLinkage:=dll.LLVMSetLinkage).restype, LLVMSetLinkage.argtypes = None, [LLVMValueRef, LLVMLinkage]
except AttributeError: pass

# const char *LLVMGetSection(LLVMValueRef Global)
try: (LLVMGetSection:=dll.LLVMGetSection).restype, LLVMGetSection.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef]
except AttributeError: pass

# void LLVMSetSection(LLVMValueRef Global, const char *Section)
try: (LLVMSetSection:=dll.LLVMSetSection).restype, LLVMSetSection.argtypes = None, [LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMVisibility LLVMGetVisibility(LLVMValueRef Global)
try: (LLVMGetVisibility:=dll.LLVMGetVisibility).restype, LLVMGetVisibility.argtypes = LLVMVisibility, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetVisibility(LLVMValueRef Global, LLVMVisibility Viz)
try: (LLVMSetVisibility:=dll.LLVMSetVisibility).restype, LLVMSetVisibility.argtypes = None, [LLVMValueRef, LLVMVisibility]
except AttributeError: pass

# LLVMDLLStorageClass LLVMGetDLLStorageClass(LLVMValueRef Global)
try: (LLVMGetDLLStorageClass:=dll.LLVMGetDLLStorageClass).restype, LLVMGetDLLStorageClass.argtypes = LLVMDLLStorageClass, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetDLLStorageClass(LLVMValueRef Global, LLVMDLLStorageClass Class)
try: (LLVMSetDLLStorageClass:=dll.LLVMSetDLLStorageClass).restype, LLVMSetDLLStorageClass.argtypes = None, [LLVMValueRef, LLVMDLLStorageClass]
except AttributeError: pass

# LLVMUnnamedAddr LLVMGetUnnamedAddress(LLVMValueRef Global)
try: (LLVMGetUnnamedAddress:=dll.LLVMGetUnnamedAddress).restype, LLVMGetUnnamedAddress.argtypes = LLVMUnnamedAddr, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetUnnamedAddress(LLVMValueRef Global, LLVMUnnamedAddr UnnamedAddr)
try: (LLVMSetUnnamedAddress:=dll.LLVMSetUnnamedAddress).restype, LLVMSetUnnamedAddress.argtypes = None, [LLVMValueRef, LLVMUnnamedAddr]
except AttributeError: pass

# LLVMTypeRef LLVMGlobalGetValueType(LLVMValueRef Global)
try: (LLVMGlobalGetValueType:=dll.LLVMGlobalGetValueType).restype, LLVMGlobalGetValueType.argtypes = LLVMTypeRef, [LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMHasUnnamedAddr(LLVMValueRef Global)
try: (LLVMHasUnnamedAddr:=dll.LLVMHasUnnamedAddr).restype, LLVMHasUnnamedAddr.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetUnnamedAddr(LLVMValueRef Global, LLVMBool HasUnnamedAddr)
try: (LLVMSetUnnamedAddr:=dll.LLVMSetUnnamedAddr).restype, LLVMSetUnnamedAddr.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# unsigned int LLVMGetAlignment(LLVMValueRef V)
try: (LLVMGetAlignment:=dll.LLVMGetAlignment).restype, LLVMGetAlignment.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetAlignment(LLVMValueRef V, unsigned int Bytes)
try: (LLVMSetAlignment:=dll.LLVMSetAlignment).restype, LLVMSetAlignment.argtypes = None, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# void LLVMGlobalSetMetadata(LLVMValueRef Global, unsigned int Kind, LLVMMetadataRef MD)
try: (LLVMGlobalSetMetadata:=dll.LLVMGlobalSetMetadata).restype, LLVMGlobalSetMetadata.argtypes = None, [LLVMValueRef, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError: pass

# void LLVMGlobalEraseMetadata(LLVMValueRef Global, unsigned int Kind)
try: (LLVMGlobalEraseMetadata:=dll.LLVMGlobalEraseMetadata).restype, LLVMGlobalEraseMetadata.argtypes = None, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# void LLVMGlobalClearMetadata(LLVMValueRef Global)
try: (LLVMGlobalClearMetadata:=dll.LLVMGlobalClearMetadata).restype, LLVMGlobalClearMetadata.argtypes = None, [LLVMValueRef]
except AttributeError: pass

class struct_LLVMOpaqueValueMetadataEntry(Struct): pass
LLVMValueMetadataEntry = struct_LLVMOpaqueValueMetadataEntry
# LLVMValueMetadataEntry *LLVMGlobalCopyAllMetadata(LLVMValueRef Value, size_t *NumEntries)
try: (LLVMGlobalCopyAllMetadata:=dll.LLVMGlobalCopyAllMetadata).restype, LLVMGlobalCopyAllMetadata.argtypes = ctypes.POINTER(LLVMValueMetadataEntry), [LLVMValueRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# void LLVMDisposeValueMetadataEntries(LLVMValueMetadataEntry *Entries)
try: (LLVMDisposeValueMetadataEntries:=dll.LLVMDisposeValueMetadataEntries).restype, LLVMDisposeValueMetadataEntries.argtypes = None, [ctypes.POINTER(LLVMValueMetadataEntry)]
except AttributeError: pass

# unsigned int LLVMValueMetadataEntriesGetKind(LLVMValueMetadataEntry *Entries, unsigned int Index)
try: (LLVMValueMetadataEntriesGetKind:=dll.LLVMValueMetadataEntriesGetKind).restype, LLVMValueMetadataEntriesGetKind.argtypes = ctypes.c_uint32, [ctypes.POINTER(LLVMValueMetadataEntry), ctypes.c_uint32]
except AttributeError: pass

# LLVMMetadataRef LLVMValueMetadataEntriesGetMetadata(LLVMValueMetadataEntry *Entries, unsigned int Index)
try: (LLVMValueMetadataEntriesGetMetadata:=dll.LLVMValueMetadataEntriesGetMetadata).restype, LLVMValueMetadataEntriesGetMetadata.argtypes = LLVMMetadataRef, [ctypes.POINTER(LLVMValueMetadataEntry), ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMAddGlobal(LLVMModuleRef M, LLVMTypeRef Ty, const char *Name)
try: (LLVMAddGlobal:=dll.LLVMAddGlobal).restype, LLVMAddGlobal.argtypes = LLVMValueRef, [LLVMModuleRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMAddGlobalInAddressSpace(LLVMModuleRef M, LLVMTypeRef Ty, const char *Name, unsigned int AddressSpace)
try: (LLVMAddGlobalInAddressSpace:=dll.LLVMAddGlobalInAddressSpace).restype, LLVMAddGlobalInAddressSpace.argtypes = LLVMValueRef, [LLVMModuleRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMGetNamedGlobal(LLVMModuleRef M, const char *Name)
try: (LLVMGetNamedGlobal:=dll.LLVMGetNamedGlobal).restype, LLVMGetNamedGlobal.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMGetNamedGlobalWithLength(LLVMModuleRef M, const char *Name, size_t Length)
try: (LLVMGetNamedGlobalWithLength:=dll.LLVMGetNamedGlobalWithLength).restype, LLVMGetNamedGlobalWithLength.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMValueRef LLVMGetFirstGlobal(LLVMModuleRef M)
try: (LLVMGetFirstGlobal:=dll.LLVMGetFirstGlobal).restype, LLVMGetFirstGlobal.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

# LLVMValueRef LLVMGetLastGlobal(LLVMModuleRef M)
try: (LLVMGetLastGlobal:=dll.LLVMGetLastGlobal).restype, LLVMGetLastGlobal.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

# LLVMValueRef LLVMGetNextGlobal(LLVMValueRef GlobalVar)
try: (LLVMGetNextGlobal:=dll.LLVMGetNextGlobal).restype, LLVMGetNextGlobal.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetPreviousGlobal(LLVMValueRef GlobalVar)
try: (LLVMGetPreviousGlobal:=dll.LLVMGetPreviousGlobal).restype, LLVMGetPreviousGlobal.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# void LLVMDeleteGlobal(LLVMValueRef GlobalVar)
try: (LLVMDeleteGlobal:=dll.LLVMDeleteGlobal).restype, LLVMDeleteGlobal.argtypes = None, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetInitializer(LLVMValueRef GlobalVar)
try: (LLVMGetInitializer:=dll.LLVMGetInitializer).restype, LLVMGetInitializer.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetInitializer(LLVMValueRef GlobalVar, LLVMValueRef ConstantVal)
try: (LLVMSetInitializer:=dll.LLVMSetInitializer).restype, LLVMSetInitializer.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMIsThreadLocal(LLVMValueRef GlobalVar)
try: (LLVMIsThreadLocal:=dll.LLVMIsThreadLocal).restype, LLVMIsThreadLocal.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetThreadLocal(LLVMValueRef GlobalVar, LLVMBool IsThreadLocal)
try: (LLVMSetThreadLocal:=dll.LLVMSetThreadLocal).restype, LLVMSetThreadLocal.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# LLVMBool LLVMIsGlobalConstant(LLVMValueRef GlobalVar)
try: (LLVMIsGlobalConstant:=dll.LLVMIsGlobalConstant).restype, LLVMIsGlobalConstant.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetGlobalConstant(LLVMValueRef GlobalVar, LLVMBool IsConstant)
try: (LLVMSetGlobalConstant:=dll.LLVMSetGlobalConstant).restype, LLVMSetGlobalConstant.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# LLVMThreadLocalMode LLVMGetThreadLocalMode(LLVMValueRef GlobalVar)
try: (LLVMGetThreadLocalMode:=dll.LLVMGetThreadLocalMode).restype, LLVMGetThreadLocalMode.argtypes = LLVMThreadLocalMode, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetThreadLocalMode(LLVMValueRef GlobalVar, LLVMThreadLocalMode Mode)
try: (LLVMSetThreadLocalMode:=dll.LLVMSetThreadLocalMode).restype, LLVMSetThreadLocalMode.argtypes = None, [LLVMValueRef, LLVMThreadLocalMode]
except AttributeError: pass

# LLVMBool LLVMIsExternallyInitialized(LLVMValueRef GlobalVar)
try: (LLVMIsExternallyInitialized:=dll.LLVMIsExternallyInitialized).restype, LLVMIsExternallyInitialized.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetExternallyInitialized(LLVMValueRef GlobalVar, LLVMBool IsExtInit)
try: (LLVMSetExternallyInitialized:=dll.LLVMSetExternallyInitialized).restype, LLVMSetExternallyInitialized.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# LLVMValueRef LLVMAddAlias2(LLVMModuleRef M, LLVMTypeRef ValueTy, unsigned int AddrSpace, LLVMValueRef Aliasee, const char *Name)
try: (LLVMAddAlias2:=dll.LLVMAddAlias2).restype, LLVMAddAlias2.argtypes = LLVMValueRef, [LLVMModuleRef, LLVMTypeRef, ctypes.c_uint32, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMGetNamedGlobalAlias(LLVMModuleRef M, const char *Name, size_t NameLen)
try: (LLVMGetNamedGlobalAlias:=dll.LLVMGetNamedGlobalAlias).restype, LLVMGetNamedGlobalAlias.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMValueRef LLVMGetFirstGlobalAlias(LLVMModuleRef M)
try: (LLVMGetFirstGlobalAlias:=dll.LLVMGetFirstGlobalAlias).restype, LLVMGetFirstGlobalAlias.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

# LLVMValueRef LLVMGetLastGlobalAlias(LLVMModuleRef M)
try: (LLVMGetLastGlobalAlias:=dll.LLVMGetLastGlobalAlias).restype, LLVMGetLastGlobalAlias.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

# LLVMValueRef LLVMGetNextGlobalAlias(LLVMValueRef GA)
try: (LLVMGetNextGlobalAlias:=dll.LLVMGetNextGlobalAlias).restype, LLVMGetNextGlobalAlias.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetPreviousGlobalAlias(LLVMValueRef GA)
try: (LLVMGetPreviousGlobalAlias:=dll.LLVMGetPreviousGlobalAlias).restype, LLVMGetPreviousGlobalAlias.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMAliasGetAliasee(LLVMValueRef Alias)
try: (LLVMAliasGetAliasee:=dll.LLVMAliasGetAliasee).restype, LLVMAliasGetAliasee.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# void LLVMAliasSetAliasee(LLVMValueRef Alias, LLVMValueRef Aliasee)
try: (LLVMAliasSetAliasee:=dll.LLVMAliasSetAliasee).restype, LLVMAliasSetAliasee.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# void LLVMDeleteFunction(LLVMValueRef Fn)
try: (LLVMDeleteFunction:=dll.LLVMDeleteFunction).restype, LLVMDeleteFunction.argtypes = None, [LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMHasPersonalityFn(LLVMValueRef Fn)
try: (LLVMHasPersonalityFn:=dll.LLVMHasPersonalityFn).restype, LLVMHasPersonalityFn.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetPersonalityFn(LLVMValueRef Fn)
try: (LLVMGetPersonalityFn:=dll.LLVMGetPersonalityFn).restype, LLVMGetPersonalityFn.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetPersonalityFn(LLVMValueRef Fn, LLVMValueRef PersonalityFn)
try: (LLVMSetPersonalityFn:=dll.LLVMSetPersonalityFn).restype, LLVMSetPersonalityFn.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# unsigned int LLVMLookupIntrinsicID(const char *Name, size_t NameLen)
try: (LLVMLookupIntrinsicID:=dll.LLVMLookupIntrinsicID).restype, LLVMLookupIntrinsicID.argtypes = ctypes.c_uint32, [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# unsigned int LLVMGetIntrinsicID(LLVMValueRef Fn)
try: (LLVMGetIntrinsicID:=dll.LLVMGetIntrinsicID).restype, LLVMGetIntrinsicID.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetIntrinsicDeclaration(LLVMModuleRef Mod, unsigned int ID, LLVMTypeRef *ParamTypes, size_t ParamCount)
try: (LLVMGetIntrinsicDeclaration:=dll.LLVMGetIntrinsicDeclaration).restype, LLVMGetIntrinsicDeclaration.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.c_uint32, ctypes.POINTER(LLVMTypeRef), size_t]
except AttributeError: pass

# LLVMTypeRef LLVMIntrinsicGetType(LLVMContextRef Ctx, unsigned int ID, LLVMTypeRef *ParamTypes, size_t ParamCount)
try: (LLVMIntrinsicGetType:=dll.LLVMIntrinsicGetType).restype, LLVMIntrinsicGetType.argtypes = LLVMTypeRef, [LLVMContextRef, ctypes.c_uint32, ctypes.POINTER(LLVMTypeRef), size_t]
except AttributeError: pass

# const char *LLVMIntrinsicGetName(unsigned int ID, size_t *NameLength)
try: (LLVMIntrinsicGetName:=dll.LLVMIntrinsicGetName).restype, LLVMIntrinsicGetName.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_uint32, ctypes.POINTER(size_t)]
except AttributeError: pass

# char *LLVMIntrinsicCopyOverloadedName(unsigned int ID, LLVMTypeRef *ParamTypes, size_t ParamCount, size_t *NameLength)
try: (LLVMIntrinsicCopyOverloadedName:=dll.LLVMIntrinsicCopyOverloadedName).restype, LLVMIntrinsicCopyOverloadedName.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_uint32, ctypes.POINTER(LLVMTypeRef), size_t, ctypes.POINTER(size_t)]
except AttributeError: pass

# char *LLVMIntrinsicCopyOverloadedName2(LLVMModuleRef Mod, unsigned int ID, LLVMTypeRef *ParamTypes, size_t ParamCount, size_t *NameLength)
try: (LLVMIntrinsicCopyOverloadedName2:=dll.LLVMIntrinsicCopyOverloadedName2).restype, LLVMIntrinsicCopyOverloadedName2.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef, ctypes.c_uint32, ctypes.POINTER(LLVMTypeRef), size_t, ctypes.POINTER(size_t)]
except AttributeError: pass

# LLVMBool LLVMIntrinsicIsOverloaded(unsigned int ID)
try: (LLVMIntrinsicIsOverloaded:=dll.LLVMIntrinsicIsOverloaded).restype, LLVMIntrinsicIsOverloaded.argtypes = LLVMBool, [ctypes.c_uint32]
except AttributeError: pass

# unsigned int LLVMGetFunctionCallConv(LLVMValueRef Fn)
try: (LLVMGetFunctionCallConv:=dll.LLVMGetFunctionCallConv).restype, LLVMGetFunctionCallConv.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetFunctionCallConv(LLVMValueRef Fn, unsigned int CC)
try: (LLVMSetFunctionCallConv:=dll.LLVMSetFunctionCallConv).restype, LLVMSetFunctionCallConv.argtypes = None, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# const char *LLVMGetGC(LLVMValueRef Fn)
try: (LLVMGetGC:=dll.LLVMGetGC).restype, LLVMGetGC.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef]
except AttributeError: pass

# void LLVMSetGC(LLVMValueRef Fn, const char *Name)
try: (LLVMSetGC:=dll.LLVMSetGC).restype, LLVMSetGC.argtypes = None, [LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMGetPrefixData(LLVMValueRef Fn)
try: (LLVMGetPrefixData:=dll.LLVMGetPrefixData).restype, LLVMGetPrefixData.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMHasPrefixData(LLVMValueRef Fn)
try: (LLVMHasPrefixData:=dll.LLVMHasPrefixData).restype, LLVMHasPrefixData.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetPrefixData(LLVMValueRef Fn, LLVMValueRef prefixData)
try: (LLVMSetPrefixData:=dll.LLVMSetPrefixData).restype, LLVMSetPrefixData.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetPrologueData(LLVMValueRef Fn)
try: (LLVMGetPrologueData:=dll.LLVMGetPrologueData).restype, LLVMGetPrologueData.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMHasPrologueData(LLVMValueRef Fn)
try: (LLVMHasPrologueData:=dll.LLVMHasPrologueData).restype, LLVMHasPrologueData.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetPrologueData(LLVMValueRef Fn, LLVMValueRef prologueData)
try: (LLVMSetPrologueData:=dll.LLVMSetPrologueData).restype, LLVMSetPrologueData.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# void LLVMAddAttributeAtIndex(LLVMValueRef F, LLVMAttributeIndex Idx, LLVMAttributeRef A)
try: (LLVMAddAttributeAtIndex:=dll.LLVMAddAttributeAtIndex).restype, LLVMAddAttributeAtIndex.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, LLVMAttributeRef]
except AttributeError: pass

# unsigned int LLVMGetAttributeCountAtIndex(LLVMValueRef F, LLVMAttributeIndex Idx)
try: (LLVMGetAttributeCountAtIndex:=dll.LLVMGetAttributeCountAtIndex).restype, LLVMGetAttributeCountAtIndex.argtypes = ctypes.c_uint32, [LLVMValueRef, LLVMAttributeIndex]
except AttributeError: pass

# void LLVMGetAttributesAtIndex(LLVMValueRef F, LLVMAttributeIndex Idx, LLVMAttributeRef *Attrs)
try: (LLVMGetAttributesAtIndex:=dll.LLVMGetAttributesAtIndex).restype, LLVMGetAttributesAtIndex.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(LLVMAttributeRef)]
except AttributeError: pass

# LLVMAttributeRef LLVMGetEnumAttributeAtIndex(LLVMValueRef F, LLVMAttributeIndex Idx, unsigned int KindID)
try: (LLVMGetEnumAttributeAtIndex:=dll.LLVMGetEnumAttributeAtIndex).restype, LLVMGetEnumAttributeAtIndex.argtypes = LLVMAttributeRef, [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError: pass

# LLVMAttributeRef LLVMGetStringAttributeAtIndex(LLVMValueRef F, LLVMAttributeIndex Idx, const char *K, unsigned int KLen)
try: (LLVMGetStringAttributeAtIndex:=dll.LLVMGetStringAttributeAtIndex).restype, LLVMGetStringAttributeAtIndex.argtypes = LLVMAttributeRef, [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

# void LLVMRemoveEnumAttributeAtIndex(LLVMValueRef F, LLVMAttributeIndex Idx, unsigned int KindID)
try: (LLVMRemoveEnumAttributeAtIndex:=dll.LLVMRemoveEnumAttributeAtIndex).restype, LLVMRemoveEnumAttributeAtIndex.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError: pass

# void LLVMRemoveStringAttributeAtIndex(LLVMValueRef F, LLVMAttributeIndex Idx, const char *K, unsigned int KLen)
try: (LLVMRemoveStringAttributeAtIndex:=dll.LLVMRemoveStringAttributeAtIndex).restype, LLVMRemoveStringAttributeAtIndex.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

# void LLVMAddTargetDependentFunctionAttr(LLVMValueRef Fn, const char *A, const char *V)
try: (LLVMAddTargetDependentFunctionAttr:=dll.LLVMAddTargetDependentFunctionAttr).restype, LLVMAddTargetDependentFunctionAttr.argtypes = None, [LLVMValueRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# unsigned int LLVMCountParams(LLVMValueRef Fn)
try: (LLVMCountParams:=dll.LLVMCountParams).restype, LLVMCountParams.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# void LLVMGetParams(LLVMValueRef Fn, LLVMValueRef *Params)
try: (LLVMGetParams:=dll.LLVMGetParams).restype, LLVMGetParams.argtypes = None, [LLVMValueRef, ctypes.POINTER(LLVMValueRef)]
except AttributeError: pass

# LLVMValueRef LLVMGetParam(LLVMValueRef Fn, unsigned int Index)
try: (LLVMGetParam:=dll.LLVMGetParam).restype, LLVMGetParam.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMGetParamParent(LLVMValueRef Inst)
try: (LLVMGetParamParent:=dll.LLVMGetParamParent).restype, LLVMGetParamParent.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetFirstParam(LLVMValueRef Fn)
try: (LLVMGetFirstParam:=dll.LLVMGetFirstParam).restype, LLVMGetFirstParam.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetLastParam(LLVMValueRef Fn)
try: (LLVMGetLastParam:=dll.LLVMGetLastParam).restype, LLVMGetLastParam.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetNextParam(LLVMValueRef Arg)
try: (LLVMGetNextParam:=dll.LLVMGetNextParam).restype, LLVMGetNextParam.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetPreviousParam(LLVMValueRef Arg)
try: (LLVMGetPreviousParam:=dll.LLVMGetPreviousParam).restype, LLVMGetPreviousParam.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetParamAlignment(LLVMValueRef Arg, unsigned int Align)
try: (LLVMSetParamAlignment:=dll.LLVMSetParamAlignment).restype, LLVMSetParamAlignment.argtypes = None, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMAddGlobalIFunc(LLVMModuleRef M, const char *Name, size_t NameLen, LLVMTypeRef Ty, unsigned int AddrSpace, LLVMValueRef Resolver)
try: (LLVMAddGlobalIFunc:=dll.LLVMAddGlobalIFunc).restype, LLVMAddGlobalIFunc.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMTypeRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetNamedGlobalIFunc(LLVMModuleRef M, const char *Name, size_t NameLen)
try: (LLVMGetNamedGlobalIFunc:=dll.LLVMGetNamedGlobalIFunc).restype, LLVMGetNamedGlobalIFunc.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMValueRef LLVMGetFirstGlobalIFunc(LLVMModuleRef M)
try: (LLVMGetFirstGlobalIFunc:=dll.LLVMGetFirstGlobalIFunc).restype, LLVMGetFirstGlobalIFunc.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

# LLVMValueRef LLVMGetLastGlobalIFunc(LLVMModuleRef M)
try: (LLVMGetLastGlobalIFunc:=dll.LLVMGetLastGlobalIFunc).restype, LLVMGetLastGlobalIFunc.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

# LLVMValueRef LLVMGetNextGlobalIFunc(LLVMValueRef IFunc)
try: (LLVMGetNextGlobalIFunc:=dll.LLVMGetNextGlobalIFunc).restype, LLVMGetNextGlobalIFunc.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetPreviousGlobalIFunc(LLVMValueRef IFunc)
try: (LLVMGetPreviousGlobalIFunc:=dll.LLVMGetPreviousGlobalIFunc).restype, LLVMGetPreviousGlobalIFunc.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetGlobalIFuncResolver(LLVMValueRef IFunc)
try: (LLVMGetGlobalIFuncResolver:=dll.LLVMGetGlobalIFuncResolver).restype, LLVMGetGlobalIFuncResolver.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetGlobalIFuncResolver(LLVMValueRef IFunc, LLVMValueRef Resolver)
try: (LLVMSetGlobalIFuncResolver:=dll.LLVMSetGlobalIFuncResolver).restype, LLVMSetGlobalIFuncResolver.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# void LLVMEraseGlobalIFunc(LLVMValueRef IFunc)
try: (LLVMEraseGlobalIFunc:=dll.LLVMEraseGlobalIFunc).restype, LLVMEraseGlobalIFunc.argtypes = None, [LLVMValueRef]
except AttributeError: pass

# void LLVMRemoveGlobalIFunc(LLVMValueRef IFunc)
try: (LLVMRemoveGlobalIFunc:=dll.LLVMRemoveGlobalIFunc).restype, LLVMRemoveGlobalIFunc.argtypes = None, [LLVMValueRef]
except AttributeError: pass

# LLVMMetadataRef LLVMMDStringInContext2(LLVMContextRef C, const char *Str, size_t SLen)
try: (LLVMMDStringInContext2:=dll.LLVMMDStringInContext2).restype, LLVMMDStringInContext2.argtypes = LLVMMetadataRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMMDNodeInContext2(LLVMContextRef C, LLVMMetadataRef *MDs, size_t Count)
try: (LLVMMDNodeInContext2:=dll.LLVMMDNodeInContext2).restype, LLVMMDNodeInContext2.argtypes = LLVMMetadataRef, [LLVMContextRef, ctypes.POINTER(LLVMMetadataRef), size_t]
except AttributeError: pass

# LLVMValueRef LLVMMetadataAsValue(LLVMContextRef C, LLVMMetadataRef MD)
try: (LLVMMetadataAsValue:=dll.LLVMMetadataAsValue).restype, LLVMMetadataAsValue.argtypes = LLVMValueRef, [LLVMContextRef, LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMValueAsMetadata(LLVMValueRef Val)
try: (LLVMValueAsMetadata:=dll.LLVMValueAsMetadata).restype, LLVMValueAsMetadata.argtypes = LLVMMetadataRef, [LLVMValueRef]
except AttributeError: pass

# const char *LLVMGetMDString(LLVMValueRef V, unsigned int *Length)
try: (LLVMGetMDString:=dll.LLVMGetMDString).restype, LLVMGetMDString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

# unsigned int LLVMGetMDNodeNumOperands(LLVMValueRef V)
try: (LLVMGetMDNodeNumOperands:=dll.LLVMGetMDNodeNumOperands).restype, LLVMGetMDNodeNumOperands.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# void LLVMGetMDNodeOperands(LLVMValueRef V, LLVMValueRef *Dest)
try: (LLVMGetMDNodeOperands:=dll.LLVMGetMDNodeOperands).restype, LLVMGetMDNodeOperands.argtypes = None, [LLVMValueRef, ctypes.POINTER(LLVMValueRef)]
except AttributeError: pass

# void LLVMReplaceMDNodeOperandWith(LLVMValueRef V, unsigned int Index, LLVMMetadataRef Replacement)
try: (LLVMReplaceMDNodeOperandWith:=dll.LLVMReplaceMDNodeOperandWith).restype, LLVMReplaceMDNodeOperandWith.argtypes = None, [LLVMValueRef, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError: pass

# LLVMValueRef LLVMMDStringInContext(LLVMContextRef C, const char *Str, unsigned int SLen)
try: (LLVMMDStringInContext:=dll.LLVMMDStringInContext).restype, LLVMMDStringInContext.argtypes = LLVMValueRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMMDString(const char *Str, unsigned int SLen)
try: (LLVMMDString:=dll.LLVMMDString).restype, LLVMMDString.argtypes = LLVMValueRef, [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMMDNodeInContext(LLVMContextRef C, LLVMValueRef *Vals, unsigned int Count)
try: (LLVMMDNodeInContext:=dll.LLVMMDNodeInContext).restype, LLVMMDNodeInContext.argtypes = LLVMValueRef, [LLVMContextRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMMDNode(LLVMValueRef *Vals, unsigned int Count)
try: (LLVMMDNode:=dll.LLVMMDNode).restype, LLVMMDNode.argtypes = LLVMValueRef, [ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

class struct_LLVMOpaqueOperandBundle(Struct): pass
LLVMOperandBundleRef = ctypes.POINTER(struct_LLVMOpaqueOperandBundle)
# LLVMOperandBundleRef LLVMCreateOperandBundle(const char *Tag, size_t TagLen, LLVMValueRef *Args, unsigned int NumArgs)
try: (LLVMCreateOperandBundle:=dll.LLVMCreateOperandBundle).restype, LLVMCreateOperandBundle.argtypes = LLVMOperandBundleRef, [ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

# void LLVMDisposeOperandBundle(LLVMOperandBundleRef Bundle)
try: (LLVMDisposeOperandBundle:=dll.LLVMDisposeOperandBundle).restype, LLVMDisposeOperandBundle.argtypes = None, [LLVMOperandBundleRef]
except AttributeError: pass

# const char *LLVMGetOperandBundleTag(LLVMOperandBundleRef Bundle, size_t *Len)
try: (LLVMGetOperandBundleTag:=dll.LLVMGetOperandBundleTag).restype, LLVMGetOperandBundleTag.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOperandBundleRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# unsigned int LLVMGetNumOperandBundleArgs(LLVMOperandBundleRef Bundle)
try: (LLVMGetNumOperandBundleArgs:=dll.LLVMGetNumOperandBundleArgs).restype, LLVMGetNumOperandBundleArgs.argtypes = ctypes.c_uint32, [LLVMOperandBundleRef]
except AttributeError: pass

# LLVMValueRef LLVMGetOperandBundleArgAtIndex(LLVMOperandBundleRef Bundle, unsigned int Index)
try: (LLVMGetOperandBundleArgAtIndex:=dll.LLVMGetOperandBundleArgAtIndex).restype, LLVMGetOperandBundleArgAtIndex.argtypes = LLVMValueRef, [LLVMOperandBundleRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMBasicBlockAsValue(LLVMBasicBlockRef BB)
try: (LLVMBasicBlockAsValue:=dll.LLVMBasicBlockAsValue).restype, LLVMBasicBlockAsValue.argtypes = LLVMValueRef, [LLVMBasicBlockRef]
except AttributeError: pass

# LLVMBool LLVMValueIsBasicBlock(LLVMValueRef Val)
try: (LLVMValueIsBasicBlock:=dll.LLVMValueIsBasicBlock).restype, LLVMValueIsBasicBlock.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# LLVMBasicBlockRef LLVMValueAsBasicBlock(LLVMValueRef Val)
try: (LLVMValueAsBasicBlock:=dll.LLVMValueAsBasicBlock).restype, LLVMValueAsBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

# const char *LLVMGetBasicBlockName(LLVMBasicBlockRef BB)
try: (LLVMGetBasicBlockName:=dll.LLVMGetBasicBlockName).restype, LLVMGetBasicBlockName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMBasicBlockRef]
except AttributeError: pass

# LLVMValueRef LLVMGetBasicBlockParent(LLVMBasicBlockRef BB)
try: (LLVMGetBasicBlockParent:=dll.LLVMGetBasicBlockParent).restype, LLVMGetBasicBlockParent.argtypes = LLVMValueRef, [LLVMBasicBlockRef]
except AttributeError: pass

# LLVMValueRef LLVMGetBasicBlockTerminator(LLVMBasicBlockRef BB)
try: (LLVMGetBasicBlockTerminator:=dll.LLVMGetBasicBlockTerminator).restype, LLVMGetBasicBlockTerminator.argtypes = LLVMValueRef, [LLVMBasicBlockRef]
except AttributeError: pass

# unsigned int LLVMCountBasicBlocks(LLVMValueRef Fn)
try: (LLVMCountBasicBlocks:=dll.LLVMCountBasicBlocks).restype, LLVMCountBasicBlocks.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# void LLVMGetBasicBlocks(LLVMValueRef Fn, LLVMBasicBlockRef *BasicBlocks)
try: (LLVMGetBasicBlocks:=dll.LLVMGetBasicBlocks).restype, LLVMGetBasicBlocks.argtypes = None, [LLVMValueRef, ctypes.POINTER(LLVMBasicBlockRef)]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetFirstBasicBlock(LLVMValueRef Fn)
try: (LLVMGetFirstBasicBlock:=dll.LLVMGetFirstBasicBlock).restype, LLVMGetFirstBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetLastBasicBlock(LLVMValueRef Fn)
try: (LLVMGetLastBasicBlock:=dll.LLVMGetLastBasicBlock).restype, LLVMGetLastBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetNextBasicBlock(LLVMBasicBlockRef BB)
try: (LLVMGetNextBasicBlock:=dll.LLVMGetNextBasicBlock).restype, LLVMGetNextBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMBasicBlockRef]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetPreviousBasicBlock(LLVMBasicBlockRef BB)
try: (LLVMGetPreviousBasicBlock:=dll.LLVMGetPreviousBasicBlock).restype, LLVMGetPreviousBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMBasicBlockRef]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetEntryBasicBlock(LLVMValueRef Fn)
try: (LLVMGetEntryBasicBlock:=dll.LLVMGetEntryBasicBlock).restype, LLVMGetEntryBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

class struct_LLVMOpaqueBuilder(Struct): pass
LLVMBuilderRef = ctypes.POINTER(struct_LLVMOpaqueBuilder)
# void LLVMInsertExistingBasicBlockAfterInsertBlock(LLVMBuilderRef Builder, LLVMBasicBlockRef BB)
try: (LLVMInsertExistingBasicBlockAfterInsertBlock:=dll.LLVMInsertExistingBasicBlockAfterInsertBlock).restype, LLVMInsertExistingBasicBlockAfterInsertBlock.argtypes = None, [LLVMBuilderRef, LLVMBasicBlockRef]
except AttributeError: pass

# void LLVMAppendExistingBasicBlock(LLVMValueRef Fn, LLVMBasicBlockRef BB)
try: (LLVMAppendExistingBasicBlock:=dll.LLVMAppendExistingBasicBlock).restype, LLVMAppendExistingBasicBlock.argtypes = None, [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

# LLVMBasicBlockRef LLVMCreateBasicBlockInContext(LLVMContextRef C, const char *Name)
try: (LLVMCreateBasicBlockInContext:=dll.LLVMCreateBasicBlockInContext).restype, LLVMCreateBasicBlockInContext.argtypes = LLVMBasicBlockRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBasicBlockRef LLVMAppendBasicBlockInContext(LLVMContextRef C, LLVMValueRef Fn, const char *Name)
try: (LLVMAppendBasicBlockInContext:=dll.LLVMAppendBasicBlockInContext).restype, LLVMAppendBasicBlockInContext.argtypes = LLVMBasicBlockRef, [LLVMContextRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBasicBlockRef LLVMAppendBasicBlock(LLVMValueRef Fn, const char *Name)
try: (LLVMAppendBasicBlock:=dll.LLVMAppendBasicBlock).restype, LLVMAppendBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBasicBlockRef LLVMInsertBasicBlockInContext(LLVMContextRef C, LLVMBasicBlockRef BB, const char *Name)
try: (LLVMInsertBasicBlockInContext:=dll.LLVMInsertBasicBlockInContext).restype, LLVMInsertBasicBlockInContext.argtypes = LLVMBasicBlockRef, [LLVMContextRef, LLVMBasicBlockRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBasicBlockRef LLVMInsertBasicBlock(LLVMBasicBlockRef InsertBeforeBB, const char *Name)
try: (LLVMInsertBasicBlock:=dll.LLVMInsertBasicBlock).restype, LLVMInsertBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMBasicBlockRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMDeleteBasicBlock(LLVMBasicBlockRef BB)
try: (LLVMDeleteBasicBlock:=dll.LLVMDeleteBasicBlock).restype, LLVMDeleteBasicBlock.argtypes = None, [LLVMBasicBlockRef]
except AttributeError: pass

# void LLVMRemoveBasicBlockFromParent(LLVMBasicBlockRef BB)
try: (LLVMRemoveBasicBlockFromParent:=dll.LLVMRemoveBasicBlockFromParent).restype, LLVMRemoveBasicBlockFromParent.argtypes = None, [LLVMBasicBlockRef]
except AttributeError: pass

# void LLVMMoveBasicBlockBefore(LLVMBasicBlockRef BB, LLVMBasicBlockRef MovePos)
try: (LLVMMoveBasicBlockBefore:=dll.LLVMMoveBasicBlockBefore).restype, LLVMMoveBasicBlockBefore.argtypes = None, [LLVMBasicBlockRef, LLVMBasicBlockRef]
except AttributeError: pass

# void LLVMMoveBasicBlockAfter(LLVMBasicBlockRef BB, LLVMBasicBlockRef MovePos)
try: (LLVMMoveBasicBlockAfter:=dll.LLVMMoveBasicBlockAfter).restype, LLVMMoveBasicBlockAfter.argtypes = None, [LLVMBasicBlockRef, LLVMBasicBlockRef]
except AttributeError: pass

# LLVMValueRef LLVMGetFirstInstruction(LLVMBasicBlockRef BB)
try: (LLVMGetFirstInstruction:=dll.LLVMGetFirstInstruction).restype, LLVMGetFirstInstruction.argtypes = LLVMValueRef, [LLVMBasicBlockRef]
except AttributeError: pass

# LLVMValueRef LLVMGetLastInstruction(LLVMBasicBlockRef BB)
try: (LLVMGetLastInstruction:=dll.LLVMGetLastInstruction).restype, LLVMGetLastInstruction.argtypes = LLVMValueRef, [LLVMBasicBlockRef]
except AttributeError: pass

# int LLVMHasMetadata(LLVMValueRef Val)
try: (LLVMHasMetadata:=dll.LLVMHasMetadata).restype, LLVMHasMetadata.argtypes = ctypes.c_int32, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetMetadata(LLVMValueRef Val, unsigned int KindID)
try: (LLVMGetMetadata:=dll.LLVMGetMetadata).restype, LLVMGetMetadata.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# void LLVMSetMetadata(LLVMValueRef Val, unsigned int KindID, LLVMValueRef Node)
try: (LLVMSetMetadata:=dll.LLVMSetMetadata).restype, LLVMSetMetadata.argtypes = None, [LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError: pass

# LLVMValueMetadataEntry *LLVMInstructionGetAllMetadataOtherThanDebugLoc(LLVMValueRef Instr, size_t *NumEntries)
try: (LLVMInstructionGetAllMetadataOtherThanDebugLoc:=dll.LLVMInstructionGetAllMetadataOtherThanDebugLoc).restype, LLVMInstructionGetAllMetadataOtherThanDebugLoc.argtypes = ctypes.POINTER(LLVMValueMetadataEntry), [LLVMValueRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetInstructionParent(LLVMValueRef Inst)
try: (LLVMGetInstructionParent:=dll.LLVMGetInstructionParent).restype, LLVMGetInstructionParent.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetNextInstruction(LLVMValueRef Inst)
try: (LLVMGetNextInstruction:=dll.LLVMGetNextInstruction).restype, LLVMGetNextInstruction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetPreviousInstruction(LLVMValueRef Inst)
try: (LLVMGetPreviousInstruction:=dll.LLVMGetPreviousInstruction).restype, LLVMGetPreviousInstruction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# void LLVMInstructionRemoveFromParent(LLVMValueRef Inst)
try: (LLVMInstructionRemoveFromParent:=dll.LLVMInstructionRemoveFromParent).restype, LLVMInstructionRemoveFromParent.argtypes = None, [LLVMValueRef]
except AttributeError: pass

# void LLVMInstructionEraseFromParent(LLVMValueRef Inst)
try: (LLVMInstructionEraseFromParent:=dll.LLVMInstructionEraseFromParent).restype, LLVMInstructionEraseFromParent.argtypes = None, [LLVMValueRef]
except AttributeError: pass

# void LLVMDeleteInstruction(LLVMValueRef Inst)
try: (LLVMDeleteInstruction:=dll.LLVMDeleteInstruction).restype, LLVMDeleteInstruction.argtypes = None, [LLVMValueRef]
except AttributeError: pass

# LLVMOpcode LLVMGetInstructionOpcode(LLVMValueRef Inst)
try: (LLVMGetInstructionOpcode:=dll.LLVMGetInstructionOpcode).restype, LLVMGetInstructionOpcode.argtypes = LLVMOpcode, [LLVMValueRef]
except AttributeError: pass

# LLVMIntPredicate LLVMGetICmpPredicate(LLVMValueRef Inst)
try: (LLVMGetICmpPredicate:=dll.LLVMGetICmpPredicate).restype, LLVMGetICmpPredicate.argtypes = LLVMIntPredicate, [LLVMValueRef]
except AttributeError: pass

# LLVMRealPredicate LLVMGetFCmpPredicate(LLVMValueRef Inst)
try: (LLVMGetFCmpPredicate:=dll.LLVMGetFCmpPredicate).restype, LLVMGetFCmpPredicate.argtypes = LLVMRealPredicate, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMInstructionClone(LLVMValueRef Inst)
try: (LLVMInstructionClone:=dll.LLVMInstructionClone).restype, LLVMInstructionClone.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMIsATerminatorInst(LLVMValueRef Inst)
try: (LLVMIsATerminatorInst:=dll.LLVMIsATerminatorInst).restype, LLVMIsATerminatorInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# LLVMDbgRecordRef LLVMGetFirstDbgRecord(LLVMValueRef Inst)
try: (LLVMGetFirstDbgRecord:=dll.LLVMGetFirstDbgRecord).restype, LLVMGetFirstDbgRecord.argtypes = LLVMDbgRecordRef, [LLVMValueRef]
except AttributeError: pass

# LLVMDbgRecordRef LLVMGetLastDbgRecord(LLVMValueRef Inst)
try: (LLVMGetLastDbgRecord:=dll.LLVMGetLastDbgRecord).restype, LLVMGetLastDbgRecord.argtypes = LLVMDbgRecordRef, [LLVMValueRef]
except AttributeError: pass

# LLVMDbgRecordRef LLVMGetNextDbgRecord(LLVMDbgRecordRef DbgRecord)
try: (LLVMGetNextDbgRecord:=dll.LLVMGetNextDbgRecord).restype, LLVMGetNextDbgRecord.argtypes = LLVMDbgRecordRef, [LLVMDbgRecordRef]
except AttributeError: pass

# LLVMDbgRecordRef LLVMGetPreviousDbgRecord(LLVMDbgRecordRef DbgRecord)
try: (LLVMGetPreviousDbgRecord:=dll.LLVMGetPreviousDbgRecord).restype, LLVMGetPreviousDbgRecord.argtypes = LLVMDbgRecordRef, [LLVMDbgRecordRef]
except AttributeError: pass

# unsigned int LLVMGetNumArgOperands(LLVMValueRef Instr)
try: (LLVMGetNumArgOperands:=dll.LLVMGetNumArgOperands).restype, LLVMGetNumArgOperands.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetInstructionCallConv(LLVMValueRef Instr, unsigned int CC)
try: (LLVMSetInstructionCallConv:=dll.LLVMSetInstructionCallConv).restype, LLVMSetInstructionCallConv.argtypes = None, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# unsigned int LLVMGetInstructionCallConv(LLVMValueRef Instr)
try: (LLVMGetInstructionCallConv:=dll.LLVMGetInstructionCallConv).restype, LLVMGetInstructionCallConv.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetInstrParamAlignment(LLVMValueRef Instr, LLVMAttributeIndex Idx, unsigned int Align)
try: (LLVMSetInstrParamAlignment:=dll.LLVMSetInstrParamAlignment).restype, LLVMSetInstrParamAlignment.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError: pass

# void LLVMAddCallSiteAttribute(LLVMValueRef C, LLVMAttributeIndex Idx, LLVMAttributeRef A)
try: (LLVMAddCallSiteAttribute:=dll.LLVMAddCallSiteAttribute).restype, LLVMAddCallSiteAttribute.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, LLVMAttributeRef]
except AttributeError: pass

# unsigned int LLVMGetCallSiteAttributeCount(LLVMValueRef C, LLVMAttributeIndex Idx)
try: (LLVMGetCallSiteAttributeCount:=dll.LLVMGetCallSiteAttributeCount).restype, LLVMGetCallSiteAttributeCount.argtypes = ctypes.c_uint32, [LLVMValueRef, LLVMAttributeIndex]
except AttributeError: pass

# void LLVMGetCallSiteAttributes(LLVMValueRef C, LLVMAttributeIndex Idx, LLVMAttributeRef *Attrs)
try: (LLVMGetCallSiteAttributes:=dll.LLVMGetCallSiteAttributes).restype, LLVMGetCallSiteAttributes.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(LLVMAttributeRef)]
except AttributeError: pass

# LLVMAttributeRef LLVMGetCallSiteEnumAttribute(LLVMValueRef C, LLVMAttributeIndex Idx, unsigned int KindID)
try: (LLVMGetCallSiteEnumAttribute:=dll.LLVMGetCallSiteEnumAttribute).restype, LLVMGetCallSiteEnumAttribute.argtypes = LLVMAttributeRef, [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError: pass

# LLVMAttributeRef LLVMGetCallSiteStringAttribute(LLVMValueRef C, LLVMAttributeIndex Idx, const char *K, unsigned int KLen)
try: (LLVMGetCallSiteStringAttribute:=dll.LLVMGetCallSiteStringAttribute).restype, LLVMGetCallSiteStringAttribute.argtypes = LLVMAttributeRef, [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

# void LLVMRemoveCallSiteEnumAttribute(LLVMValueRef C, LLVMAttributeIndex Idx, unsigned int KindID)
try: (LLVMRemoveCallSiteEnumAttribute:=dll.LLVMRemoveCallSiteEnumAttribute).restype, LLVMRemoveCallSiteEnumAttribute.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError: pass

# void LLVMRemoveCallSiteStringAttribute(LLVMValueRef C, LLVMAttributeIndex Idx, const char *K, unsigned int KLen)
try: (LLVMRemoveCallSiteStringAttribute:=dll.LLVMRemoveCallSiteStringAttribute).restype, LLVMRemoveCallSiteStringAttribute.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMGetCalledFunctionType(LLVMValueRef C)
try: (LLVMGetCalledFunctionType:=dll.LLVMGetCalledFunctionType).restype, LLVMGetCalledFunctionType.argtypes = LLVMTypeRef, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetCalledValue(LLVMValueRef Instr)
try: (LLVMGetCalledValue:=dll.LLVMGetCalledValue).restype, LLVMGetCalledValue.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# unsigned int LLVMGetNumOperandBundles(LLVMValueRef C)
try: (LLVMGetNumOperandBundles:=dll.LLVMGetNumOperandBundles).restype, LLVMGetNumOperandBundles.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# LLVMOperandBundleRef LLVMGetOperandBundleAtIndex(LLVMValueRef C, unsigned int Index)
try: (LLVMGetOperandBundleAtIndex:=dll.LLVMGetOperandBundleAtIndex).restype, LLVMGetOperandBundleAtIndex.argtypes = LLVMOperandBundleRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMBool LLVMIsTailCall(LLVMValueRef CallInst)
try: (LLVMIsTailCall:=dll.LLVMIsTailCall).restype, LLVMIsTailCall.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetTailCall(LLVMValueRef CallInst, LLVMBool IsTailCall)
try: (LLVMSetTailCall:=dll.LLVMSetTailCall).restype, LLVMSetTailCall.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# LLVMTailCallKind LLVMGetTailCallKind(LLVMValueRef CallInst)
try: (LLVMGetTailCallKind:=dll.LLVMGetTailCallKind).restype, LLVMGetTailCallKind.argtypes = LLVMTailCallKind, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetTailCallKind(LLVMValueRef CallInst, LLVMTailCallKind kind)
try: (LLVMSetTailCallKind:=dll.LLVMSetTailCallKind).restype, LLVMSetTailCallKind.argtypes = None, [LLVMValueRef, LLVMTailCallKind]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetNormalDest(LLVMValueRef InvokeInst)
try: (LLVMGetNormalDest:=dll.LLVMGetNormalDest).restype, LLVMGetNormalDest.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetUnwindDest(LLVMValueRef InvokeInst)
try: (LLVMGetUnwindDest:=dll.LLVMGetUnwindDest).restype, LLVMGetUnwindDest.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetNormalDest(LLVMValueRef InvokeInst, LLVMBasicBlockRef B)
try: (LLVMSetNormalDest:=dll.LLVMSetNormalDest).restype, LLVMSetNormalDest.argtypes = None, [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

# void LLVMSetUnwindDest(LLVMValueRef InvokeInst, LLVMBasicBlockRef B)
try: (LLVMSetUnwindDest:=dll.LLVMSetUnwindDest).restype, LLVMSetUnwindDest.argtypes = None, [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetCallBrDefaultDest(LLVMValueRef CallBr)
try: (LLVMGetCallBrDefaultDest:=dll.LLVMGetCallBrDefaultDest).restype, LLVMGetCallBrDefaultDest.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

# unsigned int LLVMGetCallBrNumIndirectDests(LLVMValueRef CallBr)
try: (LLVMGetCallBrNumIndirectDests:=dll.LLVMGetCallBrNumIndirectDests).restype, LLVMGetCallBrNumIndirectDests.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetCallBrIndirectDest(LLVMValueRef CallBr, unsigned int Idx)
try: (LLVMGetCallBrIndirectDest:=dll.LLVMGetCallBrIndirectDest).restype, LLVMGetCallBrIndirectDest.argtypes = LLVMBasicBlockRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# unsigned int LLVMGetNumSuccessors(LLVMValueRef Term)
try: (LLVMGetNumSuccessors:=dll.LLVMGetNumSuccessors).restype, LLVMGetNumSuccessors.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetSuccessor(LLVMValueRef Term, unsigned int i)
try: (LLVMGetSuccessor:=dll.LLVMGetSuccessor).restype, LLVMGetSuccessor.argtypes = LLVMBasicBlockRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# void LLVMSetSuccessor(LLVMValueRef Term, unsigned int i, LLVMBasicBlockRef block)
try: (LLVMSetSuccessor:=dll.LLVMSetSuccessor).restype, LLVMSetSuccessor.argtypes = None, [LLVMValueRef, ctypes.c_uint32, LLVMBasicBlockRef]
except AttributeError: pass

# LLVMBool LLVMIsConditional(LLVMValueRef Branch)
try: (LLVMIsConditional:=dll.LLVMIsConditional).restype, LLVMIsConditional.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetCondition(LLVMValueRef Branch)
try: (LLVMGetCondition:=dll.LLVMGetCondition).restype, LLVMGetCondition.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetCondition(LLVMValueRef Branch, LLVMValueRef Cond)
try: (LLVMSetCondition:=dll.LLVMSetCondition).restype, LLVMSetCondition.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetSwitchDefaultDest(LLVMValueRef SwitchInstr)
try: (LLVMGetSwitchDefaultDest:=dll.LLVMGetSwitchDefaultDest).restype, LLVMGetSwitchDefaultDest.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

# LLVMTypeRef LLVMGetAllocatedType(LLVMValueRef Alloca)
try: (LLVMGetAllocatedType:=dll.LLVMGetAllocatedType).restype, LLVMGetAllocatedType.argtypes = LLVMTypeRef, [LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMIsInBounds(LLVMValueRef GEP)
try: (LLVMIsInBounds:=dll.LLVMIsInBounds).restype, LLVMIsInBounds.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetIsInBounds(LLVMValueRef GEP, LLVMBool InBounds)
try: (LLVMSetIsInBounds:=dll.LLVMSetIsInBounds).restype, LLVMSetIsInBounds.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# LLVMTypeRef LLVMGetGEPSourceElementType(LLVMValueRef GEP)
try: (LLVMGetGEPSourceElementType:=dll.LLVMGetGEPSourceElementType).restype, LLVMGetGEPSourceElementType.argtypes = LLVMTypeRef, [LLVMValueRef]
except AttributeError: pass

# LLVMGEPNoWrapFlags LLVMGEPGetNoWrapFlags(LLVMValueRef GEP)
try: (LLVMGEPGetNoWrapFlags:=dll.LLVMGEPGetNoWrapFlags).restype, LLVMGEPGetNoWrapFlags.argtypes = LLVMGEPNoWrapFlags, [LLVMValueRef]
except AttributeError: pass

# void LLVMGEPSetNoWrapFlags(LLVMValueRef GEP, LLVMGEPNoWrapFlags NoWrapFlags)
try: (LLVMGEPSetNoWrapFlags:=dll.LLVMGEPSetNoWrapFlags).restype, LLVMGEPSetNoWrapFlags.argtypes = None, [LLVMValueRef, LLVMGEPNoWrapFlags]
except AttributeError: pass

# void LLVMAddIncoming(LLVMValueRef PhiNode, LLVMValueRef *IncomingValues, LLVMBasicBlockRef *IncomingBlocks, unsigned int Count)
try: (LLVMAddIncoming:=dll.LLVMAddIncoming).restype, LLVMAddIncoming.argtypes = None, [LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.POINTER(LLVMBasicBlockRef), ctypes.c_uint32]
except AttributeError: pass

# unsigned int LLVMCountIncoming(LLVMValueRef PhiNode)
try: (LLVMCountIncoming:=dll.LLVMCountIncoming).restype, LLVMCountIncoming.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetIncomingValue(LLVMValueRef PhiNode, unsigned int Index)
try: (LLVMGetIncomingValue:=dll.LLVMGetIncomingValue).restype, LLVMGetIncomingValue.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetIncomingBlock(LLVMValueRef PhiNode, unsigned int Index)
try: (LLVMGetIncomingBlock:=dll.LLVMGetIncomingBlock).restype, LLVMGetIncomingBlock.argtypes = LLVMBasicBlockRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# unsigned int LLVMGetNumIndices(LLVMValueRef Inst)
try: (LLVMGetNumIndices:=dll.LLVMGetNumIndices).restype, LLVMGetNumIndices.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# const unsigned int *LLVMGetIndices(LLVMValueRef Inst)
try: (LLVMGetIndices:=dll.LLVMGetIndices).restype, LLVMGetIndices.argtypes = ctypes.POINTER(ctypes.c_uint32), [LLVMValueRef]
except AttributeError: pass

# LLVMBuilderRef LLVMCreateBuilderInContext(LLVMContextRef C)
try: (LLVMCreateBuilderInContext:=dll.LLVMCreateBuilderInContext).restype, LLVMCreateBuilderInContext.argtypes = LLVMBuilderRef, [LLVMContextRef]
except AttributeError: pass

# LLVMBuilderRef LLVMCreateBuilder(void)
try: (LLVMCreateBuilder:=dll.LLVMCreateBuilder).restype, LLVMCreateBuilder.argtypes = LLVMBuilderRef, []
except AttributeError: pass

# void LLVMPositionBuilder(LLVMBuilderRef Builder, LLVMBasicBlockRef Block, LLVMValueRef Instr)
try: (LLVMPositionBuilder:=dll.LLVMPositionBuilder).restype, LLVMPositionBuilder.argtypes = None, [LLVMBuilderRef, LLVMBasicBlockRef, LLVMValueRef]
except AttributeError: pass

# void LLVMPositionBuilderBeforeDbgRecords(LLVMBuilderRef Builder, LLVMBasicBlockRef Block, LLVMValueRef Inst)
try: (LLVMPositionBuilderBeforeDbgRecords:=dll.LLVMPositionBuilderBeforeDbgRecords).restype, LLVMPositionBuilderBeforeDbgRecords.argtypes = None, [LLVMBuilderRef, LLVMBasicBlockRef, LLVMValueRef]
except AttributeError: pass

# void LLVMPositionBuilderBefore(LLVMBuilderRef Builder, LLVMValueRef Instr)
try: (LLVMPositionBuilderBefore:=dll.LLVMPositionBuilderBefore).restype, LLVMPositionBuilderBefore.argtypes = None, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

# void LLVMPositionBuilderBeforeInstrAndDbgRecords(LLVMBuilderRef Builder, LLVMValueRef Instr)
try: (LLVMPositionBuilderBeforeInstrAndDbgRecords:=dll.LLVMPositionBuilderBeforeInstrAndDbgRecords).restype, LLVMPositionBuilderBeforeInstrAndDbgRecords.argtypes = None, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

# void LLVMPositionBuilderAtEnd(LLVMBuilderRef Builder, LLVMBasicBlockRef Block)
try: (LLVMPositionBuilderAtEnd:=dll.LLVMPositionBuilderAtEnd).restype, LLVMPositionBuilderAtEnd.argtypes = None, [LLVMBuilderRef, LLVMBasicBlockRef]
except AttributeError: pass

# LLVMBasicBlockRef LLVMGetInsertBlock(LLVMBuilderRef Builder)
try: (LLVMGetInsertBlock:=dll.LLVMGetInsertBlock).restype, LLVMGetInsertBlock.argtypes = LLVMBasicBlockRef, [LLVMBuilderRef]
except AttributeError: pass

# void LLVMClearInsertionPosition(LLVMBuilderRef Builder)
try: (LLVMClearInsertionPosition:=dll.LLVMClearInsertionPosition).restype, LLVMClearInsertionPosition.argtypes = None, [LLVMBuilderRef]
except AttributeError: pass

# void LLVMInsertIntoBuilder(LLVMBuilderRef Builder, LLVMValueRef Instr)
try: (LLVMInsertIntoBuilder:=dll.LLVMInsertIntoBuilder).restype, LLVMInsertIntoBuilder.argtypes = None, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

# void LLVMInsertIntoBuilderWithName(LLVMBuilderRef Builder, LLVMValueRef Instr, const char *Name)
try: (LLVMInsertIntoBuilderWithName:=dll.LLVMInsertIntoBuilderWithName).restype, LLVMInsertIntoBuilderWithName.argtypes = None, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMDisposeBuilder(LLVMBuilderRef Builder)
try: (LLVMDisposeBuilder:=dll.LLVMDisposeBuilder).restype, LLVMDisposeBuilder.argtypes = None, [LLVMBuilderRef]
except AttributeError: pass

# LLVMMetadataRef LLVMGetCurrentDebugLocation2(LLVMBuilderRef Builder)
try: (LLVMGetCurrentDebugLocation2:=dll.LLVMGetCurrentDebugLocation2).restype, LLVMGetCurrentDebugLocation2.argtypes = LLVMMetadataRef, [LLVMBuilderRef]
except AttributeError: pass

# void LLVMSetCurrentDebugLocation2(LLVMBuilderRef Builder, LLVMMetadataRef Loc)
try: (LLVMSetCurrentDebugLocation2:=dll.LLVMSetCurrentDebugLocation2).restype, LLVMSetCurrentDebugLocation2.argtypes = None, [LLVMBuilderRef, LLVMMetadataRef]
except AttributeError: pass

# void LLVMSetInstDebugLocation(LLVMBuilderRef Builder, LLVMValueRef Inst)
try: (LLVMSetInstDebugLocation:=dll.LLVMSetInstDebugLocation).restype, LLVMSetInstDebugLocation.argtypes = None, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

# void LLVMAddMetadataToInst(LLVMBuilderRef Builder, LLVMValueRef Inst)
try: (LLVMAddMetadataToInst:=dll.LLVMAddMetadataToInst).restype, LLVMAddMetadataToInst.argtypes = None, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

# LLVMMetadataRef LLVMBuilderGetDefaultFPMathTag(LLVMBuilderRef Builder)
try: (LLVMBuilderGetDefaultFPMathTag:=dll.LLVMBuilderGetDefaultFPMathTag).restype, LLVMBuilderGetDefaultFPMathTag.argtypes = LLVMMetadataRef, [LLVMBuilderRef]
except AttributeError: pass

# void LLVMBuilderSetDefaultFPMathTag(LLVMBuilderRef Builder, LLVMMetadataRef FPMathTag)
try: (LLVMBuilderSetDefaultFPMathTag:=dll.LLVMBuilderSetDefaultFPMathTag).restype, LLVMBuilderSetDefaultFPMathTag.argtypes = None, [LLVMBuilderRef, LLVMMetadataRef]
except AttributeError: pass

# LLVMContextRef LLVMGetBuilderContext(LLVMBuilderRef Builder)
try: (LLVMGetBuilderContext:=dll.LLVMGetBuilderContext).restype, LLVMGetBuilderContext.argtypes = LLVMContextRef, [LLVMBuilderRef]
except AttributeError: pass

# void LLVMSetCurrentDebugLocation(LLVMBuilderRef Builder, LLVMValueRef L)
try: (LLVMSetCurrentDebugLocation:=dll.LLVMSetCurrentDebugLocation).restype, LLVMSetCurrentDebugLocation.argtypes = None, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetCurrentDebugLocation(LLVMBuilderRef Builder)
try: (LLVMGetCurrentDebugLocation:=dll.LLVMGetCurrentDebugLocation).restype, LLVMGetCurrentDebugLocation.argtypes = LLVMValueRef, [LLVMBuilderRef]
except AttributeError: pass

# LLVMValueRef LLVMBuildRetVoid(LLVMBuilderRef)
try: (LLVMBuildRetVoid:=dll.LLVMBuildRetVoid).restype, LLVMBuildRetVoid.argtypes = LLVMValueRef, [LLVMBuilderRef]
except AttributeError: pass

# LLVMValueRef LLVMBuildRet(LLVMBuilderRef, LLVMValueRef V)
try: (LLVMBuildRet:=dll.LLVMBuildRet).restype, LLVMBuildRet.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMBuildAggregateRet(LLVMBuilderRef, LLVMValueRef *RetVals, unsigned int N)
try: (LLVMBuildAggregateRet:=dll.LLVMBuildAggregateRet).restype, LLVMBuildAggregateRet.argtypes = LLVMValueRef, [LLVMBuilderRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMBuildBr(LLVMBuilderRef, LLVMBasicBlockRef Dest)
try: (LLVMBuildBr:=dll.LLVMBuildBr).restype, LLVMBuildBr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMBasicBlockRef]
except AttributeError: pass

# LLVMValueRef LLVMBuildCondBr(LLVMBuilderRef, LLVMValueRef If, LLVMBasicBlockRef Then, LLVMBasicBlockRef Else)
try: (LLVMBuildCondBr:=dll.LLVMBuildCondBr).restype, LLVMBuildCondBr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef, LLVMBasicBlockRef]
except AttributeError: pass

# LLVMValueRef LLVMBuildSwitch(LLVMBuilderRef, LLVMValueRef V, LLVMBasicBlockRef Else, unsigned int NumCases)
try: (LLVMBuildSwitch:=dll.LLVMBuildSwitch).restype, LLVMBuildSwitch.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMBuildIndirectBr(LLVMBuilderRef B, LLVMValueRef Addr, unsigned int NumDests)
try: (LLVMBuildIndirectBr:=dll.LLVMBuildIndirectBr).restype, LLVMBuildIndirectBr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMBuildCallBr(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef Fn, LLVMBasicBlockRef DefaultDest, LLVMBasicBlockRef *IndirectDests, unsigned int NumIndirectDests, LLVMValueRef *Args, unsigned int NumArgs, LLVMOperandBundleRef *Bundles, unsigned int NumBundles, const char *Name)
try: (LLVMBuildCallBr:=dll.LLVMBuildCallBr).restype, LLVMBuildCallBr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, LLVMBasicBlockRef, ctypes.POINTER(LLVMBasicBlockRef), ctypes.c_uint32, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(LLVMOperandBundleRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildInvoke2(LLVMBuilderRef, LLVMTypeRef Ty, LLVMValueRef Fn, LLVMValueRef *Args, unsigned int NumArgs, LLVMBasicBlockRef Then, LLVMBasicBlockRef Catch, const char *Name)
try: (LLVMBuildInvoke2:=dll.LLVMBuildInvoke2).restype, LLVMBuildInvoke2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, LLVMBasicBlockRef, LLVMBasicBlockRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildInvokeWithOperandBundles(LLVMBuilderRef, LLVMTypeRef Ty, LLVMValueRef Fn, LLVMValueRef *Args, unsigned int NumArgs, LLVMBasicBlockRef Then, LLVMBasicBlockRef Catch, LLVMOperandBundleRef *Bundles, unsigned int NumBundles, const char *Name)
try: (LLVMBuildInvokeWithOperandBundles:=dll.LLVMBuildInvokeWithOperandBundles).restype, LLVMBuildInvokeWithOperandBundles.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, LLVMBasicBlockRef, LLVMBasicBlockRef, ctypes.POINTER(LLVMOperandBundleRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildUnreachable(LLVMBuilderRef)
try: (LLVMBuildUnreachable:=dll.LLVMBuildUnreachable).restype, LLVMBuildUnreachable.argtypes = LLVMValueRef, [LLVMBuilderRef]
except AttributeError: pass

# LLVMValueRef LLVMBuildResume(LLVMBuilderRef B, LLVMValueRef Exn)
try: (LLVMBuildResume:=dll.LLVMBuildResume).restype, LLVMBuildResume.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMBuildLandingPad(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef PersFn, unsigned int NumClauses, const char *Name)
try: (LLVMBuildLandingPad:=dll.LLVMBuildLandingPad).restype, LLVMBuildLandingPad.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildCleanupRet(LLVMBuilderRef B, LLVMValueRef CatchPad, LLVMBasicBlockRef BB)
try: (LLVMBuildCleanupRet:=dll.LLVMBuildCleanupRet).restype, LLVMBuildCleanupRet.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

# LLVMValueRef LLVMBuildCatchRet(LLVMBuilderRef B, LLVMValueRef CatchPad, LLVMBasicBlockRef BB)
try: (LLVMBuildCatchRet:=dll.LLVMBuildCatchRet).restype, LLVMBuildCatchRet.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

# LLVMValueRef LLVMBuildCatchPad(LLVMBuilderRef B, LLVMValueRef ParentPad, LLVMValueRef *Args, unsigned int NumArgs, const char *Name)
try: (LLVMBuildCatchPad:=dll.LLVMBuildCatchPad).restype, LLVMBuildCatchPad.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildCleanupPad(LLVMBuilderRef B, LLVMValueRef ParentPad, LLVMValueRef *Args, unsigned int NumArgs, const char *Name)
try: (LLVMBuildCleanupPad:=dll.LLVMBuildCleanupPad).restype, LLVMBuildCleanupPad.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildCatchSwitch(LLVMBuilderRef B, LLVMValueRef ParentPad, LLVMBasicBlockRef UnwindBB, unsigned int NumHandlers, const char *Name)
try: (LLVMBuildCatchSwitch:=dll.LLVMBuildCatchSwitch).restype, LLVMBuildCatchSwitch.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMAddCase(LLVMValueRef Switch, LLVMValueRef OnVal, LLVMBasicBlockRef Dest)
try: (LLVMAddCase:=dll.LLVMAddCase).restype, LLVMAddCase.argtypes = None, [LLVMValueRef, LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

# void LLVMAddDestination(LLVMValueRef IndirectBr, LLVMBasicBlockRef Dest)
try: (LLVMAddDestination:=dll.LLVMAddDestination).restype, LLVMAddDestination.argtypes = None, [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

# unsigned int LLVMGetNumClauses(LLVMValueRef LandingPad)
try: (LLVMGetNumClauses:=dll.LLVMGetNumClauses).restype, LLVMGetNumClauses.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetClause(LLVMValueRef LandingPad, unsigned int Idx)
try: (LLVMGetClause:=dll.LLVMGetClause).restype, LLVMGetClause.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# void LLVMAddClause(LLVMValueRef LandingPad, LLVMValueRef ClauseVal)
try: (LLVMAddClause:=dll.LLVMAddClause).restype, LLVMAddClause.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMIsCleanup(LLVMValueRef LandingPad)
try: (LLVMIsCleanup:=dll.LLVMIsCleanup).restype, LLVMIsCleanup.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetCleanup(LLVMValueRef LandingPad, LLVMBool Val)
try: (LLVMSetCleanup:=dll.LLVMSetCleanup).restype, LLVMSetCleanup.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# void LLVMAddHandler(LLVMValueRef CatchSwitch, LLVMBasicBlockRef Dest)
try: (LLVMAddHandler:=dll.LLVMAddHandler).restype, LLVMAddHandler.argtypes = None, [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

# unsigned int LLVMGetNumHandlers(LLVMValueRef CatchSwitch)
try: (LLVMGetNumHandlers:=dll.LLVMGetNumHandlers).restype, LLVMGetNumHandlers.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# void LLVMGetHandlers(LLVMValueRef CatchSwitch, LLVMBasicBlockRef *Handlers)
try: (LLVMGetHandlers:=dll.LLVMGetHandlers).restype, LLVMGetHandlers.argtypes = None, [LLVMValueRef, ctypes.POINTER(LLVMBasicBlockRef)]
except AttributeError: pass

# LLVMValueRef LLVMGetArgOperand(LLVMValueRef Funclet, unsigned int i)
try: (LLVMGetArgOperand:=dll.LLVMGetArgOperand).restype, LLVMGetArgOperand.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# void LLVMSetArgOperand(LLVMValueRef Funclet, unsigned int i, LLVMValueRef value)
try: (LLVMSetArgOperand:=dll.LLVMSetArgOperand).restype, LLVMSetArgOperand.argtypes = None, [LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMGetParentCatchSwitch(LLVMValueRef CatchPad)
try: (LLVMGetParentCatchSwitch:=dll.LLVMGetParentCatchSwitch).restype, LLVMGetParentCatchSwitch.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetParentCatchSwitch(LLVMValueRef CatchPad, LLVMValueRef CatchSwitch)
try: (LLVMSetParentCatchSwitch:=dll.LLVMSetParentCatchSwitch).restype, LLVMSetParentCatchSwitch.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMBuildAdd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildAdd:=dll.LLVMBuildAdd).restype, LLVMBuildAdd.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildNSWAdd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildNSWAdd:=dll.LLVMBuildNSWAdd).restype, LLVMBuildNSWAdd.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildNUWAdd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildNUWAdd:=dll.LLVMBuildNUWAdd).restype, LLVMBuildNUWAdd.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFAdd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildFAdd:=dll.LLVMBuildFAdd).restype, LLVMBuildFAdd.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildSub(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildSub:=dll.LLVMBuildSub).restype, LLVMBuildSub.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildNSWSub(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildNSWSub:=dll.LLVMBuildNSWSub).restype, LLVMBuildNSWSub.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildNUWSub(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildNUWSub:=dll.LLVMBuildNUWSub).restype, LLVMBuildNUWSub.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFSub(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildFSub:=dll.LLVMBuildFSub).restype, LLVMBuildFSub.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildMul(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildMul:=dll.LLVMBuildMul).restype, LLVMBuildMul.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildNSWMul(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildNSWMul:=dll.LLVMBuildNSWMul).restype, LLVMBuildNSWMul.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildNUWMul(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildNUWMul:=dll.LLVMBuildNUWMul).restype, LLVMBuildNUWMul.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFMul(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildFMul:=dll.LLVMBuildFMul).restype, LLVMBuildFMul.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildUDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildUDiv:=dll.LLVMBuildUDiv).restype, LLVMBuildUDiv.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildExactUDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildExactUDiv:=dll.LLVMBuildExactUDiv).restype, LLVMBuildExactUDiv.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildSDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildSDiv:=dll.LLVMBuildSDiv).restype, LLVMBuildSDiv.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildExactSDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildExactSDiv:=dll.LLVMBuildExactSDiv).restype, LLVMBuildExactSDiv.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildFDiv:=dll.LLVMBuildFDiv).restype, LLVMBuildFDiv.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildURem(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildURem:=dll.LLVMBuildURem).restype, LLVMBuildURem.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildSRem(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildSRem:=dll.LLVMBuildSRem).restype, LLVMBuildSRem.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFRem(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildFRem:=dll.LLVMBuildFRem).restype, LLVMBuildFRem.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildShl(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildShl:=dll.LLVMBuildShl).restype, LLVMBuildShl.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildLShr(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildLShr:=dll.LLVMBuildLShr).restype, LLVMBuildLShr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildAShr(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildAShr:=dll.LLVMBuildAShr).restype, LLVMBuildAShr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildAnd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildAnd:=dll.LLVMBuildAnd).restype, LLVMBuildAnd.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildOr(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildOr:=dll.LLVMBuildOr).restype, LLVMBuildOr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildXor(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildXor:=dll.LLVMBuildXor).restype, LLVMBuildXor.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildBinOp(LLVMBuilderRef B, LLVMOpcode Op, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildBinOp:=dll.LLVMBuildBinOp).restype, LLVMBuildBinOp.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMOpcode, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildNeg(LLVMBuilderRef, LLVMValueRef V, const char *Name)
try: (LLVMBuildNeg:=dll.LLVMBuildNeg).restype, LLVMBuildNeg.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildNSWNeg(LLVMBuilderRef B, LLVMValueRef V, const char *Name)
try: (LLVMBuildNSWNeg:=dll.LLVMBuildNSWNeg).restype, LLVMBuildNSWNeg.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildNUWNeg(LLVMBuilderRef B, LLVMValueRef V, const char *Name) __attribute__((deprecated("Use LLVMBuildNeg + LLVMSetNUW instead.")))
try: (LLVMBuildNUWNeg:=dll.LLVMBuildNUWNeg).restype, LLVMBuildNUWNeg.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFNeg(LLVMBuilderRef, LLVMValueRef V, const char *Name)
try: (LLVMBuildFNeg:=dll.LLVMBuildFNeg).restype, LLVMBuildFNeg.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildNot(LLVMBuilderRef, LLVMValueRef V, const char *Name)
try: (LLVMBuildNot:=dll.LLVMBuildNot).restype, LLVMBuildNot.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBool LLVMGetNUW(LLVMValueRef ArithInst)
try: (LLVMGetNUW:=dll.LLVMGetNUW).restype, LLVMGetNUW.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetNUW(LLVMValueRef ArithInst, LLVMBool HasNUW)
try: (LLVMSetNUW:=dll.LLVMSetNUW).restype, LLVMSetNUW.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# LLVMBool LLVMGetNSW(LLVMValueRef ArithInst)
try: (LLVMGetNSW:=dll.LLVMGetNSW).restype, LLVMGetNSW.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetNSW(LLVMValueRef ArithInst, LLVMBool HasNSW)
try: (LLVMSetNSW:=dll.LLVMSetNSW).restype, LLVMSetNSW.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# LLVMBool LLVMGetExact(LLVMValueRef DivOrShrInst)
try: (LLVMGetExact:=dll.LLVMGetExact).restype, LLVMGetExact.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetExact(LLVMValueRef DivOrShrInst, LLVMBool IsExact)
try: (LLVMSetExact:=dll.LLVMSetExact).restype, LLVMSetExact.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# LLVMBool LLVMGetNNeg(LLVMValueRef NonNegInst)
try: (LLVMGetNNeg:=dll.LLVMGetNNeg).restype, LLVMGetNNeg.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetNNeg(LLVMValueRef NonNegInst, LLVMBool IsNonNeg)
try: (LLVMSetNNeg:=dll.LLVMSetNNeg).restype, LLVMSetNNeg.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# LLVMFastMathFlags LLVMGetFastMathFlags(LLVMValueRef FPMathInst)
try: (LLVMGetFastMathFlags:=dll.LLVMGetFastMathFlags).restype, LLVMGetFastMathFlags.argtypes = LLVMFastMathFlags, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetFastMathFlags(LLVMValueRef FPMathInst, LLVMFastMathFlags FMF)
try: (LLVMSetFastMathFlags:=dll.LLVMSetFastMathFlags).restype, LLVMSetFastMathFlags.argtypes = None, [LLVMValueRef, LLVMFastMathFlags]
except AttributeError: pass

# LLVMBool LLVMCanValueUseFastMathFlags(LLVMValueRef Inst)
try: (LLVMCanValueUseFastMathFlags:=dll.LLVMCanValueUseFastMathFlags).restype, LLVMCanValueUseFastMathFlags.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMGetIsDisjoint(LLVMValueRef Inst)
try: (LLVMGetIsDisjoint:=dll.LLVMGetIsDisjoint).restype, LLVMGetIsDisjoint.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetIsDisjoint(LLVMValueRef Inst, LLVMBool IsDisjoint)
try: (LLVMSetIsDisjoint:=dll.LLVMSetIsDisjoint).restype, LLVMSetIsDisjoint.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# LLVMValueRef LLVMBuildMalloc(LLVMBuilderRef, LLVMTypeRef Ty, const char *Name)
try: (LLVMBuildMalloc:=dll.LLVMBuildMalloc).restype, LLVMBuildMalloc.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildArrayMalloc(LLVMBuilderRef, LLVMTypeRef Ty, LLVMValueRef Val, const char *Name)
try: (LLVMBuildArrayMalloc:=dll.LLVMBuildArrayMalloc).restype, LLVMBuildArrayMalloc.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildMemSet(LLVMBuilderRef B, LLVMValueRef Ptr, LLVMValueRef Val, LLVMValueRef Len, unsigned int Align)
try: (LLVMBuildMemSet:=dll.LLVMBuildMemSet).restype, LLVMBuildMemSet.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMBuildMemCpy(LLVMBuilderRef B, LLVMValueRef Dst, unsigned int DstAlign, LLVMValueRef Src, unsigned int SrcAlign, LLVMValueRef Size)
try: (LLVMBuildMemCpy:=dll.LLVMBuildMemCpy).restype, LLVMBuildMemCpy.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMBuildMemMove(LLVMBuilderRef B, LLVMValueRef Dst, unsigned int DstAlign, LLVMValueRef Src, unsigned int SrcAlign, LLVMValueRef Size)
try: (LLVMBuildMemMove:=dll.LLVMBuildMemMove).restype, LLVMBuildMemMove.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMBuildAlloca(LLVMBuilderRef, LLVMTypeRef Ty, const char *Name)
try: (LLVMBuildAlloca:=dll.LLVMBuildAlloca).restype, LLVMBuildAlloca.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildArrayAlloca(LLVMBuilderRef, LLVMTypeRef Ty, LLVMValueRef Val, const char *Name)
try: (LLVMBuildArrayAlloca:=dll.LLVMBuildArrayAlloca).restype, LLVMBuildArrayAlloca.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFree(LLVMBuilderRef, LLVMValueRef PointerVal)
try: (LLVMBuildFree:=dll.LLVMBuildFree).restype, LLVMBuildFree.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMBuildLoad2(LLVMBuilderRef, LLVMTypeRef Ty, LLVMValueRef PointerVal, const char *Name)
try: (LLVMBuildLoad2:=dll.LLVMBuildLoad2).restype, LLVMBuildLoad2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildStore(LLVMBuilderRef, LLVMValueRef Val, LLVMValueRef Ptr)
try: (LLVMBuildStore:=dll.LLVMBuildStore).restype, LLVMBuildStore.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

# LLVMValueRef LLVMBuildGEP2(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef Pointer, LLVMValueRef *Indices, unsigned int NumIndices, const char *Name)
try: (LLVMBuildGEP2:=dll.LLVMBuildGEP2).restype, LLVMBuildGEP2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildInBoundsGEP2(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef Pointer, LLVMValueRef *Indices, unsigned int NumIndices, const char *Name)
try: (LLVMBuildInBoundsGEP2:=dll.LLVMBuildInBoundsGEP2).restype, LLVMBuildInBoundsGEP2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildGEPWithNoWrapFlags(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef Pointer, LLVMValueRef *Indices, unsigned int NumIndices, const char *Name, LLVMGEPNoWrapFlags NoWrapFlags)
try: (LLVMBuildGEPWithNoWrapFlags:=dll.LLVMBuildGEPWithNoWrapFlags).restype, LLVMBuildGEPWithNoWrapFlags.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), LLVMGEPNoWrapFlags]
except AttributeError: pass

# LLVMValueRef LLVMBuildStructGEP2(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef Pointer, unsigned int Idx, const char *Name)
try: (LLVMBuildStructGEP2:=dll.LLVMBuildStructGEP2).restype, LLVMBuildStructGEP2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildGlobalString(LLVMBuilderRef B, const char *Str, const char *Name)
try: (LLVMBuildGlobalString:=dll.LLVMBuildGlobalString).restype, LLVMBuildGlobalString.argtypes = LLVMValueRef, [LLVMBuilderRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildGlobalStringPtr(LLVMBuilderRef B, const char *Str, const char *Name)
try: (LLVMBuildGlobalStringPtr:=dll.LLVMBuildGlobalStringPtr).restype, LLVMBuildGlobalStringPtr.argtypes = LLVMValueRef, [LLVMBuilderRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBool LLVMGetVolatile(LLVMValueRef MemoryAccessInst)
try: (LLVMGetVolatile:=dll.LLVMGetVolatile).restype, LLVMGetVolatile.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetVolatile(LLVMValueRef MemoryAccessInst, LLVMBool IsVolatile)
try: (LLVMSetVolatile:=dll.LLVMSetVolatile).restype, LLVMSetVolatile.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# LLVMBool LLVMGetWeak(LLVMValueRef CmpXchgInst)
try: (LLVMGetWeak:=dll.LLVMGetWeak).restype, LLVMGetWeak.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetWeak(LLVMValueRef CmpXchgInst, LLVMBool IsWeak)
try: (LLVMSetWeak:=dll.LLVMSetWeak).restype, LLVMSetWeak.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# LLVMAtomicOrdering LLVMGetOrdering(LLVMValueRef MemoryAccessInst)
try: (LLVMGetOrdering:=dll.LLVMGetOrdering).restype, LLVMGetOrdering.argtypes = LLVMAtomicOrdering, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetOrdering(LLVMValueRef MemoryAccessInst, LLVMAtomicOrdering Ordering)
try: (LLVMSetOrdering:=dll.LLVMSetOrdering).restype, LLVMSetOrdering.argtypes = None, [LLVMValueRef, LLVMAtomicOrdering]
except AttributeError: pass

# LLVMAtomicRMWBinOp LLVMGetAtomicRMWBinOp(LLVMValueRef AtomicRMWInst)
try: (LLVMGetAtomicRMWBinOp:=dll.LLVMGetAtomicRMWBinOp).restype, LLVMGetAtomicRMWBinOp.argtypes = LLVMAtomicRMWBinOp, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetAtomicRMWBinOp(LLVMValueRef AtomicRMWInst, LLVMAtomicRMWBinOp BinOp)
try: (LLVMSetAtomicRMWBinOp:=dll.LLVMSetAtomicRMWBinOp).restype, LLVMSetAtomicRMWBinOp.argtypes = None, [LLVMValueRef, LLVMAtomicRMWBinOp]
except AttributeError: pass

# LLVMValueRef LLVMBuildTrunc(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildTrunc:=dll.LLVMBuildTrunc).restype, LLVMBuildTrunc.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildZExt(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildZExt:=dll.LLVMBuildZExt).restype, LLVMBuildZExt.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildSExt(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildSExt:=dll.LLVMBuildSExt).restype, LLVMBuildSExt.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFPToUI(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildFPToUI:=dll.LLVMBuildFPToUI).restype, LLVMBuildFPToUI.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFPToSI(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildFPToSI:=dll.LLVMBuildFPToSI).restype, LLVMBuildFPToSI.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildUIToFP(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildUIToFP:=dll.LLVMBuildUIToFP).restype, LLVMBuildUIToFP.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildSIToFP(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildSIToFP:=dll.LLVMBuildSIToFP).restype, LLVMBuildSIToFP.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFPTrunc(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildFPTrunc:=dll.LLVMBuildFPTrunc).restype, LLVMBuildFPTrunc.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFPExt(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildFPExt:=dll.LLVMBuildFPExt).restype, LLVMBuildFPExt.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildPtrToInt(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildPtrToInt:=dll.LLVMBuildPtrToInt).restype, LLVMBuildPtrToInt.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildIntToPtr(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildIntToPtr:=dll.LLVMBuildIntToPtr).restype, LLVMBuildIntToPtr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildBitCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildBitCast:=dll.LLVMBuildBitCast).restype, LLVMBuildBitCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildAddrSpaceCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildAddrSpaceCast:=dll.LLVMBuildAddrSpaceCast).restype, LLVMBuildAddrSpaceCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildZExtOrBitCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildZExtOrBitCast:=dll.LLVMBuildZExtOrBitCast).restype, LLVMBuildZExtOrBitCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildSExtOrBitCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildSExtOrBitCast:=dll.LLVMBuildSExtOrBitCast).restype, LLVMBuildSExtOrBitCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildTruncOrBitCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildTruncOrBitCast:=dll.LLVMBuildTruncOrBitCast).restype, LLVMBuildTruncOrBitCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildCast(LLVMBuilderRef B, LLVMOpcode Op, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildCast:=dll.LLVMBuildCast).restype, LLVMBuildCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMOpcode, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildPointerCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildPointerCast:=dll.LLVMBuildPointerCast).restype, LLVMBuildPointerCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildIntCast2(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, LLVMBool IsSigned, const char *Name)
try: (LLVMBuildIntCast2:=dll.LLVMBuildIntCast2).restype, LLVMBuildIntCast2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, LLVMBool, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFPCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildFPCast:=dll.LLVMBuildFPCast).restype, LLVMBuildFPCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildIntCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char *Name)
try: (LLVMBuildIntCast:=dll.LLVMBuildIntCast).restype, LLVMBuildIntCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMOpcode LLVMGetCastOpcode(LLVMValueRef Src, LLVMBool SrcIsSigned, LLVMTypeRef DestTy, LLVMBool DestIsSigned)
try: (LLVMGetCastOpcode:=dll.LLVMGetCastOpcode).restype, LLVMGetCastOpcode.argtypes = LLVMOpcode, [LLVMValueRef, LLVMBool, LLVMTypeRef, LLVMBool]
except AttributeError: pass

# LLVMValueRef LLVMBuildICmp(LLVMBuilderRef, LLVMIntPredicate Op, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildICmp:=dll.LLVMBuildICmp).restype, LLVMBuildICmp.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMIntPredicate, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFCmp(LLVMBuilderRef, LLVMRealPredicate Op, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildFCmp:=dll.LLVMBuildFCmp).restype, LLVMBuildFCmp.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMRealPredicate, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildPhi(LLVMBuilderRef, LLVMTypeRef Ty, const char *Name)
try: (LLVMBuildPhi:=dll.LLVMBuildPhi).restype, LLVMBuildPhi.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildCall2(LLVMBuilderRef, LLVMTypeRef, LLVMValueRef Fn, LLVMValueRef *Args, unsigned int NumArgs, const char *Name)
try: (LLVMBuildCall2:=dll.LLVMBuildCall2).restype, LLVMBuildCall2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildCallWithOperandBundles(LLVMBuilderRef, LLVMTypeRef, LLVMValueRef Fn, LLVMValueRef *Args, unsigned int NumArgs, LLVMOperandBundleRef *Bundles, unsigned int NumBundles, const char *Name)
try: (LLVMBuildCallWithOperandBundles:=dll.LLVMBuildCallWithOperandBundles).restype, LLVMBuildCallWithOperandBundles.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(LLVMOperandBundleRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildSelect(LLVMBuilderRef, LLVMValueRef If, LLVMValueRef Then, LLVMValueRef Else, const char *Name)
try: (LLVMBuildSelect:=dll.LLVMBuildSelect).restype, LLVMBuildSelect.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildVAArg(LLVMBuilderRef, LLVMValueRef List, LLVMTypeRef Ty, const char *Name)
try: (LLVMBuildVAArg:=dll.LLVMBuildVAArg).restype, LLVMBuildVAArg.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildExtractElement(LLVMBuilderRef, LLVMValueRef VecVal, LLVMValueRef Index, const char *Name)
try: (LLVMBuildExtractElement:=dll.LLVMBuildExtractElement).restype, LLVMBuildExtractElement.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildInsertElement(LLVMBuilderRef, LLVMValueRef VecVal, LLVMValueRef EltVal, LLVMValueRef Index, const char *Name)
try: (LLVMBuildInsertElement:=dll.LLVMBuildInsertElement).restype, LLVMBuildInsertElement.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildShuffleVector(LLVMBuilderRef, LLVMValueRef V1, LLVMValueRef V2, LLVMValueRef Mask, const char *Name)
try: (LLVMBuildShuffleVector:=dll.LLVMBuildShuffleVector).restype, LLVMBuildShuffleVector.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildExtractValue(LLVMBuilderRef, LLVMValueRef AggVal, unsigned int Index, const char *Name)
try: (LLVMBuildExtractValue:=dll.LLVMBuildExtractValue).restype, LLVMBuildExtractValue.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildInsertValue(LLVMBuilderRef, LLVMValueRef AggVal, LLVMValueRef EltVal, unsigned int Index, const char *Name)
try: (LLVMBuildInsertValue:=dll.LLVMBuildInsertValue).restype, LLVMBuildInsertValue.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFreeze(LLVMBuilderRef, LLVMValueRef Val, const char *Name)
try: (LLVMBuildFreeze:=dll.LLVMBuildFreeze).restype, LLVMBuildFreeze.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildIsNull(LLVMBuilderRef, LLVMValueRef Val, const char *Name)
try: (LLVMBuildIsNull:=dll.LLVMBuildIsNull).restype, LLVMBuildIsNull.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildIsNotNull(LLVMBuilderRef, LLVMValueRef Val, const char *Name)
try: (LLVMBuildIsNotNull:=dll.LLVMBuildIsNotNull).restype, LLVMBuildIsNotNull.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildPtrDiff2(LLVMBuilderRef, LLVMTypeRef ElemTy, LLVMValueRef LHS, LLVMValueRef RHS, const char *Name)
try: (LLVMBuildPtrDiff2:=dll.LLVMBuildPtrDiff2).restype, LLVMBuildPtrDiff2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFence(LLVMBuilderRef B, LLVMAtomicOrdering ordering, LLVMBool singleThread, const char *Name)
try: (LLVMBuildFence:=dll.LLVMBuildFence).restype, LLVMBuildFence.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMAtomicOrdering, LLVMBool, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildFenceSyncScope(LLVMBuilderRef B, LLVMAtomicOrdering ordering, unsigned int SSID, const char *Name)
try: (LLVMBuildFenceSyncScope:=dll.LLVMBuildFenceSyncScope).restype, LLVMBuildFenceSyncScope.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMAtomicOrdering, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMValueRef LLVMBuildAtomicRMW(LLVMBuilderRef B, LLVMAtomicRMWBinOp op, LLVMValueRef PTR, LLVMValueRef Val, LLVMAtomicOrdering ordering, LLVMBool singleThread)
try: (LLVMBuildAtomicRMW:=dll.LLVMBuildAtomicRMW).restype, LLVMBuildAtomicRMW.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMAtomicRMWBinOp, LLVMValueRef, LLVMValueRef, LLVMAtomicOrdering, LLVMBool]
except AttributeError: pass

# LLVMValueRef LLVMBuildAtomicRMWSyncScope(LLVMBuilderRef B, LLVMAtomicRMWBinOp op, LLVMValueRef PTR, LLVMValueRef Val, LLVMAtomicOrdering ordering, unsigned int SSID)
try: (LLVMBuildAtomicRMWSyncScope:=dll.LLVMBuildAtomicRMWSyncScope).restype, LLVMBuildAtomicRMWSyncScope.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMAtomicRMWBinOp, LLVMValueRef, LLVMValueRef, LLVMAtomicOrdering, ctypes.c_uint32]
except AttributeError: pass

# LLVMValueRef LLVMBuildAtomicCmpXchg(LLVMBuilderRef B, LLVMValueRef Ptr, LLVMValueRef Cmp, LLVMValueRef New, LLVMAtomicOrdering SuccessOrdering, LLVMAtomicOrdering FailureOrdering, LLVMBool SingleThread)
try: (LLVMBuildAtomicCmpXchg:=dll.LLVMBuildAtomicCmpXchg).restype, LLVMBuildAtomicCmpXchg.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMAtomicOrdering, LLVMAtomicOrdering, LLVMBool]
except AttributeError: pass

# LLVMValueRef LLVMBuildAtomicCmpXchgSyncScope(LLVMBuilderRef B, LLVMValueRef Ptr, LLVMValueRef Cmp, LLVMValueRef New, LLVMAtomicOrdering SuccessOrdering, LLVMAtomicOrdering FailureOrdering, unsigned int SSID)
try: (LLVMBuildAtomicCmpXchgSyncScope:=dll.LLVMBuildAtomicCmpXchgSyncScope).restype, LLVMBuildAtomicCmpXchgSyncScope.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMAtomicOrdering, LLVMAtomicOrdering, ctypes.c_uint32]
except AttributeError: pass

# unsigned int LLVMGetNumMaskElements(LLVMValueRef ShuffleVectorInst)
try: (LLVMGetNumMaskElements:=dll.LLVMGetNumMaskElements).restype, LLVMGetNumMaskElements.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# int LLVMGetUndefMaskElem(void)
try: (LLVMGetUndefMaskElem:=dll.LLVMGetUndefMaskElem).restype, LLVMGetUndefMaskElem.argtypes = ctypes.c_int32, []
except AttributeError: pass

# int LLVMGetMaskValue(LLVMValueRef ShuffleVectorInst, unsigned int Elt)
try: (LLVMGetMaskValue:=dll.LLVMGetMaskValue).restype, LLVMGetMaskValue.argtypes = ctypes.c_int32, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMBool LLVMIsAtomicSingleThread(LLVMValueRef AtomicInst)
try: (LLVMIsAtomicSingleThread:=dll.LLVMIsAtomicSingleThread).restype, LLVMIsAtomicSingleThread.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetAtomicSingleThread(LLVMValueRef AtomicInst, LLVMBool SingleThread)
try: (LLVMSetAtomicSingleThread:=dll.LLVMSetAtomicSingleThread).restype, LLVMSetAtomicSingleThread.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

# LLVMBool LLVMIsAtomic(LLVMValueRef Inst)
try: (LLVMIsAtomic:=dll.LLVMIsAtomic).restype, LLVMIsAtomic.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

# unsigned int LLVMGetAtomicSyncScopeID(LLVMValueRef AtomicInst)
try: (LLVMGetAtomicSyncScopeID:=dll.LLVMGetAtomicSyncScopeID).restype, LLVMGetAtomicSyncScopeID.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetAtomicSyncScopeID(LLVMValueRef AtomicInst, unsigned int SSID)
try: (LLVMSetAtomicSyncScopeID:=dll.LLVMSetAtomicSyncScopeID).restype, LLVMSetAtomicSyncScopeID.argtypes = None, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMAtomicOrdering LLVMGetCmpXchgSuccessOrdering(LLVMValueRef CmpXchgInst)
try: (LLVMGetCmpXchgSuccessOrdering:=dll.LLVMGetCmpXchgSuccessOrdering).restype, LLVMGetCmpXchgSuccessOrdering.argtypes = LLVMAtomicOrdering, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetCmpXchgSuccessOrdering(LLVMValueRef CmpXchgInst, LLVMAtomicOrdering Ordering)
try: (LLVMSetCmpXchgSuccessOrdering:=dll.LLVMSetCmpXchgSuccessOrdering).restype, LLVMSetCmpXchgSuccessOrdering.argtypes = None, [LLVMValueRef, LLVMAtomicOrdering]
except AttributeError: pass

# LLVMAtomicOrdering LLVMGetCmpXchgFailureOrdering(LLVMValueRef CmpXchgInst)
try: (LLVMGetCmpXchgFailureOrdering:=dll.LLVMGetCmpXchgFailureOrdering).restype, LLVMGetCmpXchgFailureOrdering.argtypes = LLVMAtomicOrdering, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetCmpXchgFailureOrdering(LLVMValueRef CmpXchgInst, LLVMAtomicOrdering Ordering)
try: (LLVMSetCmpXchgFailureOrdering:=dll.LLVMSetCmpXchgFailureOrdering).restype, LLVMSetCmpXchgFailureOrdering.argtypes = None, [LLVMValueRef, LLVMAtomicOrdering]
except AttributeError: pass

class struct_LLVMOpaqueModuleProvider(Struct): pass
LLVMModuleProviderRef = ctypes.POINTER(struct_LLVMOpaqueModuleProvider)
# LLVMModuleProviderRef LLVMCreateModuleProviderForExistingModule(LLVMModuleRef M)
try: (LLVMCreateModuleProviderForExistingModule:=dll.LLVMCreateModuleProviderForExistingModule).restype, LLVMCreateModuleProviderForExistingModule.argtypes = LLVMModuleProviderRef, [LLVMModuleRef]
except AttributeError: pass

# void LLVMDisposeModuleProvider(LLVMModuleProviderRef M)
try: (LLVMDisposeModuleProvider:=dll.LLVMDisposeModuleProvider).restype, LLVMDisposeModuleProvider.argtypes = None, [LLVMModuleProviderRef]
except AttributeError: pass

# LLVMBool LLVMCreateMemoryBufferWithContentsOfFile(const char *Path, LLVMMemoryBufferRef *OutMemBuf, char **OutMessage)
try: (LLVMCreateMemoryBufferWithContentsOfFile:=dll.LLVMCreateMemoryBufferWithContentsOfFile).restype, LLVMCreateMemoryBufferWithContentsOfFile.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMMemoryBufferRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMCreateMemoryBufferWithSTDIN(LLVMMemoryBufferRef *OutMemBuf, char **OutMessage)
try: (LLVMCreateMemoryBufferWithSTDIN:=dll.LLVMCreateMemoryBufferWithSTDIN).restype, LLVMCreateMemoryBufferWithSTDIN.argtypes = LLVMBool, [ctypes.POINTER(LLVMMemoryBufferRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMMemoryBufferRef LLVMCreateMemoryBufferWithMemoryRange(const char *InputData, size_t InputDataLength, const char *BufferName, LLVMBool RequiresNullTerminator)
try: (LLVMCreateMemoryBufferWithMemoryRange:=dll.LLVMCreateMemoryBufferWithMemoryRange).restype, LLVMCreateMemoryBufferWithMemoryRange.argtypes = LLVMMemoryBufferRef, [ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), LLVMBool]
except AttributeError: pass

# LLVMMemoryBufferRef LLVMCreateMemoryBufferWithMemoryRangeCopy(const char *InputData, size_t InputDataLength, const char *BufferName)
try: (LLVMCreateMemoryBufferWithMemoryRangeCopy:=dll.LLVMCreateMemoryBufferWithMemoryRangeCopy).restype, LLVMCreateMemoryBufferWithMemoryRangeCopy.argtypes = LLVMMemoryBufferRef, [ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# const char *LLVMGetBufferStart(LLVMMemoryBufferRef MemBuf)
try: (LLVMGetBufferStart:=dll.LLVMGetBufferStart).restype, LLVMGetBufferStart.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMMemoryBufferRef]
except AttributeError: pass

# size_t LLVMGetBufferSize(LLVMMemoryBufferRef MemBuf)
try: (LLVMGetBufferSize:=dll.LLVMGetBufferSize).restype, LLVMGetBufferSize.argtypes = size_t, [LLVMMemoryBufferRef]
except AttributeError: pass

# void LLVMDisposeMemoryBuffer(LLVMMemoryBufferRef MemBuf)
try: (LLVMDisposeMemoryBuffer:=dll.LLVMDisposeMemoryBuffer).restype, LLVMDisposeMemoryBuffer.argtypes = None, [LLVMMemoryBufferRef]
except AttributeError: pass

class struct_LLVMOpaquePassManager(Struct): pass
LLVMPassManagerRef = ctypes.POINTER(struct_LLVMOpaquePassManager)
# LLVMPassManagerRef LLVMCreatePassManager(void)
try: (LLVMCreatePassManager:=dll.LLVMCreatePassManager).restype, LLVMCreatePassManager.argtypes = LLVMPassManagerRef, []
except AttributeError: pass

# LLVMPassManagerRef LLVMCreateFunctionPassManagerForModule(LLVMModuleRef M)
try: (LLVMCreateFunctionPassManagerForModule:=dll.LLVMCreateFunctionPassManagerForModule).restype, LLVMCreateFunctionPassManagerForModule.argtypes = LLVMPassManagerRef, [LLVMModuleRef]
except AttributeError: pass

# LLVMPassManagerRef LLVMCreateFunctionPassManager(LLVMModuleProviderRef MP)
try: (LLVMCreateFunctionPassManager:=dll.LLVMCreateFunctionPassManager).restype, LLVMCreateFunctionPassManager.argtypes = LLVMPassManagerRef, [LLVMModuleProviderRef]
except AttributeError: pass

# LLVMBool LLVMRunPassManager(LLVMPassManagerRef PM, LLVMModuleRef M)
try: (LLVMRunPassManager:=dll.LLVMRunPassManager).restype, LLVMRunPassManager.argtypes = LLVMBool, [LLVMPassManagerRef, LLVMModuleRef]
except AttributeError: pass

# LLVMBool LLVMInitializeFunctionPassManager(LLVMPassManagerRef FPM)
try: (LLVMInitializeFunctionPassManager:=dll.LLVMInitializeFunctionPassManager).restype, LLVMInitializeFunctionPassManager.argtypes = LLVMBool, [LLVMPassManagerRef]
except AttributeError: pass

# LLVMBool LLVMRunFunctionPassManager(LLVMPassManagerRef FPM, LLVMValueRef F)
try: (LLVMRunFunctionPassManager:=dll.LLVMRunFunctionPassManager).restype, LLVMRunFunctionPassManager.argtypes = LLVMBool, [LLVMPassManagerRef, LLVMValueRef]
except AttributeError: pass

# LLVMBool LLVMFinalizeFunctionPassManager(LLVMPassManagerRef FPM)
try: (LLVMFinalizeFunctionPassManager:=dll.LLVMFinalizeFunctionPassManager).restype, LLVMFinalizeFunctionPassManager.argtypes = LLVMBool, [LLVMPassManagerRef]
except AttributeError: pass

# void LLVMDisposePassManager(LLVMPassManagerRef PM)
try: (LLVMDisposePassManager:=dll.LLVMDisposePassManager).restype, LLVMDisposePassManager.argtypes = None, [LLVMPassManagerRef]
except AttributeError: pass

# LLVMBool LLVMStartMultithreaded(void)
try: (LLVMStartMultithreaded:=dll.LLVMStartMultithreaded).restype, LLVMStartMultithreaded.argtypes = LLVMBool, []
except AttributeError: pass

# void LLVMStopMultithreaded(void)
try: (LLVMStopMultithreaded:=dll.LLVMStopMultithreaded).restype, LLVMStopMultithreaded.argtypes = None, []
except AttributeError: pass

# LLVMBool LLVMIsMultithreaded(void)
try: (LLVMIsMultithreaded:=dll.LLVMIsMultithreaded).restype, LLVMIsMultithreaded.argtypes = LLVMBool, []
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

LLVMDIFlags = CEnum(ctypes.c_uint32)
LLVMDIFlagZero = LLVMDIFlags.define('LLVMDIFlagZero', 0)
LLVMDIFlagPrivate = LLVMDIFlags.define('LLVMDIFlagPrivate', 1)
LLVMDIFlagProtected = LLVMDIFlags.define('LLVMDIFlagProtected', 2)
LLVMDIFlagPublic = LLVMDIFlags.define('LLVMDIFlagPublic', 3)
LLVMDIFlagFwdDecl = LLVMDIFlags.define('LLVMDIFlagFwdDecl', 4)
LLVMDIFlagAppleBlock = LLVMDIFlags.define('LLVMDIFlagAppleBlock', 8)
LLVMDIFlagReservedBit4 = LLVMDIFlags.define('LLVMDIFlagReservedBit4', 16)
LLVMDIFlagVirtual = LLVMDIFlags.define('LLVMDIFlagVirtual', 32)
LLVMDIFlagArtificial = LLVMDIFlags.define('LLVMDIFlagArtificial', 64)
LLVMDIFlagExplicit = LLVMDIFlags.define('LLVMDIFlagExplicit', 128)
LLVMDIFlagPrototyped = LLVMDIFlags.define('LLVMDIFlagPrototyped', 256)
LLVMDIFlagObjcClassComplete = LLVMDIFlags.define('LLVMDIFlagObjcClassComplete', 512)
LLVMDIFlagObjectPointer = LLVMDIFlags.define('LLVMDIFlagObjectPointer', 1024)
LLVMDIFlagVector = LLVMDIFlags.define('LLVMDIFlagVector', 2048)
LLVMDIFlagStaticMember = LLVMDIFlags.define('LLVMDIFlagStaticMember', 4096)
LLVMDIFlagLValueReference = LLVMDIFlags.define('LLVMDIFlagLValueReference', 8192)
LLVMDIFlagRValueReference = LLVMDIFlags.define('LLVMDIFlagRValueReference', 16384)
LLVMDIFlagReserved = LLVMDIFlags.define('LLVMDIFlagReserved', 32768)
LLVMDIFlagSingleInheritance = LLVMDIFlags.define('LLVMDIFlagSingleInheritance', 65536)
LLVMDIFlagMultipleInheritance = LLVMDIFlags.define('LLVMDIFlagMultipleInheritance', 131072)
LLVMDIFlagVirtualInheritance = LLVMDIFlags.define('LLVMDIFlagVirtualInheritance', 196608)
LLVMDIFlagIntroducedVirtual = LLVMDIFlags.define('LLVMDIFlagIntroducedVirtual', 262144)
LLVMDIFlagBitField = LLVMDIFlags.define('LLVMDIFlagBitField', 524288)
LLVMDIFlagNoReturn = LLVMDIFlags.define('LLVMDIFlagNoReturn', 1048576)
LLVMDIFlagTypePassByValue = LLVMDIFlags.define('LLVMDIFlagTypePassByValue', 4194304)
LLVMDIFlagTypePassByReference = LLVMDIFlags.define('LLVMDIFlagTypePassByReference', 8388608)
LLVMDIFlagEnumClass = LLVMDIFlags.define('LLVMDIFlagEnumClass', 16777216)
LLVMDIFlagFixedEnum = LLVMDIFlags.define('LLVMDIFlagFixedEnum', 16777216)
LLVMDIFlagThunk = LLVMDIFlags.define('LLVMDIFlagThunk', 33554432)
LLVMDIFlagNonTrivial = LLVMDIFlags.define('LLVMDIFlagNonTrivial', 67108864)
LLVMDIFlagBigEndian = LLVMDIFlags.define('LLVMDIFlagBigEndian', 134217728)
LLVMDIFlagLittleEndian = LLVMDIFlags.define('LLVMDIFlagLittleEndian', 268435456)
LLVMDIFlagIndirectVirtualBase = LLVMDIFlags.define('LLVMDIFlagIndirectVirtualBase', 36)
LLVMDIFlagAccessibility = LLVMDIFlags.define('LLVMDIFlagAccessibility', 3)
LLVMDIFlagPtrToMemberRep = LLVMDIFlags.define('LLVMDIFlagPtrToMemberRep', 196608)

LLVMDWARFSourceLanguage = CEnum(ctypes.c_uint32)
LLVMDWARFSourceLanguageC89 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC89', 0)
LLVMDWARFSourceLanguageC = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC', 1)
LLVMDWARFSourceLanguageAda83 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageAda83', 2)
LLVMDWARFSourceLanguageC_plus_plus = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC_plus_plus', 3)
LLVMDWARFSourceLanguageCobol74 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageCobol74', 4)
LLVMDWARFSourceLanguageCobol85 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageCobol85', 5)
LLVMDWARFSourceLanguageFortran77 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageFortran77', 6)
LLVMDWARFSourceLanguageFortran90 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageFortran90', 7)
LLVMDWARFSourceLanguagePascal83 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguagePascal83', 8)
LLVMDWARFSourceLanguageModula2 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageModula2', 9)
LLVMDWARFSourceLanguageJava = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageJava', 10)
LLVMDWARFSourceLanguageC99 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC99', 11)
LLVMDWARFSourceLanguageAda95 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageAda95', 12)
LLVMDWARFSourceLanguageFortran95 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageFortran95', 13)
LLVMDWARFSourceLanguagePLI = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguagePLI', 14)
LLVMDWARFSourceLanguageObjC = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageObjC', 15)
LLVMDWARFSourceLanguageObjC_plus_plus = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageObjC_plus_plus', 16)
LLVMDWARFSourceLanguageUPC = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageUPC', 17)
LLVMDWARFSourceLanguageD = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageD', 18)
LLVMDWARFSourceLanguagePython = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguagePython', 19)
LLVMDWARFSourceLanguageOpenCL = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageOpenCL', 20)
LLVMDWARFSourceLanguageGo = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageGo', 21)
LLVMDWARFSourceLanguageModula3 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageModula3', 22)
LLVMDWARFSourceLanguageHaskell = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageHaskell', 23)
LLVMDWARFSourceLanguageC_plus_plus_03 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC_plus_plus_03', 24)
LLVMDWARFSourceLanguageC_plus_plus_11 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC_plus_plus_11', 25)
LLVMDWARFSourceLanguageOCaml = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageOCaml', 26)
LLVMDWARFSourceLanguageRust = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageRust', 27)
LLVMDWARFSourceLanguageC11 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC11', 28)
LLVMDWARFSourceLanguageSwift = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageSwift', 29)
LLVMDWARFSourceLanguageJulia = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageJulia', 30)
LLVMDWARFSourceLanguageDylan = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageDylan', 31)
LLVMDWARFSourceLanguageC_plus_plus_14 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC_plus_plus_14', 32)
LLVMDWARFSourceLanguageFortran03 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageFortran03', 33)
LLVMDWARFSourceLanguageFortran08 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageFortran08', 34)
LLVMDWARFSourceLanguageRenderScript = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageRenderScript', 35)
LLVMDWARFSourceLanguageBLISS = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageBLISS', 36)
LLVMDWARFSourceLanguageKotlin = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageKotlin', 37)
LLVMDWARFSourceLanguageZig = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageZig', 38)
LLVMDWARFSourceLanguageCrystal = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageCrystal', 39)
LLVMDWARFSourceLanguageC_plus_plus_17 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC_plus_plus_17', 40)
LLVMDWARFSourceLanguageC_plus_plus_20 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC_plus_plus_20', 41)
LLVMDWARFSourceLanguageC17 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC17', 42)
LLVMDWARFSourceLanguageFortran18 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageFortran18', 43)
LLVMDWARFSourceLanguageAda2005 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageAda2005', 44)
LLVMDWARFSourceLanguageAda2012 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageAda2012', 45)
LLVMDWARFSourceLanguageHIP = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageHIP', 46)
LLVMDWARFSourceLanguageAssembly = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageAssembly', 47)
LLVMDWARFSourceLanguageC_sharp = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC_sharp', 48)
LLVMDWARFSourceLanguageMojo = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageMojo', 49)
LLVMDWARFSourceLanguageGLSL = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageGLSL', 50)
LLVMDWARFSourceLanguageGLSL_ES = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageGLSL_ES', 51)
LLVMDWARFSourceLanguageHLSL = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageHLSL', 52)
LLVMDWARFSourceLanguageOpenCL_CPP = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageOpenCL_CPP', 53)
LLVMDWARFSourceLanguageCPP_for_OpenCL = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageCPP_for_OpenCL', 54)
LLVMDWARFSourceLanguageSYCL = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageSYCL', 55)
LLVMDWARFSourceLanguageRuby = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageRuby', 56)
LLVMDWARFSourceLanguageMove = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageMove', 57)
LLVMDWARFSourceLanguageHylo = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageHylo', 58)
LLVMDWARFSourceLanguageMetal = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageMetal', 59)
LLVMDWARFSourceLanguageMips_Assembler = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageMips_Assembler', 60)
LLVMDWARFSourceLanguageGOOGLE_RenderScript = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageGOOGLE_RenderScript', 61)
LLVMDWARFSourceLanguageBORLAND_Delphi = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageBORLAND_Delphi', 62)

LLVMDWARFEmissionKind = CEnum(ctypes.c_uint32)
LLVMDWARFEmissionNone = LLVMDWARFEmissionKind.define('LLVMDWARFEmissionNone', 0)
LLVMDWARFEmissionFull = LLVMDWARFEmissionKind.define('LLVMDWARFEmissionFull', 1)
LLVMDWARFEmissionLineTablesOnly = LLVMDWARFEmissionKind.define('LLVMDWARFEmissionLineTablesOnly', 2)

_anonenum3 = CEnum(ctypes.c_uint32)
LLVMMDStringMetadataKind = _anonenum3.define('LLVMMDStringMetadataKind', 0)
LLVMConstantAsMetadataMetadataKind = _anonenum3.define('LLVMConstantAsMetadataMetadataKind', 1)
LLVMLocalAsMetadataMetadataKind = _anonenum3.define('LLVMLocalAsMetadataMetadataKind', 2)
LLVMDistinctMDOperandPlaceholderMetadataKind = _anonenum3.define('LLVMDistinctMDOperandPlaceholderMetadataKind', 3)
LLVMMDTupleMetadataKind = _anonenum3.define('LLVMMDTupleMetadataKind', 4)
LLVMDILocationMetadataKind = _anonenum3.define('LLVMDILocationMetadataKind', 5)
LLVMDIExpressionMetadataKind = _anonenum3.define('LLVMDIExpressionMetadataKind', 6)
LLVMDIGlobalVariableExpressionMetadataKind = _anonenum3.define('LLVMDIGlobalVariableExpressionMetadataKind', 7)
LLVMGenericDINodeMetadataKind = _anonenum3.define('LLVMGenericDINodeMetadataKind', 8)
LLVMDISubrangeMetadataKind = _anonenum3.define('LLVMDISubrangeMetadataKind', 9)
LLVMDIEnumeratorMetadataKind = _anonenum3.define('LLVMDIEnumeratorMetadataKind', 10)
LLVMDIBasicTypeMetadataKind = _anonenum3.define('LLVMDIBasicTypeMetadataKind', 11)
LLVMDIDerivedTypeMetadataKind = _anonenum3.define('LLVMDIDerivedTypeMetadataKind', 12)
LLVMDICompositeTypeMetadataKind = _anonenum3.define('LLVMDICompositeTypeMetadataKind', 13)
LLVMDISubroutineTypeMetadataKind = _anonenum3.define('LLVMDISubroutineTypeMetadataKind', 14)
LLVMDIFileMetadataKind = _anonenum3.define('LLVMDIFileMetadataKind', 15)
LLVMDICompileUnitMetadataKind = _anonenum3.define('LLVMDICompileUnitMetadataKind', 16)
LLVMDISubprogramMetadataKind = _anonenum3.define('LLVMDISubprogramMetadataKind', 17)
LLVMDILexicalBlockMetadataKind = _anonenum3.define('LLVMDILexicalBlockMetadataKind', 18)
LLVMDILexicalBlockFileMetadataKind = _anonenum3.define('LLVMDILexicalBlockFileMetadataKind', 19)
LLVMDINamespaceMetadataKind = _anonenum3.define('LLVMDINamespaceMetadataKind', 20)
LLVMDIModuleMetadataKind = _anonenum3.define('LLVMDIModuleMetadataKind', 21)
LLVMDITemplateTypeParameterMetadataKind = _anonenum3.define('LLVMDITemplateTypeParameterMetadataKind', 22)
LLVMDITemplateValueParameterMetadataKind = _anonenum3.define('LLVMDITemplateValueParameterMetadataKind', 23)
LLVMDIGlobalVariableMetadataKind = _anonenum3.define('LLVMDIGlobalVariableMetadataKind', 24)
LLVMDILocalVariableMetadataKind = _anonenum3.define('LLVMDILocalVariableMetadataKind', 25)
LLVMDILabelMetadataKind = _anonenum3.define('LLVMDILabelMetadataKind', 26)
LLVMDIObjCPropertyMetadataKind = _anonenum3.define('LLVMDIObjCPropertyMetadataKind', 27)
LLVMDIImportedEntityMetadataKind = _anonenum3.define('LLVMDIImportedEntityMetadataKind', 28)
LLVMDIMacroMetadataKind = _anonenum3.define('LLVMDIMacroMetadataKind', 29)
LLVMDIMacroFileMetadataKind = _anonenum3.define('LLVMDIMacroFileMetadataKind', 30)
LLVMDICommonBlockMetadataKind = _anonenum3.define('LLVMDICommonBlockMetadataKind', 31)
LLVMDIStringTypeMetadataKind = _anonenum3.define('LLVMDIStringTypeMetadataKind', 32)
LLVMDIGenericSubrangeMetadataKind = _anonenum3.define('LLVMDIGenericSubrangeMetadataKind', 33)
LLVMDIArgListMetadataKind = _anonenum3.define('LLVMDIArgListMetadataKind', 34)
LLVMDIAssignIDMetadataKind = _anonenum3.define('LLVMDIAssignIDMetadataKind', 35)

LLVMMetadataKind = ctypes.c_uint32
LLVMDWARFTypeEncoding = ctypes.c_uint32
LLVMDWARFMacinfoRecordType = CEnum(ctypes.c_uint32)
LLVMDWARFMacinfoRecordTypeDefine = LLVMDWARFMacinfoRecordType.define('LLVMDWARFMacinfoRecordTypeDefine', 1)
LLVMDWARFMacinfoRecordTypeMacro = LLVMDWARFMacinfoRecordType.define('LLVMDWARFMacinfoRecordTypeMacro', 2)
LLVMDWARFMacinfoRecordTypeStartFile = LLVMDWARFMacinfoRecordType.define('LLVMDWARFMacinfoRecordTypeStartFile', 3)
LLVMDWARFMacinfoRecordTypeEndFile = LLVMDWARFMacinfoRecordType.define('LLVMDWARFMacinfoRecordTypeEndFile', 4)
LLVMDWARFMacinfoRecordTypeVendorExt = LLVMDWARFMacinfoRecordType.define('LLVMDWARFMacinfoRecordTypeVendorExt', 255)

# unsigned int LLVMDebugMetadataVersion(void)
try: (LLVMDebugMetadataVersion:=dll.LLVMDebugMetadataVersion).restype, LLVMDebugMetadataVersion.argtypes = ctypes.c_uint32, []
except AttributeError: pass

# unsigned int LLVMGetModuleDebugMetadataVersion(LLVMModuleRef Module)
try: (LLVMGetModuleDebugMetadataVersion:=dll.LLVMGetModuleDebugMetadataVersion).restype, LLVMGetModuleDebugMetadataVersion.argtypes = ctypes.c_uint32, [LLVMModuleRef]
except AttributeError: pass

# LLVMBool LLVMStripModuleDebugInfo(LLVMModuleRef Module)
try: (LLVMStripModuleDebugInfo:=dll.LLVMStripModuleDebugInfo).restype, LLVMStripModuleDebugInfo.argtypes = LLVMBool, [LLVMModuleRef]
except AttributeError: pass

class struct_LLVMOpaqueDIBuilder(Struct): pass
LLVMDIBuilderRef = ctypes.POINTER(struct_LLVMOpaqueDIBuilder)
# LLVMDIBuilderRef LLVMCreateDIBuilderDisallowUnresolved(LLVMModuleRef M)
try: (LLVMCreateDIBuilderDisallowUnresolved:=dll.LLVMCreateDIBuilderDisallowUnresolved).restype, LLVMCreateDIBuilderDisallowUnresolved.argtypes = LLVMDIBuilderRef, [LLVMModuleRef]
except AttributeError: pass

# LLVMDIBuilderRef LLVMCreateDIBuilder(LLVMModuleRef M)
try: (LLVMCreateDIBuilder:=dll.LLVMCreateDIBuilder).restype, LLVMCreateDIBuilder.argtypes = LLVMDIBuilderRef, [LLVMModuleRef]
except AttributeError: pass

# void LLVMDisposeDIBuilder(LLVMDIBuilderRef Builder)
try: (LLVMDisposeDIBuilder:=dll.LLVMDisposeDIBuilder).restype, LLVMDisposeDIBuilder.argtypes = None, [LLVMDIBuilderRef]
except AttributeError: pass

# void LLVMDIBuilderFinalize(LLVMDIBuilderRef Builder)
try: (LLVMDIBuilderFinalize:=dll.LLVMDIBuilderFinalize).restype, LLVMDIBuilderFinalize.argtypes = None, [LLVMDIBuilderRef]
except AttributeError: pass

# void LLVMDIBuilderFinalizeSubprogram(LLVMDIBuilderRef Builder, LLVMMetadataRef Subprogram)
try: (LLVMDIBuilderFinalizeSubprogram:=dll.LLVMDIBuilderFinalizeSubprogram).restype, LLVMDIBuilderFinalizeSubprogram.argtypes = None, [LLVMDIBuilderRef, LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateCompileUnit(LLVMDIBuilderRef Builder, LLVMDWARFSourceLanguage Lang, LLVMMetadataRef FileRef, const char *Producer, size_t ProducerLen, LLVMBool isOptimized, const char *Flags, size_t FlagsLen, unsigned int RuntimeVer, const char *SplitName, size_t SplitNameLen, LLVMDWARFEmissionKind Kind, unsigned int DWOId, LLVMBool SplitDebugInlining, LLVMBool DebugInfoForProfiling, const char *SysRoot, size_t SysRootLen, const char *SDK, size_t SDKLen)
try: (LLVMDIBuilderCreateCompileUnit:=dll.LLVMDIBuilderCreateCompileUnit).restype, LLVMDIBuilderCreateCompileUnit.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMDWARFSourceLanguage, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMBool, ctypes.POINTER(ctypes.c_char), size_t, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, LLVMDWARFEmissionKind, ctypes.c_uint32, LLVMBool, LLVMBool, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateFile(LLVMDIBuilderRef Builder, const char *Filename, size_t FilenameLen, const char *Directory, size_t DirectoryLen)
try: (LLVMDIBuilderCreateFile:=dll.LLVMDIBuilderCreateFile).restype, LLVMDIBuilderCreateFile.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateModule(LLVMDIBuilderRef Builder, LLVMMetadataRef ParentScope, const char *Name, size_t NameLen, const char *ConfigMacros, size_t ConfigMacrosLen, const char *IncludePath, size_t IncludePathLen, const char *APINotesFile, size_t APINotesFileLen)
try: (LLVMDIBuilderCreateModule:=dll.LLVMDIBuilderCreateModule).restype, LLVMDIBuilderCreateModule.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateNameSpace(LLVMDIBuilderRef Builder, LLVMMetadataRef ParentScope, const char *Name, size_t NameLen, LLVMBool ExportSymbols)
try: (LLVMDIBuilderCreateNameSpace:=dll.LLVMDIBuilderCreateNameSpace).restype, LLVMDIBuilderCreateNameSpace.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMBool]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateFunction(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name, size_t NameLen, const char *LinkageName, size_t LinkageNameLen, LLVMMetadataRef File, unsigned int LineNo, LLVMMetadataRef Ty, LLVMBool IsLocalToUnit, LLVMBool IsDefinition, unsigned int ScopeLine, LLVMDIFlags Flags, LLVMBool IsOptimized)
try: (LLVMDIBuilderCreateFunction:=dll.LLVMDIBuilderCreateFunction).restype, LLVMDIBuilderCreateFunction.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMBool, ctypes.c_uint32, LLVMDIFlags, LLVMBool]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateLexicalBlock(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef File, unsigned int Line, unsigned int Column)
try: (LLVMDIBuilderCreateLexicalBlock:=dll.LLVMDIBuilderCreateLexicalBlock).restype, LLVMDIBuilderCreateLexicalBlock.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateLexicalBlockFile(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef File, unsigned int Discriminator)
try: (LLVMDIBuilderCreateLexicalBlockFile:=dll.LLVMDIBuilderCreateLexicalBlockFile).restype, LLVMDIBuilderCreateLexicalBlockFile.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateImportedModuleFromNamespace(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef NS, LLVMMetadataRef File, unsigned int Line)
try: (LLVMDIBuilderCreateImportedModuleFromNamespace:=dll.LLVMDIBuilderCreateImportedModuleFromNamespace).restype, LLVMDIBuilderCreateImportedModuleFromNamespace.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateImportedModuleFromAlias(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef ImportedEntity, LLVMMetadataRef File, unsigned int Line, LLVMMetadataRef *Elements, unsigned int NumElements)
try: (LLVMDIBuilderCreateImportedModuleFromAlias:=dll.LLVMDIBuilderCreateImportedModuleFromAlias).restype, LLVMDIBuilderCreateImportedModuleFromAlias.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateImportedModuleFromModule(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef M, LLVMMetadataRef File, unsigned int Line, LLVMMetadataRef *Elements, unsigned int NumElements)
try: (LLVMDIBuilderCreateImportedModuleFromModule:=dll.LLVMDIBuilderCreateImportedModuleFromModule).restype, LLVMDIBuilderCreateImportedModuleFromModule.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateImportedDeclaration(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef Decl, LLVMMetadataRef File, unsigned int Line, const char *Name, size_t NameLen, LLVMMetadataRef *Elements, unsigned int NumElements)
try: (LLVMDIBuilderCreateImportedDeclaration:=dll.LLVMDIBuilderCreateImportedDeclaration).restype, LLVMDIBuilderCreateImportedDeclaration.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateDebugLocation(LLVMContextRef Ctx, unsigned int Line, unsigned int Column, LLVMMetadataRef Scope, LLVMMetadataRef InlinedAt)
try: (LLVMDIBuilderCreateDebugLocation:=dll.LLVMDIBuilderCreateDebugLocation).restype, LLVMDIBuilderCreateDebugLocation.argtypes = LLVMMetadataRef, [LLVMContextRef, ctypes.c_uint32, ctypes.c_uint32, LLVMMetadataRef, LLVMMetadataRef]
except AttributeError: pass

# unsigned int LLVMDILocationGetLine(LLVMMetadataRef Location)
try: (LLVMDILocationGetLine:=dll.LLVMDILocationGetLine).restype, LLVMDILocationGetLine.argtypes = ctypes.c_uint32, [LLVMMetadataRef]
except AttributeError: pass

# unsigned int LLVMDILocationGetColumn(LLVMMetadataRef Location)
try: (LLVMDILocationGetColumn:=dll.LLVMDILocationGetColumn).restype, LLVMDILocationGetColumn.argtypes = ctypes.c_uint32, [LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDILocationGetScope(LLVMMetadataRef Location)
try: (LLVMDILocationGetScope:=dll.LLVMDILocationGetScope).restype, LLVMDILocationGetScope.argtypes = LLVMMetadataRef, [LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDILocationGetInlinedAt(LLVMMetadataRef Location)
try: (LLVMDILocationGetInlinedAt:=dll.LLVMDILocationGetInlinedAt).restype, LLVMDILocationGetInlinedAt.argtypes = LLVMMetadataRef, [LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIScopeGetFile(LLVMMetadataRef Scope)
try: (LLVMDIScopeGetFile:=dll.LLVMDIScopeGetFile).restype, LLVMDIScopeGetFile.argtypes = LLVMMetadataRef, [LLVMMetadataRef]
except AttributeError: pass

# const char *LLVMDIFileGetDirectory(LLVMMetadataRef File, unsigned int *Len)
try: (LLVMDIFileGetDirectory:=dll.LLVMDIFileGetDirectory).restype, LLVMDIFileGetDirectory.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMMetadataRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

# const char *LLVMDIFileGetFilename(LLVMMetadataRef File, unsigned int *Len)
try: (LLVMDIFileGetFilename:=dll.LLVMDIFileGetFilename).restype, LLVMDIFileGetFilename.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMMetadataRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

# const char *LLVMDIFileGetSource(LLVMMetadataRef File, unsigned int *Len)
try: (LLVMDIFileGetSource:=dll.LLVMDIFileGetSource).restype, LLVMDIFileGetSource.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMMetadataRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderGetOrCreateTypeArray(LLVMDIBuilderRef Builder, LLVMMetadataRef *Data, size_t NumElements)
try: (LLVMDIBuilderGetOrCreateTypeArray:=dll.LLVMDIBuilderGetOrCreateTypeArray).restype, LLVMDIBuilderGetOrCreateTypeArray.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(LLVMMetadataRef), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateSubroutineType(LLVMDIBuilderRef Builder, LLVMMetadataRef File, LLVMMetadataRef *ParameterTypes, unsigned int NumParameterTypes, LLVMDIFlags Flags)
try: (LLVMDIBuilderCreateSubroutineType:=dll.LLVMDIBuilderCreateSubroutineType).restype, LLVMDIBuilderCreateSubroutineType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32, LLVMDIFlags]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateMacro(LLVMDIBuilderRef Builder, LLVMMetadataRef ParentMacroFile, unsigned int Line, LLVMDWARFMacinfoRecordType RecordType, const char *Name, size_t NameLen, const char *Value, size_t ValueLen)
try: (LLVMDIBuilderCreateMacro:=dll.LLVMDIBuilderCreateMacro).restype, LLVMDIBuilderCreateMacro.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.c_uint32, LLVMDWARFMacinfoRecordType, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateTempMacroFile(LLVMDIBuilderRef Builder, LLVMMetadataRef ParentMacroFile, unsigned int Line, LLVMMetadataRef File)
try: (LLVMDIBuilderCreateTempMacroFile:=dll.LLVMDIBuilderCreateTempMacroFile).restype, LLVMDIBuilderCreateTempMacroFile.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError: pass

int64_t = ctypes.c_int64
# LLVMMetadataRef LLVMDIBuilderCreateEnumerator(LLVMDIBuilderRef Builder, const char *Name, size_t NameLen, int64_t Value, LLVMBool IsUnsigned)
try: (LLVMDIBuilderCreateEnumerator:=dll.LLVMDIBuilderCreateEnumerator).restype, LLVMDIBuilderCreateEnumerator.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, int64_t, LLVMBool]
except AttributeError: pass

uint32_t = ctypes.c_uint32
# LLVMMetadataRef LLVMDIBuilderCreateEnumerationType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNumber, uint64_t SizeInBits, uint32_t AlignInBits, LLVMMetadataRef *Elements, unsigned int NumElements, LLVMMetadataRef ClassTy)
try: (LLVMDIBuilderCreateEnumerationType:=dll.LLVMDIBuilderCreateEnumerationType).restype, LLVMDIBuilderCreateEnumerationType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32, LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateUnionType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNumber, uint64_t SizeInBits, uint32_t AlignInBits, LLVMDIFlags Flags, LLVMMetadataRef *Elements, unsigned int NumElements, unsigned int RunTimeLang, const char *UniqueId, size_t UniqueIdLen)
try: (LLVMDIBuilderCreateUnionType:=dll.LLVMDIBuilderCreateUnionType).restype, LLVMDIBuilderCreateUnionType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, LLVMDIFlags, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateArrayType(LLVMDIBuilderRef Builder, uint64_t Size, uint32_t AlignInBits, LLVMMetadataRef Ty, LLVMMetadataRef *Subscripts, unsigned int NumSubscripts)
try: (LLVMDIBuilderCreateArrayType:=dll.LLVMDIBuilderCreateArrayType).restype, LLVMDIBuilderCreateArrayType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, uint64_t, uint32_t, LLVMMetadataRef, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateVectorType(LLVMDIBuilderRef Builder, uint64_t Size, uint32_t AlignInBits, LLVMMetadataRef Ty, LLVMMetadataRef *Subscripts, unsigned int NumSubscripts)
try: (LLVMDIBuilderCreateVectorType:=dll.LLVMDIBuilderCreateVectorType).restype, LLVMDIBuilderCreateVectorType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, uint64_t, uint32_t, LLVMMetadataRef, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateUnspecifiedType(LLVMDIBuilderRef Builder, const char *Name, size_t NameLen)
try: (LLVMDIBuilderCreateUnspecifiedType:=dll.LLVMDIBuilderCreateUnspecifiedType).restype, LLVMDIBuilderCreateUnspecifiedType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateBasicType(LLVMDIBuilderRef Builder, const char *Name, size_t NameLen, uint64_t SizeInBits, LLVMDWARFTypeEncoding Encoding, LLVMDIFlags Flags)
try: (LLVMDIBuilderCreateBasicType:=dll.LLVMDIBuilderCreateBasicType).restype, LLVMDIBuilderCreateBasicType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, uint64_t, LLVMDWARFTypeEncoding, LLVMDIFlags]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreatePointerType(LLVMDIBuilderRef Builder, LLVMMetadataRef PointeeTy, uint64_t SizeInBits, uint32_t AlignInBits, unsigned int AddressSpace, const char *Name, size_t NameLen)
try: (LLVMDIBuilderCreatePointerType:=dll.LLVMDIBuilderCreatePointerType).restype, LLVMDIBuilderCreatePointerType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, uint64_t, uint32_t, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateStructType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNumber, uint64_t SizeInBits, uint32_t AlignInBits, LLVMDIFlags Flags, LLVMMetadataRef DerivedFrom, LLVMMetadataRef *Elements, unsigned int NumElements, unsigned int RunTimeLang, LLVMMetadataRef VTableHolder, const char *UniqueId, size_t UniqueIdLen)
try: (LLVMDIBuilderCreateStructType:=dll.LLVMDIBuilderCreateStructType).restype, LLVMDIBuilderCreateStructType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, LLVMDIFlags, LLVMMetadataRef, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32, ctypes.c_uint32, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateMemberType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNo, uint64_t SizeInBits, uint32_t AlignInBits, uint64_t OffsetInBits, LLVMDIFlags Flags, LLVMMetadataRef Ty)
try: (LLVMDIBuilderCreateMemberType:=dll.LLVMDIBuilderCreateMemberType).restype, LLVMDIBuilderCreateMemberType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, uint64_t, LLVMDIFlags, LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateStaticMemberType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNumber, LLVMMetadataRef Type, LLVMDIFlags Flags, LLVMValueRef ConstantVal, uint32_t AlignInBits)
try: (LLVMDIBuilderCreateStaticMemberType:=dll.LLVMDIBuilderCreateStaticMemberType).restype, LLVMDIBuilderCreateStaticMemberType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMDIFlags, LLVMValueRef, uint32_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateMemberPointerType(LLVMDIBuilderRef Builder, LLVMMetadataRef PointeeType, LLVMMetadataRef ClassType, uint64_t SizeInBits, uint32_t AlignInBits, LLVMDIFlags Flags)
try: (LLVMDIBuilderCreateMemberPointerType:=dll.LLVMDIBuilderCreateMemberPointerType).restype, LLVMDIBuilderCreateMemberPointerType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, uint64_t, uint32_t, LLVMDIFlags]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateObjCIVar(LLVMDIBuilderRef Builder, const char *Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNo, uint64_t SizeInBits, uint32_t AlignInBits, uint64_t OffsetInBits, LLVMDIFlags Flags, LLVMMetadataRef Ty, LLVMMetadataRef PropertyNode)
try: (LLVMDIBuilderCreateObjCIVar:=dll.LLVMDIBuilderCreateObjCIVar).restype, LLVMDIBuilderCreateObjCIVar.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, uint64_t, LLVMDIFlags, LLVMMetadataRef, LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateObjCProperty(LLVMDIBuilderRef Builder, const char *Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNo, const char *GetterName, size_t GetterNameLen, const char *SetterName, size_t SetterNameLen, unsigned int PropertyAttributes, LLVMMetadataRef Ty)
try: (LLVMDIBuilderCreateObjCProperty:=dll.LLVMDIBuilderCreateObjCProperty).restype, LLVMDIBuilderCreateObjCProperty.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateObjectPointerType(LLVMDIBuilderRef Builder, LLVMMetadataRef Type, LLVMBool Implicit)
try: (LLVMDIBuilderCreateObjectPointerType:=dll.LLVMDIBuilderCreateObjectPointerType).restype, LLVMDIBuilderCreateObjectPointerType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMBool]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateQualifiedType(LLVMDIBuilderRef Builder, unsigned int Tag, LLVMMetadataRef Type)
try: (LLVMDIBuilderCreateQualifiedType:=dll.LLVMDIBuilderCreateQualifiedType).restype, LLVMDIBuilderCreateQualifiedType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateReferenceType(LLVMDIBuilderRef Builder, unsigned int Tag, LLVMMetadataRef Type)
try: (LLVMDIBuilderCreateReferenceType:=dll.LLVMDIBuilderCreateReferenceType).restype, LLVMDIBuilderCreateReferenceType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateNullPtrType(LLVMDIBuilderRef Builder)
try: (LLVMDIBuilderCreateNullPtrType:=dll.LLVMDIBuilderCreateNullPtrType).restype, LLVMDIBuilderCreateNullPtrType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateTypedef(LLVMDIBuilderRef Builder, LLVMMetadataRef Type, const char *Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNo, LLVMMetadataRef Scope, uint32_t AlignInBits)
try: (LLVMDIBuilderCreateTypedef:=dll.LLVMDIBuilderCreateTypedef).restype, LLVMDIBuilderCreateTypedef.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, uint32_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateInheritance(LLVMDIBuilderRef Builder, LLVMMetadataRef Ty, LLVMMetadataRef BaseTy, uint64_t BaseOffset, uint32_t VBPtrOffset, LLVMDIFlags Flags)
try: (LLVMDIBuilderCreateInheritance:=dll.LLVMDIBuilderCreateInheritance).restype, LLVMDIBuilderCreateInheritance.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, uint64_t, uint32_t, LLVMDIFlags]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateForwardDecl(LLVMDIBuilderRef Builder, unsigned int Tag, const char *Name, size_t NameLen, LLVMMetadataRef Scope, LLVMMetadataRef File, unsigned int Line, unsigned int RuntimeLang, uint64_t SizeInBits, uint32_t AlignInBits, const char *UniqueIdentifier, size_t UniqueIdentifierLen)
try: (LLVMDIBuilderCreateForwardDecl:=dll.LLVMDIBuilderCreateForwardDecl).restype, LLVMDIBuilderCreateForwardDecl.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.c_uint32, uint64_t, uint32_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateReplaceableCompositeType(LLVMDIBuilderRef Builder, unsigned int Tag, const char *Name, size_t NameLen, LLVMMetadataRef Scope, LLVMMetadataRef File, unsigned int Line, unsigned int RuntimeLang, uint64_t SizeInBits, uint32_t AlignInBits, LLVMDIFlags Flags, const char *UniqueIdentifier, size_t UniqueIdentifierLen)
try: (LLVMDIBuilderCreateReplaceableCompositeType:=dll.LLVMDIBuilderCreateReplaceableCompositeType).restype, LLVMDIBuilderCreateReplaceableCompositeType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.c_uint32, uint64_t, uint32_t, LLVMDIFlags, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateBitFieldMemberType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNumber, uint64_t SizeInBits, uint64_t OffsetInBits, uint64_t StorageOffsetInBits, LLVMDIFlags Flags, LLVMMetadataRef Type)
try: (LLVMDIBuilderCreateBitFieldMemberType:=dll.LLVMDIBuilderCreateBitFieldMemberType).restype, LLVMDIBuilderCreateBitFieldMemberType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint64_t, uint64_t, LLVMDIFlags, LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateClassType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNumber, uint64_t SizeInBits, uint32_t AlignInBits, uint64_t OffsetInBits, LLVMDIFlags Flags, LLVMMetadataRef DerivedFrom, LLVMMetadataRef *Elements, unsigned int NumElements, LLVMMetadataRef VTableHolder, LLVMMetadataRef TemplateParamsNode, const char *UniqueIdentifier, size_t UniqueIdentifierLen)
try: (LLVMDIBuilderCreateClassType:=dll.LLVMDIBuilderCreateClassType).restype, LLVMDIBuilderCreateClassType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, uint64_t, LLVMDIFlags, LLVMMetadataRef, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32, LLVMMetadataRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateArtificialType(LLVMDIBuilderRef Builder, LLVMMetadataRef Type)
try: (LLVMDIBuilderCreateArtificialType:=dll.LLVMDIBuilderCreateArtificialType).restype, LLVMDIBuilderCreateArtificialType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef]
except AttributeError: pass

# const char *LLVMDITypeGetName(LLVMMetadataRef DType, size_t *Length)
try: (LLVMDITypeGetName:=dll.LLVMDITypeGetName).restype, LLVMDITypeGetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMMetadataRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# uint64_t LLVMDITypeGetSizeInBits(LLVMMetadataRef DType)
try: (LLVMDITypeGetSizeInBits:=dll.LLVMDITypeGetSizeInBits).restype, LLVMDITypeGetSizeInBits.argtypes = uint64_t, [LLVMMetadataRef]
except AttributeError: pass

# uint64_t LLVMDITypeGetOffsetInBits(LLVMMetadataRef DType)
try: (LLVMDITypeGetOffsetInBits:=dll.LLVMDITypeGetOffsetInBits).restype, LLVMDITypeGetOffsetInBits.argtypes = uint64_t, [LLVMMetadataRef]
except AttributeError: pass

# uint32_t LLVMDITypeGetAlignInBits(LLVMMetadataRef DType)
try: (LLVMDITypeGetAlignInBits:=dll.LLVMDITypeGetAlignInBits).restype, LLVMDITypeGetAlignInBits.argtypes = uint32_t, [LLVMMetadataRef]
except AttributeError: pass

# unsigned int LLVMDITypeGetLine(LLVMMetadataRef DType)
try: (LLVMDITypeGetLine:=dll.LLVMDITypeGetLine).restype, LLVMDITypeGetLine.argtypes = ctypes.c_uint32, [LLVMMetadataRef]
except AttributeError: pass

# LLVMDIFlags LLVMDITypeGetFlags(LLVMMetadataRef DType)
try: (LLVMDITypeGetFlags:=dll.LLVMDITypeGetFlags).restype, LLVMDITypeGetFlags.argtypes = LLVMDIFlags, [LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderGetOrCreateSubrange(LLVMDIBuilderRef Builder, int64_t LowerBound, int64_t Count)
try: (LLVMDIBuilderGetOrCreateSubrange:=dll.LLVMDIBuilderGetOrCreateSubrange).restype, LLVMDIBuilderGetOrCreateSubrange.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, int64_t, int64_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderGetOrCreateArray(LLVMDIBuilderRef Builder, LLVMMetadataRef *Data, size_t NumElements)
try: (LLVMDIBuilderGetOrCreateArray:=dll.LLVMDIBuilderGetOrCreateArray).restype, LLVMDIBuilderGetOrCreateArray.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(LLVMMetadataRef), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateExpression(LLVMDIBuilderRef Builder, uint64_t *Addr, size_t Length)
try: (LLVMDIBuilderCreateExpression:=dll.LLVMDIBuilderCreateExpression).restype, LLVMDIBuilderCreateExpression.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(uint64_t), size_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateConstantValueExpression(LLVMDIBuilderRef Builder, uint64_t Value)
try: (LLVMDIBuilderCreateConstantValueExpression:=dll.LLVMDIBuilderCreateConstantValueExpression).restype, LLVMDIBuilderCreateConstantValueExpression.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, uint64_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateGlobalVariableExpression(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name, size_t NameLen, const char *Linkage, size_t LinkLen, LLVMMetadataRef File, unsigned int LineNo, LLVMMetadataRef Ty, LLVMBool LocalToUnit, LLVMMetadataRef Expr, LLVMMetadataRef Decl, uint32_t AlignInBits)
try: (LLVMDIBuilderCreateGlobalVariableExpression:=dll.LLVMDIBuilderCreateGlobalVariableExpression).restype, LLVMDIBuilderCreateGlobalVariableExpression.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMMetadataRef, LLVMMetadataRef, uint32_t]
except AttributeError: pass

uint16_t = ctypes.c_uint16
# uint16_t LLVMGetDINodeTag(LLVMMetadataRef MD)
try: (LLVMGetDINodeTag:=dll.LLVMGetDINodeTag).restype, LLVMGetDINodeTag.argtypes = uint16_t, [LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIGlobalVariableExpressionGetVariable(LLVMMetadataRef GVE)
try: (LLVMDIGlobalVariableExpressionGetVariable:=dll.LLVMDIGlobalVariableExpressionGetVariable).restype, LLVMDIGlobalVariableExpressionGetVariable.argtypes = LLVMMetadataRef, [LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIGlobalVariableExpressionGetExpression(LLVMMetadataRef GVE)
try: (LLVMDIGlobalVariableExpressionGetExpression:=dll.LLVMDIGlobalVariableExpressionGetExpression).restype, LLVMDIGlobalVariableExpressionGetExpression.argtypes = LLVMMetadataRef, [LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIVariableGetFile(LLVMMetadataRef Var)
try: (LLVMDIVariableGetFile:=dll.LLVMDIVariableGetFile).restype, LLVMDIVariableGetFile.argtypes = LLVMMetadataRef, [LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIVariableGetScope(LLVMMetadataRef Var)
try: (LLVMDIVariableGetScope:=dll.LLVMDIVariableGetScope).restype, LLVMDIVariableGetScope.argtypes = LLVMMetadataRef, [LLVMMetadataRef]
except AttributeError: pass

# unsigned int LLVMDIVariableGetLine(LLVMMetadataRef Var)
try: (LLVMDIVariableGetLine:=dll.LLVMDIVariableGetLine).restype, LLVMDIVariableGetLine.argtypes = ctypes.c_uint32, [LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMTemporaryMDNode(LLVMContextRef Ctx, LLVMMetadataRef *Data, size_t NumElements)
try: (LLVMTemporaryMDNode:=dll.LLVMTemporaryMDNode).restype, LLVMTemporaryMDNode.argtypes = LLVMMetadataRef, [LLVMContextRef, ctypes.POINTER(LLVMMetadataRef), size_t]
except AttributeError: pass

# void LLVMDisposeTemporaryMDNode(LLVMMetadataRef TempNode)
try: (LLVMDisposeTemporaryMDNode:=dll.LLVMDisposeTemporaryMDNode).restype, LLVMDisposeTemporaryMDNode.argtypes = None, [LLVMMetadataRef]
except AttributeError: pass

# void LLVMMetadataReplaceAllUsesWith(LLVMMetadataRef TempTargetMetadata, LLVMMetadataRef Replacement)
try: (LLVMMetadataReplaceAllUsesWith:=dll.LLVMMetadataReplaceAllUsesWith).restype, LLVMMetadataReplaceAllUsesWith.argtypes = None, [LLVMMetadataRef, LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateTempGlobalVariableFwdDecl(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name, size_t NameLen, const char *Linkage, size_t LnkLen, LLVMMetadataRef File, unsigned int LineNo, LLVMMetadataRef Ty, LLVMBool LocalToUnit, LLVMMetadataRef Decl, uint32_t AlignInBits)
try: (LLVMDIBuilderCreateTempGlobalVariableFwdDecl:=dll.LLVMDIBuilderCreateTempGlobalVariableFwdDecl).restype, LLVMDIBuilderCreateTempGlobalVariableFwdDecl.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMMetadataRef, uint32_t]
except AttributeError: pass

# LLVMDbgRecordRef LLVMDIBuilderInsertDeclareRecordBefore(LLVMDIBuilderRef Builder, LLVMValueRef Storage, LLVMMetadataRef VarInfo, LLVMMetadataRef Expr, LLVMMetadataRef DebugLoc, LLVMValueRef Instr)
try: (LLVMDIBuilderInsertDeclareRecordBefore:=dll.LLVMDIBuilderInsertDeclareRecordBefore).restype, LLVMDIBuilderInsertDeclareRecordBefore.argtypes = LLVMDbgRecordRef, [LLVMDIBuilderRef, LLVMValueRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, LLVMValueRef]
except AttributeError: pass

# LLVMDbgRecordRef LLVMDIBuilderInsertDeclareRecordAtEnd(LLVMDIBuilderRef Builder, LLVMValueRef Storage, LLVMMetadataRef VarInfo, LLVMMetadataRef Expr, LLVMMetadataRef DebugLoc, LLVMBasicBlockRef Block)
try: (LLVMDIBuilderInsertDeclareRecordAtEnd:=dll.LLVMDIBuilderInsertDeclareRecordAtEnd).restype, LLVMDIBuilderInsertDeclareRecordAtEnd.argtypes = LLVMDbgRecordRef, [LLVMDIBuilderRef, LLVMValueRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, LLVMBasicBlockRef]
except AttributeError: pass

# LLVMDbgRecordRef LLVMDIBuilderInsertDbgValueRecordBefore(LLVMDIBuilderRef Builder, LLVMValueRef Val, LLVMMetadataRef VarInfo, LLVMMetadataRef Expr, LLVMMetadataRef DebugLoc, LLVMValueRef Instr)
try: (LLVMDIBuilderInsertDbgValueRecordBefore:=dll.LLVMDIBuilderInsertDbgValueRecordBefore).restype, LLVMDIBuilderInsertDbgValueRecordBefore.argtypes = LLVMDbgRecordRef, [LLVMDIBuilderRef, LLVMValueRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, LLVMValueRef]
except AttributeError: pass

# LLVMDbgRecordRef LLVMDIBuilderInsertDbgValueRecordAtEnd(LLVMDIBuilderRef Builder, LLVMValueRef Val, LLVMMetadataRef VarInfo, LLVMMetadataRef Expr, LLVMMetadataRef DebugLoc, LLVMBasicBlockRef Block)
try: (LLVMDIBuilderInsertDbgValueRecordAtEnd:=dll.LLVMDIBuilderInsertDbgValueRecordAtEnd).restype, LLVMDIBuilderInsertDbgValueRecordAtEnd.argtypes = LLVMDbgRecordRef, [LLVMDIBuilderRef, LLVMValueRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, LLVMBasicBlockRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateAutoVariable(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNo, LLVMMetadataRef Ty, LLVMBool AlwaysPreserve, LLVMDIFlags Flags, uint32_t AlignInBits)
try: (LLVMDIBuilderCreateAutoVariable:=dll.LLVMDIBuilderCreateAutoVariable).restype, LLVMDIBuilderCreateAutoVariable.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMDIFlags, uint32_t]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateParameterVariable(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char *Name, size_t NameLen, unsigned int ArgNo, LLVMMetadataRef File, unsigned int LineNo, LLVMMetadataRef Ty, LLVMBool AlwaysPreserve, LLVMDIFlags Flags)
try: (LLVMDIBuilderCreateParameterVariable:=dll.LLVMDIBuilderCreateParameterVariable).restype, LLVMDIBuilderCreateParameterVariable.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.c_uint32, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMDIFlags]
except AttributeError: pass

# LLVMMetadataRef LLVMGetSubprogram(LLVMValueRef Func)
try: (LLVMGetSubprogram:=dll.LLVMGetSubprogram).restype, LLVMGetSubprogram.argtypes = LLVMMetadataRef, [LLVMValueRef]
except AttributeError: pass

# void LLVMSetSubprogram(LLVMValueRef Func, LLVMMetadataRef SP)
try: (LLVMSetSubprogram:=dll.LLVMSetSubprogram).restype, LLVMSetSubprogram.argtypes = None, [LLVMValueRef, LLVMMetadataRef]
except AttributeError: pass

# unsigned int LLVMDISubprogramGetLine(LLVMMetadataRef Subprogram)
try: (LLVMDISubprogramGetLine:=dll.LLVMDISubprogramGetLine).restype, LLVMDISubprogramGetLine.argtypes = ctypes.c_uint32, [LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMInstructionGetDebugLoc(LLVMValueRef Inst)
try: (LLVMInstructionGetDebugLoc:=dll.LLVMInstructionGetDebugLoc).restype, LLVMInstructionGetDebugLoc.argtypes = LLVMMetadataRef, [LLVMValueRef]
except AttributeError: pass

# void LLVMInstructionSetDebugLoc(LLVMValueRef Inst, LLVMMetadataRef Loc)
try: (LLVMInstructionSetDebugLoc:=dll.LLVMInstructionSetDebugLoc).restype, LLVMInstructionSetDebugLoc.argtypes = None, [LLVMValueRef, LLVMMetadataRef]
except AttributeError: pass

# LLVMMetadataRef LLVMDIBuilderCreateLabel(LLVMDIBuilderRef Builder, LLVMMetadataRef Context, const char *Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNo, LLVMBool AlwaysPreserve)
try: (LLVMDIBuilderCreateLabel:=dll.LLVMDIBuilderCreateLabel).restype, LLVMDIBuilderCreateLabel.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMBool]
except AttributeError: pass

# LLVMDbgRecordRef LLVMDIBuilderInsertLabelBefore(LLVMDIBuilderRef Builder, LLVMMetadataRef LabelInfo, LLVMMetadataRef Location, LLVMValueRef InsertBefore)
try: (LLVMDIBuilderInsertLabelBefore:=dll.LLVMDIBuilderInsertLabelBefore).restype, LLVMDIBuilderInsertLabelBefore.argtypes = LLVMDbgRecordRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMValueRef]
except AttributeError: pass

# LLVMDbgRecordRef LLVMDIBuilderInsertLabelAtEnd(LLVMDIBuilderRef Builder, LLVMMetadataRef LabelInfo, LLVMMetadataRef Location, LLVMBasicBlockRef InsertAtEnd)
try: (LLVMDIBuilderInsertLabelAtEnd:=dll.LLVMDIBuilderInsertLabelAtEnd).restype, LLVMDIBuilderInsertLabelAtEnd.argtypes = LLVMDbgRecordRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMBasicBlockRef]
except AttributeError: pass

# LLVMMetadataKind LLVMGetMetadataKind(LLVMMetadataRef Metadata)
try: (LLVMGetMetadataKind:=dll.LLVMGetMetadataKind).restype, LLVMGetMetadataKind.argtypes = LLVMMetadataKind, [LLVMMetadataRef]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

LLVMDisasmContextRef = ctypes.c_void_p
LLVMOpInfoCallback = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int32, ctypes.c_void_p)
LLVMSymbolLookupCallback = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
# LLVMDisasmContextRef LLVMCreateDisasm(const char *TripleName, void *DisInfo, int TagType, LLVMOpInfoCallback GetOpInfo, LLVMSymbolLookupCallback SymbolLookUp)
try: (LLVMCreateDisasm:=dll.LLVMCreateDisasm).restype, LLVMCreateDisasm.argtypes = LLVMDisasmContextRef, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, ctypes.c_int32, LLVMOpInfoCallback, LLVMSymbolLookupCallback]
except AttributeError: pass

# LLVMDisasmContextRef LLVMCreateDisasmCPU(const char *Triple, const char *CPU, void *DisInfo, int TagType, LLVMOpInfoCallback GetOpInfo, LLVMSymbolLookupCallback SymbolLookUp)
try: (LLVMCreateDisasmCPU:=dll.LLVMCreateDisasmCPU).restype, LLVMCreateDisasmCPU.argtypes = LLVMDisasmContextRef, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, ctypes.c_int32, LLVMOpInfoCallback, LLVMSymbolLookupCallback]
except AttributeError: pass

# LLVMDisasmContextRef LLVMCreateDisasmCPUFeatures(const char *Triple, const char *CPU, const char *Features, void *DisInfo, int TagType, LLVMOpInfoCallback GetOpInfo, LLVMSymbolLookupCallback SymbolLookUp)
try: (LLVMCreateDisasmCPUFeatures:=dll.LLVMCreateDisasmCPUFeatures).restype, LLVMCreateDisasmCPUFeatures.argtypes = LLVMDisasmContextRef, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, ctypes.c_int32, LLVMOpInfoCallback, LLVMSymbolLookupCallback]
except AttributeError: pass

# int LLVMSetDisasmOptions(LLVMDisasmContextRef DC, uint64_t Options)
try: (LLVMSetDisasmOptions:=dll.LLVMSetDisasmOptions).restype, LLVMSetDisasmOptions.argtypes = ctypes.c_int32, [LLVMDisasmContextRef, uint64_t]
except AttributeError: pass

# void LLVMDisasmDispose(LLVMDisasmContextRef DC)
try: (LLVMDisasmDispose:=dll.LLVMDisasmDispose).restype, LLVMDisasmDispose.argtypes = None, [LLVMDisasmContextRef]
except AttributeError: pass

# size_t LLVMDisasmInstruction(LLVMDisasmContextRef DC, uint8_t *Bytes, uint64_t BytesSize, uint64_t PC, char *OutString, size_t OutStringSize)
try: (LLVMDisasmInstruction:=dll.LLVMDisasmInstruction).restype, LLVMDisasmInstruction.argtypes = size_t, [LLVMDisasmContextRef, ctypes.POINTER(uint8_t), uint64_t, uint64_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

class struct_LLVMOpInfoSymbol1(Struct): pass
struct_LLVMOpInfoSymbol1._fields_ = [
  ('Present', uint64_t),
  ('Name', ctypes.POINTER(ctypes.c_char)),
  ('Value', uint64_t),
]
class struct_LLVMOpInfo1(Struct): pass
struct_LLVMOpInfo1._fields_ = [
  ('AddSymbol', struct_LLVMOpInfoSymbol1),
  ('SubtractSymbol', struct_LLVMOpInfoSymbol1),
  ('Value', uint64_t),
  ('VariantKind', uint64_t),
]
class struct_LLVMOpaqueError(Struct): pass
LLVMErrorRef = ctypes.POINTER(struct_LLVMOpaqueError)
LLVMErrorTypeId = ctypes.c_void_p
# LLVMErrorTypeId LLVMGetErrorTypeId(LLVMErrorRef Err)
try: (LLVMGetErrorTypeId:=dll.LLVMGetErrorTypeId).restype, LLVMGetErrorTypeId.argtypes = LLVMErrorTypeId, [LLVMErrorRef]
except AttributeError: pass

# void LLVMConsumeError(LLVMErrorRef Err)
try: (LLVMConsumeError:=dll.LLVMConsumeError).restype, LLVMConsumeError.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

# void LLVMCantFail(LLVMErrorRef Err)
try: (LLVMCantFail:=dll.LLVMCantFail).restype, LLVMCantFail.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

# char *LLVMGetErrorMessage(LLVMErrorRef Err)
try: (LLVMGetErrorMessage:=dll.LLVMGetErrorMessage).restype, LLVMGetErrorMessage.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMErrorRef]
except AttributeError: pass

# void LLVMDisposeErrorMessage(char *ErrMsg)
try: (LLVMDisposeErrorMessage:=dll.LLVMDisposeErrorMessage).restype, LLVMDisposeErrorMessage.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorTypeId LLVMGetStringErrorTypeId(void)
try: (LLVMGetStringErrorTypeId:=dll.LLVMGetStringErrorTypeId).restype, LLVMGetStringErrorTypeId.argtypes = LLVMErrorTypeId, []
except AttributeError: pass

# LLVMErrorRef LLVMCreateStringError(const char *ErrMsg)
try: (LLVMCreateStringError:=dll.LLVMCreateStringError).restype, LLVMCreateStringError.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMInstallFatalErrorHandler(LLVMFatalErrorHandler Handler)
try: (LLVMInstallFatalErrorHandler:=dll.LLVMInstallFatalErrorHandler).restype, LLVMInstallFatalErrorHandler.argtypes = None, [LLVMFatalErrorHandler]
except AttributeError: pass

# void LLVMResetFatalErrorHandler(void)
try: (LLVMResetFatalErrorHandler:=dll.LLVMResetFatalErrorHandler).restype, LLVMResetFatalErrorHandler.argtypes = None, []
except AttributeError: pass

# void LLVMEnablePrettyStackTrace(void)
try: (LLVMEnablePrettyStackTrace:=dll.LLVMEnablePrettyStackTrace).restype, LLVMEnablePrettyStackTrace.argtypes = None, []
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

# void LLVMInitializeAArch64TargetInfo(void)
try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetInfo(void)
try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetInfo(void)
try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetInfo(void)
try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetInfo(void)
try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetInfo(void)
try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetInfo(void)
try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetInfo(void)
try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetInfo(void)
try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetInfo(void)
try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetInfo(void)
try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetInfo(void)
try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetInfo(void)
try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetInfo(void)
try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetInfo(void)
try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetInfo(void)
try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetInfo(void)
try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetInfo(void)
try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetInfo(void)
try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetInfo(void)
try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetInfo(void)
try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetInfo(void)
try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Target(void)
try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTarget(void)
try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTarget(void)
try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTarget(void)
try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTarget(void)
try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTarget(void)
try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTarget(void)
try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTarget(void)
try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTarget(void)
try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Target(void)
try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTarget(void)
try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTarget(void)
try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTarget(void)
try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTarget(void)
try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTarget(void)
try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTarget(void)
try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETarget(void)
try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTarget(void)
try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Target(void)
try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTarget(void)
try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTarget(void)
try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTarget(void)
try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64TargetMC(void)
try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetMC(void)
try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetMC(void)
try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetMC(void)
try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetMC(void)
try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetMC(void)
try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetMC(void)
try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetMC(void)
try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetMC(void)
try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetMC(void)
try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetMC(void)
try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetMC(void)
try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetMC(void)
try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetMC(void)
try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetMC(void)
try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetMC(void)
try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetMC(void)
try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetMC(void)
try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetMC(void)
try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetMC(void)
try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetMC(void)
try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetMC(void)
try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmPrinter(void)
try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmPrinter(void)
try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmPrinter(void)
try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmPrinter(void)
try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmPrinter(void)
try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmPrinter(void)
try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmPrinter(void)
try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmPrinter(void)
try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmPrinter(void)
try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmPrinter(void)
try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXAsmPrinter(void)
try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmPrinter(void)
try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmPrinter(void)
try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmPrinter(void)
try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVAsmPrinter(void)
try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmPrinter(void)
try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmPrinter(void)
try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmPrinter(void)
try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmPrinter(void)
try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreAsmPrinter(void)
try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmPrinter(void)
try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmPrinter(void)
try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmParser(void)
try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmParser(void)
try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmParser(void)
try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmParser(void)
try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmParser(void)
try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmParser(void)
try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmParser(void)
try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmParser(void)
try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmParser(void)
try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmParser(void)
try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmParser(void)
try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmParser(void)
try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmParser(void)
try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmParser(void)
try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmParser(void)
try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmParser(void)
try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmParser(void)
try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmParser(void)
try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmParser(void)
try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Disassembler(void)
try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUDisassembler(void)
try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMDisassembler(void)
try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRDisassembler(void)
try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFDisassembler(void)
try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonDisassembler(void)
try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiDisassembler(void)
try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchDisassembler(void)
try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsDisassembler(void)
try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Disassembler(void)
try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCDisassembler(void)
try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVDisassembler(void)
try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcDisassembler(void)
try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZDisassembler(void)
try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEDisassembler(void)
try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyDisassembler(void)
try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Disassembler(void)
try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreDisassembler(void)
try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kDisassembler(void)
try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaDisassembler(void)
try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

class struct_LLVMOpaqueTargetData(Struct): pass
LLVMTargetDataRef = ctypes.POINTER(struct_LLVMOpaqueTargetData)
# LLVMTargetDataRef LLVMGetModuleDataLayout(LLVMModuleRef M)
try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

# void LLVMSetModuleDataLayout(LLVMModuleRef M, LLVMTargetDataRef DL)
try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetData(const char *StringRep)
try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMDisposeTargetData(LLVMTargetDataRef TD)
try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

class struct_LLVMOpaqueTargetLibraryInfotData(Struct): pass
LLVMTargetLibraryInfoRef = ctypes.POINTER(struct_LLVMOpaqueTargetLibraryInfotData)
# void LLVMAddTargetLibraryInfo(LLVMTargetLibraryInfoRef TLI, LLVMPassManagerRef PM)
try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

# char *LLVMCopyStringRepOfTargetData(LLVMTargetDataRef TD)
try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

enum_LLVMByteOrdering = CEnum(ctypes.c_uint32)
LLVMBigEndian = enum_LLVMByteOrdering.define('LLVMBigEndian', 0)
LLVMLittleEndian = enum_LLVMByteOrdering.define('LLVMLittleEndian', 1)

# enum LLVMByteOrdering LLVMByteOrder(LLVMTargetDataRef TD)
try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSize(LLVMTargetDataRef TD)
try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSizeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrType(LLVMTargetDataRef TD)
try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeInContext(LLVMContextRef C, LLVMTargetDataRef TD)
try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForASInContext(LLVMContextRef C, LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# unsigned long long LLVMSizeOfTypeInBits(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMStoreSizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMABISizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMABIAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMCallFrameAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfGlobal(LLVMTargetDataRef TD, LLVMValueRef GlobalVar)
try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

# unsigned int LLVMElementAtOffset(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned long long Offset)
try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

# unsigned long long LLVMOffsetOfElement(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned int Element)
try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

class struct_LLVMTarget(Struct): pass
LLVMTargetRef = ctypes.POINTER(struct_LLVMTarget)
# LLVMTargetRef LLVMGetFirstTarget(void)
try: (LLVMGetFirstTarget:=dll.LLVMGetFirstTarget).restype, LLVMGetFirstTarget.argtypes = LLVMTargetRef, []
except AttributeError: pass

# LLVMTargetRef LLVMGetNextTarget(LLVMTargetRef T)
try: (LLVMGetNextTarget:=dll.LLVMGetNextTarget).restype, LLVMGetNextTarget.argtypes = LLVMTargetRef, [LLVMTargetRef]
except AttributeError: pass

# LLVMTargetRef LLVMGetTargetFromName(const char *Name)
try: (LLVMGetTargetFromName:=dll.LLVMGetTargetFromName).restype, LLVMGetTargetFromName.argtypes = LLVMTargetRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBool LLVMGetTargetFromTriple(const char *Triple, LLVMTargetRef *T, char **ErrorMessage)
try: (LLVMGetTargetFromTriple:=dll.LLVMGetTargetFromTriple).restype, LLVMGetTargetFromTriple.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTargetRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# const char *LLVMGetTargetName(LLVMTargetRef T)
try: (LLVMGetTargetName:=dll.LLVMGetTargetName).restype, LLVMGetTargetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

# const char *LLVMGetTargetDescription(LLVMTargetRef T)
try: (LLVMGetTargetDescription:=dll.LLVMGetTargetDescription).restype, LLVMGetTargetDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasJIT(LLVMTargetRef T)
try: (LLVMTargetHasJIT:=dll.LLVMTargetHasJIT).restype, LLVMTargetHasJIT.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasTargetMachine(LLVMTargetRef T)
try: (LLVMTargetHasTargetMachine:=dll.LLVMTargetHasTargetMachine).restype, LLVMTargetHasTargetMachine.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasAsmBackend(LLVMTargetRef T)
try: (LLVMTargetHasAsmBackend:=dll.LLVMTargetHasAsmBackend).restype, LLVMTargetHasAsmBackend.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

class struct_LLVMOpaqueTargetMachineOptions(Struct): pass
LLVMTargetMachineOptionsRef = ctypes.POINTER(struct_LLVMOpaqueTargetMachineOptions)
# LLVMTargetMachineOptionsRef LLVMCreateTargetMachineOptions(void)
try: (LLVMCreateTargetMachineOptions:=dll.LLVMCreateTargetMachineOptions).restype, LLVMCreateTargetMachineOptions.argtypes = LLVMTargetMachineOptionsRef, []
except AttributeError: pass

# void LLVMDisposeTargetMachineOptions(LLVMTargetMachineOptionsRef Options)
try: (LLVMDisposeTargetMachineOptions:=dll.LLVMDisposeTargetMachineOptions).restype, LLVMDisposeTargetMachineOptions.argtypes = None, [LLVMTargetMachineOptionsRef]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCPU(LLVMTargetMachineOptionsRef Options, const char *CPU)
try: (LLVMTargetMachineOptionsSetCPU:=dll.LLVMTargetMachineOptionsSetCPU).restype, LLVMTargetMachineOptionsSetCPU.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetFeatures(LLVMTargetMachineOptionsRef Options, const char *Features)
try: (LLVMTargetMachineOptionsSetFeatures:=dll.LLVMTargetMachineOptionsSetFeatures).restype, LLVMTargetMachineOptionsSetFeatures.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetABI(LLVMTargetMachineOptionsRef Options, const char *ABI)
try: (LLVMTargetMachineOptionsSetABI:=dll.LLVMTargetMachineOptionsSetABI).restype, LLVMTargetMachineOptionsSetABI.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

LLVMCodeGenOptLevel = CEnum(ctypes.c_uint32)
LLVMCodeGenLevelNone = LLVMCodeGenOptLevel.define('LLVMCodeGenLevelNone', 0)
LLVMCodeGenLevelLess = LLVMCodeGenOptLevel.define('LLVMCodeGenLevelLess', 1)
LLVMCodeGenLevelDefault = LLVMCodeGenOptLevel.define('LLVMCodeGenLevelDefault', 2)
LLVMCodeGenLevelAggressive = LLVMCodeGenOptLevel.define('LLVMCodeGenLevelAggressive', 3)

# void LLVMTargetMachineOptionsSetCodeGenOptLevel(LLVMTargetMachineOptionsRef Options, LLVMCodeGenOptLevel Level)
try: (LLVMTargetMachineOptionsSetCodeGenOptLevel:=dll.LLVMTargetMachineOptionsSetCodeGenOptLevel).restype, LLVMTargetMachineOptionsSetCodeGenOptLevel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeGenOptLevel]
except AttributeError: pass

LLVMRelocMode = CEnum(ctypes.c_uint32)
LLVMRelocDefault = LLVMRelocMode.define('LLVMRelocDefault', 0)
LLVMRelocStatic = LLVMRelocMode.define('LLVMRelocStatic', 1)
LLVMRelocPIC = LLVMRelocMode.define('LLVMRelocPIC', 2)
LLVMRelocDynamicNoPic = LLVMRelocMode.define('LLVMRelocDynamicNoPic', 3)
LLVMRelocROPI = LLVMRelocMode.define('LLVMRelocROPI', 4)
LLVMRelocRWPI = LLVMRelocMode.define('LLVMRelocRWPI', 5)
LLVMRelocROPI_RWPI = LLVMRelocMode.define('LLVMRelocROPI_RWPI', 6)

# void LLVMTargetMachineOptionsSetRelocMode(LLVMTargetMachineOptionsRef Options, LLVMRelocMode Reloc)
try: (LLVMTargetMachineOptionsSetRelocMode:=dll.LLVMTargetMachineOptionsSetRelocMode).restype, LLVMTargetMachineOptionsSetRelocMode.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMRelocMode]
except AttributeError: pass

LLVMCodeModel = CEnum(ctypes.c_uint32)
LLVMCodeModelDefault = LLVMCodeModel.define('LLVMCodeModelDefault', 0)
LLVMCodeModelJITDefault = LLVMCodeModel.define('LLVMCodeModelJITDefault', 1)
LLVMCodeModelTiny = LLVMCodeModel.define('LLVMCodeModelTiny', 2)
LLVMCodeModelSmall = LLVMCodeModel.define('LLVMCodeModelSmall', 3)
LLVMCodeModelKernel = LLVMCodeModel.define('LLVMCodeModelKernel', 4)
LLVMCodeModelMedium = LLVMCodeModel.define('LLVMCodeModelMedium', 5)
LLVMCodeModelLarge = LLVMCodeModel.define('LLVMCodeModelLarge', 6)

# void LLVMTargetMachineOptionsSetCodeModel(LLVMTargetMachineOptionsRef Options, LLVMCodeModel CodeModel)
try: (LLVMTargetMachineOptionsSetCodeModel:=dll.LLVMTargetMachineOptionsSetCodeModel).restype, LLVMTargetMachineOptionsSetCodeModel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeModel]
except AttributeError: pass

class struct_LLVMOpaqueTargetMachine(Struct): pass
LLVMTargetMachineRef = ctypes.POINTER(struct_LLVMOpaqueTargetMachine)
# LLVMTargetMachineRef LLVMCreateTargetMachineWithOptions(LLVMTargetRef T, const char *Triple, LLVMTargetMachineOptionsRef Options)
try: (LLVMCreateTargetMachineWithOptions:=dll.LLVMCreateTargetMachineWithOptions).restype, LLVMCreateTargetMachineWithOptions.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineOptionsRef]
except AttributeError: pass

# LLVMTargetMachineRef LLVMCreateTargetMachine(LLVMTargetRef T, const char *Triple, const char *CPU, const char *Features, LLVMCodeGenOptLevel Level, LLVMRelocMode Reloc, LLVMCodeModel CodeModel)
try: (LLVMCreateTargetMachine:=dll.LLVMCreateTargetMachine).restype, LLVMCreateTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError: pass

# void LLVMDisposeTargetMachine(LLVMTargetMachineRef T)
try: (LLVMDisposeTargetMachine:=dll.LLVMDisposeTargetMachine).restype, LLVMDisposeTargetMachine.argtypes = None, [LLVMTargetMachineRef]
except AttributeError: pass

# LLVMTargetRef LLVMGetTargetMachineTarget(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineTarget:=dll.LLVMGetTargetMachineTarget).restype, LLVMGetTargetMachineTarget.argtypes = LLVMTargetRef, [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineTriple(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineTriple:=dll.LLVMGetTargetMachineTriple).restype, LLVMGetTargetMachineTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineCPU(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineCPU:=dll.LLVMGetTargetMachineCPU).restype, LLVMGetTargetMachineCPU.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineFeatureString(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineFeatureString:=dll.LLVMGetTargetMachineFeatureString).restype, LLVMGetTargetMachineFeatureString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetDataLayout(LLVMTargetMachineRef T)
try: (LLVMCreateTargetDataLayout:=dll.LLVMCreateTargetDataLayout).restype, LLVMCreateTargetDataLayout.argtypes = LLVMTargetDataRef, [LLVMTargetMachineRef]
except AttributeError: pass

# void LLVMSetTargetMachineAsmVerbosity(LLVMTargetMachineRef T, LLVMBool VerboseAsm)
try: (LLVMSetTargetMachineAsmVerbosity:=dll.LLVMSetTargetMachineAsmVerbosity).restype, LLVMSetTargetMachineAsmVerbosity.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineFastISel(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineFastISel:=dll.LLVMSetTargetMachineFastISel).restype, LLVMSetTargetMachineFastISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineGlobalISel(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineGlobalISel:=dll.LLVMSetTargetMachineGlobalISel).restype, LLVMSetTargetMachineGlobalISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

LLVMGlobalISelAbortMode = CEnum(ctypes.c_uint32)
LLVMGlobalISelAbortEnable = LLVMGlobalISelAbortMode.define('LLVMGlobalISelAbortEnable', 0)
LLVMGlobalISelAbortDisable = LLVMGlobalISelAbortMode.define('LLVMGlobalISelAbortDisable', 1)
LLVMGlobalISelAbortDisableWithDiag = LLVMGlobalISelAbortMode.define('LLVMGlobalISelAbortDisableWithDiag', 2)

# void LLVMSetTargetMachineGlobalISelAbort(LLVMTargetMachineRef T, LLVMGlobalISelAbortMode Mode)
try: (LLVMSetTargetMachineGlobalISelAbort:=dll.LLVMSetTargetMachineGlobalISelAbort).restype, LLVMSetTargetMachineGlobalISelAbort.argtypes = None, [LLVMTargetMachineRef, LLVMGlobalISelAbortMode]
except AttributeError: pass

# void LLVMSetTargetMachineMachineOutliner(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineMachineOutliner:=dll.LLVMSetTargetMachineMachineOutliner).restype, LLVMSetTargetMachineMachineOutliner.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

LLVMCodeGenFileType = CEnum(ctypes.c_uint32)
LLVMAssemblyFile = LLVMCodeGenFileType.define('LLVMAssemblyFile', 0)
LLVMObjectFile = LLVMCodeGenFileType.define('LLVMObjectFile', 1)

# LLVMBool LLVMTargetMachineEmitToFile(LLVMTargetMachineRef T, LLVMModuleRef M, const char *Filename, LLVMCodeGenFileType codegen, char **ErrorMessage)
try: (LLVMTargetMachineEmitToFile:=dll.LLVMTargetMachineEmitToFile).restype, LLVMTargetMachineEmitToFile.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMTargetMachineEmitToMemoryBuffer(LLVMTargetMachineRef T, LLVMModuleRef M, LLVMCodeGenFileType codegen, char **ErrorMessage, LLVMMemoryBufferRef *OutMemBuf)
try: (LLVMTargetMachineEmitToMemoryBuffer:=dll.LLVMTargetMachineEmitToMemoryBuffer).restype, LLVMTargetMachineEmitToMemoryBuffer.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

# char *LLVMGetDefaultTargetTriple(void)
try: (LLVMGetDefaultTargetTriple:=dll.LLVMGetDefaultTargetTriple).restype, LLVMGetDefaultTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# char *LLVMNormalizeTargetTriple(const char *triple)
try: (LLVMNormalizeTargetTriple:=dll.LLVMNormalizeTargetTriple).restype, LLVMNormalizeTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *LLVMGetHostCPUName(void)
try: (LLVMGetHostCPUName:=dll.LLVMGetHostCPUName).restype, LLVMGetHostCPUName.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# char *LLVMGetHostCPUFeatures(void)
try: (LLVMGetHostCPUFeatures:=dll.LLVMGetHostCPUFeatures).restype, LLVMGetHostCPUFeatures.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# void LLVMAddAnalysisPasses(LLVMTargetMachineRef T, LLVMPassManagerRef PM)
try: (LLVMAddAnalysisPasses:=dll.LLVMAddAnalysisPasses).restype, LLVMAddAnalysisPasses.argtypes = None, [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError: pass

# void LLVMLinkInMCJIT(void)
try: (LLVMLinkInMCJIT:=dll.LLVMLinkInMCJIT).restype, LLVMLinkInMCJIT.argtypes = None, []
except AttributeError: pass

# void LLVMLinkInInterpreter(void)
try: (LLVMLinkInInterpreter:=dll.LLVMLinkInInterpreter).restype, LLVMLinkInInterpreter.argtypes = None, []
except AttributeError: pass

class struct_LLVMOpaqueGenericValue(Struct): pass
LLVMGenericValueRef = ctypes.POINTER(struct_LLVMOpaqueGenericValue)
class struct_LLVMOpaqueExecutionEngine(Struct): pass
LLVMExecutionEngineRef = ctypes.POINTER(struct_LLVMOpaqueExecutionEngine)
class struct_LLVMOpaqueMCJITMemoryManager(Struct): pass
LLVMMCJITMemoryManagerRef = ctypes.POINTER(struct_LLVMOpaqueMCJITMemoryManager)
class struct_LLVMMCJITCompilerOptions(Struct): pass
struct_LLVMMCJITCompilerOptions._fields_ = [
  ('OptLevel', ctypes.c_uint32),
  ('CodeModel', LLVMCodeModel),
  ('NoFramePointerElim', LLVMBool),
  ('EnableFastISel', LLVMBool),
  ('MCJMM', LLVMMCJITMemoryManagerRef),
]
# LLVMGenericValueRef LLVMCreateGenericValueOfInt(LLVMTypeRef Ty, unsigned long long N, LLVMBool IsSigned)
try: (LLVMCreateGenericValueOfInt:=dll.LLVMCreateGenericValueOfInt).restype, LLVMCreateGenericValueOfInt.argtypes = LLVMGenericValueRef, [LLVMTypeRef, ctypes.c_uint64, LLVMBool]
except AttributeError: pass

# LLVMGenericValueRef LLVMCreateGenericValueOfPointer(void *P)
try: (LLVMCreateGenericValueOfPointer:=dll.LLVMCreateGenericValueOfPointer).restype, LLVMCreateGenericValueOfPointer.argtypes = LLVMGenericValueRef, [ctypes.c_void_p]
except AttributeError: pass

# LLVMGenericValueRef LLVMCreateGenericValueOfFloat(LLVMTypeRef Ty, double N)
try: (LLVMCreateGenericValueOfFloat:=dll.LLVMCreateGenericValueOfFloat).restype, LLVMCreateGenericValueOfFloat.argtypes = LLVMGenericValueRef, [LLVMTypeRef, ctypes.c_double]
except AttributeError: pass

# unsigned int LLVMGenericValueIntWidth(LLVMGenericValueRef GenValRef)
try: (LLVMGenericValueIntWidth:=dll.LLVMGenericValueIntWidth).restype, LLVMGenericValueIntWidth.argtypes = ctypes.c_uint32, [LLVMGenericValueRef]
except AttributeError: pass

# unsigned long long LLVMGenericValueToInt(LLVMGenericValueRef GenVal, LLVMBool IsSigned)
try: (LLVMGenericValueToInt:=dll.LLVMGenericValueToInt).restype, LLVMGenericValueToInt.argtypes = ctypes.c_uint64, [LLVMGenericValueRef, LLVMBool]
except AttributeError: pass

# void *LLVMGenericValueToPointer(LLVMGenericValueRef GenVal)
try: (LLVMGenericValueToPointer:=dll.LLVMGenericValueToPointer).restype, LLVMGenericValueToPointer.argtypes = ctypes.c_void_p, [LLVMGenericValueRef]
except AttributeError: pass

# double LLVMGenericValueToFloat(LLVMTypeRef TyRef, LLVMGenericValueRef GenVal)
try: (LLVMGenericValueToFloat:=dll.LLVMGenericValueToFloat).restype, LLVMGenericValueToFloat.argtypes = ctypes.c_double, [LLVMTypeRef, LLVMGenericValueRef]
except AttributeError: pass

# void LLVMDisposeGenericValue(LLVMGenericValueRef GenVal)
try: (LLVMDisposeGenericValue:=dll.LLVMDisposeGenericValue).restype, LLVMDisposeGenericValue.argtypes = None, [LLVMGenericValueRef]
except AttributeError: pass

# LLVMBool LLVMCreateExecutionEngineForModule(LLVMExecutionEngineRef *OutEE, LLVMModuleRef M, char **OutError)
try: (LLVMCreateExecutionEngineForModule:=dll.LLVMCreateExecutionEngineForModule).restype, LLVMCreateExecutionEngineForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMCreateInterpreterForModule(LLVMExecutionEngineRef *OutInterp, LLVMModuleRef M, char **OutError)
try: (LLVMCreateInterpreterForModule:=dll.LLVMCreateInterpreterForModule).restype, LLVMCreateInterpreterForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMCreateJITCompilerForModule(LLVMExecutionEngineRef *OutJIT, LLVMModuleRef M, unsigned int OptLevel, char **OutError)
try: (LLVMCreateJITCompilerForModule:=dll.LLVMCreateJITCompilerForModule).restype, LLVMCreateJITCompilerForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# void LLVMInitializeMCJITCompilerOptions(struct LLVMMCJITCompilerOptions *Options, size_t SizeOfOptions)
try: (LLVMInitializeMCJITCompilerOptions:=dll.LLVMInitializeMCJITCompilerOptions).restype, LLVMInitializeMCJITCompilerOptions.argtypes = None, [ctypes.POINTER(struct_LLVMMCJITCompilerOptions), size_t]
except AttributeError: pass

# LLVMBool LLVMCreateMCJITCompilerForModule(LLVMExecutionEngineRef *OutJIT, LLVMModuleRef M, struct LLVMMCJITCompilerOptions *Options, size_t SizeOfOptions, char **OutError)
try: (LLVMCreateMCJITCompilerForModule:=dll.LLVMCreateMCJITCompilerForModule).restype, LLVMCreateMCJITCompilerForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.POINTER(struct_LLVMMCJITCompilerOptions), size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# void LLVMDisposeExecutionEngine(LLVMExecutionEngineRef EE)
try: (LLVMDisposeExecutionEngine:=dll.LLVMDisposeExecutionEngine).restype, LLVMDisposeExecutionEngine.argtypes = None, [LLVMExecutionEngineRef]
except AttributeError: pass

# void LLVMRunStaticConstructors(LLVMExecutionEngineRef EE)
try: (LLVMRunStaticConstructors:=dll.LLVMRunStaticConstructors).restype, LLVMRunStaticConstructors.argtypes = None, [LLVMExecutionEngineRef]
except AttributeError: pass

# void LLVMRunStaticDestructors(LLVMExecutionEngineRef EE)
try: (LLVMRunStaticDestructors:=dll.LLVMRunStaticDestructors).restype, LLVMRunStaticDestructors.argtypes = None, [LLVMExecutionEngineRef]
except AttributeError: pass

# int LLVMRunFunctionAsMain(LLVMExecutionEngineRef EE, LLVMValueRef F, unsigned int ArgC, const char *const *ArgV, const char *const *EnvP)
try: (LLVMRunFunctionAsMain:=dll.LLVMRunFunctionAsMain).restype, LLVMRunFunctionAsMain.argtypes = ctypes.c_int32, [LLVMExecutionEngineRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMGenericValueRef LLVMRunFunction(LLVMExecutionEngineRef EE, LLVMValueRef F, unsigned int NumArgs, LLVMGenericValueRef *Args)
try: (LLVMRunFunction:=dll.LLVMRunFunction).restype, LLVMRunFunction.argtypes = LLVMGenericValueRef, [LLVMExecutionEngineRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(LLVMGenericValueRef)]
except AttributeError: pass

# void LLVMFreeMachineCodeForFunction(LLVMExecutionEngineRef EE, LLVMValueRef F)
try: (LLVMFreeMachineCodeForFunction:=dll.LLVMFreeMachineCodeForFunction).restype, LLVMFreeMachineCodeForFunction.argtypes = None, [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError: pass

# void LLVMAddModule(LLVMExecutionEngineRef EE, LLVMModuleRef M)
try: (LLVMAddModule:=dll.LLVMAddModule).restype, LLVMAddModule.argtypes = None, [LLVMExecutionEngineRef, LLVMModuleRef]
except AttributeError: pass

# LLVMBool LLVMRemoveModule(LLVMExecutionEngineRef EE, LLVMModuleRef M, LLVMModuleRef *OutMod, char **OutError)
try: (LLVMRemoveModule:=dll.LLVMRemoveModule).restype, LLVMRemoveModule.argtypes = LLVMBool, [LLVMExecutionEngineRef, LLVMModuleRef, ctypes.POINTER(LLVMModuleRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMFindFunction(LLVMExecutionEngineRef EE, const char *Name, LLVMValueRef *OutFn)
try: (LLVMFindFunction:=dll.LLVMFindFunction).restype, LLVMFindFunction.argtypes = LLVMBool, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMValueRef)]
except AttributeError: pass

# void *LLVMRecompileAndRelinkFunction(LLVMExecutionEngineRef EE, LLVMValueRef Fn)
try: (LLVMRecompileAndRelinkFunction:=dll.LLVMRecompileAndRelinkFunction).restype, LLVMRecompileAndRelinkFunction.argtypes = ctypes.c_void_p, [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMGetExecutionEngineTargetData(LLVMExecutionEngineRef EE)
try: (LLVMGetExecutionEngineTargetData:=dll.LLVMGetExecutionEngineTargetData).restype, LLVMGetExecutionEngineTargetData.argtypes = LLVMTargetDataRef, [LLVMExecutionEngineRef]
except AttributeError: pass

# LLVMTargetMachineRef LLVMGetExecutionEngineTargetMachine(LLVMExecutionEngineRef EE)
try: (LLVMGetExecutionEngineTargetMachine:=dll.LLVMGetExecutionEngineTargetMachine).restype, LLVMGetExecutionEngineTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMExecutionEngineRef]
except AttributeError: pass

# void LLVMAddGlobalMapping(LLVMExecutionEngineRef EE, LLVMValueRef Global, void *Addr)
try: (LLVMAddGlobalMapping:=dll.LLVMAddGlobalMapping).restype, LLVMAddGlobalMapping.argtypes = None, [LLVMExecutionEngineRef, LLVMValueRef, ctypes.c_void_p]
except AttributeError: pass

# void *LLVMGetPointerToGlobal(LLVMExecutionEngineRef EE, LLVMValueRef Global)
try: (LLVMGetPointerToGlobal:=dll.LLVMGetPointerToGlobal).restype, LLVMGetPointerToGlobal.argtypes = ctypes.c_void_p, [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError: pass

# uint64_t LLVMGetGlobalValueAddress(LLVMExecutionEngineRef EE, const char *Name)
try: (LLVMGetGlobalValueAddress:=dll.LLVMGetGlobalValueAddress).restype, LLVMGetGlobalValueAddress.argtypes = uint64_t, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# uint64_t LLVMGetFunctionAddress(LLVMExecutionEngineRef EE, const char *Name)
try: (LLVMGetFunctionAddress:=dll.LLVMGetFunctionAddress).restype, LLVMGetFunctionAddress.argtypes = uint64_t, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBool LLVMExecutionEngineGetErrMsg(LLVMExecutionEngineRef EE, char **OutError)
try: (LLVMExecutionEngineGetErrMsg:=dll.LLVMExecutionEngineGetErrMsg).restype, LLVMExecutionEngineGetErrMsg.argtypes = LLVMBool, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

LLVMMemoryManagerAllocateCodeSectionCallback = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_ubyte), ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char))
LLVMMemoryManagerAllocateDataSectionCallback = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_ubyte), ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32)
LLVMMemoryManagerFinalizeMemoryCallback = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
LLVMMemoryManagerDestroyCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
# LLVMMCJITMemoryManagerRef LLVMCreateSimpleMCJITMemoryManager(void *Opaque, LLVMMemoryManagerAllocateCodeSectionCallback AllocateCodeSection, LLVMMemoryManagerAllocateDataSectionCallback AllocateDataSection, LLVMMemoryManagerFinalizeMemoryCallback FinalizeMemory, LLVMMemoryManagerDestroyCallback Destroy)
try: (LLVMCreateSimpleMCJITMemoryManager:=dll.LLVMCreateSimpleMCJITMemoryManager).restype, LLVMCreateSimpleMCJITMemoryManager.argtypes = LLVMMCJITMemoryManagerRef, [ctypes.c_void_p, LLVMMemoryManagerAllocateCodeSectionCallback, LLVMMemoryManagerAllocateDataSectionCallback, LLVMMemoryManagerFinalizeMemoryCallback, LLVMMemoryManagerDestroyCallback]
except AttributeError: pass

# void LLVMDisposeMCJITMemoryManager(LLVMMCJITMemoryManagerRef MM)
try: (LLVMDisposeMCJITMemoryManager:=dll.LLVMDisposeMCJITMemoryManager).restype, LLVMDisposeMCJITMemoryManager.argtypes = None, [LLVMMCJITMemoryManagerRef]
except AttributeError: pass

class struct_LLVMOpaqueJITEventListener(Struct): pass
LLVMJITEventListenerRef = ctypes.POINTER(struct_LLVMOpaqueJITEventListener)
# LLVMJITEventListenerRef LLVMCreateGDBRegistrationListener(void)
try: (LLVMCreateGDBRegistrationListener:=dll.LLVMCreateGDBRegistrationListener).restype, LLVMCreateGDBRegistrationListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

# LLVMJITEventListenerRef LLVMCreateIntelJITEventListener(void)
try: (LLVMCreateIntelJITEventListener:=dll.LLVMCreateIntelJITEventListener).restype, LLVMCreateIntelJITEventListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

# LLVMJITEventListenerRef LLVMCreateOProfileJITEventListener(void)
try: (LLVMCreateOProfileJITEventListener:=dll.LLVMCreateOProfileJITEventListener).restype, LLVMCreateOProfileJITEventListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

# LLVMJITEventListenerRef LLVMCreatePerfJITEventListener(void)
try: (LLVMCreatePerfJITEventListener:=dll.LLVMCreatePerfJITEventListener).restype, LLVMCreatePerfJITEventListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

# LLVMBool LLVMParseIRInContext(LLVMContextRef ContextRef, LLVMMemoryBufferRef MemBuf, LLVMModuleRef *OutM, char **OutMessage)
try: (LLVMParseIRInContext:=dll.LLVMParseIRInContext).restype, LLVMParseIRInContext.argtypes = LLVMBool, [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMErrorTypeId LLVMGetErrorTypeId(LLVMErrorRef Err)
try: (LLVMGetErrorTypeId:=dll.LLVMGetErrorTypeId).restype, LLVMGetErrorTypeId.argtypes = LLVMErrorTypeId, [LLVMErrorRef]
except AttributeError: pass

# void LLVMConsumeError(LLVMErrorRef Err)
try: (LLVMConsumeError:=dll.LLVMConsumeError).restype, LLVMConsumeError.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

# void LLVMCantFail(LLVMErrorRef Err)
try: (LLVMCantFail:=dll.LLVMCantFail).restype, LLVMCantFail.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

# char *LLVMGetErrorMessage(LLVMErrorRef Err)
try: (LLVMGetErrorMessage:=dll.LLVMGetErrorMessage).restype, LLVMGetErrorMessage.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMErrorRef]
except AttributeError: pass

# void LLVMDisposeErrorMessage(char *ErrMsg)
try: (LLVMDisposeErrorMessage:=dll.LLVMDisposeErrorMessage).restype, LLVMDisposeErrorMessage.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorTypeId LLVMGetStringErrorTypeId(void)
try: (LLVMGetStringErrorTypeId:=dll.LLVMGetStringErrorTypeId).restype, LLVMGetStringErrorTypeId.argtypes = LLVMErrorTypeId, []
except AttributeError: pass

# LLVMErrorRef LLVMCreateStringError(const char *ErrMsg)
try: (LLVMCreateStringError:=dll.LLVMCreateStringError).restype, LLVMCreateStringError.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

# void LLVMInitializeAArch64TargetInfo(void)
try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetInfo(void)
try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetInfo(void)
try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetInfo(void)
try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetInfo(void)
try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetInfo(void)
try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetInfo(void)
try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetInfo(void)
try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetInfo(void)
try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetInfo(void)
try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetInfo(void)
try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetInfo(void)
try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetInfo(void)
try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetInfo(void)
try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetInfo(void)
try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetInfo(void)
try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetInfo(void)
try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetInfo(void)
try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetInfo(void)
try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetInfo(void)
try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetInfo(void)
try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetInfo(void)
try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Target(void)
try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTarget(void)
try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTarget(void)
try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTarget(void)
try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTarget(void)
try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTarget(void)
try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTarget(void)
try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTarget(void)
try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTarget(void)
try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Target(void)
try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTarget(void)
try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTarget(void)
try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTarget(void)
try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTarget(void)
try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTarget(void)
try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTarget(void)
try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETarget(void)
try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTarget(void)
try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Target(void)
try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTarget(void)
try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTarget(void)
try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTarget(void)
try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64TargetMC(void)
try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetMC(void)
try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetMC(void)
try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetMC(void)
try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetMC(void)
try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetMC(void)
try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetMC(void)
try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetMC(void)
try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetMC(void)
try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetMC(void)
try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetMC(void)
try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetMC(void)
try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetMC(void)
try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetMC(void)
try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetMC(void)
try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetMC(void)
try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetMC(void)
try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetMC(void)
try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetMC(void)
try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetMC(void)
try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetMC(void)
try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetMC(void)
try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmPrinter(void)
try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmPrinter(void)
try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmPrinter(void)
try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmPrinter(void)
try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmPrinter(void)
try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmPrinter(void)
try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmPrinter(void)
try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmPrinter(void)
try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmPrinter(void)
try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmPrinter(void)
try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXAsmPrinter(void)
try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmPrinter(void)
try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmPrinter(void)
try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmPrinter(void)
try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVAsmPrinter(void)
try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmPrinter(void)
try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmPrinter(void)
try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmPrinter(void)
try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmPrinter(void)
try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreAsmPrinter(void)
try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmPrinter(void)
try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmPrinter(void)
try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmParser(void)
try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmParser(void)
try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmParser(void)
try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmParser(void)
try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmParser(void)
try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmParser(void)
try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmParser(void)
try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmParser(void)
try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmParser(void)
try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmParser(void)
try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmParser(void)
try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmParser(void)
try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmParser(void)
try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmParser(void)
try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmParser(void)
try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmParser(void)
try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmParser(void)
try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmParser(void)
try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmParser(void)
try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Disassembler(void)
try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUDisassembler(void)
try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMDisassembler(void)
try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRDisassembler(void)
try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFDisassembler(void)
try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonDisassembler(void)
try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiDisassembler(void)
try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchDisassembler(void)
try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsDisassembler(void)
try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Disassembler(void)
try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCDisassembler(void)
try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVDisassembler(void)
try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcDisassembler(void)
try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZDisassembler(void)
try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEDisassembler(void)
try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyDisassembler(void)
try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Disassembler(void)
try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreDisassembler(void)
try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kDisassembler(void)
try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaDisassembler(void)
try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

# LLVMTargetDataRef LLVMGetModuleDataLayout(LLVMModuleRef M)
try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

# void LLVMSetModuleDataLayout(LLVMModuleRef M, LLVMTargetDataRef DL)
try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetData(const char *StringRep)
try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMDisposeTargetData(LLVMTargetDataRef TD)
try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

# void LLVMAddTargetLibraryInfo(LLVMTargetLibraryInfoRef TLI, LLVMPassManagerRef PM)
try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

# char *LLVMCopyStringRepOfTargetData(LLVMTargetDataRef TD)
try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

# enum LLVMByteOrdering LLVMByteOrder(LLVMTargetDataRef TD)
try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSize(LLVMTargetDataRef TD)
try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSizeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrType(LLVMTargetDataRef TD)
try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeInContext(LLVMContextRef C, LLVMTargetDataRef TD)
try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForASInContext(LLVMContextRef C, LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# unsigned long long LLVMSizeOfTypeInBits(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMStoreSizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMABISizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMABIAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMCallFrameAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfGlobal(LLVMTargetDataRef TD, LLVMValueRef GlobalVar)
try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

# unsigned int LLVMElementAtOffset(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned long long Offset)
try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

# unsigned long long LLVMOffsetOfElement(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned int Element)
try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTargetRef LLVMGetFirstTarget(void)
try: (LLVMGetFirstTarget:=dll.LLVMGetFirstTarget).restype, LLVMGetFirstTarget.argtypes = LLVMTargetRef, []
except AttributeError: pass

# LLVMTargetRef LLVMGetNextTarget(LLVMTargetRef T)
try: (LLVMGetNextTarget:=dll.LLVMGetNextTarget).restype, LLVMGetNextTarget.argtypes = LLVMTargetRef, [LLVMTargetRef]
except AttributeError: pass

# LLVMTargetRef LLVMGetTargetFromName(const char *Name)
try: (LLVMGetTargetFromName:=dll.LLVMGetTargetFromName).restype, LLVMGetTargetFromName.argtypes = LLVMTargetRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBool LLVMGetTargetFromTriple(const char *Triple, LLVMTargetRef *T, char **ErrorMessage)
try: (LLVMGetTargetFromTriple:=dll.LLVMGetTargetFromTriple).restype, LLVMGetTargetFromTriple.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTargetRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# const char *LLVMGetTargetName(LLVMTargetRef T)
try: (LLVMGetTargetName:=dll.LLVMGetTargetName).restype, LLVMGetTargetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

# const char *LLVMGetTargetDescription(LLVMTargetRef T)
try: (LLVMGetTargetDescription:=dll.LLVMGetTargetDescription).restype, LLVMGetTargetDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasJIT(LLVMTargetRef T)
try: (LLVMTargetHasJIT:=dll.LLVMTargetHasJIT).restype, LLVMTargetHasJIT.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasTargetMachine(LLVMTargetRef T)
try: (LLVMTargetHasTargetMachine:=dll.LLVMTargetHasTargetMachine).restype, LLVMTargetHasTargetMachine.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasAsmBackend(LLVMTargetRef T)
try: (LLVMTargetHasAsmBackend:=dll.LLVMTargetHasAsmBackend).restype, LLVMTargetHasAsmBackend.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMTargetMachineOptionsRef LLVMCreateTargetMachineOptions(void)
try: (LLVMCreateTargetMachineOptions:=dll.LLVMCreateTargetMachineOptions).restype, LLVMCreateTargetMachineOptions.argtypes = LLVMTargetMachineOptionsRef, []
except AttributeError: pass

# void LLVMDisposeTargetMachineOptions(LLVMTargetMachineOptionsRef Options)
try: (LLVMDisposeTargetMachineOptions:=dll.LLVMDisposeTargetMachineOptions).restype, LLVMDisposeTargetMachineOptions.argtypes = None, [LLVMTargetMachineOptionsRef]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCPU(LLVMTargetMachineOptionsRef Options, const char *CPU)
try: (LLVMTargetMachineOptionsSetCPU:=dll.LLVMTargetMachineOptionsSetCPU).restype, LLVMTargetMachineOptionsSetCPU.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetFeatures(LLVMTargetMachineOptionsRef Options, const char *Features)
try: (LLVMTargetMachineOptionsSetFeatures:=dll.LLVMTargetMachineOptionsSetFeatures).restype, LLVMTargetMachineOptionsSetFeatures.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetABI(LLVMTargetMachineOptionsRef Options, const char *ABI)
try: (LLVMTargetMachineOptionsSetABI:=dll.LLVMTargetMachineOptionsSetABI).restype, LLVMTargetMachineOptionsSetABI.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCodeGenOptLevel(LLVMTargetMachineOptionsRef Options, LLVMCodeGenOptLevel Level)
try: (LLVMTargetMachineOptionsSetCodeGenOptLevel:=dll.LLVMTargetMachineOptionsSetCodeGenOptLevel).restype, LLVMTargetMachineOptionsSetCodeGenOptLevel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeGenOptLevel]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetRelocMode(LLVMTargetMachineOptionsRef Options, LLVMRelocMode Reloc)
try: (LLVMTargetMachineOptionsSetRelocMode:=dll.LLVMTargetMachineOptionsSetRelocMode).restype, LLVMTargetMachineOptionsSetRelocMode.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMRelocMode]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCodeModel(LLVMTargetMachineOptionsRef Options, LLVMCodeModel CodeModel)
try: (LLVMTargetMachineOptionsSetCodeModel:=dll.LLVMTargetMachineOptionsSetCodeModel).restype, LLVMTargetMachineOptionsSetCodeModel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeModel]
except AttributeError: pass

# LLVMTargetMachineRef LLVMCreateTargetMachineWithOptions(LLVMTargetRef T, const char *Triple, LLVMTargetMachineOptionsRef Options)
try: (LLVMCreateTargetMachineWithOptions:=dll.LLVMCreateTargetMachineWithOptions).restype, LLVMCreateTargetMachineWithOptions.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineOptionsRef]
except AttributeError: pass

# LLVMTargetMachineRef LLVMCreateTargetMachine(LLVMTargetRef T, const char *Triple, const char *CPU, const char *Features, LLVMCodeGenOptLevel Level, LLVMRelocMode Reloc, LLVMCodeModel CodeModel)
try: (LLVMCreateTargetMachine:=dll.LLVMCreateTargetMachine).restype, LLVMCreateTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError: pass

# void LLVMDisposeTargetMachine(LLVMTargetMachineRef T)
try: (LLVMDisposeTargetMachine:=dll.LLVMDisposeTargetMachine).restype, LLVMDisposeTargetMachine.argtypes = None, [LLVMTargetMachineRef]
except AttributeError: pass

# LLVMTargetRef LLVMGetTargetMachineTarget(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineTarget:=dll.LLVMGetTargetMachineTarget).restype, LLVMGetTargetMachineTarget.argtypes = LLVMTargetRef, [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineTriple(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineTriple:=dll.LLVMGetTargetMachineTriple).restype, LLVMGetTargetMachineTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineCPU(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineCPU:=dll.LLVMGetTargetMachineCPU).restype, LLVMGetTargetMachineCPU.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineFeatureString(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineFeatureString:=dll.LLVMGetTargetMachineFeatureString).restype, LLVMGetTargetMachineFeatureString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetDataLayout(LLVMTargetMachineRef T)
try: (LLVMCreateTargetDataLayout:=dll.LLVMCreateTargetDataLayout).restype, LLVMCreateTargetDataLayout.argtypes = LLVMTargetDataRef, [LLVMTargetMachineRef]
except AttributeError: pass

# void LLVMSetTargetMachineAsmVerbosity(LLVMTargetMachineRef T, LLVMBool VerboseAsm)
try: (LLVMSetTargetMachineAsmVerbosity:=dll.LLVMSetTargetMachineAsmVerbosity).restype, LLVMSetTargetMachineAsmVerbosity.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineFastISel(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineFastISel:=dll.LLVMSetTargetMachineFastISel).restype, LLVMSetTargetMachineFastISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineGlobalISel(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineGlobalISel:=dll.LLVMSetTargetMachineGlobalISel).restype, LLVMSetTargetMachineGlobalISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineGlobalISelAbort(LLVMTargetMachineRef T, LLVMGlobalISelAbortMode Mode)
try: (LLVMSetTargetMachineGlobalISelAbort:=dll.LLVMSetTargetMachineGlobalISelAbort).restype, LLVMSetTargetMachineGlobalISelAbort.argtypes = None, [LLVMTargetMachineRef, LLVMGlobalISelAbortMode]
except AttributeError: pass

# void LLVMSetTargetMachineMachineOutliner(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineMachineOutliner:=dll.LLVMSetTargetMachineMachineOutliner).restype, LLVMSetTargetMachineMachineOutliner.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# LLVMBool LLVMTargetMachineEmitToFile(LLVMTargetMachineRef T, LLVMModuleRef M, const char *Filename, LLVMCodeGenFileType codegen, char **ErrorMessage)
try: (LLVMTargetMachineEmitToFile:=dll.LLVMTargetMachineEmitToFile).restype, LLVMTargetMachineEmitToFile.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMTargetMachineEmitToMemoryBuffer(LLVMTargetMachineRef T, LLVMModuleRef M, LLVMCodeGenFileType codegen, char **ErrorMessage, LLVMMemoryBufferRef *OutMemBuf)
try: (LLVMTargetMachineEmitToMemoryBuffer:=dll.LLVMTargetMachineEmitToMemoryBuffer).restype, LLVMTargetMachineEmitToMemoryBuffer.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

# char *LLVMGetDefaultTargetTriple(void)
try: (LLVMGetDefaultTargetTriple:=dll.LLVMGetDefaultTargetTriple).restype, LLVMGetDefaultTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# char *LLVMNormalizeTargetTriple(const char *triple)
try: (LLVMNormalizeTargetTriple:=dll.LLVMNormalizeTargetTriple).restype, LLVMNormalizeTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *LLVMGetHostCPUName(void)
try: (LLVMGetHostCPUName:=dll.LLVMGetHostCPUName).restype, LLVMGetHostCPUName.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# char *LLVMGetHostCPUFeatures(void)
try: (LLVMGetHostCPUFeatures:=dll.LLVMGetHostCPUFeatures).restype, LLVMGetHostCPUFeatures.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# void LLVMAddAnalysisPasses(LLVMTargetMachineRef T, LLVMPassManagerRef PM)
try: (LLVMAddAnalysisPasses:=dll.LLVMAddAnalysisPasses).restype, LLVMAddAnalysisPasses.argtypes = None, [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueExecutionSession(Struct): pass
LLVMOrcExecutionSessionRef = ctypes.POINTER(struct_LLVMOrcOpaqueExecutionSession)
LLVMOrcErrorReporterFunction = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(struct_LLVMOpaqueError))
# void LLVMOrcExecutionSessionSetErrorReporter(LLVMOrcExecutionSessionRef ES, LLVMOrcErrorReporterFunction ReportError, void *Ctx)
try: (LLVMOrcExecutionSessionSetErrorReporter:=dll.LLVMOrcExecutionSessionSetErrorReporter).restype, LLVMOrcExecutionSessionSetErrorReporter.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcErrorReporterFunction, ctypes.c_void_p]
except AttributeError: pass

class struct_LLVMOrcOpaqueSymbolStringPool(Struct): pass
LLVMOrcSymbolStringPoolRef = ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPool)
# LLVMOrcSymbolStringPoolRef LLVMOrcExecutionSessionGetSymbolStringPool(LLVMOrcExecutionSessionRef ES)
try: (LLVMOrcExecutionSessionGetSymbolStringPool:=dll.LLVMOrcExecutionSessionGetSymbolStringPool).restype, LLVMOrcExecutionSessionGetSymbolStringPool.argtypes = LLVMOrcSymbolStringPoolRef, [LLVMOrcExecutionSessionRef]
except AttributeError: pass

# void LLVMOrcSymbolStringPoolClearDeadEntries(LLVMOrcSymbolStringPoolRef SSP)
try: (LLVMOrcSymbolStringPoolClearDeadEntries:=dll.LLVMOrcSymbolStringPoolClearDeadEntries).restype, LLVMOrcSymbolStringPoolClearDeadEntries.argtypes = None, [LLVMOrcSymbolStringPoolRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueSymbolStringPoolEntry(Struct): pass
LLVMOrcSymbolStringPoolEntryRef = ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry)
# LLVMOrcSymbolStringPoolEntryRef LLVMOrcExecutionSessionIntern(LLVMOrcExecutionSessionRef ES, const char *Name)
try: (LLVMOrcExecutionSessionIntern:=dll.LLVMOrcExecutionSessionIntern).restype, LLVMOrcExecutionSessionIntern.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

LLVMOrcLookupKind = CEnum(ctypes.c_uint32)
LLVMOrcLookupKindStatic = LLVMOrcLookupKind.define('LLVMOrcLookupKindStatic', 0)
LLVMOrcLookupKindDLSym = LLVMOrcLookupKind.define('LLVMOrcLookupKindDLSym', 1)

class LLVMOrcCJITDylibSearchOrderElement(Struct): pass
class struct_LLVMOrcOpaqueJITDylib(Struct): pass
LLVMOrcJITDylibRef = ctypes.POINTER(struct_LLVMOrcOpaqueJITDylib)
LLVMOrcJITDylibLookupFlags = CEnum(ctypes.c_uint32)
LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly = LLVMOrcJITDylibLookupFlags.define('LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly', 0)
LLVMOrcJITDylibLookupFlagsMatchAllSymbols = LLVMOrcJITDylibLookupFlags.define('LLVMOrcJITDylibLookupFlagsMatchAllSymbols', 1)

LLVMOrcCJITDylibSearchOrderElement._fields_ = [
  ('JD', LLVMOrcJITDylibRef),
  ('JDLookupFlags', LLVMOrcJITDylibLookupFlags),
]
LLVMOrcCJITDylibSearchOrder = ctypes.POINTER(LLVMOrcCJITDylibSearchOrderElement)
class LLVMOrcCLookupSetElement(Struct): pass
LLVMOrcSymbolLookupFlags = CEnum(ctypes.c_uint32)
LLVMOrcSymbolLookupFlagsRequiredSymbol = LLVMOrcSymbolLookupFlags.define('LLVMOrcSymbolLookupFlagsRequiredSymbol', 0)
LLVMOrcSymbolLookupFlagsWeaklyReferencedSymbol = LLVMOrcSymbolLookupFlags.define('LLVMOrcSymbolLookupFlagsWeaklyReferencedSymbol', 1)

LLVMOrcCLookupSetElement._fields_ = [
  ('Name', LLVMOrcSymbolStringPoolEntryRef),
  ('LookupFlags', LLVMOrcSymbolLookupFlags),
]
LLVMOrcCLookupSet = ctypes.POINTER(LLVMOrcCLookupSetElement)
class LLVMOrcCSymbolMapPair(Struct): pass
class LLVMJITEvaluatedSymbol(Struct): pass
LLVMOrcExecutorAddress = ctypes.c_uint64
class LLVMJITSymbolFlags(Struct): pass
LLVMJITSymbolFlags._fields_ = [
  ('GenericFlags', uint8_t),
  ('TargetFlags', uint8_t),
]
LLVMJITEvaluatedSymbol._fields_ = [
  ('Address', LLVMOrcExecutorAddress),
  ('Flags', LLVMJITSymbolFlags),
]
LLVMOrcCSymbolMapPair._fields_ = [
  ('Name', LLVMOrcSymbolStringPoolEntryRef),
  ('Sym', LLVMJITEvaluatedSymbol),
]
LLVMOrcExecutionSessionLookupHandleResultFunction = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_LLVMOpaqueError), ctypes.POINTER(LLVMOrcCSymbolMapPair), ctypes.c_uint64, ctypes.c_void_p)
# void LLVMOrcExecutionSessionLookup(LLVMOrcExecutionSessionRef ES, LLVMOrcLookupKind K, LLVMOrcCJITDylibSearchOrder SearchOrder, size_t SearchOrderSize, LLVMOrcCLookupSet Symbols, size_t SymbolsSize, LLVMOrcExecutionSessionLookupHandleResultFunction HandleResult, void *Ctx)
try: (LLVMOrcExecutionSessionLookup:=dll.LLVMOrcExecutionSessionLookup).restype, LLVMOrcExecutionSessionLookup.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcLookupKind, LLVMOrcCJITDylibSearchOrder, size_t, LLVMOrcCLookupSet, size_t, LLVMOrcExecutionSessionLookupHandleResultFunction, ctypes.c_void_p]
except AttributeError: pass

# void LLVMOrcRetainSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S)
try: (LLVMOrcRetainSymbolStringPoolEntry:=dll.LLVMOrcRetainSymbolStringPoolEntry).restype, LLVMOrcRetainSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

# void LLVMOrcReleaseSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S)
try: (LLVMOrcReleaseSymbolStringPoolEntry:=dll.LLVMOrcReleaseSymbolStringPoolEntry).restype, LLVMOrcReleaseSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

# const char *LLVMOrcSymbolStringPoolEntryStr(LLVMOrcSymbolStringPoolEntryRef S)
try: (LLVMOrcSymbolStringPoolEntryStr:=dll.LLVMOrcSymbolStringPoolEntryStr).restype, LLVMOrcSymbolStringPoolEntryStr.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueResourceTracker(Struct): pass
LLVMOrcResourceTrackerRef = ctypes.POINTER(struct_LLVMOrcOpaqueResourceTracker)
# void LLVMOrcReleaseResourceTracker(LLVMOrcResourceTrackerRef RT)
try: (LLVMOrcReleaseResourceTracker:=dll.LLVMOrcReleaseResourceTracker).restype, LLVMOrcReleaseResourceTracker.argtypes = None, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

# void LLVMOrcResourceTrackerTransferTo(LLVMOrcResourceTrackerRef SrcRT, LLVMOrcResourceTrackerRef DstRT)
try: (LLVMOrcResourceTrackerTransferTo:=dll.LLVMOrcResourceTrackerTransferTo).restype, LLVMOrcResourceTrackerTransferTo.argtypes = None, [LLVMOrcResourceTrackerRef, LLVMOrcResourceTrackerRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcResourceTrackerRemove(LLVMOrcResourceTrackerRef RT)
try: (LLVMOrcResourceTrackerRemove:=dll.LLVMOrcResourceTrackerRemove).restype, LLVMOrcResourceTrackerRemove.argtypes = LLVMErrorRef, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueDefinitionGenerator(Struct): pass
LLVMOrcDefinitionGeneratorRef = ctypes.POINTER(struct_LLVMOrcOpaqueDefinitionGenerator)
# void LLVMOrcDisposeDefinitionGenerator(LLVMOrcDefinitionGeneratorRef DG)
try: (LLVMOrcDisposeDefinitionGenerator:=dll.LLVMOrcDisposeDefinitionGenerator).restype, LLVMOrcDisposeDefinitionGenerator.argtypes = None, [LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueMaterializationUnit(Struct): pass
LLVMOrcMaterializationUnitRef = ctypes.POINTER(struct_LLVMOrcOpaqueMaterializationUnit)
# void LLVMOrcDisposeMaterializationUnit(LLVMOrcMaterializationUnitRef MU)
try: (LLVMOrcDisposeMaterializationUnit:=dll.LLVMOrcDisposeMaterializationUnit).restype, LLVMOrcDisposeMaterializationUnit.argtypes = None, [LLVMOrcMaterializationUnitRef]
except AttributeError: pass

class LLVMOrcCSymbolFlagsMapPair(Struct): pass
LLVMOrcCSymbolFlagsMapPair._fields_ = [
  ('Name', LLVMOrcSymbolStringPoolEntryRef),
  ('Flags', LLVMJITSymbolFlags),
]
LLVMOrcCSymbolFlagsMapPairs = ctypes.POINTER(LLVMOrcCSymbolFlagsMapPair)
class struct_LLVMOrcOpaqueMaterializationResponsibility(Struct): pass
LLVMOrcMaterializationUnitMaterializeFunction = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(struct_LLVMOrcOpaqueMaterializationResponsibility))
LLVMOrcMaterializationUnitDiscardFunction = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(struct_LLVMOrcOpaqueJITDylib), ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry))
LLVMOrcMaterializationUnitDestroyFunction = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
# LLVMOrcMaterializationUnitRef LLVMOrcCreateCustomMaterializationUnit(const char *Name, void *Ctx, LLVMOrcCSymbolFlagsMapPairs Syms, size_t NumSyms, LLVMOrcSymbolStringPoolEntryRef InitSym, LLVMOrcMaterializationUnitMaterializeFunction Materialize, LLVMOrcMaterializationUnitDiscardFunction Discard, LLVMOrcMaterializationUnitDestroyFunction Destroy)
try: (LLVMOrcCreateCustomMaterializationUnit:=dll.LLVMOrcCreateCustomMaterializationUnit).restype, LLVMOrcCreateCustomMaterializationUnit.argtypes = LLVMOrcMaterializationUnitRef, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, LLVMOrcCSymbolFlagsMapPairs, size_t, LLVMOrcSymbolStringPoolEntryRef, LLVMOrcMaterializationUnitMaterializeFunction, LLVMOrcMaterializationUnitDiscardFunction, LLVMOrcMaterializationUnitDestroyFunction]
except AttributeError: pass

LLVMOrcCSymbolMapPairs = ctypes.POINTER(LLVMOrcCSymbolMapPair)
# LLVMOrcMaterializationUnitRef LLVMOrcAbsoluteSymbols(LLVMOrcCSymbolMapPairs Syms, size_t NumPairs)
try: (LLVMOrcAbsoluteSymbols:=dll.LLVMOrcAbsoluteSymbols).restype, LLVMOrcAbsoluteSymbols.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

class struct_LLVMOrcOpaqueLazyCallThroughManager(Struct): pass
LLVMOrcLazyCallThroughManagerRef = ctypes.POINTER(struct_LLVMOrcOpaqueLazyCallThroughManager)
class struct_LLVMOrcOpaqueIndirectStubsManager(Struct): pass
LLVMOrcIndirectStubsManagerRef = ctypes.POINTER(struct_LLVMOrcOpaqueIndirectStubsManager)
class LLVMOrcCSymbolAliasMapPair(Struct): pass
class LLVMOrcCSymbolAliasMapEntry(Struct): pass
LLVMOrcCSymbolAliasMapEntry._fields_ = [
  ('Name', LLVMOrcSymbolStringPoolEntryRef),
  ('Flags', LLVMJITSymbolFlags),
]
LLVMOrcCSymbolAliasMapPair._fields_ = [
  ('Name', LLVMOrcSymbolStringPoolEntryRef),
  ('Entry', LLVMOrcCSymbolAliasMapEntry),
]
LLVMOrcCSymbolAliasMapPairs = ctypes.POINTER(LLVMOrcCSymbolAliasMapPair)
# LLVMOrcMaterializationUnitRef LLVMOrcLazyReexports(LLVMOrcLazyCallThroughManagerRef LCTM, LLVMOrcIndirectStubsManagerRef ISM, LLVMOrcJITDylibRef SourceRef, LLVMOrcCSymbolAliasMapPairs CallableAliases, size_t NumPairs)
try: (LLVMOrcLazyReexports:=dll.LLVMOrcLazyReexports).restype, LLVMOrcLazyReexports.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcLazyCallThroughManagerRef, LLVMOrcIndirectStubsManagerRef, LLVMOrcJITDylibRef, LLVMOrcCSymbolAliasMapPairs, size_t]
except AttributeError: pass

LLVMOrcMaterializationResponsibilityRef = ctypes.POINTER(struct_LLVMOrcOpaqueMaterializationResponsibility)
# void LLVMOrcDisposeMaterializationResponsibility(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcDisposeMaterializationResponsibility:=dll.LLVMOrcDisposeMaterializationResponsibility).restype, LLVMOrcDisposeMaterializationResponsibility.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcJITDylibRef LLVMOrcMaterializationResponsibilityGetTargetDylib(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityGetTargetDylib:=dll.LLVMOrcMaterializationResponsibilityGetTargetDylib).restype, LLVMOrcMaterializationResponsibilityGetTargetDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcExecutionSessionRef LLVMOrcMaterializationResponsibilityGetExecutionSession(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityGetExecutionSession:=dll.LLVMOrcMaterializationResponsibilityGetExecutionSession).restype, LLVMOrcMaterializationResponsibilityGetExecutionSession.argtypes = LLVMOrcExecutionSessionRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcCSymbolFlagsMapPairs LLVMOrcMaterializationResponsibilityGetSymbols(LLVMOrcMaterializationResponsibilityRef MR, size_t *NumPairs)
try: (LLVMOrcMaterializationResponsibilityGetSymbols:=dll.LLVMOrcMaterializationResponsibilityGetSymbols).restype, LLVMOrcMaterializationResponsibilityGetSymbols.argtypes = LLVMOrcCSymbolFlagsMapPairs, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# void LLVMOrcDisposeCSymbolFlagsMap(LLVMOrcCSymbolFlagsMapPairs Pairs)
try: (LLVMOrcDisposeCSymbolFlagsMap:=dll.LLVMOrcDisposeCSymbolFlagsMap).restype, LLVMOrcDisposeCSymbolFlagsMap.argtypes = None, [LLVMOrcCSymbolFlagsMapPairs]
except AttributeError: pass

# LLVMOrcSymbolStringPoolEntryRef LLVMOrcMaterializationResponsibilityGetInitializerSymbol(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityGetInitializerSymbol:=dll.LLVMOrcMaterializationResponsibilityGetInitializerSymbol).restype, LLVMOrcMaterializationResponsibilityGetInitializerSymbol.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcSymbolStringPoolEntryRef *LLVMOrcMaterializationResponsibilityGetRequestedSymbols(LLVMOrcMaterializationResponsibilityRef MR, size_t *NumSymbols)
try: (LLVMOrcMaterializationResponsibilityGetRequestedSymbols:=dll.LLVMOrcMaterializationResponsibilityGetRequestedSymbols).restype, LLVMOrcMaterializationResponsibilityGetRequestedSymbols.argtypes = ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# void LLVMOrcDisposeSymbols(LLVMOrcSymbolStringPoolEntryRef *Symbols)
try: (LLVMOrcDisposeSymbols:=dll.LLVMOrcDisposeSymbols).restype, LLVMOrcDisposeSymbols.argtypes = None, [ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef)]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityNotifyResolved(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolMapPairs Symbols, size_t NumPairs)
try: (LLVMOrcMaterializationResponsibilityNotifyResolved:=dll.LLVMOrcMaterializationResponsibilityNotifyResolved).restype, LLVMOrcMaterializationResponsibilityNotifyResolved.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

class LLVMOrcCSymbolDependenceGroup(Struct): pass
class LLVMOrcCSymbolsList(Struct): pass
LLVMOrcCSymbolsList._fields_ = [
  ('Symbols', ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef)),
  ('Length', size_t),
]
class LLVMOrcCDependenceMapPair(Struct): pass
LLVMOrcCDependenceMapPair._fields_ = [
  ('JD', LLVMOrcJITDylibRef),
  ('Names', LLVMOrcCSymbolsList),
]
LLVMOrcCDependenceMapPairs = ctypes.POINTER(LLVMOrcCDependenceMapPair)
LLVMOrcCSymbolDependenceGroup._fields_ = [
  ('Symbols', LLVMOrcCSymbolsList),
  ('Dependencies', LLVMOrcCDependenceMapPairs),
  ('NumDependencies', size_t),
]
# LLVMErrorRef LLVMOrcMaterializationResponsibilityNotifyEmitted(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolDependenceGroup *SymbolDepGroups, size_t NumSymbolDepGroups)
try: (LLVMOrcMaterializationResponsibilityNotifyEmitted:=dll.LLVMOrcMaterializationResponsibilityNotifyEmitted).restype, LLVMOrcMaterializationResponsibilityNotifyEmitted.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcCSymbolDependenceGroup), size_t]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityDefineMaterializing(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolFlagsMapPairs Pairs, size_t NumPairs)
try: (LLVMOrcMaterializationResponsibilityDefineMaterializing:=dll.LLVMOrcMaterializationResponsibilityDefineMaterializing).restype, LLVMOrcMaterializationResponsibilityDefineMaterializing.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolFlagsMapPairs, size_t]
except AttributeError: pass

# void LLVMOrcMaterializationResponsibilityFailMaterialization(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityFailMaterialization:=dll.LLVMOrcMaterializationResponsibilityFailMaterialization).restype, LLVMOrcMaterializationResponsibilityFailMaterialization.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityReplace(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcMaterializationUnitRef MU)
try: (LLVMOrcMaterializationResponsibilityReplace:=dll.LLVMOrcMaterializationResponsibilityReplace).restype, LLVMOrcMaterializationResponsibilityReplace.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityDelegate(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcSymbolStringPoolEntryRef *Symbols, size_t NumSymbols, LLVMOrcMaterializationResponsibilityRef *Result)
try: (LLVMOrcMaterializationResponsibilityDelegate:=dll.LLVMOrcMaterializationResponsibilityDelegate).restype, LLVMOrcMaterializationResponsibilityDelegate.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), size_t, ctypes.POINTER(LLVMOrcMaterializationResponsibilityRef)]
except AttributeError: pass

# LLVMOrcJITDylibRef LLVMOrcExecutionSessionCreateBareJITDylib(LLVMOrcExecutionSessionRef ES, const char *Name)
try: (LLVMOrcExecutionSessionCreateBareJITDylib:=dll.LLVMOrcExecutionSessionCreateBareJITDylib).restype, LLVMOrcExecutionSessionCreateBareJITDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorRef LLVMOrcExecutionSessionCreateJITDylib(LLVMOrcExecutionSessionRef ES, LLVMOrcJITDylibRef *Result, const char *Name)
try: (LLVMOrcExecutionSessionCreateJITDylib:=dll.LLVMOrcExecutionSessionCreateJITDylib).restype, LLVMOrcExecutionSessionCreateJITDylib.argtypes = LLVMErrorRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(LLVMOrcJITDylibRef), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMOrcJITDylibRef LLVMOrcExecutionSessionGetJITDylibByName(LLVMOrcExecutionSessionRef ES, const char *Name)
try: (LLVMOrcExecutionSessionGetJITDylibByName:=dll.LLVMOrcExecutionSessionGetJITDylibByName).restype, LLVMOrcExecutionSessionGetJITDylibByName.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMOrcResourceTrackerRef LLVMOrcJITDylibCreateResourceTracker(LLVMOrcJITDylibRef JD)
try: (LLVMOrcJITDylibCreateResourceTracker:=dll.LLVMOrcJITDylibCreateResourceTracker).restype, LLVMOrcJITDylibCreateResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

# LLVMOrcResourceTrackerRef LLVMOrcJITDylibGetDefaultResourceTracker(LLVMOrcJITDylibRef JD)
try: (LLVMOrcJITDylibGetDefaultResourceTracker:=dll.LLVMOrcJITDylibGetDefaultResourceTracker).restype, LLVMOrcJITDylibGetDefaultResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcJITDylibDefine(LLVMOrcJITDylibRef JD, LLVMOrcMaterializationUnitRef MU)
try: (LLVMOrcJITDylibDefine:=dll.LLVMOrcJITDylibDefine).restype, LLVMOrcJITDylibDefine.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcJITDylibClear(LLVMOrcJITDylibRef JD)
try: (LLVMOrcJITDylibClear:=dll.LLVMOrcJITDylibClear).restype, LLVMOrcJITDylibClear.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

# void LLVMOrcJITDylibAddGenerator(LLVMOrcJITDylibRef JD, LLVMOrcDefinitionGeneratorRef DG)
try: (LLVMOrcJITDylibAddGenerator:=dll.LLVMOrcJITDylibAddGenerator).restype, LLVMOrcJITDylibAddGenerator.argtypes = None, [LLVMOrcJITDylibRef, LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueLookupState(Struct): pass
LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueError), ctypes.POINTER(struct_LLVMOrcOpaqueDefinitionGenerator), ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueLookupState)), LLVMOrcLookupKind, ctypes.POINTER(struct_LLVMOrcOpaqueJITDylib), LLVMOrcJITDylibLookupFlags, ctypes.POINTER(LLVMOrcCLookupSetElement), ctypes.c_uint64)
LLVMOrcDisposeCAPIDefinitionGeneratorFunction = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
# LLVMOrcDefinitionGeneratorRef LLVMOrcCreateCustomCAPIDefinitionGenerator(LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction F, void *Ctx, LLVMOrcDisposeCAPIDefinitionGeneratorFunction Dispose)
try: (LLVMOrcCreateCustomCAPIDefinitionGenerator:=dll.LLVMOrcCreateCustomCAPIDefinitionGenerator).restype, LLVMOrcCreateCustomCAPIDefinitionGenerator.argtypes = LLVMOrcDefinitionGeneratorRef, [LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction, ctypes.c_void_p, LLVMOrcDisposeCAPIDefinitionGeneratorFunction]
except AttributeError: pass

LLVMOrcLookupStateRef = ctypes.POINTER(struct_LLVMOrcOpaqueLookupState)
# void LLVMOrcLookupStateContinueLookup(LLVMOrcLookupStateRef S, LLVMErrorRef Err)
try: (LLVMOrcLookupStateContinueLookup:=dll.LLVMOrcLookupStateContinueLookup).restype, LLVMOrcLookupStateContinueLookup.argtypes = None, [LLVMOrcLookupStateRef, LLVMErrorRef]
except AttributeError: pass

LLVMOrcSymbolPredicate = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry))
# LLVMErrorRef LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(LLVMOrcDefinitionGeneratorRef *Result, char GlobalPrefx, LLVMOrcSymbolPredicate Filter, void *FilterCtx)
try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(LLVMOrcDefinitionGeneratorRef *Result, const char *FileName, char GlobalPrefix, LLVMOrcSymbolPredicate Filter, void *FilterCtx)
try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForPath).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.POINTER(ctypes.c_char), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

class struct_LLVMOrcOpaqueObjectLayer(Struct): pass
LLVMOrcObjectLayerRef = ctypes.POINTER(struct_LLVMOrcOpaqueObjectLayer)
# LLVMErrorRef LLVMOrcCreateStaticLibrarySearchGeneratorForPath(LLVMOrcDefinitionGeneratorRef *Result, LLVMOrcObjectLayerRef ObjLayer, const char *FileName, const char *TargetTriple)
try: (LLVMOrcCreateStaticLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateStaticLibrarySearchGeneratorForPath).restype, LLVMOrcCreateStaticLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), LLVMOrcObjectLayerRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class struct_LLVMOrcOpaqueThreadSafeContext(Struct): pass
LLVMOrcThreadSafeContextRef = ctypes.POINTER(struct_LLVMOrcOpaqueThreadSafeContext)
# LLVMOrcThreadSafeContextRef LLVMOrcCreateNewThreadSafeContext(void)
try: (LLVMOrcCreateNewThreadSafeContext:=dll.LLVMOrcCreateNewThreadSafeContext).restype, LLVMOrcCreateNewThreadSafeContext.argtypes = LLVMOrcThreadSafeContextRef, []
except AttributeError: pass

# LLVMContextRef LLVMOrcThreadSafeContextGetContext(LLVMOrcThreadSafeContextRef TSCtx)
try: (LLVMOrcThreadSafeContextGetContext:=dll.LLVMOrcThreadSafeContextGetContext).restype, LLVMOrcThreadSafeContextGetContext.argtypes = LLVMContextRef, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

# void LLVMOrcDisposeThreadSafeContext(LLVMOrcThreadSafeContextRef TSCtx)
try: (LLVMOrcDisposeThreadSafeContext:=dll.LLVMOrcDisposeThreadSafeContext).restype, LLVMOrcDisposeThreadSafeContext.argtypes = None, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueThreadSafeModule(Struct): pass
LLVMOrcThreadSafeModuleRef = ctypes.POINTER(struct_LLVMOrcOpaqueThreadSafeModule)
# LLVMOrcThreadSafeModuleRef LLVMOrcCreateNewThreadSafeModule(LLVMModuleRef M, LLVMOrcThreadSafeContextRef TSCtx)
try: (LLVMOrcCreateNewThreadSafeModule:=dll.LLVMOrcCreateNewThreadSafeModule).restype, LLVMOrcCreateNewThreadSafeModule.argtypes = LLVMOrcThreadSafeModuleRef, [LLVMModuleRef, LLVMOrcThreadSafeContextRef]
except AttributeError: pass

# void LLVMOrcDisposeThreadSafeModule(LLVMOrcThreadSafeModuleRef TSM)
try: (LLVMOrcDisposeThreadSafeModule:=dll.LLVMOrcDisposeThreadSafeModule).restype, LLVMOrcDisposeThreadSafeModule.argtypes = None, [LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

LLVMOrcGenericIRModuleOperationFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueError), ctypes.c_void_p, ctypes.POINTER(struct_LLVMOpaqueModule))
# LLVMErrorRef LLVMOrcThreadSafeModuleWithModuleDo(LLVMOrcThreadSafeModuleRef TSM, LLVMOrcGenericIRModuleOperationFunction F, void *Ctx)
try: (LLVMOrcThreadSafeModuleWithModuleDo:=dll.LLVMOrcThreadSafeModuleWithModuleDo).restype, LLVMOrcThreadSafeModuleWithModuleDo.argtypes = LLVMErrorRef, [LLVMOrcThreadSafeModuleRef, LLVMOrcGenericIRModuleOperationFunction, ctypes.c_void_p]
except AttributeError: pass

class struct_LLVMOrcOpaqueJITTargetMachineBuilder(Struct): pass
LLVMOrcJITTargetMachineBuilderRef = ctypes.POINTER(struct_LLVMOrcOpaqueJITTargetMachineBuilder)
# LLVMErrorRef LLVMOrcJITTargetMachineBuilderDetectHost(LLVMOrcJITTargetMachineBuilderRef *Result)
try: (LLVMOrcJITTargetMachineBuilderDetectHost:=dll.LLVMOrcJITTargetMachineBuilderDetectHost).restype, LLVMOrcJITTargetMachineBuilderDetectHost.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcJITTargetMachineBuilderRef)]
except AttributeError: pass

# LLVMOrcJITTargetMachineBuilderRef LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(LLVMTargetMachineRef TM)
try: (LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine:=dll.LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine).restype, LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine.argtypes = LLVMOrcJITTargetMachineBuilderRef, [LLVMTargetMachineRef]
except AttributeError: pass

# void LLVMOrcDisposeJITTargetMachineBuilder(LLVMOrcJITTargetMachineBuilderRef JTMB)
try: (LLVMOrcDisposeJITTargetMachineBuilder:=dll.LLVMOrcDisposeJITTargetMachineBuilder).restype, LLVMOrcDisposeJITTargetMachineBuilder.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

# char *LLVMOrcJITTargetMachineBuilderGetTargetTriple(LLVMOrcJITTargetMachineBuilderRef JTMB)
try: (LLVMOrcJITTargetMachineBuilderGetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderGetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderGetTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

# void LLVMOrcJITTargetMachineBuilderSetTargetTriple(LLVMOrcJITTargetMachineBuilderRef JTMB, const char *TargetTriple)
try: (LLVMOrcJITTargetMachineBuilderSetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderSetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderSetTargetTriple.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorRef LLVMOrcObjectLayerAddObjectFile(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcJITDylibRef JD, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcObjectLayerAddObjectFile:=dll.LLVMOrcObjectLayerAddObjectFile).restype, LLVMOrcObjectLayerAddObjectFile.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcJITDylibRef, LLVMMemoryBufferRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcObjectLayerAddObjectFileWithRT(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcResourceTrackerRef RT, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcObjectLayerAddObjectFileWithRT:=dll.LLVMOrcObjectLayerAddObjectFileWithRT).restype, LLVMOrcObjectLayerAddObjectFileWithRT.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef]
except AttributeError: pass

# void LLVMOrcObjectLayerEmit(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcMaterializationResponsibilityRef R, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcObjectLayerEmit:=dll.LLVMOrcObjectLayerEmit).restype, LLVMOrcObjectLayerEmit.argtypes = None, [LLVMOrcObjectLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMMemoryBufferRef]
except AttributeError: pass

# void LLVMOrcDisposeObjectLayer(LLVMOrcObjectLayerRef ObjLayer)
try: (LLVMOrcDisposeObjectLayer:=dll.LLVMOrcDisposeObjectLayer).restype, LLVMOrcDisposeObjectLayer.argtypes = None, [LLVMOrcObjectLayerRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueIRTransformLayer(Struct): pass
LLVMOrcIRTransformLayerRef = ctypes.POINTER(struct_LLVMOrcOpaqueIRTransformLayer)
# void LLVMOrcIRTransformLayerEmit(LLVMOrcIRTransformLayerRef IRTransformLayer, LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcThreadSafeModuleRef TSM)
try: (LLVMOrcIRTransformLayerEmit:=dll.LLVMOrcIRTransformLayerEmit).restype, LLVMOrcIRTransformLayerEmit.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

LLVMOrcIRTransformLayerTransformFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueError), ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueThreadSafeModule)), ctypes.POINTER(struct_LLVMOrcOpaqueMaterializationResponsibility))
# void LLVMOrcIRTransformLayerSetTransform(LLVMOrcIRTransformLayerRef IRTransformLayer, LLVMOrcIRTransformLayerTransformFunction TransformFunction, void *Ctx)
try: (LLVMOrcIRTransformLayerSetTransform:=dll.LLVMOrcIRTransformLayerSetTransform).restype, LLVMOrcIRTransformLayerSetTransform.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcIRTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

class struct_LLVMOrcOpaqueObjectTransformLayer(Struct): pass
LLVMOrcObjectTransformLayerRef = ctypes.POINTER(struct_LLVMOrcOpaqueObjectTransformLayer)
LLVMOrcObjectTransformLayerTransformFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueError), ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMemoryBuffer)))
# void LLVMOrcObjectTransformLayerSetTransform(LLVMOrcObjectTransformLayerRef ObjTransformLayer, LLVMOrcObjectTransformLayerTransformFunction TransformFunction, void *Ctx)
try: (LLVMOrcObjectTransformLayerSetTransform:=dll.LLVMOrcObjectTransformLayerSetTransform).restype, LLVMOrcObjectTransformLayerSetTransform.argtypes = None, [LLVMOrcObjectTransformLayerRef, LLVMOrcObjectTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

# LLVMOrcIndirectStubsManagerRef LLVMOrcCreateLocalIndirectStubsManager(const char *TargetTriple)
try: (LLVMOrcCreateLocalIndirectStubsManager:=dll.LLVMOrcCreateLocalIndirectStubsManager).restype, LLVMOrcCreateLocalIndirectStubsManager.argtypes = LLVMOrcIndirectStubsManagerRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMOrcDisposeIndirectStubsManager(LLVMOrcIndirectStubsManagerRef ISM)
try: (LLVMOrcDisposeIndirectStubsManager:=dll.LLVMOrcDisposeIndirectStubsManager).restype, LLVMOrcDisposeIndirectStubsManager.argtypes = None, [LLVMOrcIndirectStubsManagerRef]
except AttributeError: pass

LLVMOrcJITTargetAddress = ctypes.c_uint64
# LLVMErrorRef LLVMOrcCreateLocalLazyCallThroughManager(const char *TargetTriple, LLVMOrcExecutionSessionRef ES, LLVMOrcJITTargetAddress ErrorHandlerAddr, LLVMOrcLazyCallThroughManagerRef *LCTM)
try: (LLVMOrcCreateLocalLazyCallThroughManager:=dll.LLVMOrcCreateLocalLazyCallThroughManager).restype, LLVMOrcCreateLocalLazyCallThroughManager.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char), LLVMOrcExecutionSessionRef, LLVMOrcJITTargetAddress, ctypes.POINTER(LLVMOrcLazyCallThroughManagerRef)]
except AttributeError: pass

# void LLVMOrcDisposeLazyCallThroughManager(LLVMOrcLazyCallThroughManagerRef LCTM)
try: (LLVMOrcDisposeLazyCallThroughManager:=dll.LLVMOrcDisposeLazyCallThroughManager).restype, LLVMOrcDisposeLazyCallThroughManager.argtypes = None, [LLVMOrcLazyCallThroughManagerRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueDumpObjects(Struct): pass
LLVMOrcDumpObjectsRef = ctypes.POINTER(struct_LLVMOrcOpaqueDumpObjects)
# LLVMOrcDumpObjectsRef LLVMOrcCreateDumpObjects(const char *DumpDir, const char *IdentifierOverride)
try: (LLVMOrcCreateDumpObjects:=dll.LLVMOrcCreateDumpObjects).restype, LLVMOrcCreateDumpObjects.argtypes = LLVMOrcDumpObjectsRef, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMOrcDisposeDumpObjects(LLVMOrcDumpObjectsRef DumpObjects)
try: (LLVMOrcDisposeDumpObjects:=dll.LLVMOrcDisposeDumpObjects).restype, LLVMOrcDisposeDumpObjects.argtypes = None, [LLVMOrcDumpObjectsRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcDumpObjects_CallOperator(LLVMOrcDumpObjectsRef DumpObjects, LLVMMemoryBufferRef *ObjBuffer)
try: (LLVMOrcDumpObjects_CallOperator:=dll.LLVMOrcDumpObjects_CallOperator).restype, LLVMOrcDumpObjects_CallOperator.argtypes = LLVMErrorRef, [LLVMOrcDumpObjectsRef, ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOrcOpaqueObjectLayer), ctypes.c_void_p, ctypes.POINTER(struct_LLVMOrcOpaqueExecutionSession), ctypes.POINTER(ctypes.c_char))
class struct_LLVMOrcOpaqueLLJITBuilder(Struct): pass
LLVMOrcLLJITBuilderRef = ctypes.POINTER(struct_LLVMOrcOpaqueLLJITBuilder)
class struct_LLVMOrcOpaqueLLJIT(Struct): pass
LLVMOrcLLJITRef = ctypes.POINTER(struct_LLVMOrcOpaqueLLJIT)
# LLVMOrcLLJITBuilderRef LLVMOrcCreateLLJITBuilder(void)
try: (LLVMOrcCreateLLJITBuilder:=dll.LLVMOrcCreateLLJITBuilder).restype, LLVMOrcCreateLLJITBuilder.argtypes = LLVMOrcLLJITBuilderRef, []
except AttributeError: pass

# void LLVMOrcDisposeLLJITBuilder(LLVMOrcLLJITBuilderRef Builder)
try: (LLVMOrcDisposeLLJITBuilder:=dll.LLVMOrcDisposeLLJITBuilder).restype, LLVMOrcDisposeLLJITBuilder.argtypes = None, [LLVMOrcLLJITBuilderRef]
except AttributeError: pass

# void LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(LLVMOrcLLJITBuilderRef Builder, LLVMOrcJITTargetMachineBuilderRef JTMB)
try: (LLVMOrcLLJITBuilderSetJITTargetMachineBuilder:=dll.LLVMOrcLLJITBuilderSetJITTargetMachineBuilder).restype, LLVMOrcLLJITBuilderSetJITTargetMachineBuilder.argtypes = None, [LLVMOrcLLJITBuilderRef, LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

# void LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(LLVMOrcLLJITBuilderRef Builder, LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction F, void *Ctx)
try: (LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator:=dll.LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator).restype, LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator.argtypes = None, [LLVMOrcLLJITBuilderRef, LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction, ctypes.c_void_p]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateLLJIT(LLVMOrcLLJITRef *Result, LLVMOrcLLJITBuilderRef Builder)
try: (LLVMOrcCreateLLJIT:=dll.LLVMOrcCreateLLJIT).restype, LLVMOrcCreateLLJIT.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcLLJITRef), LLVMOrcLLJITBuilderRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcDisposeLLJIT(LLVMOrcLLJITRef J)
try: (LLVMOrcDisposeLLJIT:=dll.LLVMOrcDisposeLLJIT).restype, LLVMOrcDisposeLLJIT.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef]
except AttributeError: pass

# LLVMOrcExecutionSessionRef LLVMOrcLLJITGetExecutionSession(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetExecutionSession:=dll.LLVMOrcLLJITGetExecutionSession).restype, LLVMOrcLLJITGetExecutionSession.argtypes = LLVMOrcExecutionSessionRef, [LLVMOrcLLJITRef]
except AttributeError: pass

# LLVMOrcJITDylibRef LLVMOrcLLJITGetMainJITDylib(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetMainJITDylib:=dll.LLVMOrcLLJITGetMainJITDylib).restype, LLVMOrcLLJITGetMainJITDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcLLJITRef]
except AttributeError: pass

# const char *LLVMOrcLLJITGetTripleString(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetTripleString:=dll.LLVMOrcLLJITGetTripleString).restype, LLVMOrcLLJITGetTripleString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcLLJITRef]
except AttributeError: pass

# char LLVMOrcLLJITGetGlobalPrefix(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetGlobalPrefix:=dll.LLVMOrcLLJITGetGlobalPrefix).restype, LLVMOrcLLJITGetGlobalPrefix.argtypes = ctypes.c_char, [LLVMOrcLLJITRef]
except AttributeError: pass

# LLVMOrcSymbolStringPoolEntryRef LLVMOrcLLJITMangleAndIntern(LLVMOrcLLJITRef J, const char *UnmangledName)
try: (LLVMOrcLLJITMangleAndIntern:=dll.LLVMOrcLLJITMangleAndIntern).restype, LLVMOrcLLJITMangleAndIntern.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcLLJITRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorRef LLVMOrcLLJITAddObjectFile(LLVMOrcLLJITRef J, LLVMOrcJITDylibRef JD, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcLLJITAddObjectFile:=dll.LLVMOrcLLJITAddObjectFile).restype, LLVMOrcLLJITAddObjectFile.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcJITDylibRef, LLVMMemoryBufferRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcLLJITAddObjectFileWithRT(LLVMOrcLLJITRef J, LLVMOrcResourceTrackerRef RT, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcLLJITAddObjectFileWithRT:=dll.LLVMOrcLLJITAddObjectFileWithRT).restype, LLVMOrcLLJITAddObjectFileWithRT.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcLLJITAddLLVMIRModule(LLVMOrcLLJITRef J, LLVMOrcJITDylibRef JD, LLVMOrcThreadSafeModuleRef TSM)
try: (LLVMOrcLLJITAddLLVMIRModule:=dll.LLVMOrcLLJITAddLLVMIRModule).restype, LLVMOrcLLJITAddLLVMIRModule.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcJITDylibRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcLLJITAddLLVMIRModuleWithRT(LLVMOrcLLJITRef J, LLVMOrcResourceTrackerRef JD, LLVMOrcThreadSafeModuleRef TSM)
try: (LLVMOrcLLJITAddLLVMIRModuleWithRT:=dll.LLVMOrcLLJITAddLLVMIRModuleWithRT).restype, LLVMOrcLLJITAddLLVMIRModuleWithRT.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcResourceTrackerRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcLLJITLookup(LLVMOrcLLJITRef J, LLVMOrcExecutorAddress *Result, const char *Name)
try: (LLVMOrcLLJITLookup:=dll.LLVMOrcLLJITLookup).restype, LLVMOrcLLJITLookup.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, ctypes.POINTER(LLVMOrcExecutorAddress), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMOrcObjectLayerRef LLVMOrcLLJITGetObjLinkingLayer(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetObjLinkingLayer:=dll.LLVMOrcLLJITGetObjLinkingLayer).restype, LLVMOrcLLJITGetObjLinkingLayer.argtypes = LLVMOrcObjectLayerRef, [LLVMOrcLLJITRef]
except AttributeError: pass

# LLVMOrcObjectTransformLayerRef LLVMOrcLLJITGetObjTransformLayer(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetObjTransformLayer:=dll.LLVMOrcLLJITGetObjTransformLayer).restype, LLVMOrcLLJITGetObjTransformLayer.argtypes = LLVMOrcObjectTransformLayerRef, [LLVMOrcLLJITRef]
except AttributeError: pass

# LLVMOrcIRTransformLayerRef LLVMOrcLLJITGetIRTransformLayer(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetIRTransformLayer:=dll.LLVMOrcLLJITGetIRTransformLayer).restype, LLVMOrcLLJITGetIRTransformLayer.argtypes = LLVMOrcIRTransformLayerRef, [LLVMOrcLLJITRef]
except AttributeError: pass

# const char *LLVMOrcLLJITGetDataLayoutStr(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetDataLayoutStr:=dll.LLVMOrcLLJITGetDataLayoutStr).restype, LLVMOrcLLJITGetDataLayoutStr.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcLLJITRef]
except AttributeError: pass

# LLVMErrorTypeId LLVMGetErrorTypeId(LLVMErrorRef Err)
try: (LLVMGetErrorTypeId:=dll.LLVMGetErrorTypeId).restype, LLVMGetErrorTypeId.argtypes = LLVMErrorTypeId, [LLVMErrorRef]
except AttributeError: pass

# void LLVMConsumeError(LLVMErrorRef Err)
try: (LLVMConsumeError:=dll.LLVMConsumeError).restype, LLVMConsumeError.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

# void LLVMCantFail(LLVMErrorRef Err)
try: (LLVMCantFail:=dll.LLVMCantFail).restype, LLVMCantFail.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

# char *LLVMGetErrorMessage(LLVMErrorRef Err)
try: (LLVMGetErrorMessage:=dll.LLVMGetErrorMessage).restype, LLVMGetErrorMessage.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMErrorRef]
except AttributeError: pass

# void LLVMDisposeErrorMessage(char *ErrMsg)
try: (LLVMDisposeErrorMessage:=dll.LLVMDisposeErrorMessage).restype, LLVMDisposeErrorMessage.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorTypeId LLVMGetStringErrorTypeId(void)
try: (LLVMGetStringErrorTypeId:=dll.LLVMGetStringErrorTypeId).restype, LLVMGetStringErrorTypeId.argtypes = LLVMErrorTypeId, []
except AttributeError: pass

# LLVMErrorRef LLVMCreateStringError(const char *ErrMsg)
try: (LLVMCreateStringError:=dll.LLVMCreateStringError).restype, LLVMCreateStringError.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

# void LLVMInitializeAArch64TargetInfo(void)
try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetInfo(void)
try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetInfo(void)
try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetInfo(void)
try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetInfo(void)
try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetInfo(void)
try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetInfo(void)
try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetInfo(void)
try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetInfo(void)
try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetInfo(void)
try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetInfo(void)
try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetInfo(void)
try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetInfo(void)
try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetInfo(void)
try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetInfo(void)
try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetInfo(void)
try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetInfo(void)
try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetInfo(void)
try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetInfo(void)
try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetInfo(void)
try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetInfo(void)
try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetInfo(void)
try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Target(void)
try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTarget(void)
try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTarget(void)
try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTarget(void)
try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTarget(void)
try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTarget(void)
try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTarget(void)
try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTarget(void)
try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTarget(void)
try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Target(void)
try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTarget(void)
try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTarget(void)
try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTarget(void)
try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTarget(void)
try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTarget(void)
try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTarget(void)
try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETarget(void)
try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTarget(void)
try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Target(void)
try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTarget(void)
try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTarget(void)
try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTarget(void)
try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64TargetMC(void)
try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetMC(void)
try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetMC(void)
try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetMC(void)
try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetMC(void)
try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetMC(void)
try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetMC(void)
try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetMC(void)
try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetMC(void)
try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetMC(void)
try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetMC(void)
try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetMC(void)
try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetMC(void)
try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetMC(void)
try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetMC(void)
try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetMC(void)
try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetMC(void)
try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetMC(void)
try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetMC(void)
try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetMC(void)
try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetMC(void)
try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetMC(void)
try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmPrinter(void)
try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmPrinter(void)
try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmPrinter(void)
try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmPrinter(void)
try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmPrinter(void)
try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmPrinter(void)
try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmPrinter(void)
try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmPrinter(void)
try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmPrinter(void)
try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmPrinter(void)
try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXAsmPrinter(void)
try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmPrinter(void)
try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmPrinter(void)
try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmPrinter(void)
try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVAsmPrinter(void)
try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmPrinter(void)
try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmPrinter(void)
try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmPrinter(void)
try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmPrinter(void)
try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreAsmPrinter(void)
try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmPrinter(void)
try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmPrinter(void)
try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmParser(void)
try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmParser(void)
try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmParser(void)
try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmParser(void)
try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmParser(void)
try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmParser(void)
try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmParser(void)
try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmParser(void)
try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmParser(void)
try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmParser(void)
try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmParser(void)
try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmParser(void)
try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmParser(void)
try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmParser(void)
try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmParser(void)
try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmParser(void)
try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmParser(void)
try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmParser(void)
try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmParser(void)
try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Disassembler(void)
try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUDisassembler(void)
try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMDisassembler(void)
try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRDisassembler(void)
try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFDisassembler(void)
try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonDisassembler(void)
try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiDisassembler(void)
try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchDisassembler(void)
try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsDisassembler(void)
try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Disassembler(void)
try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCDisassembler(void)
try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVDisassembler(void)
try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcDisassembler(void)
try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZDisassembler(void)
try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEDisassembler(void)
try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyDisassembler(void)
try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Disassembler(void)
try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreDisassembler(void)
try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kDisassembler(void)
try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaDisassembler(void)
try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

# LLVMTargetDataRef LLVMGetModuleDataLayout(LLVMModuleRef M)
try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

# void LLVMSetModuleDataLayout(LLVMModuleRef M, LLVMTargetDataRef DL)
try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetData(const char *StringRep)
try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMDisposeTargetData(LLVMTargetDataRef TD)
try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

# void LLVMAddTargetLibraryInfo(LLVMTargetLibraryInfoRef TLI, LLVMPassManagerRef PM)
try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

# char *LLVMCopyStringRepOfTargetData(LLVMTargetDataRef TD)
try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

# enum LLVMByteOrdering LLVMByteOrder(LLVMTargetDataRef TD)
try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSize(LLVMTargetDataRef TD)
try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSizeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrType(LLVMTargetDataRef TD)
try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeInContext(LLVMContextRef C, LLVMTargetDataRef TD)
try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForASInContext(LLVMContextRef C, LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# unsigned long long LLVMSizeOfTypeInBits(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMStoreSizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMABISizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMABIAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMCallFrameAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfGlobal(LLVMTargetDataRef TD, LLVMValueRef GlobalVar)
try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

# unsigned int LLVMElementAtOffset(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned long long Offset)
try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

# unsigned long long LLVMOffsetOfElement(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned int Element)
try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTargetRef LLVMGetFirstTarget(void)
try: (LLVMGetFirstTarget:=dll.LLVMGetFirstTarget).restype, LLVMGetFirstTarget.argtypes = LLVMTargetRef, []
except AttributeError: pass

# LLVMTargetRef LLVMGetNextTarget(LLVMTargetRef T)
try: (LLVMGetNextTarget:=dll.LLVMGetNextTarget).restype, LLVMGetNextTarget.argtypes = LLVMTargetRef, [LLVMTargetRef]
except AttributeError: pass

# LLVMTargetRef LLVMGetTargetFromName(const char *Name)
try: (LLVMGetTargetFromName:=dll.LLVMGetTargetFromName).restype, LLVMGetTargetFromName.argtypes = LLVMTargetRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBool LLVMGetTargetFromTriple(const char *Triple, LLVMTargetRef *T, char **ErrorMessage)
try: (LLVMGetTargetFromTriple:=dll.LLVMGetTargetFromTriple).restype, LLVMGetTargetFromTriple.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTargetRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# const char *LLVMGetTargetName(LLVMTargetRef T)
try: (LLVMGetTargetName:=dll.LLVMGetTargetName).restype, LLVMGetTargetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

# const char *LLVMGetTargetDescription(LLVMTargetRef T)
try: (LLVMGetTargetDescription:=dll.LLVMGetTargetDescription).restype, LLVMGetTargetDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasJIT(LLVMTargetRef T)
try: (LLVMTargetHasJIT:=dll.LLVMTargetHasJIT).restype, LLVMTargetHasJIT.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasTargetMachine(LLVMTargetRef T)
try: (LLVMTargetHasTargetMachine:=dll.LLVMTargetHasTargetMachine).restype, LLVMTargetHasTargetMachine.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasAsmBackend(LLVMTargetRef T)
try: (LLVMTargetHasAsmBackend:=dll.LLVMTargetHasAsmBackend).restype, LLVMTargetHasAsmBackend.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMTargetMachineOptionsRef LLVMCreateTargetMachineOptions(void)
try: (LLVMCreateTargetMachineOptions:=dll.LLVMCreateTargetMachineOptions).restype, LLVMCreateTargetMachineOptions.argtypes = LLVMTargetMachineOptionsRef, []
except AttributeError: pass

# void LLVMDisposeTargetMachineOptions(LLVMTargetMachineOptionsRef Options)
try: (LLVMDisposeTargetMachineOptions:=dll.LLVMDisposeTargetMachineOptions).restype, LLVMDisposeTargetMachineOptions.argtypes = None, [LLVMTargetMachineOptionsRef]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCPU(LLVMTargetMachineOptionsRef Options, const char *CPU)
try: (LLVMTargetMachineOptionsSetCPU:=dll.LLVMTargetMachineOptionsSetCPU).restype, LLVMTargetMachineOptionsSetCPU.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetFeatures(LLVMTargetMachineOptionsRef Options, const char *Features)
try: (LLVMTargetMachineOptionsSetFeatures:=dll.LLVMTargetMachineOptionsSetFeatures).restype, LLVMTargetMachineOptionsSetFeatures.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetABI(LLVMTargetMachineOptionsRef Options, const char *ABI)
try: (LLVMTargetMachineOptionsSetABI:=dll.LLVMTargetMachineOptionsSetABI).restype, LLVMTargetMachineOptionsSetABI.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCodeGenOptLevel(LLVMTargetMachineOptionsRef Options, LLVMCodeGenOptLevel Level)
try: (LLVMTargetMachineOptionsSetCodeGenOptLevel:=dll.LLVMTargetMachineOptionsSetCodeGenOptLevel).restype, LLVMTargetMachineOptionsSetCodeGenOptLevel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeGenOptLevel]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetRelocMode(LLVMTargetMachineOptionsRef Options, LLVMRelocMode Reloc)
try: (LLVMTargetMachineOptionsSetRelocMode:=dll.LLVMTargetMachineOptionsSetRelocMode).restype, LLVMTargetMachineOptionsSetRelocMode.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMRelocMode]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCodeModel(LLVMTargetMachineOptionsRef Options, LLVMCodeModel CodeModel)
try: (LLVMTargetMachineOptionsSetCodeModel:=dll.LLVMTargetMachineOptionsSetCodeModel).restype, LLVMTargetMachineOptionsSetCodeModel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeModel]
except AttributeError: pass

# LLVMTargetMachineRef LLVMCreateTargetMachineWithOptions(LLVMTargetRef T, const char *Triple, LLVMTargetMachineOptionsRef Options)
try: (LLVMCreateTargetMachineWithOptions:=dll.LLVMCreateTargetMachineWithOptions).restype, LLVMCreateTargetMachineWithOptions.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineOptionsRef]
except AttributeError: pass

# LLVMTargetMachineRef LLVMCreateTargetMachine(LLVMTargetRef T, const char *Triple, const char *CPU, const char *Features, LLVMCodeGenOptLevel Level, LLVMRelocMode Reloc, LLVMCodeModel CodeModel)
try: (LLVMCreateTargetMachine:=dll.LLVMCreateTargetMachine).restype, LLVMCreateTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError: pass

# void LLVMDisposeTargetMachine(LLVMTargetMachineRef T)
try: (LLVMDisposeTargetMachine:=dll.LLVMDisposeTargetMachine).restype, LLVMDisposeTargetMachine.argtypes = None, [LLVMTargetMachineRef]
except AttributeError: pass

# LLVMTargetRef LLVMGetTargetMachineTarget(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineTarget:=dll.LLVMGetTargetMachineTarget).restype, LLVMGetTargetMachineTarget.argtypes = LLVMTargetRef, [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineTriple(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineTriple:=dll.LLVMGetTargetMachineTriple).restype, LLVMGetTargetMachineTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineCPU(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineCPU:=dll.LLVMGetTargetMachineCPU).restype, LLVMGetTargetMachineCPU.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineFeatureString(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineFeatureString:=dll.LLVMGetTargetMachineFeatureString).restype, LLVMGetTargetMachineFeatureString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetDataLayout(LLVMTargetMachineRef T)
try: (LLVMCreateTargetDataLayout:=dll.LLVMCreateTargetDataLayout).restype, LLVMCreateTargetDataLayout.argtypes = LLVMTargetDataRef, [LLVMTargetMachineRef]
except AttributeError: pass

# void LLVMSetTargetMachineAsmVerbosity(LLVMTargetMachineRef T, LLVMBool VerboseAsm)
try: (LLVMSetTargetMachineAsmVerbosity:=dll.LLVMSetTargetMachineAsmVerbosity).restype, LLVMSetTargetMachineAsmVerbosity.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineFastISel(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineFastISel:=dll.LLVMSetTargetMachineFastISel).restype, LLVMSetTargetMachineFastISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineGlobalISel(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineGlobalISel:=dll.LLVMSetTargetMachineGlobalISel).restype, LLVMSetTargetMachineGlobalISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineGlobalISelAbort(LLVMTargetMachineRef T, LLVMGlobalISelAbortMode Mode)
try: (LLVMSetTargetMachineGlobalISelAbort:=dll.LLVMSetTargetMachineGlobalISelAbort).restype, LLVMSetTargetMachineGlobalISelAbort.argtypes = None, [LLVMTargetMachineRef, LLVMGlobalISelAbortMode]
except AttributeError: pass

# void LLVMSetTargetMachineMachineOutliner(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineMachineOutliner:=dll.LLVMSetTargetMachineMachineOutliner).restype, LLVMSetTargetMachineMachineOutliner.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# LLVMBool LLVMTargetMachineEmitToFile(LLVMTargetMachineRef T, LLVMModuleRef M, const char *Filename, LLVMCodeGenFileType codegen, char **ErrorMessage)
try: (LLVMTargetMachineEmitToFile:=dll.LLVMTargetMachineEmitToFile).restype, LLVMTargetMachineEmitToFile.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMTargetMachineEmitToMemoryBuffer(LLVMTargetMachineRef T, LLVMModuleRef M, LLVMCodeGenFileType codegen, char **ErrorMessage, LLVMMemoryBufferRef *OutMemBuf)
try: (LLVMTargetMachineEmitToMemoryBuffer:=dll.LLVMTargetMachineEmitToMemoryBuffer).restype, LLVMTargetMachineEmitToMemoryBuffer.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

# char *LLVMGetDefaultTargetTriple(void)
try: (LLVMGetDefaultTargetTriple:=dll.LLVMGetDefaultTargetTriple).restype, LLVMGetDefaultTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# char *LLVMNormalizeTargetTriple(const char *triple)
try: (LLVMNormalizeTargetTriple:=dll.LLVMNormalizeTargetTriple).restype, LLVMNormalizeTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *LLVMGetHostCPUName(void)
try: (LLVMGetHostCPUName:=dll.LLVMGetHostCPUName).restype, LLVMGetHostCPUName.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# char *LLVMGetHostCPUFeatures(void)
try: (LLVMGetHostCPUFeatures:=dll.LLVMGetHostCPUFeatures).restype, LLVMGetHostCPUFeatures.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# void LLVMAddAnalysisPasses(LLVMTargetMachineRef T, LLVMPassManagerRef PM)
try: (LLVMAddAnalysisPasses:=dll.LLVMAddAnalysisPasses).restype, LLVMAddAnalysisPasses.argtypes = None, [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError: pass

# void LLVMOrcExecutionSessionSetErrorReporter(LLVMOrcExecutionSessionRef ES, LLVMOrcErrorReporterFunction ReportError, void *Ctx)
try: (LLVMOrcExecutionSessionSetErrorReporter:=dll.LLVMOrcExecutionSessionSetErrorReporter).restype, LLVMOrcExecutionSessionSetErrorReporter.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcErrorReporterFunction, ctypes.c_void_p]
except AttributeError: pass

# LLVMOrcSymbolStringPoolRef LLVMOrcExecutionSessionGetSymbolStringPool(LLVMOrcExecutionSessionRef ES)
try: (LLVMOrcExecutionSessionGetSymbolStringPool:=dll.LLVMOrcExecutionSessionGetSymbolStringPool).restype, LLVMOrcExecutionSessionGetSymbolStringPool.argtypes = LLVMOrcSymbolStringPoolRef, [LLVMOrcExecutionSessionRef]
except AttributeError: pass

# void LLVMOrcSymbolStringPoolClearDeadEntries(LLVMOrcSymbolStringPoolRef SSP)
try: (LLVMOrcSymbolStringPoolClearDeadEntries:=dll.LLVMOrcSymbolStringPoolClearDeadEntries).restype, LLVMOrcSymbolStringPoolClearDeadEntries.argtypes = None, [LLVMOrcSymbolStringPoolRef]
except AttributeError: pass

# LLVMOrcSymbolStringPoolEntryRef LLVMOrcExecutionSessionIntern(LLVMOrcExecutionSessionRef ES, const char *Name)
try: (LLVMOrcExecutionSessionIntern:=dll.LLVMOrcExecutionSessionIntern).restype, LLVMOrcExecutionSessionIntern.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMOrcExecutionSessionLookup(LLVMOrcExecutionSessionRef ES, LLVMOrcLookupKind K, LLVMOrcCJITDylibSearchOrder SearchOrder, size_t SearchOrderSize, LLVMOrcCLookupSet Symbols, size_t SymbolsSize, LLVMOrcExecutionSessionLookupHandleResultFunction HandleResult, void *Ctx)
try: (LLVMOrcExecutionSessionLookup:=dll.LLVMOrcExecutionSessionLookup).restype, LLVMOrcExecutionSessionLookup.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcLookupKind, LLVMOrcCJITDylibSearchOrder, size_t, LLVMOrcCLookupSet, size_t, LLVMOrcExecutionSessionLookupHandleResultFunction, ctypes.c_void_p]
except AttributeError: pass

# void LLVMOrcRetainSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S)
try: (LLVMOrcRetainSymbolStringPoolEntry:=dll.LLVMOrcRetainSymbolStringPoolEntry).restype, LLVMOrcRetainSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

# void LLVMOrcReleaseSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S)
try: (LLVMOrcReleaseSymbolStringPoolEntry:=dll.LLVMOrcReleaseSymbolStringPoolEntry).restype, LLVMOrcReleaseSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

# const char *LLVMOrcSymbolStringPoolEntryStr(LLVMOrcSymbolStringPoolEntryRef S)
try: (LLVMOrcSymbolStringPoolEntryStr:=dll.LLVMOrcSymbolStringPoolEntryStr).restype, LLVMOrcSymbolStringPoolEntryStr.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

# void LLVMOrcReleaseResourceTracker(LLVMOrcResourceTrackerRef RT)
try: (LLVMOrcReleaseResourceTracker:=dll.LLVMOrcReleaseResourceTracker).restype, LLVMOrcReleaseResourceTracker.argtypes = None, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

# void LLVMOrcResourceTrackerTransferTo(LLVMOrcResourceTrackerRef SrcRT, LLVMOrcResourceTrackerRef DstRT)
try: (LLVMOrcResourceTrackerTransferTo:=dll.LLVMOrcResourceTrackerTransferTo).restype, LLVMOrcResourceTrackerTransferTo.argtypes = None, [LLVMOrcResourceTrackerRef, LLVMOrcResourceTrackerRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcResourceTrackerRemove(LLVMOrcResourceTrackerRef RT)
try: (LLVMOrcResourceTrackerRemove:=dll.LLVMOrcResourceTrackerRemove).restype, LLVMOrcResourceTrackerRemove.argtypes = LLVMErrorRef, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

# void LLVMOrcDisposeDefinitionGenerator(LLVMOrcDefinitionGeneratorRef DG)
try: (LLVMOrcDisposeDefinitionGenerator:=dll.LLVMOrcDisposeDefinitionGenerator).restype, LLVMOrcDisposeDefinitionGenerator.argtypes = None, [LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

# void LLVMOrcDisposeMaterializationUnit(LLVMOrcMaterializationUnitRef MU)
try: (LLVMOrcDisposeMaterializationUnit:=dll.LLVMOrcDisposeMaterializationUnit).restype, LLVMOrcDisposeMaterializationUnit.argtypes = None, [LLVMOrcMaterializationUnitRef]
except AttributeError: pass

# LLVMOrcMaterializationUnitRef LLVMOrcCreateCustomMaterializationUnit(const char *Name, void *Ctx, LLVMOrcCSymbolFlagsMapPairs Syms, size_t NumSyms, LLVMOrcSymbolStringPoolEntryRef InitSym, LLVMOrcMaterializationUnitMaterializeFunction Materialize, LLVMOrcMaterializationUnitDiscardFunction Discard, LLVMOrcMaterializationUnitDestroyFunction Destroy)
try: (LLVMOrcCreateCustomMaterializationUnit:=dll.LLVMOrcCreateCustomMaterializationUnit).restype, LLVMOrcCreateCustomMaterializationUnit.argtypes = LLVMOrcMaterializationUnitRef, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, LLVMOrcCSymbolFlagsMapPairs, size_t, LLVMOrcSymbolStringPoolEntryRef, LLVMOrcMaterializationUnitMaterializeFunction, LLVMOrcMaterializationUnitDiscardFunction, LLVMOrcMaterializationUnitDestroyFunction]
except AttributeError: pass

# LLVMOrcMaterializationUnitRef LLVMOrcAbsoluteSymbols(LLVMOrcCSymbolMapPairs Syms, size_t NumPairs)
try: (LLVMOrcAbsoluteSymbols:=dll.LLVMOrcAbsoluteSymbols).restype, LLVMOrcAbsoluteSymbols.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

# LLVMOrcMaterializationUnitRef LLVMOrcLazyReexports(LLVMOrcLazyCallThroughManagerRef LCTM, LLVMOrcIndirectStubsManagerRef ISM, LLVMOrcJITDylibRef SourceRef, LLVMOrcCSymbolAliasMapPairs CallableAliases, size_t NumPairs)
try: (LLVMOrcLazyReexports:=dll.LLVMOrcLazyReexports).restype, LLVMOrcLazyReexports.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcLazyCallThroughManagerRef, LLVMOrcIndirectStubsManagerRef, LLVMOrcJITDylibRef, LLVMOrcCSymbolAliasMapPairs, size_t]
except AttributeError: pass

# void LLVMOrcDisposeMaterializationResponsibility(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcDisposeMaterializationResponsibility:=dll.LLVMOrcDisposeMaterializationResponsibility).restype, LLVMOrcDisposeMaterializationResponsibility.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcJITDylibRef LLVMOrcMaterializationResponsibilityGetTargetDylib(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityGetTargetDylib:=dll.LLVMOrcMaterializationResponsibilityGetTargetDylib).restype, LLVMOrcMaterializationResponsibilityGetTargetDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcExecutionSessionRef LLVMOrcMaterializationResponsibilityGetExecutionSession(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityGetExecutionSession:=dll.LLVMOrcMaterializationResponsibilityGetExecutionSession).restype, LLVMOrcMaterializationResponsibilityGetExecutionSession.argtypes = LLVMOrcExecutionSessionRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcCSymbolFlagsMapPairs LLVMOrcMaterializationResponsibilityGetSymbols(LLVMOrcMaterializationResponsibilityRef MR, size_t *NumPairs)
try: (LLVMOrcMaterializationResponsibilityGetSymbols:=dll.LLVMOrcMaterializationResponsibilityGetSymbols).restype, LLVMOrcMaterializationResponsibilityGetSymbols.argtypes = LLVMOrcCSymbolFlagsMapPairs, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# void LLVMOrcDisposeCSymbolFlagsMap(LLVMOrcCSymbolFlagsMapPairs Pairs)
try: (LLVMOrcDisposeCSymbolFlagsMap:=dll.LLVMOrcDisposeCSymbolFlagsMap).restype, LLVMOrcDisposeCSymbolFlagsMap.argtypes = None, [LLVMOrcCSymbolFlagsMapPairs]
except AttributeError: pass

# LLVMOrcSymbolStringPoolEntryRef LLVMOrcMaterializationResponsibilityGetInitializerSymbol(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityGetInitializerSymbol:=dll.LLVMOrcMaterializationResponsibilityGetInitializerSymbol).restype, LLVMOrcMaterializationResponsibilityGetInitializerSymbol.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcSymbolStringPoolEntryRef *LLVMOrcMaterializationResponsibilityGetRequestedSymbols(LLVMOrcMaterializationResponsibilityRef MR, size_t *NumSymbols)
try: (LLVMOrcMaterializationResponsibilityGetRequestedSymbols:=dll.LLVMOrcMaterializationResponsibilityGetRequestedSymbols).restype, LLVMOrcMaterializationResponsibilityGetRequestedSymbols.argtypes = ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# void LLVMOrcDisposeSymbols(LLVMOrcSymbolStringPoolEntryRef *Symbols)
try: (LLVMOrcDisposeSymbols:=dll.LLVMOrcDisposeSymbols).restype, LLVMOrcDisposeSymbols.argtypes = None, [ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef)]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityNotifyResolved(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolMapPairs Symbols, size_t NumPairs)
try: (LLVMOrcMaterializationResponsibilityNotifyResolved:=dll.LLVMOrcMaterializationResponsibilityNotifyResolved).restype, LLVMOrcMaterializationResponsibilityNotifyResolved.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityNotifyEmitted(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolDependenceGroup *SymbolDepGroups, size_t NumSymbolDepGroups)
try: (LLVMOrcMaterializationResponsibilityNotifyEmitted:=dll.LLVMOrcMaterializationResponsibilityNotifyEmitted).restype, LLVMOrcMaterializationResponsibilityNotifyEmitted.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcCSymbolDependenceGroup), size_t]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityDefineMaterializing(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolFlagsMapPairs Pairs, size_t NumPairs)
try: (LLVMOrcMaterializationResponsibilityDefineMaterializing:=dll.LLVMOrcMaterializationResponsibilityDefineMaterializing).restype, LLVMOrcMaterializationResponsibilityDefineMaterializing.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolFlagsMapPairs, size_t]
except AttributeError: pass

# void LLVMOrcMaterializationResponsibilityFailMaterialization(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityFailMaterialization:=dll.LLVMOrcMaterializationResponsibilityFailMaterialization).restype, LLVMOrcMaterializationResponsibilityFailMaterialization.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityReplace(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcMaterializationUnitRef MU)
try: (LLVMOrcMaterializationResponsibilityReplace:=dll.LLVMOrcMaterializationResponsibilityReplace).restype, LLVMOrcMaterializationResponsibilityReplace.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityDelegate(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcSymbolStringPoolEntryRef *Symbols, size_t NumSymbols, LLVMOrcMaterializationResponsibilityRef *Result)
try: (LLVMOrcMaterializationResponsibilityDelegate:=dll.LLVMOrcMaterializationResponsibilityDelegate).restype, LLVMOrcMaterializationResponsibilityDelegate.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), size_t, ctypes.POINTER(LLVMOrcMaterializationResponsibilityRef)]
except AttributeError: pass

# LLVMOrcJITDylibRef LLVMOrcExecutionSessionCreateBareJITDylib(LLVMOrcExecutionSessionRef ES, const char *Name)
try: (LLVMOrcExecutionSessionCreateBareJITDylib:=dll.LLVMOrcExecutionSessionCreateBareJITDylib).restype, LLVMOrcExecutionSessionCreateBareJITDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorRef LLVMOrcExecutionSessionCreateJITDylib(LLVMOrcExecutionSessionRef ES, LLVMOrcJITDylibRef *Result, const char *Name)
try: (LLVMOrcExecutionSessionCreateJITDylib:=dll.LLVMOrcExecutionSessionCreateJITDylib).restype, LLVMOrcExecutionSessionCreateJITDylib.argtypes = LLVMErrorRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(LLVMOrcJITDylibRef), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMOrcJITDylibRef LLVMOrcExecutionSessionGetJITDylibByName(LLVMOrcExecutionSessionRef ES, const char *Name)
try: (LLVMOrcExecutionSessionGetJITDylibByName:=dll.LLVMOrcExecutionSessionGetJITDylibByName).restype, LLVMOrcExecutionSessionGetJITDylibByName.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMOrcResourceTrackerRef LLVMOrcJITDylibCreateResourceTracker(LLVMOrcJITDylibRef JD)
try: (LLVMOrcJITDylibCreateResourceTracker:=dll.LLVMOrcJITDylibCreateResourceTracker).restype, LLVMOrcJITDylibCreateResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

# LLVMOrcResourceTrackerRef LLVMOrcJITDylibGetDefaultResourceTracker(LLVMOrcJITDylibRef JD)
try: (LLVMOrcJITDylibGetDefaultResourceTracker:=dll.LLVMOrcJITDylibGetDefaultResourceTracker).restype, LLVMOrcJITDylibGetDefaultResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcJITDylibDefine(LLVMOrcJITDylibRef JD, LLVMOrcMaterializationUnitRef MU)
try: (LLVMOrcJITDylibDefine:=dll.LLVMOrcJITDylibDefine).restype, LLVMOrcJITDylibDefine.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcJITDylibClear(LLVMOrcJITDylibRef JD)
try: (LLVMOrcJITDylibClear:=dll.LLVMOrcJITDylibClear).restype, LLVMOrcJITDylibClear.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

# void LLVMOrcJITDylibAddGenerator(LLVMOrcJITDylibRef JD, LLVMOrcDefinitionGeneratorRef DG)
try: (LLVMOrcJITDylibAddGenerator:=dll.LLVMOrcJITDylibAddGenerator).restype, LLVMOrcJITDylibAddGenerator.argtypes = None, [LLVMOrcJITDylibRef, LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

# LLVMOrcDefinitionGeneratorRef LLVMOrcCreateCustomCAPIDefinitionGenerator(LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction F, void *Ctx, LLVMOrcDisposeCAPIDefinitionGeneratorFunction Dispose)
try: (LLVMOrcCreateCustomCAPIDefinitionGenerator:=dll.LLVMOrcCreateCustomCAPIDefinitionGenerator).restype, LLVMOrcCreateCustomCAPIDefinitionGenerator.argtypes = LLVMOrcDefinitionGeneratorRef, [LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction, ctypes.c_void_p, LLVMOrcDisposeCAPIDefinitionGeneratorFunction]
except AttributeError: pass

# void LLVMOrcLookupStateContinueLookup(LLVMOrcLookupStateRef S, LLVMErrorRef Err)
try: (LLVMOrcLookupStateContinueLookup:=dll.LLVMOrcLookupStateContinueLookup).restype, LLVMOrcLookupStateContinueLookup.argtypes = None, [LLVMOrcLookupStateRef, LLVMErrorRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(LLVMOrcDefinitionGeneratorRef *Result, char GlobalPrefx, LLVMOrcSymbolPredicate Filter, void *FilterCtx)
try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(LLVMOrcDefinitionGeneratorRef *Result, const char *FileName, char GlobalPrefix, LLVMOrcSymbolPredicate Filter, void *FilterCtx)
try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForPath).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.POINTER(ctypes.c_char), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateStaticLibrarySearchGeneratorForPath(LLVMOrcDefinitionGeneratorRef *Result, LLVMOrcObjectLayerRef ObjLayer, const char *FileName, const char *TargetTriple)
try: (LLVMOrcCreateStaticLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateStaticLibrarySearchGeneratorForPath).restype, LLVMOrcCreateStaticLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), LLVMOrcObjectLayerRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMOrcThreadSafeContextRef LLVMOrcCreateNewThreadSafeContext(void)
try: (LLVMOrcCreateNewThreadSafeContext:=dll.LLVMOrcCreateNewThreadSafeContext).restype, LLVMOrcCreateNewThreadSafeContext.argtypes = LLVMOrcThreadSafeContextRef, []
except AttributeError: pass

# LLVMContextRef LLVMOrcThreadSafeContextGetContext(LLVMOrcThreadSafeContextRef TSCtx)
try: (LLVMOrcThreadSafeContextGetContext:=dll.LLVMOrcThreadSafeContextGetContext).restype, LLVMOrcThreadSafeContextGetContext.argtypes = LLVMContextRef, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

# void LLVMOrcDisposeThreadSafeContext(LLVMOrcThreadSafeContextRef TSCtx)
try: (LLVMOrcDisposeThreadSafeContext:=dll.LLVMOrcDisposeThreadSafeContext).restype, LLVMOrcDisposeThreadSafeContext.argtypes = None, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

# LLVMOrcThreadSafeModuleRef LLVMOrcCreateNewThreadSafeModule(LLVMModuleRef M, LLVMOrcThreadSafeContextRef TSCtx)
try: (LLVMOrcCreateNewThreadSafeModule:=dll.LLVMOrcCreateNewThreadSafeModule).restype, LLVMOrcCreateNewThreadSafeModule.argtypes = LLVMOrcThreadSafeModuleRef, [LLVMModuleRef, LLVMOrcThreadSafeContextRef]
except AttributeError: pass

# void LLVMOrcDisposeThreadSafeModule(LLVMOrcThreadSafeModuleRef TSM)
try: (LLVMOrcDisposeThreadSafeModule:=dll.LLVMOrcDisposeThreadSafeModule).restype, LLVMOrcDisposeThreadSafeModule.argtypes = None, [LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcThreadSafeModuleWithModuleDo(LLVMOrcThreadSafeModuleRef TSM, LLVMOrcGenericIRModuleOperationFunction F, void *Ctx)
try: (LLVMOrcThreadSafeModuleWithModuleDo:=dll.LLVMOrcThreadSafeModuleWithModuleDo).restype, LLVMOrcThreadSafeModuleWithModuleDo.argtypes = LLVMErrorRef, [LLVMOrcThreadSafeModuleRef, LLVMOrcGenericIRModuleOperationFunction, ctypes.c_void_p]
except AttributeError: pass

# LLVMErrorRef LLVMOrcJITTargetMachineBuilderDetectHost(LLVMOrcJITTargetMachineBuilderRef *Result)
try: (LLVMOrcJITTargetMachineBuilderDetectHost:=dll.LLVMOrcJITTargetMachineBuilderDetectHost).restype, LLVMOrcJITTargetMachineBuilderDetectHost.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcJITTargetMachineBuilderRef)]
except AttributeError: pass

# LLVMOrcJITTargetMachineBuilderRef LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(LLVMTargetMachineRef TM)
try: (LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine:=dll.LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine).restype, LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine.argtypes = LLVMOrcJITTargetMachineBuilderRef, [LLVMTargetMachineRef]
except AttributeError: pass

# void LLVMOrcDisposeJITTargetMachineBuilder(LLVMOrcJITTargetMachineBuilderRef JTMB)
try: (LLVMOrcDisposeJITTargetMachineBuilder:=dll.LLVMOrcDisposeJITTargetMachineBuilder).restype, LLVMOrcDisposeJITTargetMachineBuilder.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

# char *LLVMOrcJITTargetMachineBuilderGetTargetTriple(LLVMOrcJITTargetMachineBuilderRef JTMB)
try: (LLVMOrcJITTargetMachineBuilderGetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderGetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderGetTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

# void LLVMOrcJITTargetMachineBuilderSetTargetTriple(LLVMOrcJITTargetMachineBuilderRef JTMB, const char *TargetTriple)
try: (LLVMOrcJITTargetMachineBuilderSetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderSetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderSetTargetTriple.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorRef LLVMOrcObjectLayerAddObjectFile(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcJITDylibRef JD, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcObjectLayerAddObjectFile:=dll.LLVMOrcObjectLayerAddObjectFile).restype, LLVMOrcObjectLayerAddObjectFile.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcJITDylibRef, LLVMMemoryBufferRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcObjectLayerAddObjectFileWithRT(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcResourceTrackerRef RT, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcObjectLayerAddObjectFileWithRT:=dll.LLVMOrcObjectLayerAddObjectFileWithRT).restype, LLVMOrcObjectLayerAddObjectFileWithRT.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef]
except AttributeError: pass

# void LLVMOrcObjectLayerEmit(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcMaterializationResponsibilityRef R, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcObjectLayerEmit:=dll.LLVMOrcObjectLayerEmit).restype, LLVMOrcObjectLayerEmit.argtypes = None, [LLVMOrcObjectLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMMemoryBufferRef]
except AttributeError: pass

# void LLVMOrcDisposeObjectLayer(LLVMOrcObjectLayerRef ObjLayer)
try: (LLVMOrcDisposeObjectLayer:=dll.LLVMOrcDisposeObjectLayer).restype, LLVMOrcDisposeObjectLayer.argtypes = None, [LLVMOrcObjectLayerRef]
except AttributeError: pass

# void LLVMOrcIRTransformLayerEmit(LLVMOrcIRTransformLayerRef IRTransformLayer, LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcThreadSafeModuleRef TSM)
try: (LLVMOrcIRTransformLayerEmit:=dll.LLVMOrcIRTransformLayerEmit).restype, LLVMOrcIRTransformLayerEmit.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

# void LLVMOrcIRTransformLayerSetTransform(LLVMOrcIRTransformLayerRef IRTransformLayer, LLVMOrcIRTransformLayerTransformFunction TransformFunction, void *Ctx)
try: (LLVMOrcIRTransformLayerSetTransform:=dll.LLVMOrcIRTransformLayerSetTransform).restype, LLVMOrcIRTransformLayerSetTransform.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcIRTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

# void LLVMOrcObjectTransformLayerSetTransform(LLVMOrcObjectTransformLayerRef ObjTransformLayer, LLVMOrcObjectTransformLayerTransformFunction TransformFunction, void *Ctx)
try: (LLVMOrcObjectTransformLayerSetTransform:=dll.LLVMOrcObjectTransformLayerSetTransform).restype, LLVMOrcObjectTransformLayerSetTransform.argtypes = None, [LLVMOrcObjectTransformLayerRef, LLVMOrcObjectTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

# LLVMOrcIndirectStubsManagerRef LLVMOrcCreateLocalIndirectStubsManager(const char *TargetTriple)
try: (LLVMOrcCreateLocalIndirectStubsManager:=dll.LLVMOrcCreateLocalIndirectStubsManager).restype, LLVMOrcCreateLocalIndirectStubsManager.argtypes = LLVMOrcIndirectStubsManagerRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMOrcDisposeIndirectStubsManager(LLVMOrcIndirectStubsManagerRef ISM)
try: (LLVMOrcDisposeIndirectStubsManager:=dll.LLVMOrcDisposeIndirectStubsManager).restype, LLVMOrcDisposeIndirectStubsManager.argtypes = None, [LLVMOrcIndirectStubsManagerRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateLocalLazyCallThroughManager(const char *TargetTriple, LLVMOrcExecutionSessionRef ES, LLVMOrcJITTargetAddress ErrorHandlerAddr, LLVMOrcLazyCallThroughManagerRef *LCTM)
try: (LLVMOrcCreateLocalLazyCallThroughManager:=dll.LLVMOrcCreateLocalLazyCallThroughManager).restype, LLVMOrcCreateLocalLazyCallThroughManager.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char), LLVMOrcExecutionSessionRef, LLVMOrcJITTargetAddress, ctypes.POINTER(LLVMOrcLazyCallThroughManagerRef)]
except AttributeError: pass

# void LLVMOrcDisposeLazyCallThroughManager(LLVMOrcLazyCallThroughManagerRef LCTM)
try: (LLVMOrcDisposeLazyCallThroughManager:=dll.LLVMOrcDisposeLazyCallThroughManager).restype, LLVMOrcDisposeLazyCallThroughManager.argtypes = None, [LLVMOrcLazyCallThroughManagerRef]
except AttributeError: pass

# LLVMOrcDumpObjectsRef LLVMOrcCreateDumpObjects(const char *DumpDir, const char *IdentifierOverride)
try: (LLVMOrcCreateDumpObjects:=dll.LLVMOrcCreateDumpObjects).restype, LLVMOrcCreateDumpObjects.argtypes = LLVMOrcDumpObjectsRef, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMOrcDisposeDumpObjects(LLVMOrcDumpObjectsRef DumpObjects)
try: (LLVMOrcDisposeDumpObjects:=dll.LLVMOrcDisposeDumpObjects).restype, LLVMOrcDisposeDumpObjects.argtypes = None, [LLVMOrcDumpObjectsRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcDumpObjects_CallOperator(LLVMOrcDumpObjectsRef DumpObjects, LLVMMemoryBufferRef *ObjBuffer)
try: (LLVMOrcDumpObjects_CallOperator:=dll.LLVMOrcDumpObjects_CallOperator).restype, LLVMOrcDumpObjects_CallOperator.argtypes = LLVMErrorRef, [LLVMOrcDumpObjectsRef, ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

# LLVMOrcLLJITBuilderRef LLVMOrcCreateLLJITBuilder(void)
try: (LLVMOrcCreateLLJITBuilder:=dll.LLVMOrcCreateLLJITBuilder).restype, LLVMOrcCreateLLJITBuilder.argtypes = LLVMOrcLLJITBuilderRef, []
except AttributeError: pass

# void LLVMOrcDisposeLLJITBuilder(LLVMOrcLLJITBuilderRef Builder)
try: (LLVMOrcDisposeLLJITBuilder:=dll.LLVMOrcDisposeLLJITBuilder).restype, LLVMOrcDisposeLLJITBuilder.argtypes = None, [LLVMOrcLLJITBuilderRef]
except AttributeError: pass

# void LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(LLVMOrcLLJITBuilderRef Builder, LLVMOrcJITTargetMachineBuilderRef JTMB)
try: (LLVMOrcLLJITBuilderSetJITTargetMachineBuilder:=dll.LLVMOrcLLJITBuilderSetJITTargetMachineBuilder).restype, LLVMOrcLLJITBuilderSetJITTargetMachineBuilder.argtypes = None, [LLVMOrcLLJITBuilderRef, LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

# void LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(LLVMOrcLLJITBuilderRef Builder, LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction F, void *Ctx)
try: (LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator:=dll.LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator).restype, LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator.argtypes = None, [LLVMOrcLLJITBuilderRef, LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction, ctypes.c_void_p]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateLLJIT(LLVMOrcLLJITRef *Result, LLVMOrcLLJITBuilderRef Builder)
try: (LLVMOrcCreateLLJIT:=dll.LLVMOrcCreateLLJIT).restype, LLVMOrcCreateLLJIT.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcLLJITRef), LLVMOrcLLJITBuilderRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcDisposeLLJIT(LLVMOrcLLJITRef J)
try: (LLVMOrcDisposeLLJIT:=dll.LLVMOrcDisposeLLJIT).restype, LLVMOrcDisposeLLJIT.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef]
except AttributeError: pass

# LLVMOrcExecutionSessionRef LLVMOrcLLJITGetExecutionSession(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetExecutionSession:=dll.LLVMOrcLLJITGetExecutionSession).restype, LLVMOrcLLJITGetExecutionSession.argtypes = LLVMOrcExecutionSessionRef, [LLVMOrcLLJITRef]
except AttributeError: pass

# LLVMOrcJITDylibRef LLVMOrcLLJITGetMainJITDylib(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetMainJITDylib:=dll.LLVMOrcLLJITGetMainJITDylib).restype, LLVMOrcLLJITGetMainJITDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcLLJITRef]
except AttributeError: pass

# const char *LLVMOrcLLJITGetTripleString(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetTripleString:=dll.LLVMOrcLLJITGetTripleString).restype, LLVMOrcLLJITGetTripleString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcLLJITRef]
except AttributeError: pass

# char LLVMOrcLLJITGetGlobalPrefix(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetGlobalPrefix:=dll.LLVMOrcLLJITGetGlobalPrefix).restype, LLVMOrcLLJITGetGlobalPrefix.argtypes = ctypes.c_char, [LLVMOrcLLJITRef]
except AttributeError: pass

# LLVMOrcSymbolStringPoolEntryRef LLVMOrcLLJITMangleAndIntern(LLVMOrcLLJITRef J, const char *UnmangledName)
try: (LLVMOrcLLJITMangleAndIntern:=dll.LLVMOrcLLJITMangleAndIntern).restype, LLVMOrcLLJITMangleAndIntern.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcLLJITRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorRef LLVMOrcLLJITAddObjectFile(LLVMOrcLLJITRef J, LLVMOrcJITDylibRef JD, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcLLJITAddObjectFile:=dll.LLVMOrcLLJITAddObjectFile).restype, LLVMOrcLLJITAddObjectFile.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcJITDylibRef, LLVMMemoryBufferRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcLLJITAddObjectFileWithRT(LLVMOrcLLJITRef J, LLVMOrcResourceTrackerRef RT, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcLLJITAddObjectFileWithRT:=dll.LLVMOrcLLJITAddObjectFileWithRT).restype, LLVMOrcLLJITAddObjectFileWithRT.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcLLJITAddLLVMIRModule(LLVMOrcLLJITRef J, LLVMOrcJITDylibRef JD, LLVMOrcThreadSafeModuleRef TSM)
try: (LLVMOrcLLJITAddLLVMIRModule:=dll.LLVMOrcLLJITAddLLVMIRModule).restype, LLVMOrcLLJITAddLLVMIRModule.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcJITDylibRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcLLJITAddLLVMIRModuleWithRT(LLVMOrcLLJITRef J, LLVMOrcResourceTrackerRef JD, LLVMOrcThreadSafeModuleRef TSM)
try: (LLVMOrcLLJITAddLLVMIRModuleWithRT:=dll.LLVMOrcLLJITAddLLVMIRModuleWithRT).restype, LLVMOrcLLJITAddLLVMIRModuleWithRT.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcResourceTrackerRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcLLJITLookup(LLVMOrcLLJITRef J, LLVMOrcExecutorAddress *Result, const char *Name)
try: (LLVMOrcLLJITLookup:=dll.LLVMOrcLLJITLookup).restype, LLVMOrcLLJITLookup.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, ctypes.POINTER(LLVMOrcExecutorAddress), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMOrcObjectLayerRef LLVMOrcLLJITGetObjLinkingLayer(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetObjLinkingLayer:=dll.LLVMOrcLLJITGetObjLinkingLayer).restype, LLVMOrcLLJITGetObjLinkingLayer.argtypes = LLVMOrcObjectLayerRef, [LLVMOrcLLJITRef]
except AttributeError: pass

# LLVMOrcObjectTransformLayerRef LLVMOrcLLJITGetObjTransformLayer(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetObjTransformLayer:=dll.LLVMOrcLLJITGetObjTransformLayer).restype, LLVMOrcLLJITGetObjTransformLayer.argtypes = LLVMOrcObjectTransformLayerRef, [LLVMOrcLLJITRef]
except AttributeError: pass

# LLVMOrcIRTransformLayerRef LLVMOrcLLJITGetIRTransformLayer(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetIRTransformLayer:=dll.LLVMOrcLLJITGetIRTransformLayer).restype, LLVMOrcLLJITGetIRTransformLayer.argtypes = LLVMOrcIRTransformLayerRef, [LLVMOrcLLJITRef]
except AttributeError: pass

# const char *LLVMOrcLLJITGetDataLayoutStr(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITGetDataLayoutStr:=dll.LLVMOrcLLJITGetDataLayoutStr).restype, LLVMOrcLLJITGetDataLayoutStr.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcLLJITRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcLLJITEnableDebugSupport(LLVMOrcLLJITRef J)
try: (LLVMOrcLLJITEnableDebugSupport:=dll.LLVMOrcLLJITEnableDebugSupport).restype, LLVMOrcLLJITEnableDebugSupport.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

LLVMLinkerMode = CEnum(ctypes.c_uint32)
LLVMLinkerDestroySource = LLVMLinkerMode.define('LLVMLinkerDestroySource', 0)
LLVMLinkerPreserveSource_Removed = LLVMLinkerMode.define('LLVMLinkerPreserveSource_Removed', 1)

# LLVMBool LLVMLinkModules2(LLVMModuleRef Dest, LLVMModuleRef Src)
try: (LLVMLinkModules2:=dll.LLVMLinkModules2).restype, LLVMLinkModules2.argtypes = LLVMBool, [LLVMModuleRef, LLVMModuleRef]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

class struct_LLVMOpaqueSectionIterator(Struct): pass
LLVMSectionIteratorRef = ctypes.POINTER(struct_LLVMOpaqueSectionIterator)
class struct_LLVMOpaqueSymbolIterator(Struct): pass
LLVMSymbolIteratorRef = ctypes.POINTER(struct_LLVMOpaqueSymbolIterator)
class struct_LLVMOpaqueRelocationIterator(Struct): pass
LLVMRelocationIteratorRef = ctypes.POINTER(struct_LLVMOpaqueRelocationIterator)
LLVMBinaryType = CEnum(ctypes.c_uint32)
LLVMBinaryTypeArchive = LLVMBinaryType.define('LLVMBinaryTypeArchive', 0)
LLVMBinaryTypeMachOUniversalBinary = LLVMBinaryType.define('LLVMBinaryTypeMachOUniversalBinary', 1)
LLVMBinaryTypeCOFFImportFile = LLVMBinaryType.define('LLVMBinaryTypeCOFFImportFile', 2)
LLVMBinaryTypeIR = LLVMBinaryType.define('LLVMBinaryTypeIR', 3)
LLVMBinaryTypeWinRes = LLVMBinaryType.define('LLVMBinaryTypeWinRes', 4)
LLVMBinaryTypeCOFF = LLVMBinaryType.define('LLVMBinaryTypeCOFF', 5)
LLVMBinaryTypeELF32L = LLVMBinaryType.define('LLVMBinaryTypeELF32L', 6)
LLVMBinaryTypeELF32B = LLVMBinaryType.define('LLVMBinaryTypeELF32B', 7)
LLVMBinaryTypeELF64L = LLVMBinaryType.define('LLVMBinaryTypeELF64L', 8)
LLVMBinaryTypeELF64B = LLVMBinaryType.define('LLVMBinaryTypeELF64B', 9)
LLVMBinaryTypeMachO32L = LLVMBinaryType.define('LLVMBinaryTypeMachO32L', 10)
LLVMBinaryTypeMachO32B = LLVMBinaryType.define('LLVMBinaryTypeMachO32B', 11)
LLVMBinaryTypeMachO64L = LLVMBinaryType.define('LLVMBinaryTypeMachO64L', 12)
LLVMBinaryTypeMachO64B = LLVMBinaryType.define('LLVMBinaryTypeMachO64B', 13)
LLVMBinaryTypeWasm = LLVMBinaryType.define('LLVMBinaryTypeWasm', 14)
LLVMBinaryTypeOffload = LLVMBinaryType.define('LLVMBinaryTypeOffload', 15)

class struct_LLVMOpaqueBinary(Struct): pass
LLVMBinaryRef = ctypes.POINTER(struct_LLVMOpaqueBinary)
# LLVMBinaryRef LLVMCreateBinary(LLVMMemoryBufferRef MemBuf, LLVMContextRef Context, char **ErrorMessage)
try: (LLVMCreateBinary:=dll.LLVMCreateBinary).restype, LLVMCreateBinary.argtypes = LLVMBinaryRef, [LLVMMemoryBufferRef, LLVMContextRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# void LLVMDisposeBinary(LLVMBinaryRef BR)
try: (LLVMDisposeBinary:=dll.LLVMDisposeBinary).restype, LLVMDisposeBinary.argtypes = None, [LLVMBinaryRef]
except AttributeError: pass

# LLVMMemoryBufferRef LLVMBinaryCopyMemoryBuffer(LLVMBinaryRef BR)
try: (LLVMBinaryCopyMemoryBuffer:=dll.LLVMBinaryCopyMemoryBuffer).restype, LLVMBinaryCopyMemoryBuffer.argtypes = LLVMMemoryBufferRef, [LLVMBinaryRef]
except AttributeError: pass

# LLVMBinaryType LLVMBinaryGetType(LLVMBinaryRef BR)
try: (LLVMBinaryGetType:=dll.LLVMBinaryGetType).restype, LLVMBinaryGetType.argtypes = LLVMBinaryType, [LLVMBinaryRef]
except AttributeError: pass

# LLVMBinaryRef LLVMMachOUniversalBinaryCopyObjectForArch(LLVMBinaryRef BR, const char *Arch, size_t ArchLen, char **ErrorMessage)
try: (LLVMMachOUniversalBinaryCopyObjectForArch:=dll.LLVMMachOUniversalBinaryCopyObjectForArch).restype, LLVMMachOUniversalBinaryCopyObjectForArch.argtypes = LLVMBinaryRef, [LLVMBinaryRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMSectionIteratorRef LLVMObjectFileCopySectionIterator(LLVMBinaryRef BR)
try: (LLVMObjectFileCopySectionIterator:=dll.LLVMObjectFileCopySectionIterator).restype, LLVMObjectFileCopySectionIterator.argtypes = LLVMSectionIteratorRef, [LLVMBinaryRef]
except AttributeError: pass

# LLVMBool LLVMObjectFileIsSectionIteratorAtEnd(LLVMBinaryRef BR, LLVMSectionIteratorRef SI)
try: (LLVMObjectFileIsSectionIteratorAtEnd:=dll.LLVMObjectFileIsSectionIteratorAtEnd).restype, LLVMObjectFileIsSectionIteratorAtEnd.argtypes = LLVMBool, [LLVMBinaryRef, LLVMSectionIteratorRef]
except AttributeError: pass

# LLVMSymbolIteratorRef LLVMObjectFileCopySymbolIterator(LLVMBinaryRef BR)
try: (LLVMObjectFileCopySymbolIterator:=dll.LLVMObjectFileCopySymbolIterator).restype, LLVMObjectFileCopySymbolIterator.argtypes = LLVMSymbolIteratorRef, [LLVMBinaryRef]
except AttributeError: pass

# LLVMBool LLVMObjectFileIsSymbolIteratorAtEnd(LLVMBinaryRef BR, LLVMSymbolIteratorRef SI)
try: (LLVMObjectFileIsSymbolIteratorAtEnd:=dll.LLVMObjectFileIsSymbolIteratorAtEnd).restype, LLVMObjectFileIsSymbolIteratorAtEnd.argtypes = LLVMBool, [LLVMBinaryRef, LLVMSymbolIteratorRef]
except AttributeError: pass

# void LLVMDisposeSectionIterator(LLVMSectionIteratorRef SI)
try: (LLVMDisposeSectionIterator:=dll.LLVMDisposeSectionIterator).restype, LLVMDisposeSectionIterator.argtypes = None, [LLVMSectionIteratorRef]
except AttributeError: pass

# void LLVMMoveToNextSection(LLVMSectionIteratorRef SI)
try: (LLVMMoveToNextSection:=dll.LLVMMoveToNextSection).restype, LLVMMoveToNextSection.argtypes = None, [LLVMSectionIteratorRef]
except AttributeError: pass

# void LLVMMoveToContainingSection(LLVMSectionIteratorRef Sect, LLVMSymbolIteratorRef Sym)
try: (LLVMMoveToContainingSection:=dll.LLVMMoveToContainingSection).restype, LLVMMoveToContainingSection.argtypes = None, [LLVMSectionIteratorRef, LLVMSymbolIteratorRef]
except AttributeError: pass

# void LLVMDisposeSymbolIterator(LLVMSymbolIteratorRef SI)
try: (LLVMDisposeSymbolIterator:=dll.LLVMDisposeSymbolIterator).restype, LLVMDisposeSymbolIterator.argtypes = None, [LLVMSymbolIteratorRef]
except AttributeError: pass

# void LLVMMoveToNextSymbol(LLVMSymbolIteratorRef SI)
try: (LLVMMoveToNextSymbol:=dll.LLVMMoveToNextSymbol).restype, LLVMMoveToNextSymbol.argtypes = None, [LLVMSymbolIteratorRef]
except AttributeError: pass

# const char *LLVMGetSectionName(LLVMSectionIteratorRef SI)
try: (LLVMGetSectionName:=dll.LLVMGetSectionName).restype, LLVMGetSectionName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMSectionIteratorRef]
except AttributeError: pass

# uint64_t LLVMGetSectionSize(LLVMSectionIteratorRef SI)
try: (LLVMGetSectionSize:=dll.LLVMGetSectionSize).restype, LLVMGetSectionSize.argtypes = uint64_t, [LLVMSectionIteratorRef]
except AttributeError: pass

# const char *LLVMGetSectionContents(LLVMSectionIteratorRef SI)
try: (LLVMGetSectionContents:=dll.LLVMGetSectionContents).restype, LLVMGetSectionContents.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMSectionIteratorRef]
except AttributeError: pass

# uint64_t LLVMGetSectionAddress(LLVMSectionIteratorRef SI)
try: (LLVMGetSectionAddress:=dll.LLVMGetSectionAddress).restype, LLVMGetSectionAddress.argtypes = uint64_t, [LLVMSectionIteratorRef]
except AttributeError: pass

# LLVMBool LLVMGetSectionContainsSymbol(LLVMSectionIteratorRef SI, LLVMSymbolIteratorRef Sym)
try: (LLVMGetSectionContainsSymbol:=dll.LLVMGetSectionContainsSymbol).restype, LLVMGetSectionContainsSymbol.argtypes = LLVMBool, [LLVMSectionIteratorRef, LLVMSymbolIteratorRef]
except AttributeError: pass

# LLVMRelocationIteratorRef LLVMGetRelocations(LLVMSectionIteratorRef Section)
try: (LLVMGetRelocations:=dll.LLVMGetRelocations).restype, LLVMGetRelocations.argtypes = LLVMRelocationIteratorRef, [LLVMSectionIteratorRef]
except AttributeError: pass

# void LLVMDisposeRelocationIterator(LLVMRelocationIteratorRef RI)
try: (LLVMDisposeRelocationIterator:=dll.LLVMDisposeRelocationIterator).restype, LLVMDisposeRelocationIterator.argtypes = None, [LLVMRelocationIteratorRef]
except AttributeError: pass

# LLVMBool LLVMIsRelocationIteratorAtEnd(LLVMSectionIteratorRef Section, LLVMRelocationIteratorRef RI)
try: (LLVMIsRelocationIteratorAtEnd:=dll.LLVMIsRelocationIteratorAtEnd).restype, LLVMIsRelocationIteratorAtEnd.argtypes = LLVMBool, [LLVMSectionIteratorRef, LLVMRelocationIteratorRef]
except AttributeError: pass

# void LLVMMoveToNextRelocation(LLVMRelocationIteratorRef RI)
try: (LLVMMoveToNextRelocation:=dll.LLVMMoveToNextRelocation).restype, LLVMMoveToNextRelocation.argtypes = None, [LLVMRelocationIteratorRef]
except AttributeError: pass

# const char *LLVMGetSymbolName(LLVMSymbolIteratorRef SI)
try: (LLVMGetSymbolName:=dll.LLVMGetSymbolName).restype, LLVMGetSymbolName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMSymbolIteratorRef]
except AttributeError: pass

# uint64_t LLVMGetSymbolAddress(LLVMSymbolIteratorRef SI)
try: (LLVMGetSymbolAddress:=dll.LLVMGetSymbolAddress).restype, LLVMGetSymbolAddress.argtypes = uint64_t, [LLVMSymbolIteratorRef]
except AttributeError: pass

# uint64_t LLVMGetSymbolSize(LLVMSymbolIteratorRef SI)
try: (LLVMGetSymbolSize:=dll.LLVMGetSymbolSize).restype, LLVMGetSymbolSize.argtypes = uint64_t, [LLVMSymbolIteratorRef]
except AttributeError: pass

# uint64_t LLVMGetRelocationOffset(LLVMRelocationIteratorRef RI)
try: (LLVMGetRelocationOffset:=dll.LLVMGetRelocationOffset).restype, LLVMGetRelocationOffset.argtypes = uint64_t, [LLVMRelocationIteratorRef]
except AttributeError: pass

# LLVMSymbolIteratorRef LLVMGetRelocationSymbol(LLVMRelocationIteratorRef RI)
try: (LLVMGetRelocationSymbol:=dll.LLVMGetRelocationSymbol).restype, LLVMGetRelocationSymbol.argtypes = LLVMSymbolIteratorRef, [LLVMRelocationIteratorRef]
except AttributeError: pass

# uint64_t LLVMGetRelocationType(LLVMRelocationIteratorRef RI)
try: (LLVMGetRelocationType:=dll.LLVMGetRelocationType).restype, LLVMGetRelocationType.argtypes = uint64_t, [LLVMRelocationIteratorRef]
except AttributeError: pass

# const char *LLVMGetRelocationTypeName(LLVMRelocationIteratorRef RI)
try: (LLVMGetRelocationTypeName:=dll.LLVMGetRelocationTypeName).restype, LLVMGetRelocationTypeName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMRelocationIteratorRef]
except AttributeError: pass

# const char *LLVMGetRelocationValueString(LLVMRelocationIteratorRef RI)
try: (LLVMGetRelocationValueString:=dll.LLVMGetRelocationValueString).restype, LLVMGetRelocationValueString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMRelocationIteratorRef]
except AttributeError: pass

class struct_LLVMOpaqueObjectFile(Struct): pass
LLVMObjectFileRef = ctypes.POINTER(struct_LLVMOpaqueObjectFile)
# LLVMObjectFileRef LLVMCreateObjectFile(LLVMMemoryBufferRef MemBuf)
try: (LLVMCreateObjectFile:=dll.LLVMCreateObjectFile).restype, LLVMCreateObjectFile.argtypes = LLVMObjectFileRef, [LLVMMemoryBufferRef]
except AttributeError: pass

# void LLVMDisposeObjectFile(LLVMObjectFileRef ObjectFile)
try: (LLVMDisposeObjectFile:=dll.LLVMDisposeObjectFile).restype, LLVMDisposeObjectFile.argtypes = None, [LLVMObjectFileRef]
except AttributeError: pass

# LLVMSectionIteratorRef LLVMGetSections(LLVMObjectFileRef ObjectFile)
try: (LLVMGetSections:=dll.LLVMGetSections).restype, LLVMGetSections.argtypes = LLVMSectionIteratorRef, [LLVMObjectFileRef]
except AttributeError: pass

# LLVMBool LLVMIsSectionIteratorAtEnd(LLVMObjectFileRef ObjectFile, LLVMSectionIteratorRef SI)
try: (LLVMIsSectionIteratorAtEnd:=dll.LLVMIsSectionIteratorAtEnd).restype, LLVMIsSectionIteratorAtEnd.argtypes = LLVMBool, [LLVMObjectFileRef, LLVMSectionIteratorRef]
except AttributeError: pass

# LLVMSymbolIteratorRef LLVMGetSymbols(LLVMObjectFileRef ObjectFile)
try: (LLVMGetSymbols:=dll.LLVMGetSymbols).restype, LLVMGetSymbols.argtypes = LLVMSymbolIteratorRef, [LLVMObjectFileRef]
except AttributeError: pass

# LLVMBool LLVMIsSymbolIteratorAtEnd(LLVMObjectFileRef ObjectFile, LLVMSymbolIteratorRef SI)
try: (LLVMIsSymbolIteratorAtEnd:=dll.LLVMIsSymbolIteratorAtEnd).restype, LLVMIsSymbolIteratorAtEnd.argtypes = LLVMBool, [LLVMObjectFileRef, LLVMSymbolIteratorRef]
except AttributeError: pass

# LLVMErrorTypeId LLVMGetErrorTypeId(LLVMErrorRef Err)
try: (LLVMGetErrorTypeId:=dll.LLVMGetErrorTypeId).restype, LLVMGetErrorTypeId.argtypes = LLVMErrorTypeId, [LLVMErrorRef]
except AttributeError: pass

# void LLVMConsumeError(LLVMErrorRef Err)
try: (LLVMConsumeError:=dll.LLVMConsumeError).restype, LLVMConsumeError.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

# void LLVMCantFail(LLVMErrorRef Err)
try: (LLVMCantFail:=dll.LLVMCantFail).restype, LLVMCantFail.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

# char *LLVMGetErrorMessage(LLVMErrorRef Err)
try: (LLVMGetErrorMessage:=dll.LLVMGetErrorMessage).restype, LLVMGetErrorMessage.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMErrorRef]
except AttributeError: pass

# void LLVMDisposeErrorMessage(char *ErrMsg)
try: (LLVMDisposeErrorMessage:=dll.LLVMDisposeErrorMessage).restype, LLVMDisposeErrorMessage.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorTypeId LLVMGetStringErrorTypeId(void)
try: (LLVMGetStringErrorTypeId:=dll.LLVMGetStringErrorTypeId).restype, LLVMGetStringErrorTypeId.argtypes = LLVMErrorTypeId, []
except AttributeError: pass

# LLVMErrorRef LLVMCreateStringError(const char *ErrMsg)
try: (LLVMCreateStringError:=dll.LLVMCreateStringError).restype, LLVMCreateStringError.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

# void LLVMInitializeAArch64TargetInfo(void)
try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetInfo(void)
try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetInfo(void)
try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetInfo(void)
try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetInfo(void)
try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetInfo(void)
try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetInfo(void)
try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetInfo(void)
try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetInfo(void)
try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetInfo(void)
try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetInfo(void)
try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetInfo(void)
try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetInfo(void)
try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetInfo(void)
try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetInfo(void)
try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetInfo(void)
try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetInfo(void)
try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetInfo(void)
try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetInfo(void)
try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetInfo(void)
try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetInfo(void)
try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetInfo(void)
try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Target(void)
try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTarget(void)
try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTarget(void)
try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTarget(void)
try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTarget(void)
try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTarget(void)
try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTarget(void)
try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTarget(void)
try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTarget(void)
try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Target(void)
try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTarget(void)
try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTarget(void)
try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTarget(void)
try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTarget(void)
try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTarget(void)
try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTarget(void)
try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETarget(void)
try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTarget(void)
try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Target(void)
try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTarget(void)
try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTarget(void)
try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTarget(void)
try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64TargetMC(void)
try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetMC(void)
try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetMC(void)
try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetMC(void)
try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetMC(void)
try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetMC(void)
try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetMC(void)
try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetMC(void)
try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetMC(void)
try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetMC(void)
try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetMC(void)
try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetMC(void)
try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetMC(void)
try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetMC(void)
try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetMC(void)
try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetMC(void)
try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetMC(void)
try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetMC(void)
try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetMC(void)
try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetMC(void)
try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetMC(void)
try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetMC(void)
try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmPrinter(void)
try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmPrinter(void)
try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmPrinter(void)
try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmPrinter(void)
try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmPrinter(void)
try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmPrinter(void)
try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmPrinter(void)
try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmPrinter(void)
try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmPrinter(void)
try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmPrinter(void)
try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXAsmPrinter(void)
try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmPrinter(void)
try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmPrinter(void)
try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmPrinter(void)
try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVAsmPrinter(void)
try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmPrinter(void)
try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmPrinter(void)
try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmPrinter(void)
try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmPrinter(void)
try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreAsmPrinter(void)
try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmPrinter(void)
try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmPrinter(void)
try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmParser(void)
try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmParser(void)
try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmParser(void)
try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmParser(void)
try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmParser(void)
try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmParser(void)
try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmParser(void)
try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmParser(void)
try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmParser(void)
try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmParser(void)
try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmParser(void)
try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmParser(void)
try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmParser(void)
try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmParser(void)
try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmParser(void)
try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmParser(void)
try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmParser(void)
try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmParser(void)
try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmParser(void)
try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Disassembler(void)
try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUDisassembler(void)
try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMDisassembler(void)
try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRDisassembler(void)
try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFDisassembler(void)
try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonDisassembler(void)
try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiDisassembler(void)
try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchDisassembler(void)
try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsDisassembler(void)
try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Disassembler(void)
try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCDisassembler(void)
try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVDisassembler(void)
try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcDisassembler(void)
try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZDisassembler(void)
try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEDisassembler(void)
try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyDisassembler(void)
try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Disassembler(void)
try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreDisassembler(void)
try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kDisassembler(void)
try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaDisassembler(void)
try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

# LLVMTargetDataRef LLVMGetModuleDataLayout(LLVMModuleRef M)
try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

# void LLVMSetModuleDataLayout(LLVMModuleRef M, LLVMTargetDataRef DL)
try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetData(const char *StringRep)
try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMDisposeTargetData(LLVMTargetDataRef TD)
try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

# void LLVMAddTargetLibraryInfo(LLVMTargetLibraryInfoRef TLI, LLVMPassManagerRef PM)
try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

# char *LLVMCopyStringRepOfTargetData(LLVMTargetDataRef TD)
try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

# enum LLVMByteOrdering LLVMByteOrder(LLVMTargetDataRef TD)
try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSize(LLVMTargetDataRef TD)
try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSizeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrType(LLVMTargetDataRef TD)
try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeInContext(LLVMContextRef C, LLVMTargetDataRef TD)
try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForASInContext(LLVMContextRef C, LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# unsigned long long LLVMSizeOfTypeInBits(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMStoreSizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMABISizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMABIAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMCallFrameAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfGlobal(LLVMTargetDataRef TD, LLVMValueRef GlobalVar)
try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

# unsigned int LLVMElementAtOffset(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned long long Offset)
try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

# unsigned long long LLVMOffsetOfElement(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned int Element)
try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTargetRef LLVMGetFirstTarget(void)
try: (LLVMGetFirstTarget:=dll.LLVMGetFirstTarget).restype, LLVMGetFirstTarget.argtypes = LLVMTargetRef, []
except AttributeError: pass

# LLVMTargetRef LLVMGetNextTarget(LLVMTargetRef T)
try: (LLVMGetNextTarget:=dll.LLVMGetNextTarget).restype, LLVMGetNextTarget.argtypes = LLVMTargetRef, [LLVMTargetRef]
except AttributeError: pass

# LLVMTargetRef LLVMGetTargetFromName(const char *Name)
try: (LLVMGetTargetFromName:=dll.LLVMGetTargetFromName).restype, LLVMGetTargetFromName.argtypes = LLVMTargetRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBool LLVMGetTargetFromTriple(const char *Triple, LLVMTargetRef *T, char **ErrorMessage)
try: (LLVMGetTargetFromTriple:=dll.LLVMGetTargetFromTriple).restype, LLVMGetTargetFromTriple.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTargetRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# const char *LLVMGetTargetName(LLVMTargetRef T)
try: (LLVMGetTargetName:=dll.LLVMGetTargetName).restype, LLVMGetTargetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

# const char *LLVMGetTargetDescription(LLVMTargetRef T)
try: (LLVMGetTargetDescription:=dll.LLVMGetTargetDescription).restype, LLVMGetTargetDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasJIT(LLVMTargetRef T)
try: (LLVMTargetHasJIT:=dll.LLVMTargetHasJIT).restype, LLVMTargetHasJIT.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasTargetMachine(LLVMTargetRef T)
try: (LLVMTargetHasTargetMachine:=dll.LLVMTargetHasTargetMachine).restype, LLVMTargetHasTargetMachine.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasAsmBackend(LLVMTargetRef T)
try: (LLVMTargetHasAsmBackend:=dll.LLVMTargetHasAsmBackend).restype, LLVMTargetHasAsmBackend.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMTargetMachineOptionsRef LLVMCreateTargetMachineOptions(void)
try: (LLVMCreateTargetMachineOptions:=dll.LLVMCreateTargetMachineOptions).restype, LLVMCreateTargetMachineOptions.argtypes = LLVMTargetMachineOptionsRef, []
except AttributeError: pass

# void LLVMDisposeTargetMachineOptions(LLVMTargetMachineOptionsRef Options)
try: (LLVMDisposeTargetMachineOptions:=dll.LLVMDisposeTargetMachineOptions).restype, LLVMDisposeTargetMachineOptions.argtypes = None, [LLVMTargetMachineOptionsRef]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCPU(LLVMTargetMachineOptionsRef Options, const char *CPU)
try: (LLVMTargetMachineOptionsSetCPU:=dll.LLVMTargetMachineOptionsSetCPU).restype, LLVMTargetMachineOptionsSetCPU.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetFeatures(LLVMTargetMachineOptionsRef Options, const char *Features)
try: (LLVMTargetMachineOptionsSetFeatures:=dll.LLVMTargetMachineOptionsSetFeatures).restype, LLVMTargetMachineOptionsSetFeatures.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetABI(LLVMTargetMachineOptionsRef Options, const char *ABI)
try: (LLVMTargetMachineOptionsSetABI:=dll.LLVMTargetMachineOptionsSetABI).restype, LLVMTargetMachineOptionsSetABI.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCodeGenOptLevel(LLVMTargetMachineOptionsRef Options, LLVMCodeGenOptLevel Level)
try: (LLVMTargetMachineOptionsSetCodeGenOptLevel:=dll.LLVMTargetMachineOptionsSetCodeGenOptLevel).restype, LLVMTargetMachineOptionsSetCodeGenOptLevel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeGenOptLevel]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetRelocMode(LLVMTargetMachineOptionsRef Options, LLVMRelocMode Reloc)
try: (LLVMTargetMachineOptionsSetRelocMode:=dll.LLVMTargetMachineOptionsSetRelocMode).restype, LLVMTargetMachineOptionsSetRelocMode.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMRelocMode]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCodeModel(LLVMTargetMachineOptionsRef Options, LLVMCodeModel CodeModel)
try: (LLVMTargetMachineOptionsSetCodeModel:=dll.LLVMTargetMachineOptionsSetCodeModel).restype, LLVMTargetMachineOptionsSetCodeModel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeModel]
except AttributeError: pass

# LLVMTargetMachineRef LLVMCreateTargetMachineWithOptions(LLVMTargetRef T, const char *Triple, LLVMTargetMachineOptionsRef Options)
try: (LLVMCreateTargetMachineWithOptions:=dll.LLVMCreateTargetMachineWithOptions).restype, LLVMCreateTargetMachineWithOptions.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineOptionsRef]
except AttributeError: pass

# LLVMTargetMachineRef LLVMCreateTargetMachine(LLVMTargetRef T, const char *Triple, const char *CPU, const char *Features, LLVMCodeGenOptLevel Level, LLVMRelocMode Reloc, LLVMCodeModel CodeModel)
try: (LLVMCreateTargetMachine:=dll.LLVMCreateTargetMachine).restype, LLVMCreateTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError: pass

# void LLVMDisposeTargetMachine(LLVMTargetMachineRef T)
try: (LLVMDisposeTargetMachine:=dll.LLVMDisposeTargetMachine).restype, LLVMDisposeTargetMachine.argtypes = None, [LLVMTargetMachineRef]
except AttributeError: pass

# LLVMTargetRef LLVMGetTargetMachineTarget(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineTarget:=dll.LLVMGetTargetMachineTarget).restype, LLVMGetTargetMachineTarget.argtypes = LLVMTargetRef, [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineTriple(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineTriple:=dll.LLVMGetTargetMachineTriple).restype, LLVMGetTargetMachineTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineCPU(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineCPU:=dll.LLVMGetTargetMachineCPU).restype, LLVMGetTargetMachineCPU.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineFeatureString(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineFeatureString:=dll.LLVMGetTargetMachineFeatureString).restype, LLVMGetTargetMachineFeatureString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetDataLayout(LLVMTargetMachineRef T)
try: (LLVMCreateTargetDataLayout:=dll.LLVMCreateTargetDataLayout).restype, LLVMCreateTargetDataLayout.argtypes = LLVMTargetDataRef, [LLVMTargetMachineRef]
except AttributeError: pass

# void LLVMSetTargetMachineAsmVerbosity(LLVMTargetMachineRef T, LLVMBool VerboseAsm)
try: (LLVMSetTargetMachineAsmVerbosity:=dll.LLVMSetTargetMachineAsmVerbosity).restype, LLVMSetTargetMachineAsmVerbosity.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineFastISel(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineFastISel:=dll.LLVMSetTargetMachineFastISel).restype, LLVMSetTargetMachineFastISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineGlobalISel(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineGlobalISel:=dll.LLVMSetTargetMachineGlobalISel).restype, LLVMSetTargetMachineGlobalISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineGlobalISelAbort(LLVMTargetMachineRef T, LLVMGlobalISelAbortMode Mode)
try: (LLVMSetTargetMachineGlobalISelAbort:=dll.LLVMSetTargetMachineGlobalISelAbort).restype, LLVMSetTargetMachineGlobalISelAbort.argtypes = None, [LLVMTargetMachineRef, LLVMGlobalISelAbortMode]
except AttributeError: pass

# void LLVMSetTargetMachineMachineOutliner(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineMachineOutliner:=dll.LLVMSetTargetMachineMachineOutliner).restype, LLVMSetTargetMachineMachineOutliner.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# LLVMBool LLVMTargetMachineEmitToFile(LLVMTargetMachineRef T, LLVMModuleRef M, const char *Filename, LLVMCodeGenFileType codegen, char **ErrorMessage)
try: (LLVMTargetMachineEmitToFile:=dll.LLVMTargetMachineEmitToFile).restype, LLVMTargetMachineEmitToFile.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMTargetMachineEmitToMemoryBuffer(LLVMTargetMachineRef T, LLVMModuleRef M, LLVMCodeGenFileType codegen, char **ErrorMessage, LLVMMemoryBufferRef *OutMemBuf)
try: (LLVMTargetMachineEmitToMemoryBuffer:=dll.LLVMTargetMachineEmitToMemoryBuffer).restype, LLVMTargetMachineEmitToMemoryBuffer.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

# char *LLVMGetDefaultTargetTriple(void)
try: (LLVMGetDefaultTargetTriple:=dll.LLVMGetDefaultTargetTriple).restype, LLVMGetDefaultTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# char *LLVMNormalizeTargetTriple(const char *triple)
try: (LLVMNormalizeTargetTriple:=dll.LLVMNormalizeTargetTriple).restype, LLVMNormalizeTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *LLVMGetHostCPUName(void)
try: (LLVMGetHostCPUName:=dll.LLVMGetHostCPUName).restype, LLVMGetHostCPUName.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# char *LLVMGetHostCPUFeatures(void)
try: (LLVMGetHostCPUFeatures:=dll.LLVMGetHostCPUFeatures).restype, LLVMGetHostCPUFeatures.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# void LLVMAddAnalysisPasses(LLVMTargetMachineRef T, LLVMPassManagerRef PM)
try: (LLVMAddAnalysisPasses:=dll.LLVMAddAnalysisPasses).restype, LLVMAddAnalysisPasses.argtypes = None, [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError: pass

LLVMJITSymbolGenericFlags = CEnum(ctypes.c_uint32)
LLVMJITSymbolGenericFlagsNone = LLVMJITSymbolGenericFlags.define('LLVMJITSymbolGenericFlagsNone', 0)
LLVMJITSymbolGenericFlagsExported = LLVMJITSymbolGenericFlags.define('LLVMJITSymbolGenericFlagsExported', 1)
LLVMJITSymbolGenericFlagsWeak = LLVMJITSymbolGenericFlags.define('LLVMJITSymbolGenericFlagsWeak', 2)
LLVMJITSymbolGenericFlagsCallable = LLVMJITSymbolGenericFlags.define('LLVMJITSymbolGenericFlagsCallable', 4)
LLVMJITSymbolGenericFlagsMaterializationSideEffectsOnly = LLVMJITSymbolGenericFlags.define('LLVMJITSymbolGenericFlagsMaterializationSideEffectsOnly', 8)

LLVMJITSymbolTargetFlags = ctypes.c_ubyte
class struct_LLVMOrcOpaqueObjectLinkingLayer(Struct): pass
LLVMOrcObjectLinkingLayerRef = ctypes.POINTER(struct_LLVMOrcOpaqueObjectLinkingLayer)
# void LLVMOrcExecutionSessionSetErrorReporter(LLVMOrcExecutionSessionRef ES, LLVMOrcErrorReporterFunction ReportError, void *Ctx)
try: (LLVMOrcExecutionSessionSetErrorReporter:=dll.LLVMOrcExecutionSessionSetErrorReporter).restype, LLVMOrcExecutionSessionSetErrorReporter.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcErrorReporterFunction, ctypes.c_void_p]
except AttributeError: pass

# LLVMOrcSymbolStringPoolRef LLVMOrcExecutionSessionGetSymbolStringPool(LLVMOrcExecutionSessionRef ES)
try: (LLVMOrcExecutionSessionGetSymbolStringPool:=dll.LLVMOrcExecutionSessionGetSymbolStringPool).restype, LLVMOrcExecutionSessionGetSymbolStringPool.argtypes = LLVMOrcSymbolStringPoolRef, [LLVMOrcExecutionSessionRef]
except AttributeError: pass

# void LLVMOrcSymbolStringPoolClearDeadEntries(LLVMOrcSymbolStringPoolRef SSP)
try: (LLVMOrcSymbolStringPoolClearDeadEntries:=dll.LLVMOrcSymbolStringPoolClearDeadEntries).restype, LLVMOrcSymbolStringPoolClearDeadEntries.argtypes = None, [LLVMOrcSymbolStringPoolRef]
except AttributeError: pass

# LLVMOrcSymbolStringPoolEntryRef LLVMOrcExecutionSessionIntern(LLVMOrcExecutionSessionRef ES, const char *Name)
try: (LLVMOrcExecutionSessionIntern:=dll.LLVMOrcExecutionSessionIntern).restype, LLVMOrcExecutionSessionIntern.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMOrcExecutionSessionLookup(LLVMOrcExecutionSessionRef ES, LLVMOrcLookupKind K, LLVMOrcCJITDylibSearchOrder SearchOrder, size_t SearchOrderSize, LLVMOrcCLookupSet Symbols, size_t SymbolsSize, LLVMOrcExecutionSessionLookupHandleResultFunction HandleResult, void *Ctx)
try: (LLVMOrcExecutionSessionLookup:=dll.LLVMOrcExecutionSessionLookup).restype, LLVMOrcExecutionSessionLookup.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcLookupKind, LLVMOrcCJITDylibSearchOrder, size_t, LLVMOrcCLookupSet, size_t, LLVMOrcExecutionSessionLookupHandleResultFunction, ctypes.c_void_p]
except AttributeError: pass

# void LLVMOrcRetainSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S)
try: (LLVMOrcRetainSymbolStringPoolEntry:=dll.LLVMOrcRetainSymbolStringPoolEntry).restype, LLVMOrcRetainSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

# void LLVMOrcReleaseSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S)
try: (LLVMOrcReleaseSymbolStringPoolEntry:=dll.LLVMOrcReleaseSymbolStringPoolEntry).restype, LLVMOrcReleaseSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

# const char *LLVMOrcSymbolStringPoolEntryStr(LLVMOrcSymbolStringPoolEntryRef S)
try: (LLVMOrcSymbolStringPoolEntryStr:=dll.LLVMOrcSymbolStringPoolEntryStr).restype, LLVMOrcSymbolStringPoolEntryStr.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

# void LLVMOrcReleaseResourceTracker(LLVMOrcResourceTrackerRef RT)
try: (LLVMOrcReleaseResourceTracker:=dll.LLVMOrcReleaseResourceTracker).restype, LLVMOrcReleaseResourceTracker.argtypes = None, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

# void LLVMOrcResourceTrackerTransferTo(LLVMOrcResourceTrackerRef SrcRT, LLVMOrcResourceTrackerRef DstRT)
try: (LLVMOrcResourceTrackerTransferTo:=dll.LLVMOrcResourceTrackerTransferTo).restype, LLVMOrcResourceTrackerTransferTo.argtypes = None, [LLVMOrcResourceTrackerRef, LLVMOrcResourceTrackerRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcResourceTrackerRemove(LLVMOrcResourceTrackerRef RT)
try: (LLVMOrcResourceTrackerRemove:=dll.LLVMOrcResourceTrackerRemove).restype, LLVMOrcResourceTrackerRemove.argtypes = LLVMErrorRef, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

# void LLVMOrcDisposeDefinitionGenerator(LLVMOrcDefinitionGeneratorRef DG)
try: (LLVMOrcDisposeDefinitionGenerator:=dll.LLVMOrcDisposeDefinitionGenerator).restype, LLVMOrcDisposeDefinitionGenerator.argtypes = None, [LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

# void LLVMOrcDisposeMaterializationUnit(LLVMOrcMaterializationUnitRef MU)
try: (LLVMOrcDisposeMaterializationUnit:=dll.LLVMOrcDisposeMaterializationUnit).restype, LLVMOrcDisposeMaterializationUnit.argtypes = None, [LLVMOrcMaterializationUnitRef]
except AttributeError: pass

# LLVMOrcMaterializationUnitRef LLVMOrcCreateCustomMaterializationUnit(const char *Name, void *Ctx, LLVMOrcCSymbolFlagsMapPairs Syms, size_t NumSyms, LLVMOrcSymbolStringPoolEntryRef InitSym, LLVMOrcMaterializationUnitMaterializeFunction Materialize, LLVMOrcMaterializationUnitDiscardFunction Discard, LLVMOrcMaterializationUnitDestroyFunction Destroy)
try: (LLVMOrcCreateCustomMaterializationUnit:=dll.LLVMOrcCreateCustomMaterializationUnit).restype, LLVMOrcCreateCustomMaterializationUnit.argtypes = LLVMOrcMaterializationUnitRef, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, LLVMOrcCSymbolFlagsMapPairs, size_t, LLVMOrcSymbolStringPoolEntryRef, LLVMOrcMaterializationUnitMaterializeFunction, LLVMOrcMaterializationUnitDiscardFunction, LLVMOrcMaterializationUnitDestroyFunction]
except AttributeError: pass

# LLVMOrcMaterializationUnitRef LLVMOrcAbsoluteSymbols(LLVMOrcCSymbolMapPairs Syms, size_t NumPairs)
try: (LLVMOrcAbsoluteSymbols:=dll.LLVMOrcAbsoluteSymbols).restype, LLVMOrcAbsoluteSymbols.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

# LLVMOrcMaterializationUnitRef LLVMOrcLazyReexports(LLVMOrcLazyCallThroughManagerRef LCTM, LLVMOrcIndirectStubsManagerRef ISM, LLVMOrcJITDylibRef SourceRef, LLVMOrcCSymbolAliasMapPairs CallableAliases, size_t NumPairs)
try: (LLVMOrcLazyReexports:=dll.LLVMOrcLazyReexports).restype, LLVMOrcLazyReexports.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcLazyCallThroughManagerRef, LLVMOrcIndirectStubsManagerRef, LLVMOrcJITDylibRef, LLVMOrcCSymbolAliasMapPairs, size_t]
except AttributeError: pass

# void LLVMOrcDisposeMaterializationResponsibility(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcDisposeMaterializationResponsibility:=dll.LLVMOrcDisposeMaterializationResponsibility).restype, LLVMOrcDisposeMaterializationResponsibility.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcJITDylibRef LLVMOrcMaterializationResponsibilityGetTargetDylib(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityGetTargetDylib:=dll.LLVMOrcMaterializationResponsibilityGetTargetDylib).restype, LLVMOrcMaterializationResponsibilityGetTargetDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcExecutionSessionRef LLVMOrcMaterializationResponsibilityGetExecutionSession(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityGetExecutionSession:=dll.LLVMOrcMaterializationResponsibilityGetExecutionSession).restype, LLVMOrcMaterializationResponsibilityGetExecutionSession.argtypes = LLVMOrcExecutionSessionRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcCSymbolFlagsMapPairs LLVMOrcMaterializationResponsibilityGetSymbols(LLVMOrcMaterializationResponsibilityRef MR, size_t *NumPairs)
try: (LLVMOrcMaterializationResponsibilityGetSymbols:=dll.LLVMOrcMaterializationResponsibilityGetSymbols).restype, LLVMOrcMaterializationResponsibilityGetSymbols.argtypes = LLVMOrcCSymbolFlagsMapPairs, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# void LLVMOrcDisposeCSymbolFlagsMap(LLVMOrcCSymbolFlagsMapPairs Pairs)
try: (LLVMOrcDisposeCSymbolFlagsMap:=dll.LLVMOrcDisposeCSymbolFlagsMap).restype, LLVMOrcDisposeCSymbolFlagsMap.argtypes = None, [LLVMOrcCSymbolFlagsMapPairs]
except AttributeError: pass

# LLVMOrcSymbolStringPoolEntryRef LLVMOrcMaterializationResponsibilityGetInitializerSymbol(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityGetInitializerSymbol:=dll.LLVMOrcMaterializationResponsibilityGetInitializerSymbol).restype, LLVMOrcMaterializationResponsibilityGetInitializerSymbol.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcSymbolStringPoolEntryRef *LLVMOrcMaterializationResponsibilityGetRequestedSymbols(LLVMOrcMaterializationResponsibilityRef MR, size_t *NumSymbols)
try: (LLVMOrcMaterializationResponsibilityGetRequestedSymbols:=dll.LLVMOrcMaterializationResponsibilityGetRequestedSymbols).restype, LLVMOrcMaterializationResponsibilityGetRequestedSymbols.argtypes = ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# void LLVMOrcDisposeSymbols(LLVMOrcSymbolStringPoolEntryRef *Symbols)
try: (LLVMOrcDisposeSymbols:=dll.LLVMOrcDisposeSymbols).restype, LLVMOrcDisposeSymbols.argtypes = None, [ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef)]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityNotifyResolved(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolMapPairs Symbols, size_t NumPairs)
try: (LLVMOrcMaterializationResponsibilityNotifyResolved:=dll.LLVMOrcMaterializationResponsibilityNotifyResolved).restype, LLVMOrcMaterializationResponsibilityNotifyResolved.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityNotifyEmitted(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolDependenceGroup *SymbolDepGroups, size_t NumSymbolDepGroups)
try: (LLVMOrcMaterializationResponsibilityNotifyEmitted:=dll.LLVMOrcMaterializationResponsibilityNotifyEmitted).restype, LLVMOrcMaterializationResponsibilityNotifyEmitted.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcCSymbolDependenceGroup), size_t]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityDefineMaterializing(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolFlagsMapPairs Pairs, size_t NumPairs)
try: (LLVMOrcMaterializationResponsibilityDefineMaterializing:=dll.LLVMOrcMaterializationResponsibilityDefineMaterializing).restype, LLVMOrcMaterializationResponsibilityDefineMaterializing.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolFlagsMapPairs, size_t]
except AttributeError: pass

# void LLVMOrcMaterializationResponsibilityFailMaterialization(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityFailMaterialization:=dll.LLVMOrcMaterializationResponsibilityFailMaterialization).restype, LLVMOrcMaterializationResponsibilityFailMaterialization.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityReplace(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcMaterializationUnitRef MU)
try: (LLVMOrcMaterializationResponsibilityReplace:=dll.LLVMOrcMaterializationResponsibilityReplace).restype, LLVMOrcMaterializationResponsibilityReplace.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityDelegate(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcSymbolStringPoolEntryRef *Symbols, size_t NumSymbols, LLVMOrcMaterializationResponsibilityRef *Result)
try: (LLVMOrcMaterializationResponsibilityDelegate:=dll.LLVMOrcMaterializationResponsibilityDelegate).restype, LLVMOrcMaterializationResponsibilityDelegate.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), size_t, ctypes.POINTER(LLVMOrcMaterializationResponsibilityRef)]
except AttributeError: pass

# LLVMOrcJITDylibRef LLVMOrcExecutionSessionCreateBareJITDylib(LLVMOrcExecutionSessionRef ES, const char *Name)
try: (LLVMOrcExecutionSessionCreateBareJITDylib:=dll.LLVMOrcExecutionSessionCreateBareJITDylib).restype, LLVMOrcExecutionSessionCreateBareJITDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorRef LLVMOrcExecutionSessionCreateJITDylib(LLVMOrcExecutionSessionRef ES, LLVMOrcJITDylibRef *Result, const char *Name)
try: (LLVMOrcExecutionSessionCreateJITDylib:=dll.LLVMOrcExecutionSessionCreateJITDylib).restype, LLVMOrcExecutionSessionCreateJITDylib.argtypes = LLVMErrorRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(LLVMOrcJITDylibRef), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMOrcJITDylibRef LLVMOrcExecutionSessionGetJITDylibByName(LLVMOrcExecutionSessionRef ES, const char *Name)
try: (LLVMOrcExecutionSessionGetJITDylibByName:=dll.LLVMOrcExecutionSessionGetJITDylibByName).restype, LLVMOrcExecutionSessionGetJITDylibByName.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMOrcResourceTrackerRef LLVMOrcJITDylibCreateResourceTracker(LLVMOrcJITDylibRef JD)
try: (LLVMOrcJITDylibCreateResourceTracker:=dll.LLVMOrcJITDylibCreateResourceTracker).restype, LLVMOrcJITDylibCreateResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

# LLVMOrcResourceTrackerRef LLVMOrcJITDylibGetDefaultResourceTracker(LLVMOrcJITDylibRef JD)
try: (LLVMOrcJITDylibGetDefaultResourceTracker:=dll.LLVMOrcJITDylibGetDefaultResourceTracker).restype, LLVMOrcJITDylibGetDefaultResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcJITDylibDefine(LLVMOrcJITDylibRef JD, LLVMOrcMaterializationUnitRef MU)
try: (LLVMOrcJITDylibDefine:=dll.LLVMOrcJITDylibDefine).restype, LLVMOrcJITDylibDefine.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcJITDylibClear(LLVMOrcJITDylibRef JD)
try: (LLVMOrcJITDylibClear:=dll.LLVMOrcJITDylibClear).restype, LLVMOrcJITDylibClear.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

# void LLVMOrcJITDylibAddGenerator(LLVMOrcJITDylibRef JD, LLVMOrcDefinitionGeneratorRef DG)
try: (LLVMOrcJITDylibAddGenerator:=dll.LLVMOrcJITDylibAddGenerator).restype, LLVMOrcJITDylibAddGenerator.argtypes = None, [LLVMOrcJITDylibRef, LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

# LLVMOrcDefinitionGeneratorRef LLVMOrcCreateCustomCAPIDefinitionGenerator(LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction F, void *Ctx, LLVMOrcDisposeCAPIDefinitionGeneratorFunction Dispose)
try: (LLVMOrcCreateCustomCAPIDefinitionGenerator:=dll.LLVMOrcCreateCustomCAPIDefinitionGenerator).restype, LLVMOrcCreateCustomCAPIDefinitionGenerator.argtypes = LLVMOrcDefinitionGeneratorRef, [LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction, ctypes.c_void_p, LLVMOrcDisposeCAPIDefinitionGeneratorFunction]
except AttributeError: pass

# void LLVMOrcLookupStateContinueLookup(LLVMOrcLookupStateRef S, LLVMErrorRef Err)
try: (LLVMOrcLookupStateContinueLookup:=dll.LLVMOrcLookupStateContinueLookup).restype, LLVMOrcLookupStateContinueLookup.argtypes = None, [LLVMOrcLookupStateRef, LLVMErrorRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(LLVMOrcDefinitionGeneratorRef *Result, char GlobalPrefx, LLVMOrcSymbolPredicate Filter, void *FilterCtx)
try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(LLVMOrcDefinitionGeneratorRef *Result, const char *FileName, char GlobalPrefix, LLVMOrcSymbolPredicate Filter, void *FilterCtx)
try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForPath).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.POINTER(ctypes.c_char), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateStaticLibrarySearchGeneratorForPath(LLVMOrcDefinitionGeneratorRef *Result, LLVMOrcObjectLayerRef ObjLayer, const char *FileName, const char *TargetTriple)
try: (LLVMOrcCreateStaticLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateStaticLibrarySearchGeneratorForPath).restype, LLVMOrcCreateStaticLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), LLVMOrcObjectLayerRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMOrcThreadSafeContextRef LLVMOrcCreateNewThreadSafeContext(void)
try: (LLVMOrcCreateNewThreadSafeContext:=dll.LLVMOrcCreateNewThreadSafeContext).restype, LLVMOrcCreateNewThreadSafeContext.argtypes = LLVMOrcThreadSafeContextRef, []
except AttributeError: pass

# LLVMContextRef LLVMOrcThreadSafeContextGetContext(LLVMOrcThreadSafeContextRef TSCtx)
try: (LLVMOrcThreadSafeContextGetContext:=dll.LLVMOrcThreadSafeContextGetContext).restype, LLVMOrcThreadSafeContextGetContext.argtypes = LLVMContextRef, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

# void LLVMOrcDisposeThreadSafeContext(LLVMOrcThreadSafeContextRef TSCtx)
try: (LLVMOrcDisposeThreadSafeContext:=dll.LLVMOrcDisposeThreadSafeContext).restype, LLVMOrcDisposeThreadSafeContext.argtypes = None, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

# LLVMOrcThreadSafeModuleRef LLVMOrcCreateNewThreadSafeModule(LLVMModuleRef M, LLVMOrcThreadSafeContextRef TSCtx)
try: (LLVMOrcCreateNewThreadSafeModule:=dll.LLVMOrcCreateNewThreadSafeModule).restype, LLVMOrcCreateNewThreadSafeModule.argtypes = LLVMOrcThreadSafeModuleRef, [LLVMModuleRef, LLVMOrcThreadSafeContextRef]
except AttributeError: pass

# void LLVMOrcDisposeThreadSafeModule(LLVMOrcThreadSafeModuleRef TSM)
try: (LLVMOrcDisposeThreadSafeModule:=dll.LLVMOrcDisposeThreadSafeModule).restype, LLVMOrcDisposeThreadSafeModule.argtypes = None, [LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcThreadSafeModuleWithModuleDo(LLVMOrcThreadSafeModuleRef TSM, LLVMOrcGenericIRModuleOperationFunction F, void *Ctx)
try: (LLVMOrcThreadSafeModuleWithModuleDo:=dll.LLVMOrcThreadSafeModuleWithModuleDo).restype, LLVMOrcThreadSafeModuleWithModuleDo.argtypes = LLVMErrorRef, [LLVMOrcThreadSafeModuleRef, LLVMOrcGenericIRModuleOperationFunction, ctypes.c_void_p]
except AttributeError: pass

# LLVMErrorRef LLVMOrcJITTargetMachineBuilderDetectHost(LLVMOrcJITTargetMachineBuilderRef *Result)
try: (LLVMOrcJITTargetMachineBuilderDetectHost:=dll.LLVMOrcJITTargetMachineBuilderDetectHost).restype, LLVMOrcJITTargetMachineBuilderDetectHost.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcJITTargetMachineBuilderRef)]
except AttributeError: pass

# LLVMOrcJITTargetMachineBuilderRef LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(LLVMTargetMachineRef TM)
try: (LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine:=dll.LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine).restype, LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine.argtypes = LLVMOrcJITTargetMachineBuilderRef, [LLVMTargetMachineRef]
except AttributeError: pass

# void LLVMOrcDisposeJITTargetMachineBuilder(LLVMOrcJITTargetMachineBuilderRef JTMB)
try: (LLVMOrcDisposeJITTargetMachineBuilder:=dll.LLVMOrcDisposeJITTargetMachineBuilder).restype, LLVMOrcDisposeJITTargetMachineBuilder.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

# char *LLVMOrcJITTargetMachineBuilderGetTargetTriple(LLVMOrcJITTargetMachineBuilderRef JTMB)
try: (LLVMOrcJITTargetMachineBuilderGetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderGetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderGetTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

# void LLVMOrcJITTargetMachineBuilderSetTargetTriple(LLVMOrcJITTargetMachineBuilderRef JTMB, const char *TargetTriple)
try: (LLVMOrcJITTargetMachineBuilderSetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderSetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderSetTargetTriple.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorRef LLVMOrcObjectLayerAddObjectFile(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcJITDylibRef JD, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcObjectLayerAddObjectFile:=dll.LLVMOrcObjectLayerAddObjectFile).restype, LLVMOrcObjectLayerAddObjectFile.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcJITDylibRef, LLVMMemoryBufferRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcObjectLayerAddObjectFileWithRT(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcResourceTrackerRef RT, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcObjectLayerAddObjectFileWithRT:=dll.LLVMOrcObjectLayerAddObjectFileWithRT).restype, LLVMOrcObjectLayerAddObjectFileWithRT.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef]
except AttributeError: pass

# void LLVMOrcObjectLayerEmit(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcMaterializationResponsibilityRef R, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcObjectLayerEmit:=dll.LLVMOrcObjectLayerEmit).restype, LLVMOrcObjectLayerEmit.argtypes = None, [LLVMOrcObjectLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMMemoryBufferRef]
except AttributeError: pass

# void LLVMOrcDisposeObjectLayer(LLVMOrcObjectLayerRef ObjLayer)
try: (LLVMOrcDisposeObjectLayer:=dll.LLVMOrcDisposeObjectLayer).restype, LLVMOrcDisposeObjectLayer.argtypes = None, [LLVMOrcObjectLayerRef]
except AttributeError: pass

# void LLVMOrcIRTransformLayerEmit(LLVMOrcIRTransformLayerRef IRTransformLayer, LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcThreadSafeModuleRef TSM)
try: (LLVMOrcIRTransformLayerEmit:=dll.LLVMOrcIRTransformLayerEmit).restype, LLVMOrcIRTransformLayerEmit.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

# void LLVMOrcIRTransformLayerSetTransform(LLVMOrcIRTransformLayerRef IRTransformLayer, LLVMOrcIRTransformLayerTransformFunction TransformFunction, void *Ctx)
try: (LLVMOrcIRTransformLayerSetTransform:=dll.LLVMOrcIRTransformLayerSetTransform).restype, LLVMOrcIRTransformLayerSetTransform.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcIRTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

# void LLVMOrcObjectTransformLayerSetTransform(LLVMOrcObjectTransformLayerRef ObjTransformLayer, LLVMOrcObjectTransformLayerTransformFunction TransformFunction, void *Ctx)
try: (LLVMOrcObjectTransformLayerSetTransform:=dll.LLVMOrcObjectTransformLayerSetTransform).restype, LLVMOrcObjectTransformLayerSetTransform.argtypes = None, [LLVMOrcObjectTransformLayerRef, LLVMOrcObjectTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

# LLVMOrcIndirectStubsManagerRef LLVMOrcCreateLocalIndirectStubsManager(const char *TargetTriple)
try: (LLVMOrcCreateLocalIndirectStubsManager:=dll.LLVMOrcCreateLocalIndirectStubsManager).restype, LLVMOrcCreateLocalIndirectStubsManager.argtypes = LLVMOrcIndirectStubsManagerRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMOrcDisposeIndirectStubsManager(LLVMOrcIndirectStubsManagerRef ISM)
try: (LLVMOrcDisposeIndirectStubsManager:=dll.LLVMOrcDisposeIndirectStubsManager).restype, LLVMOrcDisposeIndirectStubsManager.argtypes = None, [LLVMOrcIndirectStubsManagerRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateLocalLazyCallThroughManager(const char *TargetTriple, LLVMOrcExecutionSessionRef ES, LLVMOrcJITTargetAddress ErrorHandlerAddr, LLVMOrcLazyCallThroughManagerRef *LCTM)
try: (LLVMOrcCreateLocalLazyCallThroughManager:=dll.LLVMOrcCreateLocalLazyCallThroughManager).restype, LLVMOrcCreateLocalLazyCallThroughManager.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char), LLVMOrcExecutionSessionRef, LLVMOrcJITTargetAddress, ctypes.POINTER(LLVMOrcLazyCallThroughManagerRef)]
except AttributeError: pass

# void LLVMOrcDisposeLazyCallThroughManager(LLVMOrcLazyCallThroughManagerRef LCTM)
try: (LLVMOrcDisposeLazyCallThroughManager:=dll.LLVMOrcDisposeLazyCallThroughManager).restype, LLVMOrcDisposeLazyCallThroughManager.argtypes = None, [LLVMOrcLazyCallThroughManagerRef]
except AttributeError: pass

# LLVMOrcDumpObjectsRef LLVMOrcCreateDumpObjects(const char *DumpDir, const char *IdentifierOverride)
try: (LLVMOrcCreateDumpObjects:=dll.LLVMOrcCreateDumpObjects).restype, LLVMOrcCreateDumpObjects.argtypes = LLVMOrcDumpObjectsRef, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMOrcDisposeDumpObjects(LLVMOrcDumpObjectsRef DumpObjects)
try: (LLVMOrcDisposeDumpObjects:=dll.LLVMOrcDisposeDumpObjects).restype, LLVMOrcDisposeDumpObjects.argtypes = None, [LLVMOrcDumpObjectsRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcDumpObjects_CallOperator(LLVMOrcDumpObjectsRef DumpObjects, LLVMMemoryBufferRef *ObjBuffer)
try: (LLVMOrcDumpObjects_CallOperator:=dll.LLVMOrcDumpObjects_CallOperator).restype, LLVMOrcDumpObjects_CallOperator.argtypes = LLVMErrorRef, [LLVMOrcDumpObjectsRef, ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

# LLVMErrorTypeId LLVMGetErrorTypeId(LLVMErrorRef Err)
try: (LLVMGetErrorTypeId:=dll.LLVMGetErrorTypeId).restype, LLVMGetErrorTypeId.argtypes = LLVMErrorTypeId, [LLVMErrorRef]
except AttributeError: pass

# void LLVMConsumeError(LLVMErrorRef Err)
try: (LLVMConsumeError:=dll.LLVMConsumeError).restype, LLVMConsumeError.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

# void LLVMCantFail(LLVMErrorRef Err)
try: (LLVMCantFail:=dll.LLVMCantFail).restype, LLVMCantFail.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

# char *LLVMGetErrorMessage(LLVMErrorRef Err)
try: (LLVMGetErrorMessage:=dll.LLVMGetErrorMessage).restype, LLVMGetErrorMessage.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMErrorRef]
except AttributeError: pass

# void LLVMDisposeErrorMessage(char *ErrMsg)
try: (LLVMDisposeErrorMessage:=dll.LLVMDisposeErrorMessage).restype, LLVMDisposeErrorMessage.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorTypeId LLVMGetStringErrorTypeId(void)
try: (LLVMGetStringErrorTypeId:=dll.LLVMGetStringErrorTypeId).restype, LLVMGetStringErrorTypeId.argtypes = LLVMErrorTypeId, []
except AttributeError: pass

# LLVMErrorRef LLVMCreateStringError(const char *ErrMsg)
try: (LLVMCreateStringError:=dll.LLVMCreateStringError).restype, LLVMCreateStringError.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

# void LLVMInitializeAArch64TargetInfo(void)
try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetInfo(void)
try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetInfo(void)
try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetInfo(void)
try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetInfo(void)
try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetInfo(void)
try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetInfo(void)
try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetInfo(void)
try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetInfo(void)
try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetInfo(void)
try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetInfo(void)
try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetInfo(void)
try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetInfo(void)
try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetInfo(void)
try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetInfo(void)
try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetInfo(void)
try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetInfo(void)
try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetInfo(void)
try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetInfo(void)
try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetInfo(void)
try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetInfo(void)
try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetInfo(void)
try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Target(void)
try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTarget(void)
try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTarget(void)
try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTarget(void)
try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTarget(void)
try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTarget(void)
try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTarget(void)
try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTarget(void)
try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTarget(void)
try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Target(void)
try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTarget(void)
try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTarget(void)
try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTarget(void)
try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTarget(void)
try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTarget(void)
try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTarget(void)
try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETarget(void)
try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTarget(void)
try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Target(void)
try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTarget(void)
try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTarget(void)
try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTarget(void)
try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64TargetMC(void)
try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetMC(void)
try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetMC(void)
try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetMC(void)
try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetMC(void)
try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetMC(void)
try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetMC(void)
try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetMC(void)
try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetMC(void)
try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetMC(void)
try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetMC(void)
try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetMC(void)
try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetMC(void)
try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetMC(void)
try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetMC(void)
try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetMC(void)
try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetMC(void)
try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetMC(void)
try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetMC(void)
try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetMC(void)
try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetMC(void)
try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetMC(void)
try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmPrinter(void)
try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmPrinter(void)
try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmPrinter(void)
try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmPrinter(void)
try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmPrinter(void)
try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmPrinter(void)
try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmPrinter(void)
try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmPrinter(void)
try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmPrinter(void)
try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmPrinter(void)
try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXAsmPrinter(void)
try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmPrinter(void)
try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmPrinter(void)
try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmPrinter(void)
try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVAsmPrinter(void)
try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmPrinter(void)
try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmPrinter(void)
try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmPrinter(void)
try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmPrinter(void)
try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreAsmPrinter(void)
try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmPrinter(void)
try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmPrinter(void)
try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmParser(void)
try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmParser(void)
try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmParser(void)
try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmParser(void)
try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmParser(void)
try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmParser(void)
try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmParser(void)
try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmParser(void)
try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmParser(void)
try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmParser(void)
try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmParser(void)
try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmParser(void)
try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmParser(void)
try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmParser(void)
try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmParser(void)
try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmParser(void)
try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmParser(void)
try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmParser(void)
try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmParser(void)
try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Disassembler(void)
try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUDisassembler(void)
try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMDisassembler(void)
try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRDisassembler(void)
try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFDisassembler(void)
try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonDisassembler(void)
try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiDisassembler(void)
try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchDisassembler(void)
try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsDisassembler(void)
try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Disassembler(void)
try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCDisassembler(void)
try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVDisassembler(void)
try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcDisassembler(void)
try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZDisassembler(void)
try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEDisassembler(void)
try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyDisassembler(void)
try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Disassembler(void)
try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreDisassembler(void)
try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kDisassembler(void)
try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaDisassembler(void)
try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

# LLVMTargetDataRef LLVMGetModuleDataLayout(LLVMModuleRef M)
try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

# void LLVMSetModuleDataLayout(LLVMModuleRef M, LLVMTargetDataRef DL)
try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetData(const char *StringRep)
try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMDisposeTargetData(LLVMTargetDataRef TD)
try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

# void LLVMAddTargetLibraryInfo(LLVMTargetLibraryInfoRef TLI, LLVMPassManagerRef PM)
try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

# char *LLVMCopyStringRepOfTargetData(LLVMTargetDataRef TD)
try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

# enum LLVMByteOrdering LLVMByteOrder(LLVMTargetDataRef TD)
try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSize(LLVMTargetDataRef TD)
try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSizeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrType(LLVMTargetDataRef TD)
try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeInContext(LLVMContextRef C, LLVMTargetDataRef TD)
try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForASInContext(LLVMContextRef C, LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# unsigned long long LLVMSizeOfTypeInBits(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMStoreSizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMABISizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMABIAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMCallFrameAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfGlobal(LLVMTargetDataRef TD, LLVMValueRef GlobalVar)
try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

# unsigned int LLVMElementAtOffset(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned long long Offset)
try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

# unsigned long long LLVMOffsetOfElement(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned int Element)
try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTargetRef LLVMGetFirstTarget(void)
try: (LLVMGetFirstTarget:=dll.LLVMGetFirstTarget).restype, LLVMGetFirstTarget.argtypes = LLVMTargetRef, []
except AttributeError: pass

# LLVMTargetRef LLVMGetNextTarget(LLVMTargetRef T)
try: (LLVMGetNextTarget:=dll.LLVMGetNextTarget).restype, LLVMGetNextTarget.argtypes = LLVMTargetRef, [LLVMTargetRef]
except AttributeError: pass

# LLVMTargetRef LLVMGetTargetFromName(const char *Name)
try: (LLVMGetTargetFromName:=dll.LLVMGetTargetFromName).restype, LLVMGetTargetFromName.argtypes = LLVMTargetRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBool LLVMGetTargetFromTriple(const char *Triple, LLVMTargetRef *T, char **ErrorMessage)
try: (LLVMGetTargetFromTriple:=dll.LLVMGetTargetFromTriple).restype, LLVMGetTargetFromTriple.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTargetRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# const char *LLVMGetTargetName(LLVMTargetRef T)
try: (LLVMGetTargetName:=dll.LLVMGetTargetName).restype, LLVMGetTargetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

# const char *LLVMGetTargetDescription(LLVMTargetRef T)
try: (LLVMGetTargetDescription:=dll.LLVMGetTargetDescription).restype, LLVMGetTargetDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasJIT(LLVMTargetRef T)
try: (LLVMTargetHasJIT:=dll.LLVMTargetHasJIT).restype, LLVMTargetHasJIT.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasTargetMachine(LLVMTargetRef T)
try: (LLVMTargetHasTargetMachine:=dll.LLVMTargetHasTargetMachine).restype, LLVMTargetHasTargetMachine.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasAsmBackend(LLVMTargetRef T)
try: (LLVMTargetHasAsmBackend:=dll.LLVMTargetHasAsmBackend).restype, LLVMTargetHasAsmBackend.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMTargetMachineOptionsRef LLVMCreateTargetMachineOptions(void)
try: (LLVMCreateTargetMachineOptions:=dll.LLVMCreateTargetMachineOptions).restype, LLVMCreateTargetMachineOptions.argtypes = LLVMTargetMachineOptionsRef, []
except AttributeError: pass

# void LLVMDisposeTargetMachineOptions(LLVMTargetMachineOptionsRef Options)
try: (LLVMDisposeTargetMachineOptions:=dll.LLVMDisposeTargetMachineOptions).restype, LLVMDisposeTargetMachineOptions.argtypes = None, [LLVMTargetMachineOptionsRef]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCPU(LLVMTargetMachineOptionsRef Options, const char *CPU)
try: (LLVMTargetMachineOptionsSetCPU:=dll.LLVMTargetMachineOptionsSetCPU).restype, LLVMTargetMachineOptionsSetCPU.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetFeatures(LLVMTargetMachineOptionsRef Options, const char *Features)
try: (LLVMTargetMachineOptionsSetFeatures:=dll.LLVMTargetMachineOptionsSetFeatures).restype, LLVMTargetMachineOptionsSetFeatures.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetABI(LLVMTargetMachineOptionsRef Options, const char *ABI)
try: (LLVMTargetMachineOptionsSetABI:=dll.LLVMTargetMachineOptionsSetABI).restype, LLVMTargetMachineOptionsSetABI.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCodeGenOptLevel(LLVMTargetMachineOptionsRef Options, LLVMCodeGenOptLevel Level)
try: (LLVMTargetMachineOptionsSetCodeGenOptLevel:=dll.LLVMTargetMachineOptionsSetCodeGenOptLevel).restype, LLVMTargetMachineOptionsSetCodeGenOptLevel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeGenOptLevel]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetRelocMode(LLVMTargetMachineOptionsRef Options, LLVMRelocMode Reloc)
try: (LLVMTargetMachineOptionsSetRelocMode:=dll.LLVMTargetMachineOptionsSetRelocMode).restype, LLVMTargetMachineOptionsSetRelocMode.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMRelocMode]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCodeModel(LLVMTargetMachineOptionsRef Options, LLVMCodeModel CodeModel)
try: (LLVMTargetMachineOptionsSetCodeModel:=dll.LLVMTargetMachineOptionsSetCodeModel).restype, LLVMTargetMachineOptionsSetCodeModel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeModel]
except AttributeError: pass

# LLVMTargetMachineRef LLVMCreateTargetMachineWithOptions(LLVMTargetRef T, const char *Triple, LLVMTargetMachineOptionsRef Options)
try: (LLVMCreateTargetMachineWithOptions:=dll.LLVMCreateTargetMachineWithOptions).restype, LLVMCreateTargetMachineWithOptions.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineOptionsRef]
except AttributeError: pass

# LLVMTargetMachineRef LLVMCreateTargetMachine(LLVMTargetRef T, const char *Triple, const char *CPU, const char *Features, LLVMCodeGenOptLevel Level, LLVMRelocMode Reloc, LLVMCodeModel CodeModel)
try: (LLVMCreateTargetMachine:=dll.LLVMCreateTargetMachine).restype, LLVMCreateTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError: pass

# void LLVMDisposeTargetMachine(LLVMTargetMachineRef T)
try: (LLVMDisposeTargetMachine:=dll.LLVMDisposeTargetMachine).restype, LLVMDisposeTargetMachine.argtypes = None, [LLVMTargetMachineRef]
except AttributeError: pass

# LLVMTargetRef LLVMGetTargetMachineTarget(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineTarget:=dll.LLVMGetTargetMachineTarget).restype, LLVMGetTargetMachineTarget.argtypes = LLVMTargetRef, [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineTriple(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineTriple:=dll.LLVMGetTargetMachineTriple).restype, LLVMGetTargetMachineTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineCPU(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineCPU:=dll.LLVMGetTargetMachineCPU).restype, LLVMGetTargetMachineCPU.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineFeatureString(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineFeatureString:=dll.LLVMGetTargetMachineFeatureString).restype, LLVMGetTargetMachineFeatureString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetDataLayout(LLVMTargetMachineRef T)
try: (LLVMCreateTargetDataLayout:=dll.LLVMCreateTargetDataLayout).restype, LLVMCreateTargetDataLayout.argtypes = LLVMTargetDataRef, [LLVMTargetMachineRef]
except AttributeError: pass

# void LLVMSetTargetMachineAsmVerbosity(LLVMTargetMachineRef T, LLVMBool VerboseAsm)
try: (LLVMSetTargetMachineAsmVerbosity:=dll.LLVMSetTargetMachineAsmVerbosity).restype, LLVMSetTargetMachineAsmVerbosity.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineFastISel(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineFastISel:=dll.LLVMSetTargetMachineFastISel).restype, LLVMSetTargetMachineFastISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineGlobalISel(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineGlobalISel:=dll.LLVMSetTargetMachineGlobalISel).restype, LLVMSetTargetMachineGlobalISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineGlobalISelAbort(LLVMTargetMachineRef T, LLVMGlobalISelAbortMode Mode)
try: (LLVMSetTargetMachineGlobalISelAbort:=dll.LLVMSetTargetMachineGlobalISelAbort).restype, LLVMSetTargetMachineGlobalISelAbort.argtypes = None, [LLVMTargetMachineRef, LLVMGlobalISelAbortMode]
except AttributeError: pass

# void LLVMSetTargetMachineMachineOutliner(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineMachineOutliner:=dll.LLVMSetTargetMachineMachineOutliner).restype, LLVMSetTargetMachineMachineOutliner.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# LLVMBool LLVMTargetMachineEmitToFile(LLVMTargetMachineRef T, LLVMModuleRef M, const char *Filename, LLVMCodeGenFileType codegen, char **ErrorMessage)
try: (LLVMTargetMachineEmitToFile:=dll.LLVMTargetMachineEmitToFile).restype, LLVMTargetMachineEmitToFile.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMTargetMachineEmitToMemoryBuffer(LLVMTargetMachineRef T, LLVMModuleRef M, LLVMCodeGenFileType codegen, char **ErrorMessage, LLVMMemoryBufferRef *OutMemBuf)
try: (LLVMTargetMachineEmitToMemoryBuffer:=dll.LLVMTargetMachineEmitToMemoryBuffer).restype, LLVMTargetMachineEmitToMemoryBuffer.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

# char *LLVMGetDefaultTargetTriple(void)
try: (LLVMGetDefaultTargetTriple:=dll.LLVMGetDefaultTargetTriple).restype, LLVMGetDefaultTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# char *LLVMNormalizeTargetTriple(const char *triple)
try: (LLVMNormalizeTargetTriple:=dll.LLVMNormalizeTargetTriple).restype, LLVMNormalizeTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *LLVMGetHostCPUName(void)
try: (LLVMGetHostCPUName:=dll.LLVMGetHostCPUName).restype, LLVMGetHostCPUName.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# char *LLVMGetHostCPUFeatures(void)
try: (LLVMGetHostCPUFeatures:=dll.LLVMGetHostCPUFeatures).restype, LLVMGetHostCPUFeatures.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# void LLVMAddAnalysisPasses(LLVMTargetMachineRef T, LLVMPassManagerRef PM)
try: (LLVMAddAnalysisPasses:=dll.LLVMAddAnalysisPasses).restype, LLVMAddAnalysisPasses.argtypes = None, [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError: pass

# void LLVMLinkInMCJIT(void)
try: (LLVMLinkInMCJIT:=dll.LLVMLinkInMCJIT).restype, LLVMLinkInMCJIT.argtypes = None, []
except AttributeError: pass

# void LLVMLinkInInterpreter(void)
try: (LLVMLinkInInterpreter:=dll.LLVMLinkInInterpreter).restype, LLVMLinkInInterpreter.argtypes = None, []
except AttributeError: pass

# LLVMGenericValueRef LLVMCreateGenericValueOfInt(LLVMTypeRef Ty, unsigned long long N, LLVMBool IsSigned)
try: (LLVMCreateGenericValueOfInt:=dll.LLVMCreateGenericValueOfInt).restype, LLVMCreateGenericValueOfInt.argtypes = LLVMGenericValueRef, [LLVMTypeRef, ctypes.c_uint64, LLVMBool]
except AttributeError: pass

# LLVMGenericValueRef LLVMCreateGenericValueOfPointer(void *P)
try: (LLVMCreateGenericValueOfPointer:=dll.LLVMCreateGenericValueOfPointer).restype, LLVMCreateGenericValueOfPointer.argtypes = LLVMGenericValueRef, [ctypes.c_void_p]
except AttributeError: pass

# LLVMGenericValueRef LLVMCreateGenericValueOfFloat(LLVMTypeRef Ty, double N)
try: (LLVMCreateGenericValueOfFloat:=dll.LLVMCreateGenericValueOfFloat).restype, LLVMCreateGenericValueOfFloat.argtypes = LLVMGenericValueRef, [LLVMTypeRef, ctypes.c_double]
except AttributeError: pass

# unsigned int LLVMGenericValueIntWidth(LLVMGenericValueRef GenValRef)
try: (LLVMGenericValueIntWidth:=dll.LLVMGenericValueIntWidth).restype, LLVMGenericValueIntWidth.argtypes = ctypes.c_uint32, [LLVMGenericValueRef]
except AttributeError: pass

# unsigned long long LLVMGenericValueToInt(LLVMGenericValueRef GenVal, LLVMBool IsSigned)
try: (LLVMGenericValueToInt:=dll.LLVMGenericValueToInt).restype, LLVMGenericValueToInt.argtypes = ctypes.c_uint64, [LLVMGenericValueRef, LLVMBool]
except AttributeError: pass

# void *LLVMGenericValueToPointer(LLVMGenericValueRef GenVal)
try: (LLVMGenericValueToPointer:=dll.LLVMGenericValueToPointer).restype, LLVMGenericValueToPointer.argtypes = ctypes.c_void_p, [LLVMGenericValueRef]
except AttributeError: pass

# double LLVMGenericValueToFloat(LLVMTypeRef TyRef, LLVMGenericValueRef GenVal)
try: (LLVMGenericValueToFloat:=dll.LLVMGenericValueToFloat).restype, LLVMGenericValueToFloat.argtypes = ctypes.c_double, [LLVMTypeRef, LLVMGenericValueRef]
except AttributeError: pass

# void LLVMDisposeGenericValue(LLVMGenericValueRef GenVal)
try: (LLVMDisposeGenericValue:=dll.LLVMDisposeGenericValue).restype, LLVMDisposeGenericValue.argtypes = None, [LLVMGenericValueRef]
except AttributeError: pass

# LLVMBool LLVMCreateExecutionEngineForModule(LLVMExecutionEngineRef *OutEE, LLVMModuleRef M, char **OutError)
try: (LLVMCreateExecutionEngineForModule:=dll.LLVMCreateExecutionEngineForModule).restype, LLVMCreateExecutionEngineForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMCreateInterpreterForModule(LLVMExecutionEngineRef *OutInterp, LLVMModuleRef M, char **OutError)
try: (LLVMCreateInterpreterForModule:=dll.LLVMCreateInterpreterForModule).restype, LLVMCreateInterpreterForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMCreateJITCompilerForModule(LLVMExecutionEngineRef *OutJIT, LLVMModuleRef M, unsigned int OptLevel, char **OutError)
try: (LLVMCreateJITCompilerForModule:=dll.LLVMCreateJITCompilerForModule).restype, LLVMCreateJITCompilerForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# void LLVMInitializeMCJITCompilerOptions(struct LLVMMCJITCompilerOptions *Options, size_t SizeOfOptions)
try: (LLVMInitializeMCJITCompilerOptions:=dll.LLVMInitializeMCJITCompilerOptions).restype, LLVMInitializeMCJITCompilerOptions.argtypes = None, [ctypes.POINTER(struct_LLVMMCJITCompilerOptions), size_t]
except AttributeError: pass

# LLVMBool LLVMCreateMCJITCompilerForModule(LLVMExecutionEngineRef *OutJIT, LLVMModuleRef M, struct LLVMMCJITCompilerOptions *Options, size_t SizeOfOptions, char **OutError)
try: (LLVMCreateMCJITCompilerForModule:=dll.LLVMCreateMCJITCompilerForModule).restype, LLVMCreateMCJITCompilerForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.POINTER(struct_LLVMMCJITCompilerOptions), size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# void LLVMDisposeExecutionEngine(LLVMExecutionEngineRef EE)
try: (LLVMDisposeExecutionEngine:=dll.LLVMDisposeExecutionEngine).restype, LLVMDisposeExecutionEngine.argtypes = None, [LLVMExecutionEngineRef]
except AttributeError: pass

# void LLVMRunStaticConstructors(LLVMExecutionEngineRef EE)
try: (LLVMRunStaticConstructors:=dll.LLVMRunStaticConstructors).restype, LLVMRunStaticConstructors.argtypes = None, [LLVMExecutionEngineRef]
except AttributeError: pass

# void LLVMRunStaticDestructors(LLVMExecutionEngineRef EE)
try: (LLVMRunStaticDestructors:=dll.LLVMRunStaticDestructors).restype, LLVMRunStaticDestructors.argtypes = None, [LLVMExecutionEngineRef]
except AttributeError: pass

# int LLVMRunFunctionAsMain(LLVMExecutionEngineRef EE, LLVMValueRef F, unsigned int ArgC, const char *const *ArgV, const char *const *EnvP)
try: (LLVMRunFunctionAsMain:=dll.LLVMRunFunctionAsMain).restype, LLVMRunFunctionAsMain.argtypes = ctypes.c_int32, [LLVMExecutionEngineRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMGenericValueRef LLVMRunFunction(LLVMExecutionEngineRef EE, LLVMValueRef F, unsigned int NumArgs, LLVMGenericValueRef *Args)
try: (LLVMRunFunction:=dll.LLVMRunFunction).restype, LLVMRunFunction.argtypes = LLVMGenericValueRef, [LLVMExecutionEngineRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(LLVMGenericValueRef)]
except AttributeError: pass

# void LLVMFreeMachineCodeForFunction(LLVMExecutionEngineRef EE, LLVMValueRef F)
try: (LLVMFreeMachineCodeForFunction:=dll.LLVMFreeMachineCodeForFunction).restype, LLVMFreeMachineCodeForFunction.argtypes = None, [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError: pass

# void LLVMAddModule(LLVMExecutionEngineRef EE, LLVMModuleRef M)
try: (LLVMAddModule:=dll.LLVMAddModule).restype, LLVMAddModule.argtypes = None, [LLVMExecutionEngineRef, LLVMModuleRef]
except AttributeError: pass

# LLVMBool LLVMRemoveModule(LLVMExecutionEngineRef EE, LLVMModuleRef M, LLVMModuleRef *OutMod, char **OutError)
try: (LLVMRemoveModule:=dll.LLVMRemoveModule).restype, LLVMRemoveModule.argtypes = LLVMBool, [LLVMExecutionEngineRef, LLVMModuleRef, ctypes.POINTER(LLVMModuleRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMFindFunction(LLVMExecutionEngineRef EE, const char *Name, LLVMValueRef *OutFn)
try: (LLVMFindFunction:=dll.LLVMFindFunction).restype, LLVMFindFunction.argtypes = LLVMBool, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMValueRef)]
except AttributeError: pass

# void *LLVMRecompileAndRelinkFunction(LLVMExecutionEngineRef EE, LLVMValueRef Fn)
try: (LLVMRecompileAndRelinkFunction:=dll.LLVMRecompileAndRelinkFunction).restype, LLVMRecompileAndRelinkFunction.argtypes = ctypes.c_void_p, [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMGetExecutionEngineTargetData(LLVMExecutionEngineRef EE)
try: (LLVMGetExecutionEngineTargetData:=dll.LLVMGetExecutionEngineTargetData).restype, LLVMGetExecutionEngineTargetData.argtypes = LLVMTargetDataRef, [LLVMExecutionEngineRef]
except AttributeError: pass

# LLVMTargetMachineRef LLVMGetExecutionEngineTargetMachine(LLVMExecutionEngineRef EE)
try: (LLVMGetExecutionEngineTargetMachine:=dll.LLVMGetExecutionEngineTargetMachine).restype, LLVMGetExecutionEngineTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMExecutionEngineRef]
except AttributeError: pass

# void LLVMAddGlobalMapping(LLVMExecutionEngineRef EE, LLVMValueRef Global, void *Addr)
try: (LLVMAddGlobalMapping:=dll.LLVMAddGlobalMapping).restype, LLVMAddGlobalMapping.argtypes = None, [LLVMExecutionEngineRef, LLVMValueRef, ctypes.c_void_p]
except AttributeError: pass

# void *LLVMGetPointerToGlobal(LLVMExecutionEngineRef EE, LLVMValueRef Global)
try: (LLVMGetPointerToGlobal:=dll.LLVMGetPointerToGlobal).restype, LLVMGetPointerToGlobal.argtypes = ctypes.c_void_p, [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError: pass

# uint64_t LLVMGetGlobalValueAddress(LLVMExecutionEngineRef EE, const char *Name)
try: (LLVMGetGlobalValueAddress:=dll.LLVMGetGlobalValueAddress).restype, LLVMGetGlobalValueAddress.argtypes = uint64_t, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# uint64_t LLVMGetFunctionAddress(LLVMExecutionEngineRef EE, const char *Name)
try: (LLVMGetFunctionAddress:=dll.LLVMGetFunctionAddress).restype, LLVMGetFunctionAddress.argtypes = uint64_t, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBool LLVMExecutionEngineGetErrMsg(LLVMExecutionEngineRef EE, char **OutError)
try: (LLVMExecutionEngineGetErrMsg:=dll.LLVMExecutionEngineGetErrMsg).restype, LLVMExecutionEngineGetErrMsg.argtypes = LLVMBool, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMMCJITMemoryManagerRef LLVMCreateSimpleMCJITMemoryManager(void *Opaque, LLVMMemoryManagerAllocateCodeSectionCallback AllocateCodeSection, LLVMMemoryManagerAllocateDataSectionCallback AllocateDataSection, LLVMMemoryManagerFinalizeMemoryCallback FinalizeMemory, LLVMMemoryManagerDestroyCallback Destroy)
try: (LLVMCreateSimpleMCJITMemoryManager:=dll.LLVMCreateSimpleMCJITMemoryManager).restype, LLVMCreateSimpleMCJITMemoryManager.argtypes = LLVMMCJITMemoryManagerRef, [ctypes.c_void_p, LLVMMemoryManagerAllocateCodeSectionCallback, LLVMMemoryManagerAllocateDataSectionCallback, LLVMMemoryManagerFinalizeMemoryCallback, LLVMMemoryManagerDestroyCallback]
except AttributeError: pass

# void LLVMDisposeMCJITMemoryManager(LLVMMCJITMemoryManagerRef MM)
try: (LLVMDisposeMCJITMemoryManager:=dll.LLVMDisposeMCJITMemoryManager).restype, LLVMDisposeMCJITMemoryManager.argtypes = None, [LLVMMCJITMemoryManagerRef]
except AttributeError: pass

# LLVMJITEventListenerRef LLVMCreateGDBRegistrationListener(void)
try: (LLVMCreateGDBRegistrationListener:=dll.LLVMCreateGDBRegistrationListener).restype, LLVMCreateGDBRegistrationListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

# LLVMJITEventListenerRef LLVMCreateIntelJITEventListener(void)
try: (LLVMCreateIntelJITEventListener:=dll.LLVMCreateIntelJITEventListener).restype, LLVMCreateIntelJITEventListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

# LLVMJITEventListenerRef LLVMCreateOProfileJITEventListener(void)
try: (LLVMCreateOProfileJITEventListener:=dll.LLVMCreateOProfileJITEventListener).restype, LLVMCreateOProfileJITEventListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

# LLVMJITEventListenerRef LLVMCreatePerfJITEventListener(void)
try: (LLVMCreatePerfJITEventListener:=dll.LLVMCreatePerfJITEventListener).restype, LLVMCreatePerfJITEventListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

# void LLVMOrcExecutionSessionSetErrorReporter(LLVMOrcExecutionSessionRef ES, LLVMOrcErrorReporterFunction ReportError, void *Ctx)
try: (LLVMOrcExecutionSessionSetErrorReporter:=dll.LLVMOrcExecutionSessionSetErrorReporter).restype, LLVMOrcExecutionSessionSetErrorReporter.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcErrorReporterFunction, ctypes.c_void_p]
except AttributeError: pass

# LLVMOrcSymbolStringPoolRef LLVMOrcExecutionSessionGetSymbolStringPool(LLVMOrcExecutionSessionRef ES)
try: (LLVMOrcExecutionSessionGetSymbolStringPool:=dll.LLVMOrcExecutionSessionGetSymbolStringPool).restype, LLVMOrcExecutionSessionGetSymbolStringPool.argtypes = LLVMOrcSymbolStringPoolRef, [LLVMOrcExecutionSessionRef]
except AttributeError: pass

# void LLVMOrcSymbolStringPoolClearDeadEntries(LLVMOrcSymbolStringPoolRef SSP)
try: (LLVMOrcSymbolStringPoolClearDeadEntries:=dll.LLVMOrcSymbolStringPoolClearDeadEntries).restype, LLVMOrcSymbolStringPoolClearDeadEntries.argtypes = None, [LLVMOrcSymbolStringPoolRef]
except AttributeError: pass

# LLVMOrcSymbolStringPoolEntryRef LLVMOrcExecutionSessionIntern(LLVMOrcExecutionSessionRef ES, const char *Name)
try: (LLVMOrcExecutionSessionIntern:=dll.LLVMOrcExecutionSessionIntern).restype, LLVMOrcExecutionSessionIntern.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMOrcExecutionSessionLookup(LLVMOrcExecutionSessionRef ES, LLVMOrcLookupKind K, LLVMOrcCJITDylibSearchOrder SearchOrder, size_t SearchOrderSize, LLVMOrcCLookupSet Symbols, size_t SymbolsSize, LLVMOrcExecutionSessionLookupHandleResultFunction HandleResult, void *Ctx)
try: (LLVMOrcExecutionSessionLookup:=dll.LLVMOrcExecutionSessionLookup).restype, LLVMOrcExecutionSessionLookup.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcLookupKind, LLVMOrcCJITDylibSearchOrder, size_t, LLVMOrcCLookupSet, size_t, LLVMOrcExecutionSessionLookupHandleResultFunction, ctypes.c_void_p]
except AttributeError: pass

# void LLVMOrcRetainSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S)
try: (LLVMOrcRetainSymbolStringPoolEntry:=dll.LLVMOrcRetainSymbolStringPoolEntry).restype, LLVMOrcRetainSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

# void LLVMOrcReleaseSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S)
try: (LLVMOrcReleaseSymbolStringPoolEntry:=dll.LLVMOrcReleaseSymbolStringPoolEntry).restype, LLVMOrcReleaseSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

# const char *LLVMOrcSymbolStringPoolEntryStr(LLVMOrcSymbolStringPoolEntryRef S)
try: (LLVMOrcSymbolStringPoolEntryStr:=dll.LLVMOrcSymbolStringPoolEntryStr).restype, LLVMOrcSymbolStringPoolEntryStr.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

# void LLVMOrcReleaseResourceTracker(LLVMOrcResourceTrackerRef RT)
try: (LLVMOrcReleaseResourceTracker:=dll.LLVMOrcReleaseResourceTracker).restype, LLVMOrcReleaseResourceTracker.argtypes = None, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

# void LLVMOrcResourceTrackerTransferTo(LLVMOrcResourceTrackerRef SrcRT, LLVMOrcResourceTrackerRef DstRT)
try: (LLVMOrcResourceTrackerTransferTo:=dll.LLVMOrcResourceTrackerTransferTo).restype, LLVMOrcResourceTrackerTransferTo.argtypes = None, [LLVMOrcResourceTrackerRef, LLVMOrcResourceTrackerRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcResourceTrackerRemove(LLVMOrcResourceTrackerRef RT)
try: (LLVMOrcResourceTrackerRemove:=dll.LLVMOrcResourceTrackerRemove).restype, LLVMOrcResourceTrackerRemove.argtypes = LLVMErrorRef, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

# void LLVMOrcDisposeDefinitionGenerator(LLVMOrcDefinitionGeneratorRef DG)
try: (LLVMOrcDisposeDefinitionGenerator:=dll.LLVMOrcDisposeDefinitionGenerator).restype, LLVMOrcDisposeDefinitionGenerator.argtypes = None, [LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

# void LLVMOrcDisposeMaterializationUnit(LLVMOrcMaterializationUnitRef MU)
try: (LLVMOrcDisposeMaterializationUnit:=dll.LLVMOrcDisposeMaterializationUnit).restype, LLVMOrcDisposeMaterializationUnit.argtypes = None, [LLVMOrcMaterializationUnitRef]
except AttributeError: pass

# LLVMOrcMaterializationUnitRef LLVMOrcCreateCustomMaterializationUnit(const char *Name, void *Ctx, LLVMOrcCSymbolFlagsMapPairs Syms, size_t NumSyms, LLVMOrcSymbolStringPoolEntryRef InitSym, LLVMOrcMaterializationUnitMaterializeFunction Materialize, LLVMOrcMaterializationUnitDiscardFunction Discard, LLVMOrcMaterializationUnitDestroyFunction Destroy)
try: (LLVMOrcCreateCustomMaterializationUnit:=dll.LLVMOrcCreateCustomMaterializationUnit).restype, LLVMOrcCreateCustomMaterializationUnit.argtypes = LLVMOrcMaterializationUnitRef, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, LLVMOrcCSymbolFlagsMapPairs, size_t, LLVMOrcSymbolStringPoolEntryRef, LLVMOrcMaterializationUnitMaterializeFunction, LLVMOrcMaterializationUnitDiscardFunction, LLVMOrcMaterializationUnitDestroyFunction]
except AttributeError: pass

# LLVMOrcMaterializationUnitRef LLVMOrcAbsoluteSymbols(LLVMOrcCSymbolMapPairs Syms, size_t NumPairs)
try: (LLVMOrcAbsoluteSymbols:=dll.LLVMOrcAbsoluteSymbols).restype, LLVMOrcAbsoluteSymbols.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

# LLVMOrcMaterializationUnitRef LLVMOrcLazyReexports(LLVMOrcLazyCallThroughManagerRef LCTM, LLVMOrcIndirectStubsManagerRef ISM, LLVMOrcJITDylibRef SourceRef, LLVMOrcCSymbolAliasMapPairs CallableAliases, size_t NumPairs)
try: (LLVMOrcLazyReexports:=dll.LLVMOrcLazyReexports).restype, LLVMOrcLazyReexports.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcLazyCallThroughManagerRef, LLVMOrcIndirectStubsManagerRef, LLVMOrcJITDylibRef, LLVMOrcCSymbolAliasMapPairs, size_t]
except AttributeError: pass

# void LLVMOrcDisposeMaterializationResponsibility(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcDisposeMaterializationResponsibility:=dll.LLVMOrcDisposeMaterializationResponsibility).restype, LLVMOrcDisposeMaterializationResponsibility.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcJITDylibRef LLVMOrcMaterializationResponsibilityGetTargetDylib(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityGetTargetDylib:=dll.LLVMOrcMaterializationResponsibilityGetTargetDylib).restype, LLVMOrcMaterializationResponsibilityGetTargetDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcExecutionSessionRef LLVMOrcMaterializationResponsibilityGetExecutionSession(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityGetExecutionSession:=dll.LLVMOrcMaterializationResponsibilityGetExecutionSession).restype, LLVMOrcMaterializationResponsibilityGetExecutionSession.argtypes = LLVMOrcExecutionSessionRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcCSymbolFlagsMapPairs LLVMOrcMaterializationResponsibilityGetSymbols(LLVMOrcMaterializationResponsibilityRef MR, size_t *NumPairs)
try: (LLVMOrcMaterializationResponsibilityGetSymbols:=dll.LLVMOrcMaterializationResponsibilityGetSymbols).restype, LLVMOrcMaterializationResponsibilityGetSymbols.argtypes = LLVMOrcCSymbolFlagsMapPairs, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# void LLVMOrcDisposeCSymbolFlagsMap(LLVMOrcCSymbolFlagsMapPairs Pairs)
try: (LLVMOrcDisposeCSymbolFlagsMap:=dll.LLVMOrcDisposeCSymbolFlagsMap).restype, LLVMOrcDisposeCSymbolFlagsMap.argtypes = None, [LLVMOrcCSymbolFlagsMapPairs]
except AttributeError: pass

# LLVMOrcSymbolStringPoolEntryRef LLVMOrcMaterializationResponsibilityGetInitializerSymbol(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityGetInitializerSymbol:=dll.LLVMOrcMaterializationResponsibilityGetInitializerSymbol).restype, LLVMOrcMaterializationResponsibilityGetInitializerSymbol.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMOrcSymbolStringPoolEntryRef *LLVMOrcMaterializationResponsibilityGetRequestedSymbols(LLVMOrcMaterializationResponsibilityRef MR, size_t *NumSymbols)
try: (LLVMOrcMaterializationResponsibilityGetRequestedSymbols:=dll.LLVMOrcMaterializationResponsibilityGetRequestedSymbols).restype, LLVMOrcMaterializationResponsibilityGetRequestedSymbols.argtypes = ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

# void LLVMOrcDisposeSymbols(LLVMOrcSymbolStringPoolEntryRef *Symbols)
try: (LLVMOrcDisposeSymbols:=dll.LLVMOrcDisposeSymbols).restype, LLVMOrcDisposeSymbols.argtypes = None, [ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef)]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityNotifyResolved(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolMapPairs Symbols, size_t NumPairs)
try: (LLVMOrcMaterializationResponsibilityNotifyResolved:=dll.LLVMOrcMaterializationResponsibilityNotifyResolved).restype, LLVMOrcMaterializationResponsibilityNotifyResolved.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityNotifyEmitted(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolDependenceGroup *SymbolDepGroups, size_t NumSymbolDepGroups)
try: (LLVMOrcMaterializationResponsibilityNotifyEmitted:=dll.LLVMOrcMaterializationResponsibilityNotifyEmitted).restype, LLVMOrcMaterializationResponsibilityNotifyEmitted.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcCSymbolDependenceGroup), size_t]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityDefineMaterializing(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolFlagsMapPairs Pairs, size_t NumPairs)
try: (LLVMOrcMaterializationResponsibilityDefineMaterializing:=dll.LLVMOrcMaterializationResponsibilityDefineMaterializing).restype, LLVMOrcMaterializationResponsibilityDefineMaterializing.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolFlagsMapPairs, size_t]
except AttributeError: pass

# void LLVMOrcMaterializationResponsibilityFailMaterialization(LLVMOrcMaterializationResponsibilityRef MR)
try: (LLVMOrcMaterializationResponsibilityFailMaterialization:=dll.LLVMOrcMaterializationResponsibilityFailMaterialization).restype, LLVMOrcMaterializationResponsibilityFailMaterialization.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityReplace(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcMaterializationUnitRef MU)
try: (LLVMOrcMaterializationResponsibilityReplace:=dll.LLVMOrcMaterializationResponsibilityReplace).restype, LLVMOrcMaterializationResponsibilityReplace.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcMaterializationResponsibilityDelegate(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcSymbolStringPoolEntryRef *Symbols, size_t NumSymbols, LLVMOrcMaterializationResponsibilityRef *Result)
try: (LLVMOrcMaterializationResponsibilityDelegate:=dll.LLVMOrcMaterializationResponsibilityDelegate).restype, LLVMOrcMaterializationResponsibilityDelegate.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), size_t, ctypes.POINTER(LLVMOrcMaterializationResponsibilityRef)]
except AttributeError: pass

# LLVMOrcJITDylibRef LLVMOrcExecutionSessionCreateBareJITDylib(LLVMOrcExecutionSessionRef ES, const char *Name)
try: (LLVMOrcExecutionSessionCreateBareJITDylib:=dll.LLVMOrcExecutionSessionCreateBareJITDylib).restype, LLVMOrcExecutionSessionCreateBareJITDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorRef LLVMOrcExecutionSessionCreateJITDylib(LLVMOrcExecutionSessionRef ES, LLVMOrcJITDylibRef *Result, const char *Name)
try: (LLVMOrcExecutionSessionCreateJITDylib:=dll.LLVMOrcExecutionSessionCreateJITDylib).restype, LLVMOrcExecutionSessionCreateJITDylib.argtypes = LLVMErrorRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(LLVMOrcJITDylibRef), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMOrcJITDylibRef LLVMOrcExecutionSessionGetJITDylibByName(LLVMOrcExecutionSessionRef ES, const char *Name)
try: (LLVMOrcExecutionSessionGetJITDylibByName:=dll.LLVMOrcExecutionSessionGetJITDylibByName).restype, LLVMOrcExecutionSessionGetJITDylibByName.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMOrcResourceTrackerRef LLVMOrcJITDylibCreateResourceTracker(LLVMOrcJITDylibRef JD)
try: (LLVMOrcJITDylibCreateResourceTracker:=dll.LLVMOrcJITDylibCreateResourceTracker).restype, LLVMOrcJITDylibCreateResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

# LLVMOrcResourceTrackerRef LLVMOrcJITDylibGetDefaultResourceTracker(LLVMOrcJITDylibRef JD)
try: (LLVMOrcJITDylibGetDefaultResourceTracker:=dll.LLVMOrcJITDylibGetDefaultResourceTracker).restype, LLVMOrcJITDylibGetDefaultResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcJITDylibDefine(LLVMOrcJITDylibRef JD, LLVMOrcMaterializationUnitRef MU)
try: (LLVMOrcJITDylibDefine:=dll.LLVMOrcJITDylibDefine).restype, LLVMOrcJITDylibDefine.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcJITDylibClear(LLVMOrcJITDylibRef JD)
try: (LLVMOrcJITDylibClear:=dll.LLVMOrcJITDylibClear).restype, LLVMOrcJITDylibClear.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

# void LLVMOrcJITDylibAddGenerator(LLVMOrcJITDylibRef JD, LLVMOrcDefinitionGeneratorRef DG)
try: (LLVMOrcJITDylibAddGenerator:=dll.LLVMOrcJITDylibAddGenerator).restype, LLVMOrcJITDylibAddGenerator.argtypes = None, [LLVMOrcJITDylibRef, LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

# LLVMOrcDefinitionGeneratorRef LLVMOrcCreateCustomCAPIDefinitionGenerator(LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction F, void *Ctx, LLVMOrcDisposeCAPIDefinitionGeneratorFunction Dispose)
try: (LLVMOrcCreateCustomCAPIDefinitionGenerator:=dll.LLVMOrcCreateCustomCAPIDefinitionGenerator).restype, LLVMOrcCreateCustomCAPIDefinitionGenerator.argtypes = LLVMOrcDefinitionGeneratorRef, [LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction, ctypes.c_void_p, LLVMOrcDisposeCAPIDefinitionGeneratorFunction]
except AttributeError: pass

# void LLVMOrcLookupStateContinueLookup(LLVMOrcLookupStateRef S, LLVMErrorRef Err)
try: (LLVMOrcLookupStateContinueLookup:=dll.LLVMOrcLookupStateContinueLookup).restype, LLVMOrcLookupStateContinueLookup.argtypes = None, [LLVMOrcLookupStateRef, LLVMErrorRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(LLVMOrcDefinitionGeneratorRef *Result, char GlobalPrefx, LLVMOrcSymbolPredicate Filter, void *FilterCtx)
try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(LLVMOrcDefinitionGeneratorRef *Result, const char *FileName, char GlobalPrefix, LLVMOrcSymbolPredicate Filter, void *FilterCtx)
try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForPath).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.POINTER(ctypes.c_char), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateStaticLibrarySearchGeneratorForPath(LLVMOrcDefinitionGeneratorRef *Result, LLVMOrcObjectLayerRef ObjLayer, const char *FileName, const char *TargetTriple)
try: (LLVMOrcCreateStaticLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateStaticLibrarySearchGeneratorForPath).restype, LLVMOrcCreateStaticLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), LLVMOrcObjectLayerRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMOrcThreadSafeContextRef LLVMOrcCreateNewThreadSafeContext(void)
try: (LLVMOrcCreateNewThreadSafeContext:=dll.LLVMOrcCreateNewThreadSafeContext).restype, LLVMOrcCreateNewThreadSafeContext.argtypes = LLVMOrcThreadSafeContextRef, []
except AttributeError: pass

# LLVMContextRef LLVMOrcThreadSafeContextGetContext(LLVMOrcThreadSafeContextRef TSCtx)
try: (LLVMOrcThreadSafeContextGetContext:=dll.LLVMOrcThreadSafeContextGetContext).restype, LLVMOrcThreadSafeContextGetContext.argtypes = LLVMContextRef, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

# void LLVMOrcDisposeThreadSafeContext(LLVMOrcThreadSafeContextRef TSCtx)
try: (LLVMOrcDisposeThreadSafeContext:=dll.LLVMOrcDisposeThreadSafeContext).restype, LLVMOrcDisposeThreadSafeContext.argtypes = None, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

# LLVMOrcThreadSafeModuleRef LLVMOrcCreateNewThreadSafeModule(LLVMModuleRef M, LLVMOrcThreadSafeContextRef TSCtx)
try: (LLVMOrcCreateNewThreadSafeModule:=dll.LLVMOrcCreateNewThreadSafeModule).restype, LLVMOrcCreateNewThreadSafeModule.argtypes = LLVMOrcThreadSafeModuleRef, [LLVMModuleRef, LLVMOrcThreadSafeContextRef]
except AttributeError: pass

# void LLVMOrcDisposeThreadSafeModule(LLVMOrcThreadSafeModuleRef TSM)
try: (LLVMOrcDisposeThreadSafeModule:=dll.LLVMOrcDisposeThreadSafeModule).restype, LLVMOrcDisposeThreadSafeModule.argtypes = None, [LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcThreadSafeModuleWithModuleDo(LLVMOrcThreadSafeModuleRef TSM, LLVMOrcGenericIRModuleOperationFunction F, void *Ctx)
try: (LLVMOrcThreadSafeModuleWithModuleDo:=dll.LLVMOrcThreadSafeModuleWithModuleDo).restype, LLVMOrcThreadSafeModuleWithModuleDo.argtypes = LLVMErrorRef, [LLVMOrcThreadSafeModuleRef, LLVMOrcGenericIRModuleOperationFunction, ctypes.c_void_p]
except AttributeError: pass

# LLVMErrorRef LLVMOrcJITTargetMachineBuilderDetectHost(LLVMOrcJITTargetMachineBuilderRef *Result)
try: (LLVMOrcJITTargetMachineBuilderDetectHost:=dll.LLVMOrcJITTargetMachineBuilderDetectHost).restype, LLVMOrcJITTargetMachineBuilderDetectHost.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcJITTargetMachineBuilderRef)]
except AttributeError: pass

# LLVMOrcJITTargetMachineBuilderRef LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(LLVMTargetMachineRef TM)
try: (LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine:=dll.LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine).restype, LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine.argtypes = LLVMOrcJITTargetMachineBuilderRef, [LLVMTargetMachineRef]
except AttributeError: pass

# void LLVMOrcDisposeJITTargetMachineBuilder(LLVMOrcJITTargetMachineBuilderRef JTMB)
try: (LLVMOrcDisposeJITTargetMachineBuilder:=dll.LLVMOrcDisposeJITTargetMachineBuilder).restype, LLVMOrcDisposeJITTargetMachineBuilder.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

# char *LLVMOrcJITTargetMachineBuilderGetTargetTriple(LLVMOrcJITTargetMachineBuilderRef JTMB)
try: (LLVMOrcJITTargetMachineBuilderGetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderGetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderGetTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

# void LLVMOrcJITTargetMachineBuilderSetTargetTriple(LLVMOrcJITTargetMachineBuilderRef JTMB, const char *TargetTriple)
try: (LLVMOrcJITTargetMachineBuilderSetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderSetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderSetTargetTriple.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorRef LLVMOrcObjectLayerAddObjectFile(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcJITDylibRef JD, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcObjectLayerAddObjectFile:=dll.LLVMOrcObjectLayerAddObjectFile).restype, LLVMOrcObjectLayerAddObjectFile.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcJITDylibRef, LLVMMemoryBufferRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcObjectLayerAddObjectFileWithRT(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcResourceTrackerRef RT, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcObjectLayerAddObjectFileWithRT:=dll.LLVMOrcObjectLayerAddObjectFileWithRT).restype, LLVMOrcObjectLayerAddObjectFileWithRT.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef]
except AttributeError: pass

# void LLVMOrcObjectLayerEmit(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcMaterializationResponsibilityRef R, LLVMMemoryBufferRef ObjBuffer)
try: (LLVMOrcObjectLayerEmit:=dll.LLVMOrcObjectLayerEmit).restype, LLVMOrcObjectLayerEmit.argtypes = None, [LLVMOrcObjectLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMMemoryBufferRef]
except AttributeError: pass

# void LLVMOrcDisposeObjectLayer(LLVMOrcObjectLayerRef ObjLayer)
try: (LLVMOrcDisposeObjectLayer:=dll.LLVMOrcDisposeObjectLayer).restype, LLVMOrcDisposeObjectLayer.argtypes = None, [LLVMOrcObjectLayerRef]
except AttributeError: pass

# void LLVMOrcIRTransformLayerEmit(LLVMOrcIRTransformLayerRef IRTransformLayer, LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcThreadSafeModuleRef TSM)
try: (LLVMOrcIRTransformLayerEmit:=dll.LLVMOrcIRTransformLayerEmit).restype, LLVMOrcIRTransformLayerEmit.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

# void LLVMOrcIRTransformLayerSetTransform(LLVMOrcIRTransformLayerRef IRTransformLayer, LLVMOrcIRTransformLayerTransformFunction TransformFunction, void *Ctx)
try: (LLVMOrcIRTransformLayerSetTransform:=dll.LLVMOrcIRTransformLayerSetTransform).restype, LLVMOrcIRTransformLayerSetTransform.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcIRTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

# void LLVMOrcObjectTransformLayerSetTransform(LLVMOrcObjectTransformLayerRef ObjTransformLayer, LLVMOrcObjectTransformLayerTransformFunction TransformFunction, void *Ctx)
try: (LLVMOrcObjectTransformLayerSetTransform:=dll.LLVMOrcObjectTransformLayerSetTransform).restype, LLVMOrcObjectTransformLayerSetTransform.argtypes = None, [LLVMOrcObjectTransformLayerRef, LLVMOrcObjectTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

# LLVMOrcIndirectStubsManagerRef LLVMOrcCreateLocalIndirectStubsManager(const char *TargetTriple)
try: (LLVMOrcCreateLocalIndirectStubsManager:=dll.LLVMOrcCreateLocalIndirectStubsManager).restype, LLVMOrcCreateLocalIndirectStubsManager.argtypes = LLVMOrcIndirectStubsManagerRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMOrcDisposeIndirectStubsManager(LLVMOrcIndirectStubsManagerRef ISM)
try: (LLVMOrcDisposeIndirectStubsManager:=dll.LLVMOrcDisposeIndirectStubsManager).restype, LLVMOrcDisposeIndirectStubsManager.argtypes = None, [LLVMOrcIndirectStubsManagerRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcCreateLocalLazyCallThroughManager(const char *TargetTriple, LLVMOrcExecutionSessionRef ES, LLVMOrcJITTargetAddress ErrorHandlerAddr, LLVMOrcLazyCallThroughManagerRef *LCTM)
try: (LLVMOrcCreateLocalLazyCallThroughManager:=dll.LLVMOrcCreateLocalLazyCallThroughManager).restype, LLVMOrcCreateLocalLazyCallThroughManager.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char), LLVMOrcExecutionSessionRef, LLVMOrcJITTargetAddress, ctypes.POINTER(LLVMOrcLazyCallThroughManagerRef)]
except AttributeError: pass

# void LLVMOrcDisposeLazyCallThroughManager(LLVMOrcLazyCallThroughManagerRef LCTM)
try: (LLVMOrcDisposeLazyCallThroughManager:=dll.LLVMOrcDisposeLazyCallThroughManager).restype, LLVMOrcDisposeLazyCallThroughManager.argtypes = None, [LLVMOrcLazyCallThroughManagerRef]
except AttributeError: pass

# LLVMOrcDumpObjectsRef LLVMOrcCreateDumpObjects(const char *DumpDir, const char *IdentifierOverride)
try: (LLVMOrcCreateDumpObjects:=dll.LLVMOrcCreateDumpObjects).restype, LLVMOrcCreateDumpObjects.argtypes = LLVMOrcDumpObjectsRef, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMOrcDisposeDumpObjects(LLVMOrcDumpObjectsRef DumpObjects)
try: (LLVMOrcDisposeDumpObjects:=dll.LLVMOrcDisposeDumpObjects).restype, LLVMOrcDisposeDumpObjects.argtypes = None, [LLVMOrcDumpObjectsRef]
except AttributeError: pass

# LLVMErrorRef LLVMOrcDumpObjects_CallOperator(LLVMOrcDumpObjectsRef DumpObjects, LLVMMemoryBufferRef *ObjBuffer)
try: (LLVMOrcDumpObjects_CallOperator:=dll.LLVMOrcDumpObjects_CallOperator).restype, LLVMOrcDumpObjects_CallOperator.argtypes = LLVMErrorRef, [LLVMOrcDumpObjectsRef, ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

LLVMMemoryManagerCreateContextCallback = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)
LLVMMemoryManagerNotifyTerminatingCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
# LLVMOrcObjectLayerRef LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager(LLVMOrcExecutionSessionRef ES)
try: (LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager:=dll.LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager).restype, LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager.argtypes = LLVMOrcObjectLayerRef, [LLVMOrcExecutionSessionRef]
except AttributeError: pass

# LLVMOrcObjectLayerRef LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks(LLVMOrcExecutionSessionRef ES, void *CreateContextCtx, LLVMMemoryManagerCreateContextCallback CreateContext, LLVMMemoryManagerNotifyTerminatingCallback NotifyTerminating, LLVMMemoryManagerAllocateCodeSectionCallback AllocateCodeSection, LLVMMemoryManagerAllocateDataSectionCallback AllocateDataSection, LLVMMemoryManagerFinalizeMemoryCallback FinalizeMemory, LLVMMemoryManagerDestroyCallback Destroy)
try: (LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks:=dll.LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks).restype, LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks.argtypes = LLVMOrcObjectLayerRef, [LLVMOrcExecutionSessionRef, ctypes.c_void_p, LLVMMemoryManagerCreateContextCallback, LLVMMemoryManagerNotifyTerminatingCallback, LLVMMemoryManagerAllocateCodeSectionCallback, LLVMMemoryManagerAllocateDataSectionCallback, LLVMMemoryManagerFinalizeMemoryCallback, LLVMMemoryManagerDestroyCallback]
except AttributeError: pass

# void LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(LLVMOrcObjectLayerRef RTDyldObjLinkingLayer, LLVMJITEventListenerRef Listener)
try: (LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener:=dll.LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener).restype, LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener.argtypes = None, [LLVMOrcObjectLayerRef, LLVMJITEventListenerRef]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

enum_LLVMRemarkType = CEnum(ctypes.c_uint32)
LLVMRemarkTypeUnknown = enum_LLVMRemarkType.define('LLVMRemarkTypeUnknown', 0)
LLVMRemarkTypePassed = enum_LLVMRemarkType.define('LLVMRemarkTypePassed', 1)
LLVMRemarkTypeMissed = enum_LLVMRemarkType.define('LLVMRemarkTypeMissed', 2)
LLVMRemarkTypeAnalysis = enum_LLVMRemarkType.define('LLVMRemarkTypeAnalysis', 3)
LLVMRemarkTypeAnalysisFPCommute = enum_LLVMRemarkType.define('LLVMRemarkTypeAnalysisFPCommute', 4)
LLVMRemarkTypeAnalysisAliasing = enum_LLVMRemarkType.define('LLVMRemarkTypeAnalysisAliasing', 5)
LLVMRemarkTypeFailure = enum_LLVMRemarkType.define('LLVMRemarkTypeFailure', 6)

class struct_LLVMRemarkOpaqueString(Struct): pass
LLVMRemarkStringRef = ctypes.POINTER(struct_LLVMRemarkOpaqueString)
# extern const char *LLVMRemarkStringGetData(LLVMRemarkStringRef String)
try: (LLVMRemarkStringGetData:=dll.LLVMRemarkStringGetData).restype, LLVMRemarkStringGetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMRemarkStringRef]
except AttributeError: pass

# extern uint32_t LLVMRemarkStringGetLen(LLVMRemarkStringRef String)
try: (LLVMRemarkStringGetLen:=dll.LLVMRemarkStringGetLen).restype, LLVMRemarkStringGetLen.argtypes = uint32_t, [LLVMRemarkStringRef]
except AttributeError: pass

class struct_LLVMRemarkOpaqueDebugLoc(Struct): pass
LLVMRemarkDebugLocRef = ctypes.POINTER(struct_LLVMRemarkOpaqueDebugLoc)
# extern LLVMRemarkStringRef LLVMRemarkDebugLocGetSourceFilePath(LLVMRemarkDebugLocRef DL)
try: (LLVMRemarkDebugLocGetSourceFilePath:=dll.LLVMRemarkDebugLocGetSourceFilePath).restype, LLVMRemarkDebugLocGetSourceFilePath.argtypes = LLVMRemarkStringRef, [LLVMRemarkDebugLocRef]
except AttributeError: pass

# extern uint32_t LLVMRemarkDebugLocGetSourceLine(LLVMRemarkDebugLocRef DL)
try: (LLVMRemarkDebugLocGetSourceLine:=dll.LLVMRemarkDebugLocGetSourceLine).restype, LLVMRemarkDebugLocGetSourceLine.argtypes = uint32_t, [LLVMRemarkDebugLocRef]
except AttributeError: pass

# extern uint32_t LLVMRemarkDebugLocGetSourceColumn(LLVMRemarkDebugLocRef DL)
try: (LLVMRemarkDebugLocGetSourceColumn:=dll.LLVMRemarkDebugLocGetSourceColumn).restype, LLVMRemarkDebugLocGetSourceColumn.argtypes = uint32_t, [LLVMRemarkDebugLocRef]
except AttributeError: pass

class struct_LLVMRemarkOpaqueArg(Struct): pass
LLVMRemarkArgRef = ctypes.POINTER(struct_LLVMRemarkOpaqueArg)
# extern LLVMRemarkStringRef LLVMRemarkArgGetKey(LLVMRemarkArgRef Arg)
try: (LLVMRemarkArgGetKey:=dll.LLVMRemarkArgGetKey).restype, LLVMRemarkArgGetKey.argtypes = LLVMRemarkStringRef, [LLVMRemarkArgRef]
except AttributeError: pass

# extern LLVMRemarkStringRef LLVMRemarkArgGetValue(LLVMRemarkArgRef Arg)
try: (LLVMRemarkArgGetValue:=dll.LLVMRemarkArgGetValue).restype, LLVMRemarkArgGetValue.argtypes = LLVMRemarkStringRef, [LLVMRemarkArgRef]
except AttributeError: pass

# extern LLVMRemarkDebugLocRef LLVMRemarkArgGetDebugLoc(LLVMRemarkArgRef Arg)
try: (LLVMRemarkArgGetDebugLoc:=dll.LLVMRemarkArgGetDebugLoc).restype, LLVMRemarkArgGetDebugLoc.argtypes = LLVMRemarkDebugLocRef, [LLVMRemarkArgRef]
except AttributeError: pass

class struct_LLVMRemarkOpaqueEntry(Struct): pass
LLVMRemarkEntryRef = ctypes.POINTER(struct_LLVMRemarkOpaqueEntry)
# extern void LLVMRemarkEntryDispose(LLVMRemarkEntryRef Remark)
try: (LLVMRemarkEntryDispose:=dll.LLVMRemarkEntryDispose).restype, LLVMRemarkEntryDispose.argtypes = None, [LLVMRemarkEntryRef]
except AttributeError: pass

# extern enum LLVMRemarkType LLVMRemarkEntryGetType(LLVMRemarkEntryRef Remark)
try: (LLVMRemarkEntryGetType:=dll.LLVMRemarkEntryGetType).restype, LLVMRemarkEntryGetType.argtypes = enum_LLVMRemarkType, [LLVMRemarkEntryRef]
except AttributeError: pass

# extern LLVMRemarkStringRef LLVMRemarkEntryGetPassName(LLVMRemarkEntryRef Remark)
try: (LLVMRemarkEntryGetPassName:=dll.LLVMRemarkEntryGetPassName).restype, LLVMRemarkEntryGetPassName.argtypes = LLVMRemarkStringRef, [LLVMRemarkEntryRef]
except AttributeError: pass

# extern LLVMRemarkStringRef LLVMRemarkEntryGetRemarkName(LLVMRemarkEntryRef Remark)
try: (LLVMRemarkEntryGetRemarkName:=dll.LLVMRemarkEntryGetRemarkName).restype, LLVMRemarkEntryGetRemarkName.argtypes = LLVMRemarkStringRef, [LLVMRemarkEntryRef]
except AttributeError: pass

# extern LLVMRemarkStringRef LLVMRemarkEntryGetFunctionName(LLVMRemarkEntryRef Remark)
try: (LLVMRemarkEntryGetFunctionName:=dll.LLVMRemarkEntryGetFunctionName).restype, LLVMRemarkEntryGetFunctionName.argtypes = LLVMRemarkStringRef, [LLVMRemarkEntryRef]
except AttributeError: pass

# extern LLVMRemarkDebugLocRef LLVMRemarkEntryGetDebugLoc(LLVMRemarkEntryRef Remark)
try: (LLVMRemarkEntryGetDebugLoc:=dll.LLVMRemarkEntryGetDebugLoc).restype, LLVMRemarkEntryGetDebugLoc.argtypes = LLVMRemarkDebugLocRef, [LLVMRemarkEntryRef]
except AttributeError: pass

# extern uint64_t LLVMRemarkEntryGetHotness(LLVMRemarkEntryRef Remark)
try: (LLVMRemarkEntryGetHotness:=dll.LLVMRemarkEntryGetHotness).restype, LLVMRemarkEntryGetHotness.argtypes = uint64_t, [LLVMRemarkEntryRef]
except AttributeError: pass

# extern uint32_t LLVMRemarkEntryGetNumArgs(LLVMRemarkEntryRef Remark)
try: (LLVMRemarkEntryGetNumArgs:=dll.LLVMRemarkEntryGetNumArgs).restype, LLVMRemarkEntryGetNumArgs.argtypes = uint32_t, [LLVMRemarkEntryRef]
except AttributeError: pass

# extern LLVMRemarkArgRef LLVMRemarkEntryGetFirstArg(LLVMRemarkEntryRef Remark)
try: (LLVMRemarkEntryGetFirstArg:=dll.LLVMRemarkEntryGetFirstArg).restype, LLVMRemarkEntryGetFirstArg.argtypes = LLVMRemarkArgRef, [LLVMRemarkEntryRef]
except AttributeError: pass

# extern LLVMRemarkArgRef LLVMRemarkEntryGetNextArg(LLVMRemarkArgRef It, LLVMRemarkEntryRef Remark)
try: (LLVMRemarkEntryGetNextArg:=dll.LLVMRemarkEntryGetNextArg).restype, LLVMRemarkEntryGetNextArg.argtypes = LLVMRemarkArgRef, [LLVMRemarkArgRef, LLVMRemarkEntryRef]
except AttributeError: pass

class struct_LLVMRemarkOpaqueParser(Struct): pass
LLVMRemarkParserRef = ctypes.POINTER(struct_LLVMRemarkOpaqueParser)
# extern LLVMRemarkParserRef LLVMRemarkParserCreateYAML(const void *Buf, uint64_t Size)
try: (LLVMRemarkParserCreateYAML:=dll.LLVMRemarkParserCreateYAML).restype, LLVMRemarkParserCreateYAML.argtypes = LLVMRemarkParserRef, [ctypes.c_void_p, uint64_t]
except AttributeError: pass

# extern LLVMRemarkParserRef LLVMRemarkParserCreateBitstream(const void *Buf, uint64_t Size)
try: (LLVMRemarkParserCreateBitstream:=dll.LLVMRemarkParserCreateBitstream).restype, LLVMRemarkParserCreateBitstream.argtypes = LLVMRemarkParserRef, [ctypes.c_void_p, uint64_t]
except AttributeError: pass

# extern LLVMRemarkEntryRef LLVMRemarkParserGetNext(LLVMRemarkParserRef Parser)
try: (LLVMRemarkParserGetNext:=dll.LLVMRemarkParserGetNext).restype, LLVMRemarkParserGetNext.argtypes = LLVMRemarkEntryRef, [LLVMRemarkParserRef]
except AttributeError: pass

# extern LLVMBool LLVMRemarkParserHasError(LLVMRemarkParserRef Parser)
try: (LLVMRemarkParserHasError:=dll.LLVMRemarkParserHasError).restype, LLVMRemarkParserHasError.argtypes = LLVMBool, [LLVMRemarkParserRef]
except AttributeError: pass

# extern const char *LLVMRemarkParserGetErrorMessage(LLVMRemarkParserRef Parser)
try: (LLVMRemarkParserGetErrorMessage:=dll.LLVMRemarkParserGetErrorMessage).restype, LLVMRemarkParserGetErrorMessage.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMRemarkParserRef]
except AttributeError: pass

# extern void LLVMRemarkParserDispose(LLVMRemarkParserRef Parser)
try: (LLVMRemarkParserDispose:=dll.LLVMRemarkParserDispose).restype, LLVMRemarkParserDispose.argtypes = None, [LLVMRemarkParserRef]
except AttributeError: pass

# extern uint32_t LLVMRemarkVersion(void)
try: (LLVMRemarkVersion:=dll.LLVMRemarkVersion).restype, LLVMRemarkVersion.argtypes = uint32_t, []
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

# LLVMBool LLVMLoadLibraryPermanently(const char *Filename)
try: (LLVMLoadLibraryPermanently:=dll.LLVMLoadLibraryPermanently).restype, LLVMLoadLibraryPermanently.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMParseCommandLineOptions(int argc, const char *const *argv, const char *Overview)
try: (LLVMParseCommandLineOptions:=dll.LLVMParseCommandLineOptions).restype, LLVMParseCommandLineOptions.argtypes = None, [ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void *LLVMSearchForAddressOfSymbol(const char *symbolName)
try: (LLVMSearchForAddressOfSymbol:=dll.LLVMSearchForAddressOfSymbol).restype, LLVMSearchForAddressOfSymbol.argtypes = ctypes.c_void_p, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMAddSymbol(const char *symbolName, void *symbolValue)
try: (LLVMAddSymbol:=dll.LLVMAddSymbol).restype, LLVMAddSymbol.argtypes = None, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

# void LLVMInitializeAArch64TargetInfo(void)
try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetInfo(void)
try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetInfo(void)
try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetInfo(void)
try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetInfo(void)
try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetInfo(void)
try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetInfo(void)
try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetInfo(void)
try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetInfo(void)
try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetInfo(void)
try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetInfo(void)
try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetInfo(void)
try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetInfo(void)
try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetInfo(void)
try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetInfo(void)
try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetInfo(void)
try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetInfo(void)
try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetInfo(void)
try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetInfo(void)
try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetInfo(void)
try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetInfo(void)
try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetInfo(void)
try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Target(void)
try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTarget(void)
try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTarget(void)
try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTarget(void)
try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTarget(void)
try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTarget(void)
try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTarget(void)
try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTarget(void)
try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTarget(void)
try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Target(void)
try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTarget(void)
try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTarget(void)
try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTarget(void)
try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTarget(void)
try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTarget(void)
try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTarget(void)
try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETarget(void)
try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTarget(void)
try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Target(void)
try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTarget(void)
try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTarget(void)
try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTarget(void)
try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64TargetMC(void)
try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetMC(void)
try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetMC(void)
try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetMC(void)
try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetMC(void)
try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetMC(void)
try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetMC(void)
try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetMC(void)
try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetMC(void)
try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetMC(void)
try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetMC(void)
try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetMC(void)
try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetMC(void)
try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetMC(void)
try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetMC(void)
try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetMC(void)
try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetMC(void)
try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetMC(void)
try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetMC(void)
try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetMC(void)
try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetMC(void)
try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetMC(void)
try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmPrinter(void)
try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmPrinter(void)
try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmPrinter(void)
try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmPrinter(void)
try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmPrinter(void)
try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmPrinter(void)
try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmPrinter(void)
try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmPrinter(void)
try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmPrinter(void)
try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmPrinter(void)
try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXAsmPrinter(void)
try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmPrinter(void)
try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmPrinter(void)
try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmPrinter(void)
try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVAsmPrinter(void)
try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmPrinter(void)
try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmPrinter(void)
try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmPrinter(void)
try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmPrinter(void)
try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreAsmPrinter(void)
try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmPrinter(void)
try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmPrinter(void)
try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmParser(void)
try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmParser(void)
try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmParser(void)
try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmParser(void)
try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmParser(void)
try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmParser(void)
try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmParser(void)
try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmParser(void)
try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmParser(void)
try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmParser(void)
try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmParser(void)
try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmParser(void)
try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmParser(void)
try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmParser(void)
try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmParser(void)
try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmParser(void)
try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmParser(void)
try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmParser(void)
try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmParser(void)
try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Disassembler(void)
try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUDisassembler(void)
try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMDisassembler(void)
try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRDisassembler(void)
try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFDisassembler(void)
try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonDisassembler(void)
try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiDisassembler(void)
try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchDisassembler(void)
try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsDisassembler(void)
try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Disassembler(void)
try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCDisassembler(void)
try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVDisassembler(void)
try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcDisassembler(void)
try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZDisassembler(void)
try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEDisassembler(void)
try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyDisassembler(void)
try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Disassembler(void)
try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreDisassembler(void)
try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kDisassembler(void)
try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaDisassembler(void)
try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

# LLVMTargetDataRef LLVMGetModuleDataLayout(LLVMModuleRef M)
try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

# void LLVMSetModuleDataLayout(LLVMModuleRef M, LLVMTargetDataRef DL)
try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetData(const char *StringRep)
try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMDisposeTargetData(LLVMTargetDataRef TD)
try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

# void LLVMAddTargetLibraryInfo(LLVMTargetLibraryInfoRef TLI, LLVMPassManagerRef PM)
try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

# char *LLVMCopyStringRepOfTargetData(LLVMTargetDataRef TD)
try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

# enum LLVMByteOrdering LLVMByteOrder(LLVMTargetDataRef TD)
try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSize(LLVMTargetDataRef TD)
try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSizeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrType(LLVMTargetDataRef TD)
try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeInContext(LLVMContextRef C, LLVMTargetDataRef TD)
try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForASInContext(LLVMContextRef C, LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# unsigned long long LLVMSizeOfTypeInBits(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMStoreSizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMABISizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMABIAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMCallFrameAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfGlobal(LLVMTargetDataRef TD, LLVMValueRef GlobalVar)
try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

# unsigned int LLVMElementAtOffset(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned long long Offset)
try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

# unsigned long long LLVMOffsetOfElement(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned int Element)
try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

# void LLVMInitializeAArch64TargetInfo(void)
try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetInfo(void)
try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetInfo(void)
try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetInfo(void)
try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetInfo(void)
try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetInfo(void)
try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetInfo(void)
try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetInfo(void)
try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetInfo(void)
try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetInfo(void)
try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetInfo(void)
try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetInfo(void)
try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetInfo(void)
try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetInfo(void)
try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetInfo(void)
try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetInfo(void)
try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetInfo(void)
try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetInfo(void)
try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetInfo(void)
try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetInfo(void)
try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetInfo(void)
try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetInfo(void)
try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Target(void)
try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTarget(void)
try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTarget(void)
try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTarget(void)
try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTarget(void)
try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTarget(void)
try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTarget(void)
try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTarget(void)
try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTarget(void)
try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Target(void)
try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTarget(void)
try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTarget(void)
try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTarget(void)
try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTarget(void)
try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTarget(void)
try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTarget(void)
try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETarget(void)
try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTarget(void)
try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Target(void)
try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTarget(void)
try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTarget(void)
try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTarget(void)
try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64TargetMC(void)
try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetMC(void)
try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetMC(void)
try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetMC(void)
try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetMC(void)
try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetMC(void)
try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetMC(void)
try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetMC(void)
try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetMC(void)
try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetMC(void)
try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetMC(void)
try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetMC(void)
try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetMC(void)
try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetMC(void)
try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetMC(void)
try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetMC(void)
try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetMC(void)
try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetMC(void)
try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetMC(void)
try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetMC(void)
try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetMC(void)
try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetMC(void)
try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmPrinter(void)
try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmPrinter(void)
try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmPrinter(void)
try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmPrinter(void)
try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmPrinter(void)
try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmPrinter(void)
try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmPrinter(void)
try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmPrinter(void)
try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmPrinter(void)
try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmPrinter(void)
try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXAsmPrinter(void)
try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmPrinter(void)
try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmPrinter(void)
try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmPrinter(void)
try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVAsmPrinter(void)
try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmPrinter(void)
try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmPrinter(void)
try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmPrinter(void)
try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmPrinter(void)
try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreAsmPrinter(void)
try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmPrinter(void)
try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmPrinter(void)
try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmParser(void)
try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmParser(void)
try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmParser(void)
try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmParser(void)
try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmParser(void)
try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmParser(void)
try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmParser(void)
try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmParser(void)
try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmParser(void)
try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmParser(void)
try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmParser(void)
try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmParser(void)
try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmParser(void)
try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmParser(void)
try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmParser(void)
try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmParser(void)
try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmParser(void)
try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmParser(void)
try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmParser(void)
try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Disassembler(void)
try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUDisassembler(void)
try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMDisassembler(void)
try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRDisassembler(void)
try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFDisassembler(void)
try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonDisassembler(void)
try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiDisassembler(void)
try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchDisassembler(void)
try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsDisassembler(void)
try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Disassembler(void)
try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCDisassembler(void)
try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVDisassembler(void)
try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcDisassembler(void)
try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZDisassembler(void)
try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEDisassembler(void)
try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyDisassembler(void)
try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Disassembler(void)
try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreDisassembler(void)
try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kDisassembler(void)
try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaDisassembler(void)
try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

# LLVMTargetDataRef LLVMGetModuleDataLayout(LLVMModuleRef M)
try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

# void LLVMSetModuleDataLayout(LLVMModuleRef M, LLVMTargetDataRef DL)
try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetData(const char *StringRep)
try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMDisposeTargetData(LLVMTargetDataRef TD)
try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

# void LLVMAddTargetLibraryInfo(LLVMTargetLibraryInfoRef TLI, LLVMPassManagerRef PM)
try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

# char *LLVMCopyStringRepOfTargetData(LLVMTargetDataRef TD)
try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

# enum LLVMByteOrdering LLVMByteOrder(LLVMTargetDataRef TD)
try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSize(LLVMTargetDataRef TD)
try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSizeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrType(LLVMTargetDataRef TD)
try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeInContext(LLVMContextRef C, LLVMTargetDataRef TD)
try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForASInContext(LLVMContextRef C, LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# unsigned long long LLVMSizeOfTypeInBits(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMStoreSizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMABISizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMABIAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMCallFrameAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfGlobal(LLVMTargetDataRef TD, LLVMValueRef GlobalVar)
try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

# unsigned int LLVMElementAtOffset(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned long long Offset)
try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

# unsigned long long LLVMOffsetOfElement(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned int Element)
try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTargetRef LLVMGetFirstTarget(void)
try: (LLVMGetFirstTarget:=dll.LLVMGetFirstTarget).restype, LLVMGetFirstTarget.argtypes = LLVMTargetRef, []
except AttributeError: pass

# LLVMTargetRef LLVMGetNextTarget(LLVMTargetRef T)
try: (LLVMGetNextTarget:=dll.LLVMGetNextTarget).restype, LLVMGetNextTarget.argtypes = LLVMTargetRef, [LLVMTargetRef]
except AttributeError: pass

# LLVMTargetRef LLVMGetTargetFromName(const char *Name)
try: (LLVMGetTargetFromName:=dll.LLVMGetTargetFromName).restype, LLVMGetTargetFromName.argtypes = LLVMTargetRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBool LLVMGetTargetFromTriple(const char *Triple, LLVMTargetRef *T, char **ErrorMessage)
try: (LLVMGetTargetFromTriple:=dll.LLVMGetTargetFromTriple).restype, LLVMGetTargetFromTriple.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTargetRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# const char *LLVMGetTargetName(LLVMTargetRef T)
try: (LLVMGetTargetName:=dll.LLVMGetTargetName).restype, LLVMGetTargetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

# const char *LLVMGetTargetDescription(LLVMTargetRef T)
try: (LLVMGetTargetDescription:=dll.LLVMGetTargetDescription).restype, LLVMGetTargetDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasJIT(LLVMTargetRef T)
try: (LLVMTargetHasJIT:=dll.LLVMTargetHasJIT).restype, LLVMTargetHasJIT.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasTargetMachine(LLVMTargetRef T)
try: (LLVMTargetHasTargetMachine:=dll.LLVMTargetHasTargetMachine).restype, LLVMTargetHasTargetMachine.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasAsmBackend(LLVMTargetRef T)
try: (LLVMTargetHasAsmBackend:=dll.LLVMTargetHasAsmBackend).restype, LLVMTargetHasAsmBackend.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMTargetMachineOptionsRef LLVMCreateTargetMachineOptions(void)
try: (LLVMCreateTargetMachineOptions:=dll.LLVMCreateTargetMachineOptions).restype, LLVMCreateTargetMachineOptions.argtypes = LLVMTargetMachineOptionsRef, []
except AttributeError: pass

# void LLVMDisposeTargetMachineOptions(LLVMTargetMachineOptionsRef Options)
try: (LLVMDisposeTargetMachineOptions:=dll.LLVMDisposeTargetMachineOptions).restype, LLVMDisposeTargetMachineOptions.argtypes = None, [LLVMTargetMachineOptionsRef]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCPU(LLVMTargetMachineOptionsRef Options, const char *CPU)
try: (LLVMTargetMachineOptionsSetCPU:=dll.LLVMTargetMachineOptionsSetCPU).restype, LLVMTargetMachineOptionsSetCPU.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetFeatures(LLVMTargetMachineOptionsRef Options, const char *Features)
try: (LLVMTargetMachineOptionsSetFeatures:=dll.LLVMTargetMachineOptionsSetFeatures).restype, LLVMTargetMachineOptionsSetFeatures.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetABI(LLVMTargetMachineOptionsRef Options, const char *ABI)
try: (LLVMTargetMachineOptionsSetABI:=dll.LLVMTargetMachineOptionsSetABI).restype, LLVMTargetMachineOptionsSetABI.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCodeGenOptLevel(LLVMTargetMachineOptionsRef Options, LLVMCodeGenOptLevel Level)
try: (LLVMTargetMachineOptionsSetCodeGenOptLevel:=dll.LLVMTargetMachineOptionsSetCodeGenOptLevel).restype, LLVMTargetMachineOptionsSetCodeGenOptLevel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeGenOptLevel]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetRelocMode(LLVMTargetMachineOptionsRef Options, LLVMRelocMode Reloc)
try: (LLVMTargetMachineOptionsSetRelocMode:=dll.LLVMTargetMachineOptionsSetRelocMode).restype, LLVMTargetMachineOptionsSetRelocMode.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMRelocMode]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCodeModel(LLVMTargetMachineOptionsRef Options, LLVMCodeModel CodeModel)
try: (LLVMTargetMachineOptionsSetCodeModel:=dll.LLVMTargetMachineOptionsSetCodeModel).restype, LLVMTargetMachineOptionsSetCodeModel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeModel]
except AttributeError: pass

# LLVMTargetMachineRef LLVMCreateTargetMachineWithOptions(LLVMTargetRef T, const char *Triple, LLVMTargetMachineOptionsRef Options)
try: (LLVMCreateTargetMachineWithOptions:=dll.LLVMCreateTargetMachineWithOptions).restype, LLVMCreateTargetMachineWithOptions.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineOptionsRef]
except AttributeError: pass

# LLVMTargetMachineRef LLVMCreateTargetMachine(LLVMTargetRef T, const char *Triple, const char *CPU, const char *Features, LLVMCodeGenOptLevel Level, LLVMRelocMode Reloc, LLVMCodeModel CodeModel)
try: (LLVMCreateTargetMachine:=dll.LLVMCreateTargetMachine).restype, LLVMCreateTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError: pass

# void LLVMDisposeTargetMachine(LLVMTargetMachineRef T)
try: (LLVMDisposeTargetMachine:=dll.LLVMDisposeTargetMachine).restype, LLVMDisposeTargetMachine.argtypes = None, [LLVMTargetMachineRef]
except AttributeError: pass

# LLVMTargetRef LLVMGetTargetMachineTarget(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineTarget:=dll.LLVMGetTargetMachineTarget).restype, LLVMGetTargetMachineTarget.argtypes = LLVMTargetRef, [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineTriple(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineTriple:=dll.LLVMGetTargetMachineTriple).restype, LLVMGetTargetMachineTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineCPU(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineCPU:=dll.LLVMGetTargetMachineCPU).restype, LLVMGetTargetMachineCPU.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineFeatureString(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineFeatureString:=dll.LLVMGetTargetMachineFeatureString).restype, LLVMGetTargetMachineFeatureString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetDataLayout(LLVMTargetMachineRef T)
try: (LLVMCreateTargetDataLayout:=dll.LLVMCreateTargetDataLayout).restype, LLVMCreateTargetDataLayout.argtypes = LLVMTargetDataRef, [LLVMTargetMachineRef]
except AttributeError: pass

# void LLVMSetTargetMachineAsmVerbosity(LLVMTargetMachineRef T, LLVMBool VerboseAsm)
try: (LLVMSetTargetMachineAsmVerbosity:=dll.LLVMSetTargetMachineAsmVerbosity).restype, LLVMSetTargetMachineAsmVerbosity.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineFastISel(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineFastISel:=dll.LLVMSetTargetMachineFastISel).restype, LLVMSetTargetMachineFastISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineGlobalISel(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineGlobalISel:=dll.LLVMSetTargetMachineGlobalISel).restype, LLVMSetTargetMachineGlobalISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineGlobalISelAbort(LLVMTargetMachineRef T, LLVMGlobalISelAbortMode Mode)
try: (LLVMSetTargetMachineGlobalISelAbort:=dll.LLVMSetTargetMachineGlobalISelAbort).restype, LLVMSetTargetMachineGlobalISelAbort.argtypes = None, [LLVMTargetMachineRef, LLVMGlobalISelAbortMode]
except AttributeError: pass

# void LLVMSetTargetMachineMachineOutliner(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineMachineOutliner:=dll.LLVMSetTargetMachineMachineOutliner).restype, LLVMSetTargetMachineMachineOutliner.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# LLVMBool LLVMTargetMachineEmitToFile(LLVMTargetMachineRef T, LLVMModuleRef M, const char *Filename, LLVMCodeGenFileType codegen, char **ErrorMessage)
try: (LLVMTargetMachineEmitToFile:=dll.LLVMTargetMachineEmitToFile).restype, LLVMTargetMachineEmitToFile.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMTargetMachineEmitToMemoryBuffer(LLVMTargetMachineRef T, LLVMModuleRef M, LLVMCodeGenFileType codegen, char **ErrorMessage, LLVMMemoryBufferRef *OutMemBuf)
try: (LLVMTargetMachineEmitToMemoryBuffer:=dll.LLVMTargetMachineEmitToMemoryBuffer).restype, LLVMTargetMachineEmitToMemoryBuffer.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

# char *LLVMGetDefaultTargetTriple(void)
try: (LLVMGetDefaultTargetTriple:=dll.LLVMGetDefaultTargetTriple).restype, LLVMGetDefaultTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# char *LLVMNormalizeTargetTriple(const char *triple)
try: (LLVMNormalizeTargetTriple:=dll.LLVMNormalizeTargetTriple).restype, LLVMNormalizeTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *LLVMGetHostCPUName(void)
try: (LLVMGetHostCPUName:=dll.LLVMGetHostCPUName).restype, LLVMGetHostCPUName.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# char *LLVMGetHostCPUFeatures(void)
try: (LLVMGetHostCPUFeatures:=dll.LLVMGetHostCPUFeatures).restype, LLVMGetHostCPUFeatures.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# void LLVMAddAnalysisPasses(LLVMTargetMachineRef T, LLVMPassManagerRef PM)
try: (LLVMAddAnalysisPasses:=dll.LLVMAddAnalysisPasses).restype, LLVMAddAnalysisPasses.argtypes = None, [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError: pass

# LLVMErrorTypeId LLVMGetErrorTypeId(LLVMErrorRef Err)
try: (LLVMGetErrorTypeId:=dll.LLVMGetErrorTypeId).restype, LLVMGetErrorTypeId.argtypes = LLVMErrorTypeId, [LLVMErrorRef]
except AttributeError: pass

# void LLVMConsumeError(LLVMErrorRef Err)
try: (LLVMConsumeError:=dll.LLVMConsumeError).restype, LLVMConsumeError.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

# void LLVMCantFail(LLVMErrorRef Err)
try: (LLVMCantFail:=dll.LLVMCantFail).restype, LLVMCantFail.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

# char *LLVMGetErrorMessage(LLVMErrorRef Err)
try: (LLVMGetErrorMessage:=dll.LLVMGetErrorMessage).restype, LLVMGetErrorMessage.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMErrorRef]
except AttributeError: pass

# void LLVMDisposeErrorMessage(char *ErrMsg)
try: (LLVMDisposeErrorMessage:=dll.LLVMDisposeErrorMessage).restype, LLVMDisposeErrorMessage.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMErrorTypeId LLVMGetStringErrorTypeId(void)
try: (LLVMGetStringErrorTypeId:=dll.LLVMGetStringErrorTypeId).restype, LLVMGetStringErrorTypeId.argtypes = LLVMErrorTypeId, []
except AttributeError: pass

# LLVMErrorRef LLVMCreateStringError(const char *ErrMsg)
try: (LLVMCreateStringError:=dll.LLVMCreateStringError).restype, LLVMCreateStringError.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

# void LLVMInitializeAArch64TargetInfo(void)
try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetInfo(void)
try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetInfo(void)
try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetInfo(void)
try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetInfo(void)
try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetInfo(void)
try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetInfo(void)
try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetInfo(void)
try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetInfo(void)
try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetInfo(void)
try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetInfo(void)
try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetInfo(void)
try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetInfo(void)
try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetInfo(void)
try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetInfo(void)
try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetInfo(void)
try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetInfo(void)
try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetInfo(void)
try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetInfo(void)
try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetInfo(void)
try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetInfo(void)
try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetInfo(void)
try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Target(void)
try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTarget(void)
try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTarget(void)
try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTarget(void)
try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTarget(void)
try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTarget(void)
try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTarget(void)
try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTarget(void)
try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTarget(void)
try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Target(void)
try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTarget(void)
try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTarget(void)
try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTarget(void)
try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTarget(void)
try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTarget(void)
try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTarget(void)
try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETarget(void)
try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTarget(void)
try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Target(void)
try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTarget(void)
try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTarget(void)
try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTarget(void)
try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64TargetMC(void)
try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUTargetMC(void)
try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMTargetMC(void)
try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRTargetMC(void)
try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFTargetMC(void)
try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonTargetMC(void)
try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiTargetMC(void)
try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchTargetMC(void)
try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsTargetMC(void)
try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430TargetMC(void)
try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXTargetMC(void)
try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCTargetMC(void)
try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVTargetMC(void)
try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcTargetMC(void)
try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVTargetMC(void)
try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZTargetMC(void)
try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVETargetMC(void)
try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyTargetMC(void)
try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86TargetMC(void)
try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreTargetMC(void)
try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kTargetMC(void)
try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaTargetMC(void)
try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmPrinter(void)
try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmPrinter(void)
try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmPrinter(void)
try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmPrinter(void)
try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmPrinter(void)
try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmPrinter(void)
try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmPrinter(void)
try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmPrinter(void)
try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmPrinter(void)
try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmPrinter(void)
try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeNVPTXAsmPrinter(void)
try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmPrinter(void)
try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmPrinter(void)
try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmPrinter(void)
try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSPIRVAsmPrinter(void)
try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmPrinter(void)
try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmPrinter(void)
try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmPrinter(void)
try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmPrinter(void)
try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreAsmPrinter(void)
try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmPrinter(void)
try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmPrinter(void)
try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64AsmParser(void)
try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUAsmParser(void)
try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMAsmParser(void)
try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRAsmParser(void)
try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFAsmParser(void)
try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonAsmParser(void)
try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiAsmParser(void)
try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchAsmParser(void)
try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsAsmParser(void)
try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430AsmParser(void)
try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCAsmParser(void)
try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVAsmParser(void)
try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcAsmParser(void)
try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZAsmParser(void)
try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEAsmParser(void)
try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyAsmParser(void)
try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86AsmParser(void)
try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kAsmParser(void)
try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaAsmParser(void)
try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAArch64Disassembler(void)
try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAMDGPUDisassembler(void)
try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeARMDisassembler(void)
try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeAVRDisassembler(void)
try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeBPFDisassembler(void)
try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeHexagonDisassembler(void)
try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLanaiDisassembler(void)
try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeLoongArchDisassembler(void)
try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMipsDisassembler(void)
try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeMSP430Disassembler(void)
try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializePowerPCDisassembler(void)
try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeRISCVDisassembler(void)
try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSparcDisassembler(void)
try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeSystemZDisassembler(void)
try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeVEDisassembler(void)
try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeWebAssemblyDisassembler(void)
try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeX86Disassembler(void)
try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXCoreDisassembler(void)
try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeM68kDisassembler(void)
try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

# void LLVMInitializeXtensaDisassembler(void)
try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

# LLVMTargetDataRef LLVMGetModuleDataLayout(LLVMModuleRef M)
try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

# void LLVMSetModuleDataLayout(LLVMModuleRef M, LLVMTargetDataRef DL)
try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetData(const char *StringRep)
try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMDisposeTargetData(LLVMTargetDataRef TD)
try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

# void LLVMAddTargetLibraryInfo(LLVMTargetLibraryInfoRef TLI, LLVMPassManagerRef PM)
try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

# char *LLVMCopyStringRepOfTargetData(LLVMTargetDataRef TD)
try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

# enum LLVMByteOrdering LLVMByteOrder(LLVMTargetDataRef TD)
try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSize(LLVMTargetDataRef TD)
try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

# unsigned int LLVMPointerSizeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrType(LLVMTargetDataRef TD)
try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForAS(LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeInContext(LLVMContextRef C, LLVMTargetDataRef TD)
try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

# LLVMTypeRef LLVMIntPtrTypeForASInContext(LLVMContextRef C, LLVMTargetDataRef TD, unsigned int AS)
try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

# unsigned long long LLVMSizeOfTypeInBits(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMStoreSizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned long long LLVMABISizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMABIAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMCallFrameAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)
try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

# unsigned int LLVMPreferredAlignmentOfGlobal(LLVMTargetDataRef TD, LLVMValueRef GlobalVar)
try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

# unsigned int LLVMElementAtOffset(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned long long Offset)
try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

# unsigned long long LLVMOffsetOfElement(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned int Element)
try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

# LLVMTargetRef LLVMGetFirstTarget(void)
try: (LLVMGetFirstTarget:=dll.LLVMGetFirstTarget).restype, LLVMGetFirstTarget.argtypes = LLVMTargetRef, []
except AttributeError: pass

# LLVMTargetRef LLVMGetNextTarget(LLVMTargetRef T)
try: (LLVMGetNextTarget:=dll.LLVMGetNextTarget).restype, LLVMGetNextTarget.argtypes = LLVMTargetRef, [LLVMTargetRef]
except AttributeError: pass

# LLVMTargetRef LLVMGetTargetFromName(const char *Name)
try: (LLVMGetTargetFromName:=dll.LLVMGetTargetFromName).restype, LLVMGetTargetFromName.argtypes = LLVMTargetRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# LLVMBool LLVMGetTargetFromTriple(const char *Triple, LLVMTargetRef *T, char **ErrorMessage)
try: (LLVMGetTargetFromTriple:=dll.LLVMGetTargetFromTriple).restype, LLVMGetTargetFromTriple.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTargetRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# const char *LLVMGetTargetName(LLVMTargetRef T)
try: (LLVMGetTargetName:=dll.LLVMGetTargetName).restype, LLVMGetTargetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

# const char *LLVMGetTargetDescription(LLVMTargetRef T)
try: (LLVMGetTargetDescription:=dll.LLVMGetTargetDescription).restype, LLVMGetTargetDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasJIT(LLVMTargetRef T)
try: (LLVMTargetHasJIT:=dll.LLVMTargetHasJIT).restype, LLVMTargetHasJIT.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasTargetMachine(LLVMTargetRef T)
try: (LLVMTargetHasTargetMachine:=dll.LLVMTargetHasTargetMachine).restype, LLVMTargetHasTargetMachine.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMBool LLVMTargetHasAsmBackend(LLVMTargetRef T)
try: (LLVMTargetHasAsmBackend:=dll.LLVMTargetHasAsmBackend).restype, LLVMTargetHasAsmBackend.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

# LLVMTargetMachineOptionsRef LLVMCreateTargetMachineOptions(void)
try: (LLVMCreateTargetMachineOptions:=dll.LLVMCreateTargetMachineOptions).restype, LLVMCreateTargetMachineOptions.argtypes = LLVMTargetMachineOptionsRef, []
except AttributeError: pass

# void LLVMDisposeTargetMachineOptions(LLVMTargetMachineOptionsRef Options)
try: (LLVMDisposeTargetMachineOptions:=dll.LLVMDisposeTargetMachineOptions).restype, LLVMDisposeTargetMachineOptions.argtypes = None, [LLVMTargetMachineOptionsRef]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCPU(LLVMTargetMachineOptionsRef Options, const char *CPU)
try: (LLVMTargetMachineOptionsSetCPU:=dll.LLVMTargetMachineOptionsSetCPU).restype, LLVMTargetMachineOptionsSetCPU.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetFeatures(LLVMTargetMachineOptionsRef Options, const char *Features)
try: (LLVMTargetMachineOptionsSetFeatures:=dll.LLVMTargetMachineOptionsSetFeatures).restype, LLVMTargetMachineOptionsSetFeatures.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetABI(LLVMTargetMachineOptionsRef Options, const char *ABI)
try: (LLVMTargetMachineOptionsSetABI:=dll.LLVMTargetMachineOptionsSetABI).restype, LLVMTargetMachineOptionsSetABI.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCodeGenOptLevel(LLVMTargetMachineOptionsRef Options, LLVMCodeGenOptLevel Level)
try: (LLVMTargetMachineOptionsSetCodeGenOptLevel:=dll.LLVMTargetMachineOptionsSetCodeGenOptLevel).restype, LLVMTargetMachineOptionsSetCodeGenOptLevel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeGenOptLevel]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetRelocMode(LLVMTargetMachineOptionsRef Options, LLVMRelocMode Reloc)
try: (LLVMTargetMachineOptionsSetRelocMode:=dll.LLVMTargetMachineOptionsSetRelocMode).restype, LLVMTargetMachineOptionsSetRelocMode.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMRelocMode]
except AttributeError: pass

# void LLVMTargetMachineOptionsSetCodeModel(LLVMTargetMachineOptionsRef Options, LLVMCodeModel CodeModel)
try: (LLVMTargetMachineOptionsSetCodeModel:=dll.LLVMTargetMachineOptionsSetCodeModel).restype, LLVMTargetMachineOptionsSetCodeModel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeModel]
except AttributeError: pass

# LLVMTargetMachineRef LLVMCreateTargetMachineWithOptions(LLVMTargetRef T, const char *Triple, LLVMTargetMachineOptionsRef Options)
try: (LLVMCreateTargetMachineWithOptions:=dll.LLVMCreateTargetMachineWithOptions).restype, LLVMCreateTargetMachineWithOptions.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineOptionsRef]
except AttributeError: pass

# LLVMTargetMachineRef LLVMCreateTargetMachine(LLVMTargetRef T, const char *Triple, const char *CPU, const char *Features, LLVMCodeGenOptLevel Level, LLVMRelocMode Reloc, LLVMCodeModel CodeModel)
try: (LLVMCreateTargetMachine:=dll.LLVMCreateTargetMachine).restype, LLVMCreateTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError: pass

# void LLVMDisposeTargetMachine(LLVMTargetMachineRef T)
try: (LLVMDisposeTargetMachine:=dll.LLVMDisposeTargetMachine).restype, LLVMDisposeTargetMachine.argtypes = None, [LLVMTargetMachineRef]
except AttributeError: pass

# LLVMTargetRef LLVMGetTargetMachineTarget(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineTarget:=dll.LLVMGetTargetMachineTarget).restype, LLVMGetTargetMachineTarget.argtypes = LLVMTargetRef, [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineTriple(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineTriple:=dll.LLVMGetTargetMachineTriple).restype, LLVMGetTargetMachineTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineCPU(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineCPU:=dll.LLVMGetTargetMachineCPU).restype, LLVMGetTargetMachineCPU.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# char *LLVMGetTargetMachineFeatureString(LLVMTargetMachineRef T)
try: (LLVMGetTargetMachineFeatureString:=dll.LLVMGetTargetMachineFeatureString).restype, LLVMGetTargetMachineFeatureString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

# LLVMTargetDataRef LLVMCreateTargetDataLayout(LLVMTargetMachineRef T)
try: (LLVMCreateTargetDataLayout:=dll.LLVMCreateTargetDataLayout).restype, LLVMCreateTargetDataLayout.argtypes = LLVMTargetDataRef, [LLVMTargetMachineRef]
except AttributeError: pass

# void LLVMSetTargetMachineAsmVerbosity(LLVMTargetMachineRef T, LLVMBool VerboseAsm)
try: (LLVMSetTargetMachineAsmVerbosity:=dll.LLVMSetTargetMachineAsmVerbosity).restype, LLVMSetTargetMachineAsmVerbosity.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineFastISel(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineFastISel:=dll.LLVMSetTargetMachineFastISel).restype, LLVMSetTargetMachineFastISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineGlobalISel(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineGlobalISel:=dll.LLVMSetTargetMachineGlobalISel).restype, LLVMSetTargetMachineGlobalISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# void LLVMSetTargetMachineGlobalISelAbort(LLVMTargetMachineRef T, LLVMGlobalISelAbortMode Mode)
try: (LLVMSetTargetMachineGlobalISelAbort:=dll.LLVMSetTargetMachineGlobalISelAbort).restype, LLVMSetTargetMachineGlobalISelAbort.argtypes = None, [LLVMTargetMachineRef, LLVMGlobalISelAbortMode]
except AttributeError: pass

# void LLVMSetTargetMachineMachineOutliner(LLVMTargetMachineRef T, LLVMBool Enable)
try: (LLVMSetTargetMachineMachineOutliner:=dll.LLVMSetTargetMachineMachineOutliner).restype, LLVMSetTargetMachineMachineOutliner.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

# LLVMBool LLVMTargetMachineEmitToFile(LLVMTargetMachineRef T, LLVMModuleRef M, const char *Filename, LLVMCodeGenFileType codegen, char **ErrorMessage)
try: (LLVMTargetMachineEmitToFile:=dll.LLVMTargetMachineEmitToFile).restype, LLVMTargetMachineEmitToFile.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# LLVMBool LLVMTargetMachineEmitToMemoryBuffer(LLVMTargetMachineRef T, LLVMModuleRef M, LLVMCodeGenFileType codegen, char **ErrorMessage, LLVMMemoryBufferRef *OutMemBuf)
try: (LLVMTargetMachineEmitToMemoryBuffer:=dll.LLVMTargetMachineEmitToMemoryBuffer).restype, LLVMTargetMachineEmitToMemoryBuffer.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

# char *LLVMGetDefaultTargetTriple(void)
try: (LLVMGetDefaultTargetTriple:=dll.LLVMGetDefaultTargetTriple).restype, LLVMGetDefaultTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# char *LLVMNormalizeTargetTriple(const char *triple)
try: (LLVMNormalizeTargetTriple:=dll.LLVMNormalizeTargetTriple).restype, LLVMNormalizeTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *LLVMGetHostCPUName(void)
try: (LLVMGetHostCPUName:=dll.LLVMGetHostCPUName).restype, LLVMGetHostCPUName.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# char *LLVMGetHostCPUFeatures(void)
try: (LLVMGetHostCPUFeatures:=dll.LLVMGetHostCPUFeatures).restype, LLVMGetHostCPUFeatures.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# void LLVMAddAnalysisPasses(LLVMTargetMachineRef T, LLVMPassManagerRef PM)
try: (LLVMAddAnalysisPasses:=dll.LLVMAddAnalysisPasses).restype, LLVMAddAnalysisPasses.argtypes = None, [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError: pass

class struct_LLVMOpaquePassBuilderOptions(Struct): pass
LLVMPassBuilderOptionsRef = ctypes.POINTER(struct_LLVMOpaquePassBuilderOptions)
# LLVMErrorRef LLVMRunPasses(LLVMModuleRef M, const char *Passes, LLVMTargetMachineRef TM, LLVMPassBuilderOptionsRef Options)
try: (LLVMRunPasses:=dll.LLVMRunPasses).restype, LLVMRunPasses.argtypes = LLVMErrorRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineRef, LLVMPassBuilderOptionsRef]
except AttributeError: pass

# LLVMErrorRef LLVMRunPassesOnFunction(LLVMValueRef F, const char *Passes, LLVMTargetMachineRef TM, LLVMPassBuilderOptionsRef Options)
try: (LLVMRunPassesOnFunction:=dll.LLVMRunPassesOnFunction).restype, LLVMRunPassesOnFunction.argtypes = LLVMErrorRef, [LLVMValueRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineRef, LLVMPassBuilderOptionsRef]
except AttributeError: pass

# LLVMPassBuilderOptionsRef LLVMCreatePassBuilderOptions(void)
try: (LLVMCreatePassBuilderOptions:=dll.LLVMCreatePassBuilderOptions).restype, LLVMCreatePassBuilderOptions.argtypes = LLVMPassBuilderOptionsRef, []
except AttributeError: pass

# void LLVMPassBuilderOptionsSetVerifyEach(LLVMPassBuilderOptionsRef Options, LLVMBool VerifyEach)
try: (LLVMPassBuilderOptionsSetVerifyEach:=dll.LLVMPassBuilderOptionsSetVerifyEach).restype, LLVMPassBuilderOptionsSetVerifyEach.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

# void LLVMPassBuilderOptionsSetDebugLogging(LLVMPassBuilderOptionsRef Options, LLVMBool DebugLogging)
try: (LLVMPassBuilderOptionsSetDebugLogging:=dll.LLVMPassBuilderOptionsSetDebugLogging).restype, LLVMPassBuilderOptionsSetDebugLogging.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

# void LLVMPassBuilderOptionsSetAAPipeline(LLVMPassBuilderOptionsRef Options, const char *AAPipeline)
try: (LLVMPassBuilderOptionsSetAAPipeline:=dll.LLVMPassBuilderOptionsSetAAPipeline).restype, LLVMPassBuilderOptionsSetAAPipeline.argtypes = None, [LLVMPassBuilderOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void LLVMPassBuilderOptionsSetLoopInterleaving(LLVMPassBuilderOptionsRef Options, LLVMBool LoopInterleaving)
try: (LLVMPassBuilderOptionsSetLoopInterleaving:=dll.LLVMPassBuilderOptionsSetLoopInterleaving).restype, LLVMPassBuilderOptionsSetLoopInterleaving.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

# void LLVMPassBuilderOptionsSetLoopVectorization(LLVMPassBuilderOptionsRef Options, LLVMBool LoopVectorization)
try: (LLVMPassBuilderOptionsSetLoopVectorization:=dll.LLVMPassBuilderOptionsSetLoopVectorization).restype, LLVMPassBuilderOptionsSetLoopVectorization.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

# void LLVMPassBuilderOptionsSetSLPVectorization(LLVMPassBuilderOptionsRef Options, LLVMBool SLPVectorization)
try: (LLVMPassBuilderOptionsSetSLPVectorization:=dll.LLVMPassBuilderOptionsSetSLPVectorization).restype, LLVMPassBuilderOptionsSetSLPVectorization.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

# void LLVMPassBuilderOptionsSetLoopUnrolling(LLVMPassBuilderOptionsRef Options, LLVMBool LoopUnrolling)
try: (LLVMPassBuilderOptionsSetLoopUnrolling:=dll.LLVMPassBuilderOptionsSetLoopUnrolling).restype, LLVMPassBuilderOptionsSetLoopUnrolling.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

# void LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll(LLVMPassBuilderOptionsRef Options, LLVMBool ForgetAllSCEVInLoopUnroll)
try: (LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll:=dll.LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll).restype, LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

# void LLVMPassBuilderOptionsSetLicmMssaOptCap(LLVMPassBuilderOptionsRef Options, unsigned int LicmMssaOptCap)
try: (LLVMPassBuilderOptionsSetLicmMssaOptCap:=dll.LLVMPassBuilderOptionsSetLicmMssaOptCap).restype, LLVMPassBuilderOptionsSetLicmMssaOptCap.argtypes = None, [LLVMPassBuilderOptionsRef, ctypes.c_uint32]
except AttributeError: pass

# void LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap(LLVMPassBuilderOptionsRef Options, unsigned int LicmMssaNoAccForPromotionCap)
try: (LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap:=dll.LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap).restype, LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap.argtypes = None, [LLVMPassBuilderOptionsRef, ctypes.c_uint32]
except AttributeError: pass

# void LLVMPassBuilderOptionsSetCallGraphProfile(LLVMPassBuilderOptionsRef Options, LLVMBool CallGraphProfile)
try: (LLVMPassBuilderOptionsSetCallGraphProfile:=dll.LLVMPassBuilderOptionsSetCallGraphProfile).restype, LLVMPassBuilderOptionsSetCallGraphProfile.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

# void LLVMPassBuilderOptionsSetMergeFunctions(LLVMPassBuilderOptionsRef Options, LLVMBool MergeFunctions)
try: (LLVMPassBuilderOptionsSetMergeFunctions:=dll.LLVMPassBuilderOptionsSetMergeFunctions).restype, LLVMPassBuilderOptionsSetMergeFunctions.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

# void LLVMPassBuilderOptionsSetInlinerThreshold(LLVMPassBuilderOptionsRef Options, int Threshold)
try: (LLVMPassBuilderOptionsSetInlinerThreshold:=dll.LLVMPassBuilderOptionsSetInlinerThreshold).restype, LLVMPassBuilderOptionsSetInlinerThreshold.argtypes = None, [LLVMPassBuilderOptionsRef, ctypes.c_int32]
except AttributeError: pass

# void LLVMDisposePassBuilderOptions(LLVMPassBuilderOptionsRef Options)
try: (LLVMDisposePassBuilderOptions:=dll.LLVMDisposePassBuilderOptions).restype, LLVMDisposePassBuilderOptions.argtypes = None, [LLVMPassBuilderOptionsRef]
except AttributeError: pass

# extern intmax_t imaxabs(intmax_t __n) __attribute__((nothrow)) __attribute__((const))
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

# extern imaxdiv_t imaxdiv(intmax_t __numer, intmax_t __denom) __attribute__((nothrow)) __attribute__((const))
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t strtoimax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoimax") __attribute__((nothrow))
try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t strtoumax(const char *restrict __nptr, char **restrict __endptr, int __base) asm("__isoc23_strtoumax") __attribute__((nothrow))
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern intmax_t wcstoimax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoimax") __attribute__((nothrow))
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern uintmax_t wcstoumax(const __gwchar_t *restrict __nptr, __gwchar_t **restrict __endptr, int __base) asm("__isoc23_wcstoumax") __attribute__((nothrow))
try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

class llvm_blake3_chunk_state(Struct): pass
llvm_blake3_chunk_state._fields_ = [
  ('cv', (uint32_t * 8)),
  ('chunk_counter', uint64_t),
  ('buf', (uint8_t * 64)),
  ('buf_len', uint8_t),
  ('blocks_compressed', uint8_t),
  ('flags', uint8_t),
]
class llvm_blake3_hasher(Struct): pass
llvm_blake3_hasher._fields_ = [
  ('key', (uint32_t * 8)),
  ('chunk', llvm_blake3_chunk_state),
  ('cv_stack_len', uint8_t),
  ('cv_stack', (uint8_t * 1760)),
]
# const char *llvm_blake3_version(void)
try: (llvm_blake3_version:=dll.llvm_blake3_version).restype, llvm_blake3_version.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# void llvm_blake3_hasher_init(llvm_blake3_hasher *self)
try: (llvm_blake3_hasher_init:=dll.llvm_blake3_hasher_init).restype, llvm_blake3_hasher_init.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher)]
except AttributeError: pass

# void llvm_blake3_hasher_init_keyed(llvm_blake3_hasher *self, const uint8_t key[32])
try: (llvm_blake3_hasher_init_keyed:=dll.llvm_blake3_hasher_init_keyed).restype, llvm_blake3_hasher_init_keyed.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher), (uint8_t * 32)]
except AttributeError: pass

# void llvm_blake3_hasher_init_derive_key(llvm_blake3_hasher *self, const char *context)
try: (llvm_blake3_hasher_init_derive_key:=dll.llvm_blake3_hasher_init_derive_key).restype, llvm_blake3_hasher_init_derive_key.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void llvm_blake3_hasher_init_derive_key_raw(llvm_blake3_hasher *self, const void *context, size_t context_len)
try: (llvm_blake3_hasher_init_derive_key_raw:=dll.llvm_blake3_hasher_init_derive_key_raw).restype, llvm_blake3_hasher_init_derive_key_raw.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher), ctypes.c_void_p, size_t]
except AttributeError: pass

# void llvm_blake3_hasher_update(llvm_blake3_hasher *self, const void *input, size_t input_len)
try: (llvm_blake3_hasher_update:=dll.llvm_blake3_hasher_update).restype, llvm_blake3_hasher_update.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher), ctypes.c_void_p, size_t]
except AttributeError: pass

# void llvm_blake3_hasher_finalize(const llvm_blake3_hasher *self, uint8_t *out, size_t out_len)
try: (llvm_blake3_hasher_finalize:=dll.llvm_blake3_hasher_finalize).restype, llvm_blake3_hasher_finalize.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher), ctypes.POINTER(uint8_t), size_t]
except AttributeError: pass

# void llvm_blake3_hasher_finalize_seek(const llvm_blake3_hasher *self, uint64_t seek, uint8_t *out, size_t out_len)
try: (llvm_blake3_hasher_finalize_seek:=dll.llvm_blake3_hasher_finalize_seek).restype, llvm_blake3_hasher_finalize_seek.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher), uint64_t, ctypes.POINTER(uint8_t), size_t]
except AttributeError: pass

# void llvm_blake3_hasher_reset(llvm_blake3_hasher *self)
try: (llvm_blake3_hasher_reset:=dll.llvm_blake3_hasher_reset).restype, llvm_blake3_hasher_reset.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher)]
except AttributeError: pass

# extern int select(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, struct timeval *restrict __timeout)
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# extern int pselect(int __nfds, fd_set *restrict __readfds, fd_set *restrict __writefds, fd_set *restrict __exceptfds, const struct timespec *restrict __timeout, const __sigset_t *restrict __sigmask)
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

lto_bool_t = ctypes.c_bool
lto_symbol_attributes = CEnum(ctypes.c_uint32)
LTO_SYMBOL_ALIGNMENT_MASK = lto_symbol_attributes.define('LTO_SYMBOL_ALIGNMENT_MASK', 31)
LTO_SYMBOL_PERMISSIONS_MASK = lto_symbol_attributes.define('LTO_SYMBOL_PERMISSIONS_MASK', 224)
LTO_SYMBOL_PERMISSIONS_CODE = lto_symbol_attributes.define('LTO_SYMBOL_PERMISSIONS_CODE', 160)
LTO_SYMBOL_PERMISSIONS_DATA = lto_symbol_attributes.define('LTO_SYMBOL_PERMISSIONS_DATA', 192)
LTO_SYMBOL_PERMISSIONS_RODATA = lto_symbol_attributes.define('LTO_SYMBOL_PERMISSIONS_RODATA', 128)
LTO_SYMBOL_DEFINITION_MASK = lto_symbol_attributes.define('LTO_SYMBOL_DEFINITION_MASK', 1792)
LTO_SYMBOL_DEFINITION_REGULAR = lto_symbol_attributes.define('LTO_SYMBOL_DEFINITION_REGULAR', 256)
LTO_SYMBOL_DEFINITION_TENTATIVE = lto_symbol_attributes.define('LTO_SYMBOL_DEFINITION_TENTATIVE', 512)
LTO_SYMBOL_DEFINITION_WEAK = lto_symbol_attributes.define('LTO_SYMBOL_DEFINITION_WEAK', 768)
LTO_SYMBOL_DEFINITION_UNDEFINED = lto_symbol_attributes.define('LTO_SYMBOL_DEFINITION_UNDEFINED', 1024)
LTO_SYMBOL_DEFINITION_WEAKUNDEF = lto_symbol_attributes.define('LTO_SYMBOL_DEFINITION_WEAKUNDEF', 1280)
LTO_SYMBOL_SCOPE_MASK = lto_symbol_attributes.define('LTO_SYMBOL_SCOPE_MASK', 14336)
LTO_SYMBOL_SCOPE_INTERNAL = lto_symbol_attributes.define('LTO_SYMBOL_SCOPE_INTERNAL', 2048)
LTO_SYMBOL_SCOPE_HIDDEN = lto_symbol_attributes.define('LTO_SYMBOL_SCOPE_HIDDEN', 4096)
LTO_SYMBOL_SCOPE_PROTECTED = lto_symbol_attributes.define('LTO_SYMBOL_SCOPE_PROTECTED', 8192)
LTO_SYMBOL_SCOPE_DEFAULT = lto_symbol_attributes.define('LTO_SYMBOL_SCOPE_DEFAULT', 6144)
LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN = lto_symbol_attributes.define('LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN', 10240)
LTO_SYMBOL_COMDAT = lto_symbol_attributes.define('LTO_SYMBOL_COMDAT', 16384)
LTO_SYMBOL_ALIAS = lto_symbol_attributes.define('LTO_SYMBOL_ALIAS', 32768)

lto_debug_model = CEnum(ctypes.c_uint32)
LTO_DEBUG_MODEL_NONE = lto_debug_model.define('LTO_DEBUG_MODEL_NONE', 0)
LTO_DEBUG_MODEL_DWARF = lto_debug_model.define('LTO_DEBUG_MODEL_DWARF', 1)

lto_codegen_model = CEnum(ctypes.c_uint32)
LTO_CODEGEN_PIC_MODEL_STATIC = lto_codegen_model.define('LTO_CODEGEN_PIC_MODEL_STATIC', 0)
LTO_CODEGEN_PIC_MODEL_DYNAMIC = lto_codegen_model.define('LTO_CODEGEN_PIC_MODEL_DYNAMIC', 1)
LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC = lto_codegen_model.define('LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC', 2)
LTO_CODEGEN_PIC_MODEL_DEFAULT = lto_codegen_model.define('LTO_CODEGEN_PIC_MODEL_DEFAULT', 3)

class struct_LLVMOpaqueLTOModule(Struct): pass
lto_module_t = ctypes.POINTER(struct_LLVMOpaqueLTOModule)
class struct_LLVMOpaqueLTOCodeGenerator(Struct): pass
lto_code_gen_t = ctypes.POINTER(struct_LLVMOpaqueLTOCodeGenerator)
class struct_LLVMOpaqueThinLTOCodeGenerator(Struct): pass
thinlto_code_gen_t = ctypes.POINTER(struct_LLVMOpaqueThinLTOCodeGenerator)
# extern const char *lto_get_version(void)
try: (lto_get_version:=dll.lto_get_version).restype, lto_get_version.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# extern const char *lto_get_error_message(void)
try: (lto_get_error_message:=dll.lto_get_error_message).restype, lto_get_error_message.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# extern lto_bool_t lto_module_is_object_file(const char *path)
try: (lto_module_is_object_file:=dll.lto_module_is_object_file).restype, lto_module_is_object_file.argtypes = lto_bool_t, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern lto_bool_t lto_module_is_object_file_for_target(const char *path, const char *target_triple_prefix)
try: (lto_module_is_object_file_for_target:=dll.lto_module_is_object_file_for_target).restype, lto_module_is_object_file_for_target.argtypes = lto_bool_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern lto_bool_t lto_module_has_objc_category(const void *mem, size_t length)
try: (lto_module_has_objc_category:=dll.lto_module_has_objc_category).restype, lto_module_has_objc_category.argtypes = lto_bool_t, [ctypes.c_void_p, size_t]
except AttributeError: pass

# extern lto_bool_t lto_module_is_object_file_in_memory(const void *mem, size_t length)
try: (lto_module_is_object_file_in_memory:=dll.lto_module_is_object_file_in_memory).restype, lto_module_is_object_file_in_memory.argtypes = lto_bool_t, [ctypes.c_void_p, size_t]
except AttributeError: pass

# extern lto_bool_t lto_module_is_object_file_in_memory_for_target(const void *mem, size_t length, const char *target_triple_prefix)
try: (lto_module_is_object_file_in_memory_for_target:=dll.lto_module_is_object_file_in_memory_for_target).restype, lto_module_is_object_file_in_memory_for_target.argtypes = lto_bool_t, [ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern lto_module_t lto_module_create(const char *path)
try: (lto_module_create:=dll.lto_module_create).restype, lto_module_create.argtypes = lto_module_t, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern lto_module_t lto_module_create_from_memory(const void *mem, size_t length)
try: (lto_module_create_from_memory:=dll.lto_module_create_from_memory).restype, lto_module_create_from_memory.argtypes = lto_module_t, [ctypes.c_void_p, size_t]
except AttributeError: pass

# extern lto_module_t lto_module_create_from_memory_with_path(const void *mem, size_t length, const char *path)
try: (lto_module_create_from_memory_with_path:=dll.lto_module_create_from_memory_with_path).restype, lto_module_create_from_memory_with_path.argtypes = lto_module_t, [ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern lto_module_t lto_module_create_in_local_context(const void *mem, size_t length, const char *path)
try: (lto_module_create_in_local_context:=dll.lto_module_create_in_local_context).restype, lto_module_create_in_local_context.argtypes = lto_module_t, [ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern lto_module_t lto_module_create_in_codegen_context(const void *mem, size_t length, const char *path, lto_code_gen_t cg)
try: (lto_module_create_in_codegen_context:=dll.lto_module_create_in_codegen_context).restype, lto_module_create_in_codegen_context.argtypes = lto_module_t, [ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char), lto_code_gen_t]
except AttributeError: pass

# extern lto_module_t lto_module_create_from_fd(int fd, const char *path, size_t file_size)
try: (lto_module_create_from_fd:=dll.lto_module_create_from_fd).restype, lto_module_create_from_fd.argtypes = lto_module_t, [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

off_t = ctypes.c_int64
# extern lto_module_t lto_module_create_from_fd_at_offset(int fd, const char *path, size_t file_size, size_t map_size, off_t offset)
try: (lto_module_create_from_fd_at_offset:=dll.lto_module_create_from_fd_at_offset).restype, lto_module_create_from_fd_at_offset.argtypes = lto_module_t, [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t, size_t, off_t]
except AttributeError: pass

# extern void lto_module_dispose(lto_module_t mod)
try: (lto_module_dispose:=dll.lto_module_dispose).restype, lto_module_dispose.argtypes = None, [lto_module_t]
except AttributeError: pass

# extern const char *lto_module_get_target_triple(lto_module_t mod)
try: (lto_module_get_target_triple:=dll.lto_module_get_target_triple).restype, lto_module_get_target_triple.argtypes = ctypes.POINTER(ctypes.c_char), [lto_module_t]
except AttributeError: pass

# extern void lto_module_set_target_triple(lto_module_t mod, const char *triple)
try: (lto_module_set_target_triple:=dll.lto_module_set_target_triple).restype, lto_module_set_target_triple.argtypes = None, [lto_module_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern unsigned int lto_module_get_num_symbols(lto_module_t mod)
try: (lto_module_get_num_symbols:=dll.lto_module_get_num_symbols).restype, lto_module_get_num_symbols.argtypes = ctypes.c_uint32, [lto_module_t]
except AttributeError: pass

# extern const char *lto_module_get_symbol_name(lto_module_t mod, unsigned int index)
try: (lto_module_get_symbol_name:=dll.lto_module_get_symbol_name).restype, lto_module_get_symbol_name.argtypes = ctypes.POINTER(ctypes.c_char), [lto_module_t, ctypes.c_uint32]
except AttributeError: pass

# extern lto_symbol_attributes lto_module_get_symbol_attribute(lto_module_t mod, unsigned int index)
try: (lto_module_get_symbol_attribute:=dll.lto_module_get_symbol_attribute).restype, lto_module_get_symbol_attribute.argtypes = lto_symbol_attributes, [lto_module_t, ctypes.c_uint32]
except AttributeError: pass

# extern const char *lto_module_get_linkeropts(lto_module_t mod)
try: (lto_module_get_linkeropts:=dll.lto_module_get_linkeropts).restype, lto_module_get_linkeropts.argtypes = ctypes.POINTER(ctypes.c_char), [lto_module_t]
except AttributeError: pass

# extern lto_bool_t lto_module_get_macho_cputype(lto_module_t mod, unsigned int *out_cputype, unsigned int *out_cpusubtype)
try: (lto_module_get_macho_cputype:=dll.lto_module_get_macho_cputype).restype, lto_module_get_macho_cputype.argtypes = lto_bool_t, [lto_module_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

# extern lto_bool_t lto_module_has_ctor_dtor(lto_module_t mod)
try: (lto_module_has_ctor_dtor:=dll.lto_module_has_ctor_dtor).restype, lto_module_has_ctor_dtor.argtypes = lto_bool_t, [lto_module_t]
except AttributeError: pass

lto_codegen_diagnostic_severity_t = CEnum(ctypes.c_uint32)
LTO_DS_ERROR = lto_codegen_diagnostic_severity_t.define('LTO_DS_ERROR', 0)
LTO_DS_WARNING = lto_codegen_diagnostic_severity_t.define('LTO_DS_WARNING', 1)
LTO_DS_REMARK = lto_codegen_diagnostic_severity_t.define('LTO_DS_REMARK', 3)
LTO_DS_NOTE = lto_codegen_diagnostic_severity_t.define('LTO_DS_NOTE', 2)

lto_diagnostic_handler_t = ctypes.CFUNCTYPE(None, lto_codegen_diagnostic_severity_t, ctypes.POINTER(ctypes.c_char), ctypes.c_void_p)
# extern void lto_codegen_set_diagnostic_handler(lto_code_gen_t, lto_diagnostic_handler_t, void *)
try: (lto_codegen_set_diagnostic_handler:=dll.lto_codegen_set_diagnostic_handler).restype, lto_codegen_set_diagnostic_handler.argtypes = None, [lto_code_gen_t, lto_diagnostic_handler_t, ctypes.c_void_p]
except AttributeError: pass

# extern lto_code_gen_t lto_codegen_create(void)
try: (lto_codegen_create:=dll.lto_codegen_create).restype, lto_codegen_create.argtypes = lto_code_gen_t, []
except AttributeError: pass

# extern lto_code_gen_t lto_codegen_create_in_local_context(void)
try: (lto_codegen_create_in_local_context:=dll.lto_codegen_create_in_local_context).restype, lto_codegen_create_in_local_context.argtypes = lto_code_gen_t, []
except AttributeError: pass

# extern void lto_codegen_dispose(lto_code_gen_t)
try: (lto_codegen_dispose:=dll.lto_codegen_dispose).restype, lto_codegen_dispose.argtypes = None, [lto_code_gen_t]
except AttributeError: pass

# extern lto_bool_t lto_codegen_add_module(lto_code_gen_t cg, lto_module_t mod)
try: (lto_codegen_add_module:=dll.lto_codegen_add_module).restype, lto_codegen_add_module.argtypes = lto_bool_t, [lto_code_gen_t, lto_module_t]
except AttributeError: pass

# extern void lto_codegen_set_module(lto_code_gen_t cg, lto_module_t mod)
try: (lto_codegen_set_module:=dll.lto_codegen_set_module).restype, lto_codegen_set_module.argtypes = None, [lto_code_gen_t, lto_module_t]
except AttributeError: pass

# extern lto_bool_t lto_codegen_set_debug_model(lto_code_gen_t cg, lto_debug_model)
try: (lto_codegen_set_debug_model:=dll.lto_codegen_set_debug_model).restype, lto_codegen_set_debug_model.argtypes = lto_bool_t, [lto_code_gen_t, lto_debug_model]
except AttributeError: pass

# extern lto_bool_t lto_codegen_set_pic_model(lto_code_gen_t cg, lto_codegen_model)
try: (lto_codegen_set_pic_model:=dll.lto_codegen_set_pic_model).restype, lto_codegen_set_pic_model.argtypes = lto_bool_t, [lto_code_gen_t, lto_codegen_model]
except AttributeError: pass

# extern void lto_codegen_set_cpu(lto_code_gen_t cg, const char *cpu)
try: (lto_codegen_set_cpu:=dll.lto_codegen_set_cpu).restype, lto_codegen_set_cpu.argtypes = None, [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern void lto_codegen_set_assembler_path(lto_code_gen_t cg, const char *path)
try: (lto_codegen_set_assembler_path:=dll.lto_codegen_set_assembler_path).restype, lto_codegen_set_assembler_path.argtypes = None, [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern void lto_codegen_set_assembler_args(lto_code_gen_t cg, const char **args, int nargs)
try: (lto_codegen_set_assembler_args:=dll.lto_codegen_set_assembler_args).restype, lto_codegen_set_assembler_args.argtypes = None, [lto_code_gen_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern void lto_codegen_add_must_preserve_symbol(lto_code_gen_t cg, const char *symbol)
try: (lto_codegen_add_must_preserve_symbol:=dll.lto_codegen_add_must_preserve_symbol).restype, lto_codegen_add_must_preserve_symbol.argtypes = None, [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern lto_bool_t lto_codegen_write_merged_modules(lto_code_gen_t cg, const char *path)
try: (lto_codegen_write_merged_modules:=dll.lto_codegen_write_merged_modules).restype, lto_codegen_write_merged_modules.argtypes = lto_bool_t, [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern const void *lto_codegen_compile(lto_code_gen_t cg, size_t *length)
try: (lto_codegen_compile:=dll.lto_codegen_compile).restype, lto_codegen_compile.argtypes = ctypes.c_void_p, [lto_code_gen_t, ctypes.POINTER(size_t)]
except AttributeError: pass

# extern lto_bool_t lto_codegen_compile_to_file(lto_code_gen_t cg, const char **name)
try: (lto_codegen_compile_to_file:=dll.lto_codegen_compile_to_file).restype, lto_codegen_compile_to_file.argtypes = lto_bool_t, [lto_code_gen_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# extern lto_bool_t lto_codegen_optimize(lto_code_gen_t cg)
try: (lto_codegen_optimize:=dll.lto_codegen_optimize).restype, lto_codegen_optimize.argtypes = lto_bool_t, [lto_code_gen_t]
except AttributeError: pass

# extern const void *lto_codegen_compile_optimized(lto_code_gen_t cg, size_t *length)
try: (lto_codegen_compile_optimized:=dll.lto_codegen_compile_optimized).restype, lto_codegen_compile_optimized.argtypes = ctypes.c_void_p, [lto_code_gen_t, ctypes.POINTER(size_t)]
except AttributeError: pass

# extern unsigned int lto_api_version(void)
try: (lto_api_version:=dll.lto_api_version).restype, lto_api_version.argtypes = ctypes.c_uint32, []
except AttributeError: pass

# extern void lto_set_debug_options(const char *const *options, int number)
try: (lto_set_debug_options:=dll.lto_set_debug_options).restype, lto_set_debug_options.argtypes = None, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern void lto_codegen_debug_options(lto_code_gen_t cg, const char *)
try: (lto_codegen_debug_options:=dll.lto_codegen_debug_options).restype, lto_codegen_debug_options.argtypes = None, [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern void lto_codegen_debug_options_array(lto_code_gen_t cg, const char *const *, int number)
try: (lto_codegen_debug_options_array:=dll.lto_codegen_debug_options_array).restype, lto_codegen_debug_options_array.argtypes = None, [lto_code_gen_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern void lto_initialize_disassembler(void)
try: (lto_initialize_disassembler:=dll.lto_initialize_disassembler).restype, lto_initialize_disassembler.argtypes = None, []
except AttributeError: pass

# extern void lto_codegen_set_should_internalize(lto_code_gen_t cg, lto_bool_t ShouldInternalize)
try: (lto_codegen_set_should_internalize:=dll.lto_codegen_set_should_internalize).restype, lto_codegen_set_should_internalize.argtypes = None, [lto_code_gen_t, lto_bool_t]
except AttributeError: pass

# extern void lto_codegen_set_should_embed_uselists(lto_code_gen_t cg, lto_bool_t ShouldEmbedUselists)
try: (lto_codegen_set_should_embed_uselists:=dll.lto_codegen_set_should_embed_uselists).restype, lto_codegen_set_should_embed_uselists.argtypes = None, [lto_code_gen_t, lto_bool_t]
except AttributeError: pass

class struct_LLVMOpaqueLTOInput(Struct): pass
lto_input_t = ctypes.POINTER(struct_LLVMOpaqueLTOInput)
# extern lto_input_t lto_input_create(const void *buffer, size_t buffer_size, const char *path)
try: (lto_input_create:=dll.lto_input_create).restype, lto_input_create.argtypes = lto_input_t, [ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern void lto_input_dispose(lto_input_t input)
try: (lto_input_dispose:=dll.lto_input_dispose).restype, lto_input_dispose.argtypes = None, [lto_input_t]
except AttributeError: pass

# extern unsigned int lto_input_get_num_dependent_libraries(lto_input_t input)
try: (lto_input_get_num_dependent_libraries:=dll.lto_input_get_num_dependent_libraries).restype, lto_input_get_num_dependent_libraries.argtypes = ctypes.c_uint32, [lto_input_t]
except AttributeError: pass

# extern const char *lto_input_get_dependent_library(lto_input_t input, size_t index, size_t *size)
try: (lto_input_get_dependent_library:=dll.lto_input_get_dependent_library).restype, lto_input_get_dependent_library.argtypes = ctypes.POINTER(ctypes.c_char), [lto_input_t, size_t, ctypes.POINTER(size_t)]
except AttributeError: pass

# extern const char *const *lto_runtime_lib_symbols_list(size_t *size)
try: (lto_runtime_lib_symbols_list:=dll.lto_runtime_lib_symbols_list).restype, lto_runtime_lib_symbols_list.argtypes = ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), [ctypes.POINTER(size_t)]
except AttributeError: pass

class LTOObjectBuffer(Struct): pass
LTOObjectBuffer._fields_ = [
  ('Buffer', ctypes.POINTER(ctypes.c_char)),
  ('Size', size_t),
]
# extern thinlto_code_gen_t thinlto_create_codegen(void)
try: (thinlto_create_codegen:=dll.thinlto_create_codegen).restype, thinlto_create_codegen.argtypes = thinlto_code_gen_t, []
except AttributeError: pass

# extern void thinlto_codegen_dispose(thinlto_code_gen_t cg)
try: (thinlto_codegen_dispose:=dll.thinlto_codegen_dispose).restype, thinlto_codegen_dispose.argtypes = None, [thinlto_code_gen_t]
except AttributeError: pass

# extern void thinlto_codegen_add_module(thinlto_code_gen_t cg, const char *identifier, const char *data, int length)
try: (thinlto_codegen_add_module:=dll.thinlto_codegen_add_module).restype, thinlto_codegen_add_module.argtypes = None, [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError: pass

# extern void thinlto_codegen_process(thinlto_code_gen_t cg)
try: (thinlto_codegen_process:=dll.thinlto_codegen_process).restype, thinlto_codegen_process.argtypes = None, [thinlto_code_gen_t]
except AttributeError: pass

# extern unsigned int thinlto_module_get_num_objects(thinlto_code_gen_t cg)
try: (thinlto_module_get_num_objects:=dll.thinlto_module_get_num_objects).restype, thinlto_module_get_num_objects.argtypes = ctypes.c_uint32, [thinlto_code_gen_t]
except AttributeError: pass

# extern LTOObjectBuffer thinlto_module_get_object(thinlto_code_gen_t cg, unsigned int index)
try: (thinlto_module_get_object:=dll.thinlto_module_get_object).restype, thinlto_module_get_object.argtypes = LTOObjectBuffer, [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError: pass

# unsigned int thinlto_module_get_num_object_files(thinlto_code_gen_t cg)
try: (thinlto_module_get_num_object_files:=dll.thinlto_module_get_num_object_files).restype, thinlto_module_get_num_object_files.argtypes = ctypes.c_uint32, [thinlto_code_gen_t]
except AttributeError: pass

# const char *thinlto_module_get_object_file(thinlto_code_gen_t cg, unsigned int index)
try: (thinlto_module_get_object_file:=dll.thinlto_module_get_object_file).restype, thinlto_module_get_object_file.argtypes = ctypes.POINTER(ctypes.c_char), [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError: pass

# extern lto_bool_t thinlto_codegen_set_pic_model(thinlto_code_gen_t cg, lto_codegen_model)
try: (thinlto_codegen_set_pic_model:=dll.thinlto_codegen_set_pic_model).restype, thinlto_codegen_set_pic_model.argtypes = lto_bool_t, [thinlto_code_gen_t, lto_codegen_model]
except AttributeError: pass

# extern void thinlto_codegen_set_savetemps_dir(thinlto_code_gen_t cg, const char *save_temps_dir)
try: (thinlto_codegen_set_savetemps_dir:=dll.thinlto_codegen_set_savetemps_dir).restype, thinlto_codegen_set_savetemps_dir.argtypes = None, [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void thinlto_set_generated_objects_dir(thinlto_code_gen_t cg, const char *save_temps_dir)
try: (thinlto_set_generated_objects_dir:=dll.thinlto_set_generated_objects_dir).restype, thinlto_set_generated_objects_dir.argtypes = None, [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern void thinlto_codegen_set_cpu(thinlto_code_gen_t cg, const char *cpu)
try: (thinlto_codegen_set_cpu:=dll.thinlto_codegen_set_cpu).restype, thinlto_codegen_set_cpu.argtypes = None, [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern void thinlto_codegen_disable_codegen(thinlto_code_gen_t cg, lto_bool_t disable)
try: (thinlto_codegen_disable_codegen:=dll.thinlto_codegen_disable_codegen).restype, thinlto_codegen_disable_codegen.argtypes = None, [thinlto_code_gen_t, lto_bool_t]
except AttributeError: pass

# extern void thinlto_codegen_set_codegen_only(thinlto_code_gen_t cg, lto_bool_t codegen_only)
try: (thinlto_codegen_set_codegen_only:=dll.thinlto_codegen_set_codegen_only).restype, thinlto_codegen_set_codegen_only.argtypes = None, [thinlto_code_gen_t, lto_bool_t]
except AttributeError: pass

# extern void thinlto_debug_options(const char *const *options, int number)
try: (thinlto_debug_options:=dll.thinlto_debug_options).restype, thinlto_debug_options.argtypes = None, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

# extern lto_bool_t lto_module_is_thinlto(lto_module_t mod)
try: (lto_module_is_thinlto:=dll.lto_module_is_thinlto).restype, lto_module_is_thinlto.argtypes = lto_bool_t, [lto_module_t]
except AttributeError: pass

# extern void thinlto_codegen_add_must_preserve_symbol(thinlto_code_gen_t cg, const char *name, int length)
try: (thinlto_codegen_add_must_preserve_symbol:=dll.thinlto_codegen_add_must_preserve_symbol).restype, thinlto_codegen_add_must_preserve_symbol.argtypes = None, [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError: pass

# extern void thinlto_codegen_add_cross_referenced_symbol(thinlto_code_gen_t cg, const char *name, int length)
try: (thinlto_codegen_add_cross_referenced_symbol:=dll.thinlto_codegen_add_cross_referenced_symbol).restype, thinlto_codegen_add_cross_referenced_symbol.argtypes = None, [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError: pass

# extern void thinlto_codegen_set_cache_dir(thinlto_code_gen_t cg, const char *cache_dir)
try: (thinlto_codegen_set_cache_dir:=dll.thinlto_codegen_set_cache_dir).restype, thinlto_codegen_set_cache_dir.argtypes = None, [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# extern void thinlto_codegen_set_cache_pruning_interval(thinlto_code_gen_t cg, int interval)
try: (thinlto_codegen_set_cache_pruning_interval:=dll.thinlto_codegen_set_cache_pruning_interval).restype, thinlto_codegen_set_cache_pruning_interval.argtypes = None, [thinlto_code_gen_t, ctypes.c_int32]
except AttributeError: pass

# extern void thinlto_codegen_set_final_cache_size_relative_to_available_space(thinlto_code_gen_t cg, unsigned int percentage)
try: (thinlto_codegen_set_final_cache_size_relative_to_available_space:=dll.thinlto_codegen_set_final_cache_size_relative_to_available_space).restype, thinlto_codegen_set_final_cache_size_relative_to_available_space.argtypes = None, [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError: pass

# extern void thinlto_codegen_set_cache_entry_expiration(thinlto_code_gen_t cg, unsigned int expiration)
try: (thinlto_codegen_set_cache_entry_expiration:=dll.thinlto_codegen_set_cache_entry_expiration).restype, thinlto_codegen_set_cache_entry_expiration.argtypes = None, [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError: pass

# extern void thinlto_codegen_set_cache_size_bytes(thinlto_code_gen_t cg, unsigned int max_size_bytes)
try: (thinlto_codegen_set_cache_size_bytes:=dll.thinlto_codegen_set_cache_size_bytes).restype, thinlto_codegen_set_cache_size_bytes.argtypes = None, [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError: pass

# extern void thinlto_codegen_set_cache_size_megabytes(thinlto_code_gen_t cg, unsigned int max_size_megabytes)
try: (thinlto_codegen_set_cache_size_megabytes:=dll.thinlto_codegen_set_cache_size_megabytes).restype, thinlto_codegen_set_cache_size_megabytes.argtypes = None, [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError: pass

# extern void thinlto_codegen_set_cache_size_files(thinlto_code_gen_t cg, unsigned int max_size_files)
try: (thinlto_codegen_set_cache_size_files:=dll.thinlto_codegen_set_cache_size_files).restype, thinlto_codegen_set_cache_size_files.argtypes = None, [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError: pass

LLVMDisassembler_Option_UseMarkup = 1
LLVMDisassembler_Option_PrintImmHex = 2
LLVMDisassembler_Option_AsmPrinterVariant = 4
LLVMDisassembler_Option_SetInstrComments = 8
LLVMDisassembler_Option_PrintLatency = 16
LLVMDisassembler_Option_Color = 32
LLVMDisassembler_VariantKind_None = 0
LLVMDisassembler_VariantKind_ARM_HI16 = 1
LLVMDisassembler_VariantKind_ARM_LO16 = 2
LLVMDisassembler_VariantKind_ARM64_PAGE = 1
LLVMDisassembler_VariantKind_ARM64_PAGEOFF = 2
LLVMDisassembler_VariantKind_ARM64_GOTPAGE = 3
LLVMDisassembler_VariantKind_ARM64_GOTPAGEOFF = 4
LLVMDisassembler_VariantKind_ARM64_TLVP = 5
LLVMDisassembler_VariantKind_ARM64_TLVOFF = 6
LLVMDisassembler_ReferenceType_InOut_None = 0
LLVMDisassembler_ReferenceType_In_Branch = 1
LLVMDisassembler_ReferenceType_In_PCrel_Load = 2
LLVMDisassembler_ReferenceType_In_ARM64_ADRP = 0x100000001
LLVMDisassembler_ReferenceType_In_ARM64_ADDXri = 0x100000002
LLVMDisassembler_ReferenceType_In_ARM64_LDRXui = 0x100000003
LLVMDisassembler_ReferenceType_In_ARM64_LDRXl = 0x100000004
LLVMDisassembler_ReferenceType_In_ARM64_ADR = 0x100000005
LLVMDisassembler_ReferenceType_Out_SymbolStub = 1
LLVMDisassembler_ReferenceType_Out_LitPool_SymAddr = 2
LLVMDisassembler_ReferenceType_Out_LitPool_CstrAddr = 3
LLVMDisassembler_ReferenceType_Out_Objc_CFString_Ref = 4
LLVMDisassembler_ReferenceType_Out_Objc_Message = 5
LLVMDisassembler_ReferenceType_Out_Objc_Message_Ref = 6
LLVMDisassembler_ReferenceType_Out_Objc_Selector_Ref = 7
LLVMDisassembler_ReferenceType_Out_Objc_Class_Ref = 8
LLVMDisassembler_ReferenceType_DeMangled_Name = 9
LLVMErrorSuccess = 0
REMARKS_API_VERSION = 1
LLVM_BLAKE3_VERSION_STRING = "1.3.1"
LLVM_BLAKE3_KEY_LEN = 32
LLVM_BLAKE3_OUT_LEN = 32
LLVM_BLAKE3_BLOCK_LEN = 64
LLVM_BLAKE3_CHUNK_LEN = 1024
LLVM_BLAKE3_MAX_DEPTH = 54
LTO_API_VERSION = 29