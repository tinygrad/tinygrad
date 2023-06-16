import ctypes
import subprocess
import numpy as np

# https://en.wikipedia.org/wiki/X86_calling_conventions
# https://pixelclear.github.io/technical/post/2021/01/02/x86-Assembly-tutorial-part1.html
# https://www.cs.princeton.edu/courses/archive/spr11/cos217/lectures/15AssemblyFunctions.pdf
# https://stackoverflow.com/questions/71704813/writing-and-linking-shared-libraries-in-assembly-32-bit
# FLoating point: https://my.eng.utah.edu/~cs4400/sse-fp.pdf

print(subprocess.run(["as", "-o", "Add.o", "Add.s"]))
print(subprocess.run(["ld", "-shared", "Add.o", "-o", "Add.so"]))

lib = ctypes.CDLL('./Add.so')

fxn = lib['_add']
fxn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
fxn.restype = ctypes.c_int64

a = np.asarray([1.1,2.2,3.3,4.4], dtype=np.float32)
b = np.asarray([1.1,2.2,3.3,4.4], dtype=np.float32)
r = np.zeros((4), dtype=np.float32)

buf_addr = [buf.ctypes.data for buf in [r, a, b]]
print("bufs:", buf_addr)

res = fxn(*buf_addr)
print("return value", res)
print(r)
