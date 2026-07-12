#!/usr/bin/env python3
"""Clean register remap: identify all accumulators, remap group 3 to be consecutive, apply all 4 rpt3."""
import struct, ctypes, tempfile
from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from tinygrad.runtime.autogen import mesa
from tinygrad.helpers import data64

dev = Device['QCOM']

src = (
    '#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n'
    'const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n'
    '__attribute__((reqd_work_group_size(128, 1, 1)))\n'
    '__kernel void gemm_h(read_only image2d_t A, read_only image2d_t B, __global half *C) {\n'
    '  int lid = get_local_id(0);\n'
    '  int row = get_group_id(1) * 4 + (lid >> 5);\n'
    '  int col4 = get_group_id(0) * 32 + (lid & 31);\n'
    '  half4 acc0=(half4)(0), acc1=(half4)(0), acc2=(half4)(0), acc3=(half4)(0);\n'
    '  for (int k4 = 0; k4 < 256; k4++) {\n'
    '    half4 a = read_imageh(A, smp, (int2)(k4, row));\n'
    '    half4 b0 = read_imageh(B, smp, (int2)(col4, k4*4));\n'
    '    half4 b1 = read_imageh(B, smp, (int2)(col4, k4*4+1));\n'
    '    half4 b2 = read_imageh(B, smp, (int2)(col4, k4*4+2));\n'
    '    half4 b3 = read_imageh(B, smp, (int2)(col4, k4*4+3));\n'
    '    acc0 += a.xxxx * b0;\n'
    '    acc1 += a.yyyy * b1;\n'
    '    acc2 += a.zzzz * b2;\n'
    '    acc3 += a.wwww * b3;\n'
    '  }\n'
    '  half4 r = acc0 + acc1 + acc2 + acc3;\n'
    '  vstore4(r, 0, C + row*1024 + col4*4);\n'
    '}\n'
)

lib = bytearray(dev.compiler.compile_cached(src))
image_offset = struct.unpack_from('<I', lib, 0xc0)[0]
image_size = struct.unpack_from('<I', lib, 0x100)[0]
shader = bytearray(lib[image_offset:image_offset+image_size])
total = image_size // 8

def ri(buf, line):
    off = line * 8
    return struct.unpack_from('<I', buf, off+4)[0], struct.unpack_from('<I', buf, off)[0]

def wi(buf, line, hi, lo):
    off = line * 8
    struct.pack_into('<I', buf, off, lo)
    struct.pack_into('<I', buf, off+4, hi)

def rn(r):
    return "hr%d.%s" % (r // 4, "xyzw"[r % 4])

# Map all accumulator registers from MAD instructions
print("=== ACCUMULATOR ANALYSIS ===")
for i in range(total):
    hi, lo = ri(shader, i)
    if not ((hi >> 24) in (0x63, 0x73) and ((hi >> 24) & 0x0F) == 0x3): continue
    dst = hi & 0xFF
    src1 = lo & 0xFF
    rpt_byte = (hi >> 8) & 0xFF
    src2_hi = (hi >> 16) & 0xFF
    src2 = (src2_hi << 1) | ((rpt_byte >> 7) & 1)
    src3 = (lo >> 16) & 0xFF
    rpt = rpt_byte & 0x7F
    sy = (hi >> 28) == 0x7
    r_flag = (lo >> 29) & 1
    flags = ("(sy)" if sy else "") + ("(rpt%d)" % rpt if rpt else "")
    rf = "(r)" if r_flag else ""
    print("  L%02d: %smad.f16 %s, %s, %s%s, %s%s" % (i, flags, rn(dst), rn(src1), rf, rn(src2), rf, rn(src3)))

# From the analysis, we know:
# Group 0 (src1=hr0.x=0): accs at hr14.z..hr15.y = 58,59,60,61 - CONSECUTIVE
# Group 1 (src1=hr0.y=1): accs at hr13.z..hr14.y = 54,55,56,57 - CONSECUTIVE (already rpt3)
# Group 2 (src1=hr0.z=2): accs at hr12.z..hr13.y = 50,51,52,53 - CONSECUTIVE
# Group 3 (src1=hr0.w=3): split as (rpt1) hr10.z,hr10.w + (rpt1) hr12.x,hr12.y = 42,43,48,49 - NOT CONSECUTIVE

# Fix: remap 48->44, 49->45 (making group 3 = 42,43,44,45)
# Check 44 and 45 are free
used = set()
for i in range(total):
    hi, lo = ri(shader, i)
    if hi == 0 and lo == 0: continue
    used.add(hi & 0xFF)
    used.add(lo & 0xFF)
    used.add((lo >> 16) & 0xFF)

print("\nRegs 44,45 in use:", 44 in used, 45 in used)

# Do the remap
for old_r, new_r in [(48, 44), (49, 45)]:
    for i in range(total):
        hi, lo = ri(shader, i)
        if hi == 0 and lo == 0: continue
        changed = False
        if (hi & 0xFF) == old_r:
            hi = (hi & 0xFFFFFF00) | new_r; changed = True
        if (lo & 0xFF) == old_r:
            lo = (lo & 0xFFFFFF00) | new_r; changed = True
        if ((lo >> 16) & 0xFF) == old_r:
            lo = (lo & 0xFF00FFFF) | (new_r << 16); changed = True
        if changed:
            wi(shader, i, hi, lo)

print("Remapped 48->44, 49->45")

# Now find groups of 4 consecutive non-rpt MADs and convert to rpt3
patched = 0
i = 0
while i < total - 3:
    hi0, lo0 = ri(shader, i)
    if not ((hi0 >> 24) in (0x63, 0x73) and ((hi0 >> 24) & 0x0F) == 0x3):
        i += 1; continue
    dst0 = hi0 & 0xFF
    rpt0 = (hi0 >> 8) & 0x7F
    src1_0 = lo0 & 0xFF
    src3_0 = (lo0 >> 16) & 0xFF
    src2_hi0 = (hi0 >> 16) & 0xFF
    rpt_byte0 = (hi0 >> 8) & 0xFF
    src2_0 = (src2_hi0 << 1) | ((rpt_byte0 >> 7) & 1)
    
    if rpt0 > 0 or dst0 != src3_0:
        i += 1; continue
    
    # Check next 3
    ok = True
    for j in range(1, 4):
        hj, lj = ri(shader, i+j)
        if not ((hj >> 24) in (0x63, 0x73) and ((hj >> 24) & 0x0F) == 0x3):
            ok = False; break
        dj = hj & 0xFF
        s1j = lj & 0xFF
        rpj = (hj >> 8) & 0x7F
        s3j = (lj >> 16) & 0xFF
        s2j_hi = (hj >> 16) & 0xFF
        rpj_byte = (hj >> 8) & 0xFF
        s2j = (s2j_hi << 1) | ((rpj_byte >> 7) & 1)
        if rpj != 0 or s1j != src1_0 or dj != dst0+j or s2j != src2_0+j or s3j != dst0+j:
            ok = False; break
    
    if ok:
        rpt_byte_new = ((hi0 >> 8) & 0x80) | 3
        hi_new = (hi0 & 0xFFFF00FF) | (rpt_byte_new << 8)
        lo_new = lo0 | 0x20000000
        wi(shader, i, hi_new, lo_new)
        for j in range(1, 4):
            wi(shader, i+j, 0, 0)
        patched += 1
        print("  rpt3 at line %d: %s" % (i, rn(dst0)))
        i += 4
    else:
        i += 1

# Also merge (rpt1)+(rpt1) into (rpt3) where registers are consecutive
for i in range(total - 1):
    hi0, lo0 = ri(shader, i)
    hi1, lo1 = ri(shader, i+1)
    if hi0 == 0 or hi1 == 0: continue
    if not ((hi0 >> 24) in (0x63, 0x73) and ((hi0 >> 24) & 0x0F) == 0x3): continue
    if not ((hi1 >> 24) in (0x63, 0x73) and ((hi1 >> 24) & 0x0F) == 0x3): continue
    rpt0 = (hi0 >> 8) & 0x7F
    rpt1v = (hi1 >> 8) & 0x7F
    if rpt0 != 1 or rpt1v != 1: continue
    dst0 = hi0 & 0xFF
    dst1 = hi1 & 0xFF
    src1_0 = lo0 & 0xFF
    src1_1 = lo1 & 0xFF
    src2_0 = ((hi0 >> 16) & 0xFF) * 2 + (((hi0 >> 8) & 0xFF) >> 7)
    src2_1 = ((hi1 >> 16) & 0xFF) * 2 + (((hi1 >> 8) & 0xFF) >> 7)
    if src1_0 != src1_1: continue
    if dst1 != dst0 + 2: continue
    if src2_1 != src2_0 + 2: continue
    # Merge: change first to rpt3, NOP second
    rpt_byte_new = ((hi0 >> 8) & 0x80) | 3
    hi_new = (hi0 & 0xFFFF00FF) | (rpt_byte_new << 8)
    wi(shader, i, hi_new, lo0)
    wi(shader, i+1, 0, 0)
    patched += 1
    print("  Merged rpt1+rpt1 -> rpt3 at line %d: %s" % (i, rn(dst0)))

print("Total patched: %d groups" % patched)

# Write back
lib[image_offset:image_offset+image_size] = shader

# Benchmark
a_imgdt = dtypes.imageh((1024, 256))
b_imgdt = dtypes.imageh((1024, 256))
a_buf = Buffer(dev.device, 256*1024*4, dtypes.half, preallocate=True)
b_buf = Buffer(dev.device, 256*1024*4, dtypes.half, preallocate=True)
c_buf = Buffer(dev.device, 1024*1024, dtypes.half, preallocate=True)
ctypes.memset(int(a_buf._buf.va_addr), 0, a_buf.nbytes)
ctypes.memset(int(b_buf._buf.va_addr), 0, b_buf.nbytes)

prg = dev.runtime('gemm_h', bytes(lib), [[(0, a_imgdt)], [(1, b_imgdt)], [(2, dtypes.half.ptr())]])
gs = (1024//128, 1024//4, 1)
ls = (128, 1, 1)

for _ in range(5):
    prg(a_buf._buf, b_buf._buf, c_buf._buf, global_size=gs, local_size=ls, wait=True)

times = []
for _ in range(30):
    t = prg(a_buf._buf, b_buf._buf, c_buf._buf, global_size=gs, local_size=ls, wait=True)
    if t: times.append(t)

if times:
    best = min(times)
    gflops = 2*1024*1024*1024 / best / 1e9
    print("\n*** RESULT: %.1f GFLOPS (%.0fus) ***" % (gflops, best*1e6))
