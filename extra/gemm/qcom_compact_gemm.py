#!/usr/bin/env python3
"""Compact the GEMM loop by removing NOP instructions and adjusting branch offsets."""
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
image_size_orig = struct.unpack_from('<I', lib, 0x100)[0]
shader = bytearray(lib[image_offset:image_offset+image_size_orig])
total = image_size_orig // 8

def ri(buf, line):
    off = line * 8
    return struct.unpack_from('<I', buf, off+4)[0], struct.unpack_from('<I', buf, off)[0]

def wi(buf, line, hi, lo):
    off = line * 8
    struct.pack_into('<I', buf, off, lo)
    struct.pack_into('<I', buf, off+4, hi)

def rn(r):
    return "hr%d.%s" % (r // 4, "xyzw"[r % 4])

def get_disasm(binary):
    with tempfile.TemporaryFile('w+', buffering=1) as tf:
        @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
        def hd(data, n, instr):
            fst, snd = data64(ctypes.cast(instr, ctypes.POINTER(ctypes.c_uint64)).contents.value)
            print(f"{n:04} [{fst:08x}_{snd:08x}] ", end="", flush=True, file=tf)
        libc = ctypes.CDLL(None)
        libc.setlinebuf(fp:=ctypes.cast(libc.fdopen(tf.fileno(), b"w"), ctypes.POINTER(mesa.struct__IO_FILE)))
        mesa.ir3_isa_disasm(bytes(binary), len(binary), fp, mesa.struct_isa_decode_options(630, True, 0, True, pre_instr_cb=hd))
        tf.seek(0)
        return tf.read()

# Step 1: Apply register remap (48->44, 49->45)
for old_r, new_r in [(48, 44), (49, 45)]:
    for i in range(total):
        hi, lo = ri(shader, i)
        if hi == 0 and lo == 0: continue
        changed = False
        if (hi & 0xFF) == old_r: hi = (hi & 0xFFFFFF00) | new_r; changed = True
        if (lo & 0xFF) == old_r: lo = (lo & 0xFFFFFF00) | new_r; changed = True
        if ((lo >> 16) & 0xFF) == old_r: lo = (lo & 0xFF00FFFF) | (new_r << 16); changed = True
        if changed: wi(shader, i, hi, lo)

# Step 2: Convert all eligible MAD groups to (rpt3)
# First convert 4x scalar -> rpt3
i = 0
while i < total - 3:
    hi0, lo0 = ri(shader, i)
    if not ((hi0 >> 24) in (0x63, 0x73) and ((hi0 >> 24) & 0x0F) == 0x3): i += 1; continue
    dst0, rpt0 = hi0 & 0xFF, (hi0 >> 8) & 0x7F
    src1_0, src3_0 = lo0 & 0xFF, (lo0 >> 16) & 0xFF
    src2_0 = ((hi0 >> 16) & 0xFF) * 2 + (((hi0 >> 8) & 0xFF) >> 7)
    if rpt0 > 0 or dst0 != src3_0: i += 1; continue
    ok = True
    for j in range(1, 4):
        hj, lj = ri(shader, i+j)
        if not ((hj >> 24) in (0x63, 0x73) and ((hj >> 24) & 0x0F) == 0x3): ok = False; break
        dj, rpj = hj & 0xFF, (hj >> 8) & 0x7F
        s1j, s3j = lj & 0xFF, (lj >> 16) & 0xFF
        s2j = ((hj >> 16) & 0xFF) * 2 + (((hj >> 8) & 0xFF) >> 7)
        if rpj != 0 or s1j != src1_0 or dj != dst0+j or s2j != src2_0+j or s3j != dst0+j: ok = False; break
    if ok:
        rpt_byte_new = ((hi0 >> 8) & 0x80) | 3
        hi_new = (hi0 & 0xFFFF00FF) | (rpt_byte_new << 8)
        wi(shader, i, hi_new, lo0 | 0x20000000)
        for j in range(1, 4): wi(shader, i+j, 0, 0)
        i += 4
    else:
        i += 1

# Merge (rpt1)+(rpt1) -> (rpt3)
for i in range(total - 1):
    hi0, lo0 = ri(shader, i)
    hi1, lo1 = ri(shader, i+1)
    if hi0 == 0 or hi1 == 0: continue
    if not ((hi0 >> 24) in (0x63, 0x73) and ((hi0 >> 24) & 0x0F) == 0x3): continue
    if not ((hi1 >> 24) in (0x63, 0x73) and ((hi1 >> 24) & 0x0F) == 0x3): continue
    rpt0 = (hi0 >> 8) & 0x7F
    rpt1v = (hi1 >> 8) & 0x7F
    if rpt0 != 1 or rpt1v != 1: continue
    dst0, dst1 = hi0 & 0xFF, hi1 & 0xFF
    src1_0, src1_1 = lo0 & 0xFF, lo1 & 0xFF
    src2_0 = ((hi0 >> 16) & 0xFF) * 2 + (((hi0 >> 8) & 0xFF) >> 7)
    src2_1 = ((hi1 >> 16) & 0xFF) * 2 + (((hi1 >> 8) & 0xFF) >> 7)
    if src1_0 != src1_1 or dst1 != dst0 + 2 or src2_1 != src2_0 + 2: continue
    rpt_byte_new = ((hi0 >> 8) & 0x80) | 3
    wi(shader, i, (hi0 & 0xFFFF00FF) | (rpt_byte_new << 8), lo0)
    wi(shader, i+1, 0, 0)

# Step 3: COMPACT - remove NOP instructions from the loop body
# Find the branch and loop target
branch_line = None
for i in range(total):
    hi, lo = ri(shader, i)
    if (hi >> 20) == 0x009:
        branch_line = i
        br_offset_raw = lo
        br_offset = struct.unpack('<i', struct.pack('<I', lo))[0]
        target_line = i + 1 + br_offset

if branch_line is None:
    print("ERROR: no branch found")
    exit(1)

print("Branch at line %d, target line %d (offset %d)" % (branch_line, target_line, br_offset))

# Count NOPs in the LOOP (between target_line and branch_line inclusive)
loop_nops = []
for i in range(target_line, branch_line + 1):
    hi, lo = ri(shader, i)
    if hi == 0 and lo == 0:
        loop_nops.append(i)

print("Loop body: lines %d-%d (%d instrs), %d NOPs to remove" % (
    target_line, branch_line, branch_line - target_line + 1, len(loop_nops)))

# Build new instruction stream: remove NOPs from the loop body
# Also need to handle: some "NOPs" are actually (nop2), (nop3) etc which are
# instruction modifiers, not standalone NOPs. Only remove pure 00000000_00000000 NOPs.
new_instrs = []
old_to_new = {}  # map old line numbers to new line numbers

for i in range(total):
    hi, lo = ri(shader, i)
    # Remove pure NOPs that are inside the loop
    if hi == 0 and lo == 0 and target_line <= i <= branch_line:
        continue  # skip this NOP
    old_to_new[i] = len(new_instrs)
    new_instrs.append((hi, lo))

new_total = len(new_instrs)
print("Compacted: %d -> %d instructions (removed %d)" % (total, new_total, total - new_total))

# Fix the branch offset
if branch_line in old_to_new and target_line in old_to_new:
    new_branch = old_to_new[branch_line]
    new_target = old_to_new[target_line]
    new_br_offset = new_target - new_branch - 1
    # Update the branch instruction
    br_hi, br_lo = new_instrs[new_branch]
    new_instrs[new_branch] = (br_hi, struct.unpack('<I', struct.pack('<i', new_br_offset))[0])
    print("Branch: old offset %d -> new offset %d" % (br_offset, new_br_offset))

# Build new shader binary - KEEP SAME SIZE by padding with NOPs at the end
new_shader = bytearray()
for hi, lo in new_instrs:
    new_shader += struct.pack('<II', lo, hi)

# Pad to original size with end + nop instructions
while len(new_shader) < image_size_orig:
    new_shader += struct.pack('<II', 0x00000000, 0x00000000)  # nop padding

new_image_size = image_size_orig  # keep same size!
print("New shader: %d bytes = %d real instrs + %d padding" % (new_image_size, new_total, (image_size_orig - new_total*8)//8))

# Don't resize - just replace shader in-place
lib_new = bytearray(lib)
lib_new[image_offset:image_offset+image_size_orig] = new_shader
# image_size stays the same - no need to update

# Verify disassembly
print("\n=== COMPACTED KERNEL ===")
asm = get_disasm(bytes(new_shader))
mad_count = asm.count('mad.f16')
rpt3_count = asm.count('(rpt3)mad.f16')
isam_count = asm.count('isam')
nop_count = asm.count('nop')
print("instrs=%d mad=%d rpt3=%d isam=%d nop=%d" % (new_total, mad_count, rpt3_count, isam_count, nop_count))

for line in asm.strip().split('\n'):
    if line.strip():
        print(line[:120])

# Benchmark
a_imgdt = dtypes.imageh((1024, 256))
b_imgdt = dtypes.imageh((1024, 256))
a_buf = Buffer(dev.device, 256*1024*4, dtypes.half, preallocate=True)
b_buf = Buffer(dev.device, 256*1024*4, dtypes.half, preallocate=True)
c_buf = Buffer(dev.device, 1024*1024, dtypes.half, preallocate=True)
ctypes.memset(int(a_buf._buf.va_addr), 0, a_buf.nbytes)
ctypes.memset(int(b_buf._buf.va_addr), 0, b_buf.nbytes)

try:
    prg = dev.runtime('gemm_h', bytes(lib_new), [[(0, a_imgdt)], [(1, b_imgdt)], [(2, dtypes.half.ptr())]])
    gs = (1024 // 128, 1024 // 4, 1)
    ls = (128, 1, 1)

    for _ in range(5):
        prg(a_buf._buf, b_buf._buf, c_buf._buf, global_size=gs, local_size=ls, wait=True)

    times = []
    for _ in range(30):
        t = prg(a_buf._buf, b_buf._buf, c_buf._buf, global_size=gs, local_size=ls, wait=True)
        if t: times.append(t)

    if times:
        best = min(times)
        gflops = 2 * 1024 * 1024 * 1024 / best / 1e9
        print("\n*** COMPACTED: %.1f GFLOPS (%.0fus) ***" % (gflops, best * 1e6))
except Exception as e:
    print("ERROR: %s" % str(e)[:200])
