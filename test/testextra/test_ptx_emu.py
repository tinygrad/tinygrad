# tests for the PTX emulator in test/mockgpu/nv/emu.py, these run the emulator directly and need no device
import ctypes, unittest
import numpy as np
from test.mockgpu.nv.emu import ptx_run

HDR = """.version 7.5
.target sm_86
.address_size 64
.visible .entry k (%s)
{
%s
	ret;
}"""

def kernel(nparams:int, body:str) -> str:
  params = ",\n".join(f"\t.param .u64 arg{i}" for i in range(nparams))
  loads = "\n".join(f"\tld.param.u64 %pb{i}, [arg{i}+0];" for i in range(nparams))
  regs = f"\t.reg .u64 %pb<{nparams}>;\n"
  return HDR % (params, regs + body.replace("LOADPARAMS", loads))

def run(src:str, bufs:list[np.ndarray], block=(1,1,1), grid=(1,1,1), smem=0):
  ptrs = [b.ctypes.data for b in bufs]
  args = (ctypes.c_void_p * len(ptrs))(*ptrs)
  ptx_run(ctypes.c_char_p(src.encode()), len(ptrs), args, *block, *grid, smem)

# index the calling thread's element of a 4 byte buffer in %pbN, leaving the address in %addr
IDX = """	.reg .u32 %tx;
	.reg .u64 %ix;
	.reg .u64 %addr<8>;
	mov.u32 %tx, %tid.x;
	cvt.u64.u32 %ix, %tx;
"""
def at(param:int, reg:int, sz:int=4) -> str: return f"\tmad.lo.u64 %addr{reg}, %ix, {sz}, %pb{param};\n"

class TestPTXEmuArith(unittest.TestCase):
  def _binop(self, op:str, ty:str, a:np.ndarray, b:np.ndarray, out_dtype) -> np.ndarray:
    n, sz = len(a), int(ty[1:]) // 8
    body = IDX + "LOADPARAMS\n" + f"""	.reg .{ty} %v<4>;
{at(0,0,sz)}{at(1,1,sz)}{at(2,2,sz)}	ld.global.{ty} %v0, [%addr0+0];
	ld.global.{ty} %v1, [%addr1+0];
	{op}.{ty} %v2, %v0, %v1;
	st.global.{ty} [%addr2+0], %v2;
"""
    out = np.zeros(n, dtype=out_dtype)
    run(kernel(3, body), [a, b, out], block=(n, 1, 1))
    return out

  def test_sdiv_srem_truncate(self):
    # ptx integer division truncates toward zero, unlike python's floor division
    a = np.array([7, -7, 7, -7, 1, -2147483648, 2147483647, 5], dtype=np.int32)
    b = np.array([3, 3, -3, -3, -2147483648, 1, -3, -1], dtype=np.int32)
    q = self._binop("div", "s32", a, b, np.int32)
    r = self._binop("rem", "s32", a, b, np.int32)
    exp_q = np.array([int(x / y) if y else 0 for x, y in zip(a.tolist(), b.tolist())], dtype=np.int32)
    np.testing.assert_array_equal(q, exp_q)
    np.testing.assert_array_equal(r, (a - b * exp_q).astype(np.int32))

  def test_udiv_urem(self):
    a = np.array([7, 0, 4294967295, 100], dtype=np.uint32)
    b = np.array([3, 5, 7, 100], dtype=np.uint32)
    np.testing.assert_array_equal(self._binop("div", "u32", a, b, np.uint32), a // b)
    np.testing.assert_array_equal(self._binop("rem", "u32", a, b, np.uint32), a % b)

  def test_int_wrapping(self):
    a = np.array([2147483647, -2147483648, 65536], dtype=np.int32)
    b = np.array([1, -1, 65536], dtype=np.int32)
    np.testing.assert_array_equal(self._binop("add", "s32", a, b, np.int32), (a + b).astype(np.int32))
    np.testing.assert_array_equal(self._binop("mul.lo", "s32", a, b, np.int32), (a * b).astype(np.int32))

  def test_float_nan_inf(self):
    inf, nan = float("inf"), float("nan")
    a = np.array([1.0, inf, nan, 1.0, -1.0, nan], dtype=np.float32)
    b = np.array([0.0, inf, 2.0, nan, 0.0, nan], dtype=np.float32)
    # ptx min/max return the operand that is not NaN
    np.testing.assert_array_equal(self._binop("max", "f32", a, b, np.float32), np.fmax(a, b))
    np.testing.assert_array_equal(self._binop("min", "f32", a, b, np.float32), np.fmin(a, b))
    got = self._binop("div.rn", "f32", a, b, np.float32)
    with np.errstate(all="ignore"): np.testing.assert_array_equal(got, a / b)

  def test_mul_hi_wide_64(self):
    a = np.array([2**63, 2**40 + 7, 12345678901234567, 0xffffffffffffffff], dtype=np.uint64)
    b = np.array([3, 2**33, 987654321, 0xffffffffffffffff], dtype=np.uint64)
    got = self._binop("mul.hi", "u64", a, b, np.uint64)
    exp = np.array([(int(x) * int(y)) >> 64 for x, y in zip(a.tolist(), b.tolist())], dtype=np.uint64)
    np.testing.assert_array_equal(got, exp)

class TestPTXEmuLiterals(unittest.TestCase):
  def test_hex_literals(self):
    # 0x03FF must not lose its trailing F to a suffix strip, and 0x3FU must lose its U
    out = np.zeros(4, dtype=np.uint32)
    body = """	.reg .u32 %v<4>;
	.reg .u64 %addr<4>;
LOADPARAMS
	mov.b32 %v0, 0x03FF;
	mov.b32 %v1, 0x3FU;
	mov.b32 %v2, 255U;
	mov.b32 %v3, 0xdeadbeef;
	st.global.u32 [%pb0+0], %v0;
	st.global.u32 [%pb0+4], %v1;
	st.global.u32 [%pb0+8], %v2;
	st.global.u32 [%pb0+12], %v3;
"""
    run(kernel(1, body), [out])
    np.testing.assert_array_equal(out, np.array([0x3FF, 0x3F, 255, 0xdeadbeef], dtype=np.uint32))

  def test_float_literals(self):
    out = np.zeros(2, dtype=np.float32)
    body = """	.reg .f32 %v<2>;
LOADPARAMS
	mov.b32 %v0, 0f3F800000;
	mov.b32 %v1, 0fBFC00000;
	st.global.f32 [%pb0+0], %v0;
	st.global.f32 [%pb0+4], %v1;
"""
    run(kernel(1, body), [out])
    np.testing.assert_array_equal(out, np.array([1.0, -1.5], dtype=np.float32))

  def test_operands_without_leading_space(self):
    # nvrtc emits inline asm like "mma...f32{%r1, %r2}" with no space before the first operand
    out = np.zeros(2, dtype=np.uint32)
    body = """	.reg .u32 %v<2>;
LOADPARAMS
	mov.b32%v0, 7;
	mov.b32 %v1, 9;
	st.global.u32 [%pb0+0], %v0;
	st.global.u32 [%pb0+4], %v1;
"""
    run(kernel(1, body), [out])
    np.testing.assert_array_equal(out, np.array([7, 9], dtype=np.uint32))

  def test_negative_memory_offset(self):
    src = np.arange(8, dtype=np.uint32)
    out = np.zeros(1, dtype=np.uint32)
    body = """	.reg .u32 %v<1>;
	.reg .u64 %a<1>;
LOADPARAMS
	add.u64 %a0, %pb0, 16;
	ld.global.u32 %v0, [%a0+-8];
	st.global.u32 [%pb1+0], %v0;
"""
    run(kernel(2, body), [src, out])
    np.testing.assert_array_equal(out, np.array([2], dtype=np.uint32))

  def test_inline_asm_local_registers(self):
    # nvrtc's inline asm declares its own registers with no % prefix and no space after .reg
    out = np.zeros(1, dtype=np.float32)
    body = """	.reg .b16 %rs<2>;
LOADPARAMS
	mov.b16 %rs0, 0x4000;
	{.reg.b16         h;
 .reg.b32         f;
	mov.b16 h, %rs0;
	cvt.f32.f16 f, h;
	st.global.f32 [%pb0+0], f;}
"""
    run(kernel(1, body), [out])
    np.testing.assert_array_equal(out, np.array([2.0], dtype=np.float32))

  def test_label_before_inline_asm(self):
    # a label is not semicolon terminated, so it must not swallow the brace of a following inline asm block
    out = np.zeros(1, dtype=np.uint32)
    body = """	.reg .u32 %v<2>;
	.reg .pred %p<1>;
LOADPARAMS
	mov.u32 %v0, 0;
	setp.eq.u32 %p0, %v0, 0;
	@%p0 bra $TARGET;
	mov.u32 %v0, 99;
$TARGET:
	{.reg.b32 t;
	mov.b32 t, 7;
	mov.b32 %v1, t;}
	st.global.u32 [%pb0+0], %v1;
"""
    run(kernel(1, body), [out])
    np.testing.assert_array_equal(out, np.array([7], dtype=np.uint32))

class TestPTXEmuMemory(unittest.TestCase):
  def test_signed_narrow_load_sign_extends(self):
    # ld.s8 into a wider register must sign extend, ld.u8 must not
    src = np.array([-128, -1, 0, 127], dtype=np.int8)
    out = np.zeros(4, dtype=np.int32)
    body = IDX + "LOADPARAMS\n" + f"""	.reg .s16 %h<2>;
	.reg .s32 %w<2>;
	mad.lo.u64 %addr0, %ix, 1, %pb0;
{at(1,1)}	ld.global.s8 %h0, [%addr0+0];
	cvt.s32.s16 %w0, %h0;
	st.global.s32 [%addr1+0], %w0;
"""
    run(kernel(2, body), [src, out], block=(4, 1, 1))
    np.testing.assert_array_equal(out, src.astype(np.int32))

  def test_byte_and_unaligned_access(self):
    # single byte stores and loads at odd addresses exercise the non typed gather path
    src = np.arange(16, dtype=np.uint8)
    out = np.zeros(16, dtype=np.uint8)
    body = IDX + "LOADPARAMS\n" + """	.reg .u16 %v<2>;
	mad.lo.u64 %addr0, %ix, 1, %pb0;
	mad.lo.u64 %addr1, %ix, 1, %pb1;
	ld.global.u8 %v0, [%addr0+0];
	add.u16 %v1, %v0, 1;
	st.global.u8 [%addr1+0], %v1;
"""
    run(kernel(2, body), [src, out], block=(16, 1, 1))
    np.testing.assert_array_equal(out, (src + 1).astype(np.uint8))

  def test_unaligned_word_access(self):
    # a 4 byte load at a 1 byte offset cannot use a typed view
    src = np.arange(32, dtype=np.uint8)
    out = np.zeros(4, dtype=np.uint32)
    body = IDX + "LOADPARAMS\n" + f"""	.reg .u32 %v<1>;
	mad.lo.u64 %addr0, %ix, 4, %pb0;
	add.u64 %addr0, %addr0, 1;
{at(1,1)}	ld.global.u32 %v0, [%addr0+0];
	st.global.u32 [%addr1+0], %v0;
"""
    run(kernel(2, body), [src, out], block=(4, 1, 1))
    exp = np.array([int.from_bytes(src[1 + 4 * i:5 + 4 * i].tobytes(), "little") for i in range(4)], dtype=np.uint32)
    np.testing.assert_array_equal(out, exp)

  def test_vector_load_store(self):
    src = np.arange(16, dtype=np.float32)
    out = np.zeros(16, dtype=np.float32)
    body = IDX + "LOADPARAMS\n" + f"""	.reg .f32 %v<4>;
{at(0,0,16)}{at(1,1,16)}	ld.global.v4.f32 {{%v0, %v1, %v2, %v3}}, [%addr0+0];
	st.global.v4.f32 [%addr1+0], {{%v3, %v2, %v1, %v0}};
"""
    run(kernel(2, body), [src, out], block=(4, 1, 1))
    np.testing.assert_array_equal(out, src.reshape(4, 4)[:, ::-1].ravel())

  def test_shared_memory_barrier(self):
    # every thread writes its index, then after the barrier reads the slot of the thread on the far side of the block
    n = 64
    out = np.zeros(n, dtype=np.int32)
    body = IDX + "LOADPARAMS\n" + f"""	.shared .align 4 .b8 buf[{n * 4}];
	.reg .u64 %sb<2>;
	.reg .u32 %r<4>;
	mov.u64 %sb0, buf[0];
	mad.lo.u64 %addr0, %ix, 4, %sb0;
	st.shared.u32 [%addr0+0], %tx;
	bar.sync 0;
	mov.u32 %r0, {n - 1};
	sub.u32 %r1, %r0, %tx;
	cvt.u64.u32 %addr1, %r1;
	mad.lo.u64 %addr1, %addr1, 4, %sb0;
	ld.shared.u32 %r2, [%addr1+0];
{at(0,2)}	st.global.u32 [%addr2+0], %r2;
"""
    run(kernel(1, body), [out], block=(n, 1, 1))
    np.testing.assert_array_equal(out, np.arange(n)[::-1])

  def test_local_memory_is_per_thread(self):
    # nvrtc spills to a .local depot (sin/cos need one), and .local is per thread, not one buffer for the whole grid
    n = 32
    out = np.zeros(n, dtype=np.int32)
    body = IDX + "LOADPARAMS\n" + f"""	.local .align 4 .b8 depot[4];
	.reg .u64 %lb<2>;
	.reg .u32 %r<2>;
	mov.u64 %lb0, depot[0];
	st.local.u32 [%lb0+0], %tx;
	ld.local.u32 %r0, [%lb0+0];
{at(0,0)}	st.global.u32 [%addr0+0], %r0;
"""
    run(kernel(1, body), [out], block=(n, 1, 1))
    np.testing.assert_array_equal(out, np.arange(n))

  def test_shared_memory_is_per_block(self):
    # each block must get its own shared allocation even though the whole grid runs at once
    nb, bs = 8, 32
    out = np.zeros(nb * bs, dtype=np.int32)
    body = """	.reg .u32 %tx, %bx, %v<2>;
	.reg .u64 %ix, %bi, %sb, %addr<2>;
LOADPARAMS
	.shared .align 4 .b8 buf[4];
	mov.u32 %tx, %tid.x;
	mov.u32 %bx, %ctaid.x;
	mov.u64 %sb, buf[0];
	setp.eq.u32 %p0, %tx, 0;
	@%p0 st.shared.u32 [%sb+0], %bx;
	bar.sync 0;
	ld.shared.u32 %v0, [%sb+0];
	cvt.u64.u32 %ix, %tx;
	cvt.u64.u32 %bi, %bx;
	mad.lo.u64 %addr0, %bi, 128, %pb0;
	mad.lo.u64 %addr1, %ix, 4, %addr0;
	st.global.u32 [%addr1+0], %v0;
"""
    body = "\t.reg .pred %p<1>;\n" + body
    run(kernel(1, body), [out], block=(bs, 1, 1), grid=(nb, 1, 1))
    np.testing.assert_array_equal(out, np.repeat(np.arange(nb), bs))

class TestPTXEmuControlFlow(unittest.TestCase):
  def test_divergent_loop(self):
    # thread i loops i times, so the lanes leave the loop at different points and have to reconverge
    n = 32
    out = np.zeros(n, dtype=np.int32)
    body = IDX + "LOADPARAMS\n" + f"""	.reg .pred %p<1>;
	.reg .u32 %acc, %i;
	mov.u32 %acc, 0;
	mov.u32 %i, 0;
$LOOP:
	setp.ge.u32 %p0, %i, %tx;
	@%p0 bra $DONE;
	add.u32 %acc, %acc, %i;
	add.u32 %i, %i, 1;
	bra $LOOP;
$DONE:
{at(0,0)}	st.global.u32 [%addr0+0], %acc;
"""
    run(kernel(1, body), [out], block=(n, 1, 1))
    np.testing.assert_array_equal(out, np.array([sum(range(i)) for i in range(n)]))

  def test_early_exit(self):
    # odd lanes return before the store, so their slot keeps its original value
    n = 16
    out = np.full(n, -1, dtype=np.int32)
    body = IDX + "LOADPARAMS\n" + f"""	.reg .pred %p<1>;
	.reg .u32 %r<2>;
	and.b32 %r0, %tx, 1;
	setp.ne.u32 %p0, %r0, 0;
	@%p0 ret;
{at(0,0)}	st.global.u32 [%addr0+0], %tx;
"""
    run(kernel(1, body), [out], block=(n, 1, 1))
    exp = np.where(np.arange(n) % 2 == 0, np.arange(n), -1).astype(np.int32)
    np.testing.assert_array_equal(out, exp)

  def test_grid_is_chunked_consistently(self):
    # more lanes than one chunk, every block must still see the right ctaid
    nb = 4096
    out = np.zeros(nb * 32, dtype=np.uint32)
    body = """	.reg .u32 %tx, %bx;
	.reg .u64 %ix, %bi, %addr<2>;
LOADPARAMS
	mov.u32 %tx, %tid.x;
	mov.u32 %bx, %ctaid.x;
	cvt.u64.u32 %ix, %tx;
	cvt.u64.u32 %bi, %bx;
	mad.lo.u64 %addr0, %bi, 128, %pb0;
	mad.lo.u64 %addr1, %ix, 4, %addr0;
	st.global.u32 [%addr1+0], %bx;
"""
    run(kernel(1, body), [out], block=(32, 1, 1), grid=(nb, 1, 1))
    np.testing.assert_array_equal(out, np.repeat(np.arange(nb, dtype=np.uint32), 32))

if __name__ == "__main__":
  unittest.main()
