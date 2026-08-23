import ctypes, unittest
from test.mockgpu.ptx.emu import parse_ptx, ptx_run

PLUS_F32 = r"""
.version 7.5
.target sm_80
.address_size 64
.visible .entry E_3 (
	.param .u64 data0,
	.param .u64 data1,
	.param .u64 data2
)
.maxntid 3
{
	.reg .s32 %cast_s32_<1>;
	.reg .u64 %dat_u64_<3>;
	.reg .u64 %bidx_u64_<3>;
	.reg .f32 %val_f32_<2>;
	.reg .f32 %alu_f32_<1>;
	.reg .u32 %lidx0;
	mov.b32		%cast_s32_0, 3;
	ld.param.u64	%dat_u64_0, [data0+0];
	ld.param.u64	%dat_u64_1, [data1+0];
	ld.param.u64	%dat_u64_2, [data2+0];
	mov.u32		%lidx0, %tid.x;
	cvt.s64.s32	%bidx_u64_0, %lidx0;
	mad.lo.s64	%bidx_u64_0, %bidx_u64_0, 4, %dat_u64_1;
	ld.global.f32	%val_f32_0, [%bidx_u64_0+0];
	cvt.s64.s32	%bidx_u64_1, %lidx0;
	mad.lo.s64	%bidx_u64_1, %bidx_u64_1, 4, %dat_u64_2;
	ld.global.f32	%val_f32_1, [%bidx_u64_1+0];
	cvt.s64.s32	%bidx_u64_2, %lidx0;
	mad.lo.s64	%bidx_u64_2, %bidx_u64_2, 4, %dat_u64_0;
	add.f32		%alu_f32_0, %val_f32_0, %val_f32_1;
	st.global.f32	[%bidx_u64_2+0], %alu_f32_0;
	ret;
}
"""

class TestPTXEmu(unittest.TestCase):
  def test_parse_plus(self):
    k = parse_ptx(PLUS_F32)
    self.assertEqual(k.name, "E_3")
    self.assertEqual([n for n,_ in k.params], ["data0","data1","data2"])
    self.assertTrue(any(i.op=='add' for i in k.insts))

  def test_plus_f32(self):
    out = (ctypes.c_float*3)(0,0,0)
    a = (ctypes.c_float*3)(1,2,3)
    b = (ctypes.c_float*3)(4,5,6)
    args = (ctypes.c_void_p*3)(ctypes.addressof(out), ctypes.addressof(a), ctypes.addressof(b))
    ptx_run(PLUS_F32, 3, args, 3, 1, 1, 1, 1, 1, 0)
    self.assertEqual(list(out), [5.0, 7.0, 9.0])

  def test_shared_reduce(self):
    src = r"""
.visible .entry r (
	.param .u64 data0,
	.param .u64 data1
)
{
	.reg .u64 %a, %b, %p, %q;
	.reg .f32 %v, %acc, %t;
	.reg .u32 %tid, %off;
	.reg .s32 %i;
	.reg .pred %p0;
	.shared .align 16 .b8 sm[16];
	ld.param.u64 %a, [data0+0];
	ld.param.u64 %b, [data1+0];
	mov.u64 %p, sm[0];
	mov.u32 %tid, %tid.x;
	cvt.s64.s32 %q, %tid;
	mad.lo.s64 %q, %q, 4, %b;
	ld.global.f32 %v, [%q+0];
	cvt.s64.s32 %q, %tid;
	mad.lo.s64 %q, %q, 4, %p;
	st.shared.f32 [%q+0], %v;
	bar.sync 0;
	mov.b32 %acc, 0f00000000;
	mov.u32 %i, -1;
	bra END;
LOOP:
	cvt.s64.s32 %q, %i;
	mad.lo.s64 %q, %q, 4, %p;
	ld.shared.f32 %t, [%q+0];
	add.f32 %acc, %acc, %t;
END:
	add.s32 %i, %i, 1;
	setp.lt.s32 %p0, %i, 4;
	@%p0 bra LOOP;
	setp.eq.s32 %p0, %tid, 0;
	@!%p0 bra SKIP;
	st.global.f32 [%a+0], %acc;
SKIP:
	ret;
}
"""
    out = (ctypes.c_float*1)(0)
    inp = (ctypes.c_float*4)(1,2,3,4)
    args = (ctypes.c_void_p*2)(ctypes.addressof(out), ctypes.addressof(inp))
    ptx_run(src, 2, args, 4, 1, 1, 1, 1, 1, 0)
    self.assertEqual(out[0], 10.0)

if __name__ == '__main__':
  unittest.main()
