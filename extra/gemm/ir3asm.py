"""ir3 assembler for Adreno a6xx (A630).

Constructs complete QCOM shader binaries from instruction listings.
Uses a compiled "donor" kernel for the binary envelope (header, metadata,
buffer descriptors, sampler info) and replaces the shader instructions
and register counts.

Encoding reference: derived from Mesa ir3 disassembly of known-good shaders.
Instruction format: 64 bits (8 bytes) stored as two little-endian 32-bit words.
"""
import struct

# ============================================================
# HELPERS
# ============================================================

def _hreg(name):
    """Parse 'hr3.z' -> half-register number 14."""
    if isinstance(name, int): return name
    r, c = name.replace('hr','').replace('r','').split('.')
    return int(r) * 4 + 'xyzw'.index(c)

def _freg(name):
    """Parse 'r3.z' -> full-register number 14."""
    if isinstance(name, int): return name
    r, c = name.replace('r','').split('.')
    return int(r) * 4 + 'xyzw'.index(c)

def _pack(lo, hi):
    return struct.pack('<II', lo & 0xFFFFFFFF, hi & 0xFFFFFFFF)

# ============================================================
# CAT0: FLOW CONTROL
# ============================================================

def NOP(rpt=0):
    """(rptN)nop"""
    return _pack(0, (rpt & 0x7F) << 8)

def NOP_SS(rpt=0):
    """(ss)(rptN)nop -- wait until prior instructions have consumed their sources."""
    return _pack(0, 0x1000 | ((rpt & 0x7F) << 8))

def END():
    """end"""
    return _pack(0, 0x03000000)

def BR(offset, inv=True):
    """br !p0.x, #offset  (inv=True means branch when predicate is FALSE)
    offset is signed, relative to the branch instruction."""
    return struct.pack('<iI', offset, 0x00900000 if inv else 0x00800000)

def JUMP(offset):
    """jump #offset. Offset is signed, relative to the jump instruction."""
    return struct.pack('<iI', offset, 0x01000000)

# ============================================================
# CAT1: MOVE / CONVERT
# ============================================================

def MOV_S32(dst, imm, sy=False):
    """(sy?)mov.s32s32 rDST, #imm"""
    return _pack(imm, ((0x30 if sy else 0x20) << 24) | (0x55 << 16) | (0x40 << 8) | (_freg(dst) & 0xFF))

def MOV_F32(dst, src, rpt=0, sy=False, ss=False, r=False):
    """(sy?)(ss?)(rptN?)mov.f32f32 rDST, (r?)rSRC"""
    return _pack(_freg(src), (0x30044000 if sy else 0x20044000) | (0x1000 if ss else 0) |
                 (0x800 if r else 0) | ((rpt & 0x7F) << 8) | (_freg(dst) & 0xFF))

def MOV_H(dst, src, rpt=0, r=False):
    """(rptN?)mov.f16f16 hrDST, (r?)hrSRC."""
    return _pack(_hreg(src), 0x20000000 | (0x800 if r else 0) | ((rpt & 0x7F) << 8) | (_hreg(dst) & 0xFF))

def MOV_H_IMM(dst, imm_u16=0, rpt=0):
    """(rptN?)mov.f16f16 hrDST, h(imm) -- imm is raw fp16 bits (0=zero, 0x3c00=1.0)."""
    return _pack(imm_u16, 0x20400000 | ((rpt & 0x7F) << 8) | (_hreg(dst) & 0xFF))

def COV_F16F32(dst, src, sy=False, rpt=0, r=False):
    """(sy?)(rptN?)cov.f16f32 rDST, (r?)hrSRC"""
    return _pack(_hreg(src), ((0x30 if sy else 0x20) << 24) | 0x004000 | (0x800 if r else 0) |
                 ((rpt & 0x7f) << 8) | (_freg(dst) & 0xFF))

# ============================================================
# CAT2: INTEGER / FLOAT ALU (2 operands)
# ============================================================

def ADD_S(dst, src1, imm, nop=0, ss=False):
    """(ss?)(nopN?)add.s rDST, rSRC1, #imm  (signed immediate add)"""
    d, s = _freg(dst), _freg(src1)
    hi_base = 0x42300000 | (d & 0xFF)
    if nop > 0:
        hi_base = (hi_base & 0xFF00FFFF) | (0x38 << 16) | ((nop & 0x7) << 11)
    if ss: hi_base |= 0x1000
    lo = ((0x27 if imm < 0 else 0x20) << 24) | ((imm & 0xFF) << 16) | (s & 0xFF)
    return _pack(lo, hi_base)

def ADD_S_REG(dst, src1, src2, nop=0):
    """(nopN?)add.s rDST, rSRC1, rSRC2"""
    d, s1, s2 = _freg(dst), _freg(src1), _freg(src2)
    hi_base = 0x42300000 | (d & 0xFF)
    if nop > 0:
        hi_base = (hi_base & 0xFF00FFFF) | (0x38 << 16) | ((nop & 0x7) << 11)
    return _pack(((s2 & 0xFF) << 16) | (s1 & 0xFF), hi_base)

def ADD_S_CONST_REG(dst, const_src, src2, nop=0):
    """(nopN?)add.s rDST, cSRC1, rSRC2"""
    d, c1, s2 = _freg(dst), _freg(const_src.replace('c', 'r', 1)), _freg(src2)
    hi_base = 0x42300000 | (d & 0xFF)
    if nop > 0:
        hi_base = (hi_base & 0xFF00FFFF) | (0x38 << 16) | ((nop & 0x7) << 11)
    return _pack(((s2 & 0xFF) << 16) | 0x1000 | (c1 & 0xFF), hi_base)

def ADD_F(dst, src1, src2, rpt=0, r1=False, r2=False, sy=False):
    """Vector-capable add.f; full registers use the same scalar indices."""
    hi = (0x50100000 if sy else 0x40100000) | (0x800 if r1 else 0) | (0x80000 if r2 else 0)
    return _pack(((_hreg(src2) & 0xFF) << 16) | (_hreg(src1) & 0xFF),
                 hi | ((rpt & 0x7f) << 8) | (_hreg(dst) & 0xFF))

def SUB_F(dst, src1, src2, rpt=0, r1=False, r2=False, sy=False):
    """Vector-capable add.f with a negated second source."""
    hi = (0x50100000 if sy else 0x40100000) | (0x800 if r1 else 0) | (0x80000 if r2 else 0)
    return _pack(0x40000000 | ((_hreg(src2) & 0xFF) << 16) | (_hreg(src1) & 0xFF),
                 hi | ((rpt & 0x7f) << 8) | (_hreg(dst) & 0xFF))

def ADD_U(dst, src1_const, src2):
    """add.u rDST, cSRC1, rSRC2  -- src1 is constant register"""
    # From: 42100008_00031050 = add.u r2.x, c20.x, r0.w
    return _pack((_freg(src2) << 16) | 0x1050, 0x42100000 | (_freg(dst) & 0xFF))

def CMPS_S_EQ(src1, imm, nop=0):
    """(nopN?)cmps.s.eq p0.x, rSRC1, #imm"""
    hi = 0x42b400f8
    if nop > 0:
        hi = (hi & 0xFF00FFFF) | (0xb4 << 16) | ((nop & 0x7) << 11)
    # Integer immediates use the low bits of the source descriptor for bits 8+.
    # Keeping this fixed at 0x20 silently truncated loop bounds above 255.
    lo = ((0x20 | (imm >> 8)) << 24) | ((imm & 0xFF) << 16) | (_freg(src1) & 0xFF)
    return _pack(lo, hi)

def CMPS_S_LT_REG(src1, src2, nop=0):
    """(nopN?)cmps.s.lt p0.x, rSRC1, rSRC2"""
    hi = 0x42b000f8
    if nop > 0: hi = (hi & 0xFF00FFFF) | (0xb0 << 16) | ((nop & 0x7) << 11)
    return _pack(((_freg(src2) & 0xff) << 16) | (_freg(src1) & 0xff), hi)

def SHL_B(dst, src, imm, jp=False, ss=False, nop=0):
    """(ss?)(jp?)(nopN?)shl.b rDST, rSRC, #imm"""
    hi = (0x4ed00000 if jp else 0x46d00000) | (_freg(dst) & 0xFF)
    if ss: hi |= 1 << 12
    if nop & 1: hi |= 1 << 11
    if nop & 2: hi |= 1 << 19
    return _pack((0x20 << 24) | ((imm & 0xFF) << 16) | (_freg(src) & 0xFF), hi)

def SHR_B(dst, src, imm):
    """shr.b rDST, rSRC, #imm"""
    return _pack((0x20 << 24) | ((imm & 0xFF) << 16) | (_freg(src) & 0xFF), 0x46f00000 | (_freg(dst) & 0xFF))

def AND_B(dst, src, imm, nop=0):
    """(nopN?)and.b rDST, rSRC, #imm"""
    hi = 0x43900000 | (_freg(dst) & 0xFF)
    if nop & 1: hi |= 1 << 11
    if nop & 2: hi |= 1 << 19
    return _pack((0x20 << 24) | ((imm & 0xFF) << 16) | (_freg(src) & 0xFF), hi)

def AND_B_CONST(dst, src, const_src, nop=0):
    """(nopN?)and.b rDST, rSRC, cSRC2"""
    d, s, c = _freg(dst), _freg(const_src.replace('c', 'r', 1)), _freg(src)
    hi = 0x43900000 | (d & 0xFF)
    if nop & 1: hi |= 1 << 11
    if nop & 2: hi |= 1 << 19
    return _pack((0x10 << 24) | ((c & 0xFF) << 16) | (s & 0xFF), hi)

def OR_B(dst, src, imm, ss=False):
    """(ss?)or.b rDST, rSRC, #imm"""
    return _pack((0x20 << 24) | ((imm & 0xFF) << 16) | (_freg(src) & 0xFF),
                 0x43b00000 | (0x1000 if ss else 0) | (_freg(dst) & 0xFF))

def CMPS_U_LT(dst, src1, src2_const):
    """cmps.u.lt rDST, rSRC1, cSRC2"""
    # From: 42900010_10500008 = cmps.u.lt r4.x, r2.x, c20.x
    return _pack(0x10500000 | (_freg(src1) & 0xFF), 0x42900000 | (_freg(dst) & 0xFF))

def CMPS_U_LT_REG(dst, src1, src2, sy=False):
    """(sy?)cmps.u.lt rDST, rSRC1, rSRC2"""
    hi = (0x52900000 if sy else 0x42900000) | (_freg(dst) & 0xff)
    return _pack(((_freg(src2) & 0xff) << 16) | (_freg(src1) & 0xff), hi)

# ============================================================
# CAT3: MAD (3 operands)
# ============================================================

def MAD_F16(dst, src1, src2, src3, rpt=0, sy=False, r=False, r1=False, r3=False):
    """(sy?)(rptN?)mad.f16 hrDST, (r1?)hrSRC1, (r?)hrSRC2, (r?)hrSRC3
    When rpt>0, r1 auto-increments src1 and r auto-increments src2/src3/dst."""
    d, s1, s2, s3 = _hreg(dst), _hreg(src1), _hreg(src2), _hreg(src3)
    hi = ((0x73 if sy else 0x63) << 24) | ((s2 >> 1) << 16) | ((((s2 & 1) << 7) | (0x08 if r1 else 0) | (rpt & 0x7F)) << 8) | (d & 0xFF)
    lo = (0x20000000 if (r or r3) else 0) | ((s3 & 0xFF) << 16) | (0x8000 if r else 0) | (s1 & 0xFF)
    return _pack(lo, hi)

def MAD_F32(dst, src1, src2, src3, rpt=0, sy=False, r=False, r1=False):
    """(sy?)(rptN?)mad.f32 rDST, rSRC1, (r?)rSRC2, (r?)rSRC3"""
    d, s1, s2, s3 = _freg(dst), _freg(src1), _freg(src2), _freg(src3)
    hi = ((0x73 if sy else 0x63) << 24) | (0x80 << 16) | ((s2 >> 1) << 16) | \
         ((((s2 & 1) << 7) | (0x08 if r1 else 0) | (rpt & 0x7F)) << 8) | (d & 0xFF)
    lo = (0x20000000 if r else 0) | ((s3 & 0xFF) << 16) | (0x8000 if r else 0) | (s1 & 0xFF)
    return _pack(lo, hi)

def DP4ACC(dst, src1, src2, src3, sy=False, mixed=False, signed=None):
    """A6xx packed 4x int8 dot product accumulated into a full int32 register.

    ``mixed=False`` selects unsigned*unsigned. ``mixed=True`` selects the
    pre-A7xx mixed signedness mode used by A630 (signed lhs, unsigned rhs).
    The instruction has no repeat form on this generation.
    """
    if signed is not None: mixed = signed
    d, s1, s2, s3 = _freg(dst), _freg(src1), _freg(src2), _freg(src3)
    hi = ((0x76 if sy else 0x66) << 24) | (0x80 << 16) | ((s2 >> 1) << 16)
    hi |= (((s2 & 1) << 7) | 0x40) << 8
    # AL-OP is bit 13 and the pre-A7 signed/unsigned selector is bit 14.
    lo = ((s3 & 0xff) << 16) | 0x2000 | (0x4000 if mixed else 0) | (s1 & 0xff)
    return _pack(lo, hi | (d & 0xff))

# ============================================================
# CAT3: SHLG / SHRM (shift with merge)
# ============================================================

def SHLG(dst, imm, src1, src2, nop=0):
    """(nopN?)shlg rDST, #imm, rSRC1, rSRC2.

    This covers the packed image-coordinate forms emitted by the a6xx compiler
    for GEMM kernels. The low byte encodes the shift immediate and bits 23:16
    encode src2; the remaining source mode bits are pattern-specific.
    """
    d, s1, s2 = _freg(dst), _freg(src1), _freg(src2)
    if (s1, s2) in ((_freg('r0.y'), _freg('r0.z')), (_freg('r0.z'), _freg('r0.x'))):
        hi_mid, lo_mid = 0x80, 0xb0
        if (s1, s2) == (_freg('r0.z'), _freg('r0.x')): hi_mid = 0x81
    elif (s1, s2) in ((_freg('r0.w'), _freg('r0.x')), (_freg('r0.w'), _freg('r0.y'))):
        hi_mid, lo_mid = 0x81, 0x30
    else:
        raise ValueError('unsupported SHLG source pattern %s, %s' % (src1, src2))
    hi = (0x65 << 24) | (hi_mid << 16) | (0x84 << 8) | (d & 0xFF)
    lo = ((s2 & 0xFF) << 16) | (lo_mid << 8) | (imm & 0xFF)
    return _pack(lo, hi)

def SHLG_IMM(dst, imm, src, merge):
    """shlg rDST, #imm, rSRC, #merge.

    Observed in compiler address generation for widened column stores, e.g.
    65b08402_10803002 = shlg r0.z, 2, r24.y, 128.
    """
    d, s = _freg(dst), _freg(src)
    hi = (0x65 << 24) | ((0x80 | ((s >> 1) & 0x7f)) << 16) | (0x84 << 8) | (d & 0xff)
    lo = (0x10 << 24) | ((merge & 0xffff) << 16) | 0x3000 | (imm & 0xff)
    return _pack(lo, hi)

def SHRM(dst, shift, src1, merge):
    """shrm rDST, #shift, rSRC1, #merge.

    Observed compiler form for subgroup row offsets, e.g.
    64000402_100c3003 = shrm r0.z, 3, r0.x, 12.
    """
    d, s1 = _freg(dst), _freg(src1)
    if s1 != _freg('r0.x'):
        raise ValueError('unsupported SHRM source %s' % src1)
    hi = 0x64000400 | (d & 0xFF)
    lo = (0x10 << 24) | ((merge & 0xFF) << 16) | 0x3000 | (shift & 0xFF)
    return _pack(lo, hi)

# ============================================================
# CAT5: TEXTURE (ISAM)
# ============================================================

def ISAM_F16(dst, coord, tex=0, samp=0, sy=False, wrmask=0xf):
    """isam.1d (f16)(xyzw) hrDST, rCOORD, s#SAMP, t#TEX
    dst: first half-register of the xyzw quad
    coord: full-register containing the (int2) coordinate pair"""
    return _pack((tex * 2) << 24 | ((samp & 0x7) << 21) | (_freg(coord) * 2 + 1),
                 (0xb0000000 if sy else 0xa0000000) | ((wrmask & 0xf) << 8) | (_hreg(dst) & 0xFF))

def ISAM_F32(dst, coord, tex=0, samp=0):
    """isam.1d (f32)(xyzw) rDST, rCOORD, s#SAMP, t#TEX"""
    return _pack((tex * 2) << 24 | ((samp & 0x7) << 21) | (_freg(coord) * 2 + 1), 0xa0001f00 | (_freg(dst) & 0xFF))

def ISAM_U32(dst, coord, tex=0, samp=0):
    """isam.1d (u32)(xyzw) rDST, rCOORD, s#SAMP, t#TEX"""
    return _pack((tex * 2) << 24 | ((samp & 0x7) << 21) | (_freg(coord) * 2 + 1), 0xa0003f00 | (_freg(dst) & 0xFF))

def COV_S32S16(dst, src, rpt=0, r=False, sy=False):
    """cov.s32s16 hDST, rSRC, optionally repeating over four packed lanes."""
    hi = (0x30150000 if sy else 0x20150000) | ((rpt & 0x7) << 8) | (0x800 if r else 0) | (_hreg(dst) & 0xff)
    return _pack(_freg(src) & 0xff, hi)

def SHRG_H(dst, src, shift=16, rpt=0, r=False):
    """shrg hDST, #shift, rSRC, #0 for extracting packed high half lanes."""
    s = _freg(src)
    hi = 0x65004400 | (((s >> 1) & 0x7f) << 16) | ((rpt & 0x7) << 8) | (_hreg(dst) & 0xff)
    lo = 0x10003000 | (0x8000 if r else 0) | (shift & 0xff)
    return _pack(lo, hi)

def QUAD_BRCST(dst, src, idx, typ=3, wrmask=1, sy=False, jp=False):
    """quad_shuffle.brcst.{typ} DST, SRC, IDX"""
    half = typ in (0, 2, 4, 6)
    d = _hreg(dst) if half else _freg(dst)
    s = _hreg(src) if half else _freg(src)
    i = _hreg(idx) if half else _freg(idx)
    lo = (0 if half else 1) | ((s & 0xff) << 1) | ((i & 0xff) << 9)
    hi = 0xa7e00000 | ((typ & 7) << 12) | ((wrmask & 0xf) << 8) | (d & 0xff)
    if jp: hi |= 1 << 27
    if sy: hi |= 1 << 28
    return _pack(lo, hi)

# ============================================================
# CAT6: LOAD / STORE
# ============================================================

def STG_F16(addr, data_hreg, count=4, sy=False):
    """(sy?)stg.f16 g[rADDR], hrDATA, count"""
    # Encoding from compiled kernels:
    # c0c01100_04800000 = stg.f16 g[r2.x], hr0.x, 4
    # c0c01500_04800008 = stg.f16 g[r2.z], hr1.x, 4
    # c0c01900_04800010 = stg.f16 g[r3.x], hr2.x, 4
    # c0c01d00_04800018 = stg.f16 g[r3.z], hr3.x, 4
    # hi pattern: c0c0XX00 where XX encodes the address register
    # lo pattern: 048000YY where YY encodes the data register
    a, d = _freg(addr), _hreg(data_hreg)
    # addr encoding: r2.x=8 -> 0x11, r2.z=10 -> 0x15, r3.x=12 -> 0x19, r3.z=14 -> 0x1d
    # Pattern: (addr * 2 + 1) = 17,21,25,29 = 0x11,0x15,0x19,0x1d
    addr_enc = a * 2 + 1
    hi = (0xd0c00000 if sy else 0xc0c00000) | (addr_enc << 8)
    lo = 0x04800000 | ((d * 2) & 0xFF)
    return _pack(lo, hi)

def STG_U32(addr, data_reg, count=1, sy=False):
    """(sy?)stg.u32 g[rADDR], rDATA, count"""
    a, d = _freg(addr), _freg(data_reg)
    hi = (0xd0c00000 if sy else 0xc0c00000) | (3 << 17) | ((a * 2 + 1) << 8)
    lo = ((count & 0x7) << 24) | 0x00800000 | ((d << 1) & 0x1FE)
    return _pack(lo, hi)

def STG_F32(addr, data_reg, count=4, sy=False):
    """(sy?)stg.f32 g[rADDR], rDATA, count"""
    a, d = _freg(addr), _freg(data_reg)
    hi = (0xd0c00000 if sy else 0xc0c00000) | (1 << 17) | ((a * 2 + 1) << 8)
    lo = ((count & 0x7) << 24) | 0x00800000 | ((d << 1) & 0x1FE)
    return _pack(lo, hi)

def STIB_F32(data_reg, coord_reg, sy=False):
    """Typed 2D image store of float4 data to integer (x,y) coordinates."""
    hi = (0xd0220000 if sy else 0xc0220000) | (_freg(data_reg) & 0xff)
    lo = ((_freg(coord_reg) & 0xff) << 24) | 0x00677a00
    return _pack(lo, hi)

def GETFIBERID(dst):
    """getfiberid.u32 rDST"""
    return _pack(0x00c98000, 0xc0260000 | (_freg(dst) & 0xff))

def SHFL(dst, src, idx, mode=7, typ=2, sy=False, jp=False):
    """shfl.{mode}.{typ} DST, SRC, IDX

    mode: xor=1, up=2, down=3, rup=6, rdown=7.
    typ: f16=0, f32=1, u16=2, u32=3, s16=4, s32=5.
    idx can be an immediate int or a full register. For half types, dst/src are
    half-register indices; SRC2 is always a full register/immediate per Mesa.
    """
    d = _hreg(dst) if typ in (0, 2, 4, 6) else _freg(dst)
    s = _hreg(src) if typ in (0, 2, 4, 6) else _freg(src)
    if isinstance(idx, int):
        idx_im, idx_bits = 1, idx & 0xff
    else:
        idx_im, idx_bits = 0, _freg(idx) & 0xff
    lo = ((s & 0xff) << 1) | (idx_im << 23) | (idx_bits << 24)
    hi = (0xc0000000 | (0x1b << 22) | (2 << 20) | ((typ & 7) << 17) |
          ((mode & 7) << 13) | (d & 0xff))
    if jp: hi |= 1 << 27
    if sy: hi |= 1 << 28
    return _pack(lo, hi)

# ============================================================
# CAT3 SPECIAL: SAD.S32
# ============================================================

def SAD_S32(dst, src1_const, src2, src3, nop=0):
    """(nopN?)sad.s32 rDST, cSRC1, (neg)rSRC2, rSRC3"""
    # From: 67888009_40101051 = sad.s32 r2.y, c20.y, (neg)r4.y, r4.x
    d, s2, s3 = _freg(dst), _freg(src2), _freg(src3)
    hi_src2 = 0x80 | ((s2 >> 1) & 0xF)
    # Observed nop3 form uses 0x88 in the third byte; plain sad.s32 uses 0x80.
    hi_nop = 0x88 if nop > 0 else 0x80
    hi = (0x67 << 24) | (hi_src2 << 16) | (hi_nop << 8) | (d & 0xFF)
    lo = 0x40000000 | (s3 << 16) | 0x1051
    return _pack(lo, hi)

# ============================================================
# BINARY ENVELOPE
# ============================================================

def get_envelope(dev, src):
    """Compile an OpenCL kernel and return the binary as a mutable envelope."""
    lib = bytearray(dev.compiler.compile_cached(src))
    img_off = struct.unpack_from('<I', lib, 0xc0)[0]
    img_sz = struct.unpack_from('<I', lib, 0x100)[0]
    reg_off = struct.unpack_from('<I', lib, 0x34)[0]
    return lib, img_off, img_sz, reg_off

def inject(lib, img_off, img_sz, reg_off, shader_bytes, fregs, hregs, mergedregs=None):
    """Replace shader binary and register counts in the envelope."""
    lib = bytearray(lib)
    shader = bytearray(shader_bytes)
    if len(shader) > img_sz:
        raise ValueError(f"shader is {len(shader)} bytes but donor image is only {img_sz} bytes")
    # Pad to original size
    while len(shader) < img_sz:
        shader += NOP()
    lib[img_off:img_off+img_sz] = shader[:img_sz]
    if mergedregs is True: fregs |= 1 << 31
    if mergedregs is False: hregs |= 1 << 31
    struct.pack_into('<I', lib, reg_off + 0x14, fregs)
    struct.pack_into('<I', lib, reg_off + 0x18, hregs)
    return bytes(lib)

def disasm(shader_bytes, gpu_id=630):
    """Disassemble shader binary using Mesa's ir3_isa_disasm."""
    import ctypes, tempfile
    from tinygrad.runtime.autogen import mesa
    from tinygrad.helpers import data64
    with tempfile.TemporaryFile('w+', buffering=1) as tf:
        @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
        def hd(data, n, instr):
            fst, snd = data64(ctypes.cast(instr, ctypes.POINTER(ctypes.c_uint64)).contents.value)
            print(f"{n:04} [{fst:08x}_{snd:08x}] ", end="", flush=True, file=tf)
        libc = ctypes.CDLL(None)
        libc.setlinebuf(fp:=ctypes.cast(libc.fdopen(tf.fileno(), b"w"), ctypes.POINTER(mesa.struct__IO_FILE)))
        mesa.ir3_isa_disasm(bytes(shader_bytes), len(shader_bytes), fp, mesa.struct_isa_decode_options(gpu_id, True, 0, True, pre_instr_cb=hd))
        tf.seek(0)
        return tf.read()

def assemble(instr_list):
    """Assemble a list of instruction bytes into a shader binary."""
    return b''.join(instr_list)
