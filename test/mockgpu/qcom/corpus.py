"""Checked A630 IR3 regression literals.

These are fixed little-endian 64-bit machine words, never captured UOps.
Each entry records the disassembly used to select it and its provenance class.
"""
import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class IR3Literal:
  name: str
  words: tuple[int, ...]
  disassembly: tuple[str, ...]
  provenance: str
  category: str

  def __post_init__(self):
    if not self.words or len(self.words) != len(self.disassembly) or any(not 0 <= word < 1 << 64 for word in self.words):
      raise ValueError(f"invalid IR3 literal {self.name}")


def _literal(name, words, disassembly, provenance, category):
  return IR3Literal(name, tuple(words), tuple(disassembly), provenance, category)


def ir3_program(*instructions: int | IR3Literal) -> bytes:
  words = [word for instruction in instructions for word in (instruction.words if isinstance(instruction, IR3Literal) else (instruction,))]
  return struct.pack(f"<{len(words)}Q", *words)


END = _literal("end", (0x0300000000000000,), ("end",), "A630 ISA control encoding", "control")

# Cat0 and Cat1: compiler-emitted scalar setup, conversion, and control flow.
LOG2_ZERO = _literal("log2_zero", (0x9050000200000002,), ("log2 r0.z, r0.z",), "A630 ISA regression literal", "cat4-sfu")
MOVS_BROADCAST_IMMEDIATE = _literal("movs_immediate", (0x200440C0AE800004,), ("movs.f32f32 r48.x, r1.x, 93",),
  "Mesa compiler output", "cat1-movs")
MOVS_BROADCAST_A0 = _literal("movs_a0", (0x201100C0C000040B,), ("movs.s16s16 hr48.x, hr2.w, a0.x",),
  "Mesa compiler output", "cat1-movs")
SHRG_HALF_DEST = _literal("shrg_half_dest", (0x6502C401000B3010,), ("shrg hr0.y, 16, r1.y, r2.w",),
  "Mesa compiler output", "cat3")
CLZ_ZERO = _literal("clz_zero", (0x46B0080A00000006,), ("clz.b r2.z, r1.z",), "Mesa compiler output", "cat2")
SIGNED_BYTE_TO_HALF = _literal("signed_byte_to_half", (0x3019000000000000, 0x2010000000000000),
  ("cov.u8s16 hr0.x, hr0.x", "cov.s16f16 hr0.x, hr0.x"), "Mesa compiler output", "cat1-conversion")
MULL_REPEAT = _literal("mull_repeat", (0x46580103000300C1,), ("(rpt1)mull.u r0.w, r48.y, (r)r0.w",),
  "Mesa compiler output", "cat2-repeat")
RELATIVE_DESTINATION = _literal("relative_destination", (0x2017400400000008,), ("mov.s32s32 r<a0.x + 4>, r2.x",),
  "Mesa IR3 disassembler regression", "cat1-relative")
RELATIVE_SOURCE_REPEAT = _literal("relative_source_repeat", (0x200CCB1000000804,),
  ("(rpt3) mov.u32u32 r4.x, (r)r<a0.x + 4>",), "Mesa IR3 disassembler regression", "cat1-relative")

# Cat2/Cat3/Cat4 ALU literals, including compiler-generated repeat sequences.
FLOAT_LUT_MUL_PI = _literal("float_lut_mul_pi", (0x4070000028052803,), ("mul.f r0.x, (2.0), (pi)",),
  "A630 ISA regression literal", "cat2-float-lut")
REPEATED_ADD = _literal("repeated_add", (0x42180B2000100000,), ("(rpt3)add.u r8.x, (r)r0.x, (r)r4.x",),
  "A630 ISA regression literal", "cat2-repeat")
HALF_SOURCE_MUL = _literal("half_source_mul", (0x4060400800040000,), ("mul.f r2.x, hr0.x, hr1.x",),
  "A630 ISA regression literal", "cat2-half-source")
FP8_HALF_COMPARE = _literal("fp8_half_compare", (0x40A0000C10140003,), ("cmps.f.lt hr3.x, hr0.w, hc5.x",),
  "Mesa compiler fp8 conversion output", "cat2-half-constant")
FLOAT_ALU_COMPARE = _literal("float_alu_compare", (0x5010000E0002000A, 0x4070000E0002000E, 0x40B04000000A000E),
  ("(sy)add.f r3.z, r2.z, r0.z", "mul.f r3.z, r3.z, r0.z", "cmps.f.lt hr0.x, r3.z, r2.z"),
  "Mesa compiler output", "cat2-float")
BACKWARD_BRANCH = _literal("backward_branch", (0x4250000020010000, 0x429500F820000000, 0x00800000FFFFFFFE),
  ("sub.u r0.x, r0.x, 1", "cmps.u.ne p0.x, r0.x, 0", "br p0.x, #-2"), "A630 ISA regression literal", "control")
MESA_STD_HOT_PREHEADER = _literal("mesa_std_hot_preheader", (0x202CC00000000004, 0x202CC00100000005, 0x204CC00300000000,
  0x204CC00400000000, 0x0000030000000000, 0xC006000201800001),
  ("mov.u32u32 r0.x, c1.x", "mov.u32u32 r0.y, c1.y", "mov.u32u32 r0.w, 0", "mov.u32u32 r1.x, 0", "nop",
   "ldg.u32 r0.z, g[r0.x], 1"), "Mesa compiler output from x.std() backward shape (15,25,35)", "preheader-cat1-cat6")
MESA_STD_HOT_LOOP = _literal("mesa_std_hot_loop", (0x47100005201F0003, 0x46D8000620020003, 0x46D0000520020005,
  0x4218000700061002, 0x650184050005301E, 0x4290500010020007, 0x4210080320010003, 0x4210000500051003,
  0x2009400900000000, 0x42BB00F810100003, 0x4218080800050009, 0x0000020000000000, 0xC006000A0181C001,
  0x5018080A4002000A, 0x638500040004000A, 0x0080000000000002, 0x01000000FFFFFFF0),
  ("ashr.b r1.y, r0.w, 31", "shl.b r1.z, r0.w, 2", "shl.b r1.y, r1.y, 2", "add.u r1.w, c0.z, r1.z",
   "shrg r1.y, 30, r0.w, r1.y", "cmps.u.lt hr0.x, r1.w, c0.z", "add.u r0.w, r0.w, 1", "add.u r1.y, c0.w, r1.y",
   "mov.s16s32 r2.y, hr0.x", "cmps.s.ge p0.x, r0.w, c4.x", "add.u r2.x, r2.y, r1.y", "nop",
   "ldg.u32 r2.z, g[r1.w], 1", "(sy)add.f r2.z, r2.z, r0.z", "mad.f32 r1.x, r2.z, r2.z, r1.x", "br p0.x, #2", "jump #-16"),
  "Mesa compiler output from x.std() backward shape (15,25,35)", "natural-loop-cat2-cat3-cat6")
MADSH_MAGIC_DIVIDE = _literal("madsh_magic_divide", (0x6182000C00080000, 0x46F000102011000C),
  ("madsh.m16 r3.x, r0.x, r1.x, r2.x", "shr.b r4.x, r3.x, 17"), "Mesa compiler output", "cat3-madsh")
MADSH_REPEAT = _literal("madsh_repeat", (0x61878B2020198014,),
  ("(rpt3)madsh.m16 r8.x, (r)r5.x, (r)r3.w, (r)r6.y",), "Mesa compiler output", "cat3-madsh-repeat")
SFU_EXP2_REPEAT = _literal("sfu_exp2_repeat", (0x80700B1000000000,), ("(rpt3)exp2 r4.x, (r)r0.x",),
  "A630 ISA regression literal", "cat4-sfu-repeat")
BITWISE_SHIFT_COMPARE = _literal("bitwise_shift_compare", (0x53B0000E000A0002, 0x43900002000A0002, 0x43F00002000E0002,
  0x43D0000F00000002, 0x46D000162003000F, 0x4710001720040016, 0x42B00B1E20000017),
  ("(sy)or.b r3.z, r0.z, r2.z", "and.b r0.z, r0.z, r2.z", "xor.b r0.z, r0.z, r3.z", "not.b r3.w, r0.z",
   "shl.b r5.z, r3.w, 3", "ashr.b r5.w, r5.z, 4", "(rpt3)cmps.s.lt r7.z, (r)r5.w, 0"),
  "Mesa compiler bitwise/select kernel output", "cat2-bitwise-shift-compare")
SWZ_SWAP = _literal("swz_swap", (0x240CC00400000400,), ("swz.u32u32 r1.x, r0.x, r0.x, r1.x",),
  "A630 ISA regression literal", "cat1-swizzle")
UNSUPPORTED_UL_ADD = _literal("unsupported_ul_add", (0x4210200420010000,), ("(ul)add.u r1.x, r0.x, 1",),
  "A630 ISA invalid-modifier regression", "cat2-invalid")
INVALID_FLOAT_LUT = _literal("invalid_float_lut", (0x407000002805280C,), ("mul.f r0.x, invalid-float-lut(12), (pi)",),
  "A630 ISA invalid-immediate regression", "cat2-invalid")

# Cat6/Cat7 memory, image, and synchronization literals.
GLOBAL_LOAD = _literal("global_load", (0xC006000001810001,), ("ldg.u32 r0.x, g[r1.x], 1",),
  "A630 ISA regression literal", "cat6-global")
PREDICATED_GLOBAL_LOAD = _literal("predicated_global_load", (0x0682000000000000, 0xC006000001810001, 0x0782000000000000),
  ("predt", "ldg.u32 r0.x, g[r1.x], 1", "prede"), "A630 ISA regression literal", "control-and-cat6")
PRIVATE_SIGNED_SPILL = _literal("private_signed_spill", (0xC1465BA001803E2A, 0xC086000801860001),
  ("stp.u32 p[r11.y-96], r5.y, 1", "ldp.u32 r2.x, p[r6.x], 1"), "Mesa compiler output", "cat6-private")
PRIVATE_LANE_SPILL = _literal("private_lane_spill", (0xC146071001800000, 0xC08600000180C021),
  ("stp.u32 p[r0.w+16], r0.x, 1", "ldp.u32 r0.x, p[r0.w+16], 1"), "Mesa compiler output", "cat6-private")
IMAGE_STORE = _literal("image_store", (0xC020000004677A00,), ("stib.b.typed.2d.f16.4.imm hr0.x, r1.x, 0",),
  "Mesa compiler output", "cat6-image")
IMAGE_SAMPLE = _literal("image_sample", (0xA0001F0B0000000F,), ("isam.1d (f32)(xyzw)r2.w, r1.w, s#0, t#0",),
  "Mesa compiler output", "cat5-image")
PARTIAL_WAVE_ADD = _literal("partial_wave_add", (0x4210000420010008,), ("add.u r1.x, r2.x, 1",),
  "A630 ISA regression literal", "cat2")
PREDICATION = _literal("predication", (0x0682000000000000, 0x4210000020010000, 0x0702000000000000,
  0x4210000420020004, 0x0782000000000000, 0x4210000820030008),
  ("predt", "add.u r0.x, r0.x, 1", "predf", "add.u r1.x, r1.x, 2", "prede", "add.u r2.x, r2.x, 3"),
  "A630 ISA regression literal", "control-and-cat2")
DIVERGENT_BRANCH = _literal("divergent_branch", (0x0080000000000003, 0x42100000200A0000, 0x0100000000000003,
  0x4210000020140000, 0x0000000000000000, 0x4210000420010000),
  ("br p0.x, #3", "add.u r0.x, r0.x, 10", "jump #3", "add.u r0.x, r0.x, 20", "nop", "add.u r1.x, r0.x, 1"),
  "A630 ISA regression literal", "control-and-cat2")
SHARED_BARRIER = _literal("shared_barrier", (0x46D0000820020000, 0x4210000C20010000, 0xC106110001800018,
  0xE002000000000000, 0xC046001401840001),
  ("shl.b r2.x, r0.x, 2", "add.u r3.x, r0.x, 1", "stl.u32 l[r2.x], r3.x, 1", "bar", "ldl.u32 r5.x, l[r4.x], 1"),
  "A630 ISA regression literal", "cat2-cat6-cat7")
GLOBAL_ATOMIC_ADD = _literal("global_atomic_add", (0xC416000C08010001,),
  ("atomic.g.add.untyped.1d.u32.1.g r3.x, r1.x, r2.x",), "A630 ISA regression literal", "cat6-atomic")
