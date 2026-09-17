import ctypes, struct, unittest
from types import SimpleNamespace
from typing import ClassVar, cast
from tinygrad.device import TinyELF
from tinygrad.codegen import to_program
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.nir import IR3Renderer
from tinygrad.runtime.support.compiler_mesa import IR3Compiler
from tinygrad.runtime.ops_qcom import QCOMDevice, QCOMProgramData
from tinygrad.uop.ops import KernelInfo, UOp
from test.mockgpu.qcom.emu import Memory, decode, f32bits, run_scalar, run_thread

class TestA630Scalar(unittest.TestCase):
  renderer:ClassVar[IR3Renderer]
  artifact:ClassVar[TinyELF]
  program:ClassVar[bytes]
  @classmethod
  def setUpClass(cls):
    cls.renderer = IR3Renderer(Target('QCOM', 'IR3', 'a630'))
    output, source = UOp.param(0, dtypes.float, 1), UOp.param(1, dtypes.float, 1)
    index = UOp.const(0)
    sink = output.index(index).store(source.index(index).load()+1.0).sink(arg=KernelInfo(name='a630_add_one'))
    cls.artifact = to_program(sink, cls.renderer).to_elf()
    cls.program = IR3Compiler.unpack_lib(cls.artifact.lib)[3]

  def arguments(self, artifact:TinyELF, addresses:tuple[int, ...]) -> tuple[int, ...]:
    # Use the real runtime's compiled metadata; this unit test does not exercise a device or command queue.
    data = QCOMProgramData(cast(QCOMDevice, SimpleNamespace(renderer=self.renderer)), artifact)
    args = bytearray(data.kernargs_alloc_size)
    for value,offset,size in data.consts_info: struct.pack_into('<I' if size == 4 else '<H', args, offset, value)
    for index,address in enumerate(addresses): struct.pack_into('<Q', args, data.buf_off+index*8, address)
    return tuple(struct.unpack('<'+'I'*(len(args)//4), args))

  def test_real_compiler_instructions(self):
    names = [instruction.name for instruction in decode(self.program)]
    for required in ('mov', 'ldg', 'add.f', 'stg', 'end'): self.assertIn(required, names)

  def test_add_one_executes_compiled_a630(self):
    for value in (0.0, 1.5, -2.0, 100.0):
      with self.subTest(value=value):
        source, output = (ctypes.c_float*1)(value), (ctypes.c_float*1)(-999.0)
        out_address, in_address = ctypes.addressof(output), ctypes.addressof(source)
        constants = self.arguments(self.artifact, (out_address, in_address))
        memory = Memory(((out_address, ctypes.sizeof(output)), (in_address, ctypes.sizeof(source))))
        run_scalar(self.program, constants, memory)
        self.assertEqual(output[0], value+1.0)
        self.assertEqual(source[0], value)

  def test_scalar_arithmetic(self):
    output, source = UOp.param(0, dtypes.float, 1), UOp.param(1, dtypes.float, 1)
    index = UOp.const(0)
    value = source.index(index).load()
    cases = (('multiply', value*3.0, lambda x: x*3.0),
             ('add constant', value+0.33333334, lambda x: x+ctypes.c_float(0.33333334).value),
             ('negate', -value, lambda x: -x), ('subtract', value-1.0, lambda x: x-1.0))
    for label,expression,reference in cases:
      artifact = to_program(output.index(index).store(expression).sink(arg=KernelInfo(name='a630_arithmetic')), self.renderer).to_elf()
      program = IR3Compiler.unpack_lib(artifact.lib)[3]
      for x in (0.0, -0.0, 1.5, -2.0, 100.0):
        with self.subTest(operation=label, x=x):
          inp, out = (ctypes.c_float*1)(x), (ctypes.c_float*1)(-999.0)
          addresses = (ctypes.addressof(out), ctypes.addressof(inp))
          constants = self.arguments(artifact, addresses)
          run_scalar(program, constants, Memory(tuple((address, 4) for address in addresses)))
          self.assertEqual(struct.pack('<f', out[0]), struct.pack('<f', ctypes.c_float(reference(x)).value))
          self.assertEqual(struct.pack('<f', inp[0]), struct.pack('<f', x))

  def test_compiled_vector_add(self):
    count = 8
    output, source = UOp.param(0, dtypes.float, count), UOp.param(1, dtypes.float, count)
    index = UOp.range(count, 0)
    sink = output.index(index).store(source.index(index).load()+1.0).end(index).sink(arg=KernelInfo(name='a630_vector'))
    program = to_program(sink, self.renderer)
    artifact = program.to_elf()
    data = QCOMProgramData(cast(QCOMDevice, SimpleNamespace(renderer=self.renderer)), artifact)
    self.assertEqual(program.arg.global_size, (1, 1, 1))
    self.assertEqual(program.arg.local_size[1:], (1, 1))
    inp, out = (ctypes.c_float*count)(*range(count)), (ctypes.c_float*count)(*([-999.0]*count))
    addresses = ctypes.addressof(out), ctypes.addressof(inp)
    constants = self.arguments(artifact, addresses)
    memory = Memory(tuple((address, count*4) for address in addresses))
    for local_x in range(program.arg.local_size[0]):
      run_scalar(data.image, constants, memory, {data.lid: local_x, data.lid+1: 0, data.lid+2: 0})
    self.assertEqual(list(out), [float(i+1) for i in range(count)])
    self.assertEqual(list(inp), [float(i) for i in range(count)])

  def test_a630_mad_rounds_product_before_addition(self):
    output, source = UOp.param(0, dtypes.float, 1), UOp.param(1, dtypes.float, 3)
    a,b,c = [source.index(UOp.const(i)).load() for i in range(3)]
    sink = output.index(UOp.const(0)).store(a*b+c).sink(arg=KernelInfo(name='a630_mad'))
    artifact = to_program(sink, self.renderer).to_elf()
    program = IR3Compiler.unpack_lib(artifact.lib)[3]
    self.assertIn('mad.f32', [instruction.name for instruction in decode(program)])
    # Unfused rounds (1+2^-23)*(1-2^-23) to 1 before subtracting 1; fused would leave -2^-46.
    inp, out = (ctypes.c_float*3)(1+2**-23, 1-2**-23, -1), (ctypes.c_float*1)(-999)
    addresses = ctypes.addressof(out), ctypes.addressof(inp)
    run_scalar(program, self.arguments(artifact, addresses), Memory(((addresses[0], 4), (addresses[1], 12))))
    self.assertEqual(struct.pack('<f', out[0]), b'\x00\x00\x00\x00')

  def test_predication_gates_writes(self):
    # Mesa IR3 category-0 encodings plus the MOV encoding emitted by the compiler smoke kernel.
    mov = 0x202cc00300000000  # mov.u32u32 r0.w, c0.x
    prede, end = 0x0782000000000000, 0x0300000000000000
    for instruction,mode in ((0x0682000000000000, True), (0x0702000000000000, False)):
      for predicate in (0, 1):
        with self.subTest(mode=mode, predicate=predicate):
          program = struct.pack('<4Q', instruction, mov, prede, end)
          thread = run_scalar(program, (99,), Memory(()), {248: predicate, 3: 37})
          self.assertEqual(thread.regs[3], 99 if bool(predicate) == mode else 37)

  def test_jump_point_refreshes_predication_mask(self):
    predt, prede, end = 0x0682000000000000, 0x0782000000000000, 0x0300000000000000
    write_predicate = 0x202cc0f800000000  # mov.u32u32 p0.x, c0.x
    mov = 0x202cc00300000001  # mov.u32u32 r0.w, c0.y
    for jump_point in (False, True):
      with self.subTest(jump_point=jump_point):
        program = struct.pack('<5Q', predt, write_predicate, mov | (int(jump_point) << 59), prede, end)
        thread = run_scalar(program, (0, 99), Memory(()), {248: 1, 3: 37})
        self.assertEqual(thread.regs[3], 37 if jump_point else 99)

  def test_signed_inline_immediate(self):
    # Compiled add.u r2.z, r0.x, -1. Mesa exposes the raw 11-bit immediate as 2047.
    program = struct.pack('<2Q', 0x4210000a27ff0000, 0x0300000000000000)
    for value in (0, 1, 8):
      with self.subTest(value=value):
        thread = run_scalar(program, (), Memory(()), {0: value})
        self.assertEqual(thread.regs[10], (value-1) & 0xffffffff)

  def test_parallel_register_moves(self):
    # Category-1 multi-move encodings, Mesa ir3-cat1.xml: all sources are read before any destination is written.
    base, end = (1 << 61) | (2 << 57) | (3 << 50) | (3 << 46), 0x0300000000000000
    cases = ((base | (1 << 16) | (0 << 8) | 1, [20,10,30,40]),
             (base | (1 << 40) | (0 << 24) | (1 << 16) | (2 << 8) | 3, [40,30,20,10]),
             (base | (2 << 40) | (0 << 24) | (1 << 16) | (2 << 8) | (3 << 32), [40,30,20,10]))
    for instruction,expected in cases:
      with self.subTest(instruction=hex(instruction)):
        thread = run_scalar(struct.pack('<2Q', instruction, end), (), Memory(()), {0:10, 1:20, 2:30, 3:40})
        self.assertEqual(thread.regs[:4], expected)

  def test_call_and_return(self):
    # Start at a call whose helper precedes the kernel, as in vendor-compiled OpenCL math libraries.
    mov, ret, call, end = 0x202cc00000000000, 0x0200000000000000, 0x01800000fffffffe, 0x0300000000000000
    worker = run_thread(struct.pack('<4Q', mov, ret, call, end), (19,), Memory(()), start=2)
    with self.assertRaises(StopIteration) as finished: next(worker)
    self.assertEqual(finished.exception.value.regs[0], 19)

  def test_relative_constant_address(self):
    # mova a0.x, 1; mov.s32s32 r1.w, c<a0.x+96>; end, as emitted by the OpenCL math library.
    program = struct.pack('<3Q', 0x205100f400000001, 0x2015600700000c60, 0x0300000000000000)
    constants = [0]*128
    constants[96], constants[97] = 123, 456
    thread = run_scalar(program, tuple(constants), Memory(()), {96:789})
    self.assertEqual(thread.regs[7], 456)

  def test_half_constant_modes(self):
    program = struct.pack('<2Q', 0x2020000000000001, 0x0300000000000000)  # mov.f16f16 hr0.x, hc0.y; end
    for demotion,expected in ((False, 0x3800), (True, 0x3c00)):
      with self.subTest(demotion=demotion):
        worker = run_thread(program, (0x38000000, 0x3f800000), Memory(()), constant_demotion=demotion)
        with self.assertRaises(StopIteration) as finished: next(worker)
        self.assertEqual(finished.exception.value.half_regs[0], expected)

  def test_byte_conversion_sign_extends(self):
    # mov.u8u8 hr0.y, 253; cov.u8s16 hr0.x, hr0.y; end. The COV instruction sign-extends despite the U8 name.
    program = struct.pack('<3Q', 0x20598001000000fd, 0x2019000000000001, 0x0300000000000000)
    self.assertEqual(run_scalar(program, (), Memory(())).half_regs[0], 0xfffd)

  def test_inverted_comparison(self):
    # Qualcomm emits bit 42 for !(a < b), including the unordered/NaN case. Mesa displays this comparison flag as (sat).
    program = struct.pack('<2Q', 0x40b004f800010000, 0x0300000000000000)
    for value in (-2.0, 0.0, 1.0, 2.0, float('nan'), float('inf')):
      with self.subTest(value=value):
        thread = run_scalar(program, (), Memory(()), {0:f32bits(value), 1:f32bits(1.0)})
        self.assertEqual(thread.regs[248], int(not value < 1.0))

  def test_clz_zero_sentinel(self):
    program = struct.pack('<2Q', 0x46b0000100000000, 0x0300000000000000)  # clz.b r0.y, r0.x; end
    for value,expected in ((0, 0xffffffff), (1, 31), (0x80000000, 0), (0xffffffff, 0)):
      with self.subTest(value=value):
        self.assertEqual(run_scalar(program, (), Memory(()), {0:value}).regs[1], expected)

  def test_mad_u16_full_register_form(self):
    # Emitted for (uint_a & 65535) * (uint_b & 65535) + uint_c; Mesa 25 labels these full-GPR sources as half.
    program = struct.pack('<2Q', 0x7003400200030002, 0x0300000000000000)
    for a,b,c in ((3,4,5), (0x12345678,0xabcd9876,0x34567890)):
      with self.subTest(a=a, b=b, c=c):
        result = run_scalar(program, (), Memory(()), {2:a, 6:b, 3:c})
        self.assertEqual(result.regs[2], ((a & 65535)*(b & 65535)+c) & 0xffffffff)

  def test_negative_private_store_offset(self):
    # Compiled stp.f32 p[r45.w-60], r0.x, 1. Mesa's callback returns raw 13-bit offset 8132.
    scratch = ctypes.create_string_buffer(64)
    pointer = ctypes.addressof(scratch)
    program = struct.pack('<2Q', 13926097360389160448, 0x0300000000000000)
    worker = run_thread(program, (), Memory(((pointer, 64),)), {183:60, 0:f32bits(3.5)}, private=pointer)
    with self.assertRaises(StopIteration): next(worker)
    self.assertEqual(ctypes.c_float.from_buffer(scratch).value, 3.5)
    self.assertEqual(scratch.raw[4:], bytes(60))

  def test_integer_to_float_rounding(self):
    # COV's default mode is toward zero; mode 1 is nearest-even. These integers lie exactly between adjacent float32 values.
    for value,expected in ((16777219, (16777218,16777220,16777220,16777218)),
                           (-16777219, (-16777218,-16777220,-16777218,-16777220))):
      for rounding,target in enumerate(expected):
        with self.subTest(value=value, rounding=rounding):
          program = struct.pack('<2Q', 0x2014400100000000 | (rounding << 55), 0x0300000000000000)
          self.assertEqual(run_scalar(program, (), Memory(()), {0:value}).regs[1], f32bits(target))

  def test_signed_selection_includes_zero(self):
    # Vendor-compiled indexing uses SEL.S32 to implement index >= 0 ? index : index+length.
    program = struct.pack('<2Q', 0x6580800300020000, 0x0300000000000000)
    for selector,expected in ((-2, 9), (0, 7), (2, 7)):
      with self.subTest(selector=selector):
        self.assertEqual(run_scalar(program, (), Memory(()), {0:7, 1:selector, 2:9}).regs[3], expected)

if __name__ == '__main__': unittest.main()
