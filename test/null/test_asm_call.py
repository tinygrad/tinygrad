import unittest
import ctypes, subprocess, tempfile, pickle, sys, itertools
from tinygrad import dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Context
from tinygrad.codegen import to_program
from tinygrad.device import Device, TinyELF
from tinygrad.engine.realize import get_runtime
from tinygrad.uop.ops import UOp, KernelInfo
from test.mockgpu.amd import call
from test.mockgpu.amd.call import cfg_regions


class TestCallJIT(unittest.TestCase):
  def test_default_names_resolve_to_their_own_module(self):
    out = UOp.param(0, dtypes.int, (2,))
    value = UOp.param(1, dtypes.int, addrspace=AddrSpace.ALU)
    for backend, name in itertools.product(("CPU:LLVM", "CPU:CLANG"), ("test", "asm_waves")):
      with self.subTest(backend=backend, name=name), Context(DEV=backend):
        renderer = Device["CPU"].renderer
        with Context(ALLOW_DEVICE_USAGE=0):
          programs = []
          for v in (7, 19):
            calls = (out[0].store(v).sink().call(out, name=name), out[1].store(value).sink().call(out, UOp.const(v+1, dtypes.int), name=name))
            programs.append(to_program(UOp.sink(*(c.sink().call(out, name=name) for c in calls), arg=KernelInfo(name)), renderer))
        self.assertEqual(programs[0].to_elf().name, programs[1].to_elf().name)
        self.assertNotEqual(programs[0].src[3].arg, programs[1].src[3].arg)
        self.assertTrue(all(not p.src[3].src for p in programs))
        if backend == "CPU:CLANG": self.assertIn("extern void __tinygrad_call_", programs[0].src[2].arg)
        for prg, expected in zip(programs, (7, 19)):
          result = (ctypes.c_int * 2)()
          get_runtime("CPU", prg)(ctypes.addressof(result))
          self.assertEqual(list(result), [expected, expected+1])

  @Context(DEV="CPU:LLVM")
  def test_static_functions_are_not_exported(self):
    from tinygrad.runtime.support.elf import elf_sections, elf_symbols
    compiler = Device["CPU"].compiler
    lib = compiler.compile_cached("define internal void @asm_fn_local(ptr %out) noinline { store i32 41, ptr %out\n ret void }\n"
                                  "define void @asm_fn_entry(ptr %out) noinline { call void @asm_fn_local(ptr %out)\n"
                                  "%x = load i32, ptr %out\n %y = add i32 %x, 1\n store i32 %y, ptr %out\n ret void }")
    # ELF permits local and global symbols to share a name; their table indices still identify distinct functions.
    lib = lib.replace(b"asm_fn_entry\x00", b"asm_fn_local\x00")
    runtime = Device["CPU"].runtime(TinyELF(lib, "asm_fn_entry", Device["CPU"].renderer.target, ()))
    symbols = elf_symbols(*elf_sections(lib))
    self.assertEqual([symbols[i][0] for i in runtime.exports], ["asm_fn_local"])
    self.assertEqual(sum(name == "asm_fn_local" for name,_ in symbols), 2)
    result = ctypes.c_int()
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(runtime.entry)(ctypes.addressof(result))
    self.assertEqual(result.value, 42)

  @Context(DEV="CPU:LLVM")
  def test_program_preserves_elf(self):
    out = UOp.param(0, dtypes.int, (1,))
    prg = to_program(out[0].store(7).sink().call(out, name="store").sink(arg=KernelInfo(name="jit_store")), Device["CPU"].renderer)
    with tempfile.NamedTemporaryFile() as f:
      f.write(prg.src[3].arg)
      f.flush()
      subprocess.run(["objdump", "-d", f.name], check=True)
    result = ctypes.c_int()
    runtime = get_runtime("CPU", prg)
    self.assertEqual(runtime.profile_key, prg.key)
    runtime(ctypes.addressof(result))
    self.assertEqual(result.value, 7)
    # The compiled artifact loads in another process without reconstructing its renderer or compiler caches.
    subprocess.run([sys.executable, "-c", """
import ctypes, pickle, sys
from tinygrad.device import Device
obj = pickle.loads(sys.stdin.buffer.read())
out = ctypes.c_int()
Device['CPU'].runtime(obj)(ctypes.addressof(out))
assert out.value == 7
"""], input=pickle.dumps(prg.to_elf()), check=True)


class TestAsmCFG(unittest.TestCase):
  def test_diamond_with_unreachable_block(self):
    paths = {0:{1:0, 2:1}, 1:{3:0}, 2:{3:0}, 3:{}, 99:{1:0}}
    regions = cfg_regions(paths, 0)
    self.assertEqual(regions[0], (0, []))
    self.assertCountEqual(regions[1:3], [(1, []), (2, [])])
    self.assertEqual(regions[3:], [(3, [])])

  def test_nested_loops(self):
    paths = {0:{1:0}, 1:{2:0, 5:1}, 2:{3:0, 4:1}, 3:{2:0}, 4:{1:0}, 5:{}}
    self.assertEqual(cfg_regions(paths, 0), [(0, []), (1, [(1, []), (2, [(2, []), (3, [])]), (4, [])]), (5, [])])

  def test_multiple_backedges(self):
    paths = {0:{1:0}, 1:{2:0, 3:1}, 2:{1:0}, 3:{1:0, 4:1}, 4:{}}
    regions = cfg_regions(paths, 0)
    self.assertEqual([pc for pc,_ in regions], [0, 1, 4])
    self.assertEqual(regions[1][1][0], (1, []))
    self.assertCountEqual(regions[1][1][1:], [(2, []), (3, [])])

  def test_self_loop(self):
    self.assertEqual(cfg_regions({0:{0:0, 1:1}, 1:{}}, 0), [(0, [(0, [])]), (1, [])])

  def test_irreducible_loop(self):
    with self.assertRaisesRegex(call.IrreducibleCFG, "irreducible"):
      cfg_regions({0:{1:0, 2:1}, 1:{2:0}, 2:{1:0, 3:1}, 3:{}}, 0)

  def test_irreducible_inside_loop(self):
    # The outer SCC has one entry, but its nested cycle has two.
    with self.assertRaisesRegex(call.IrreducibleCFG, "irreducible"):
      cfg_regions({0:{1:0}, 1:{2:0, 3:1}, 2:{3:0}, 3:{2:0, 1:1}}, 0)

  def test_nested_loop_exits(self):
    # The inner loop can leave the outer loop directly, or continue through its tail.
    paths = {0:{1:0}, 1:{2:0, 5:1}, 2:{3:0, 4:1}, 3:{2:0, 5:1}, 4:{1:0}, 5:{}}
    self.assertEqual(cfg_regions(paths, 0), [(0, []), (1, [(1, []), (2, [(2, []), (3, [])]), (4, [])]), (5, [])])

  def test_long_chain(self):
    paths = {i:{i+1:0} for i in range(4999)} | {4999:{}}
    self.assertEqual(cfg_regions(paths, 0), [(i, []) for i in range(5000)])

  def test_long_chain_of_loops(self):
    paths = {i:{i:0, i+1:1} for i in range(4999)} | {4999:{4999:0}}
    self.assertEqual(cfg_regions(paths, 0), [(i, [(i, [])]) for i in range(5000)])

  def test_loop_lift_has_stable_cache_key(self):
    from tinygrad.runtime.autogen.amd.rdna3.ins import s_mov_b32, s_add_u32, s_cmp_lt_i32, s_cbranch_scc1, s_endpgm
    from tinygrad.renderer.amd.dsl import s
    code = b"".join((s_mov_b32(s[2], 0).to_bytes(), s_add_u32(s[2], s[2], 1).to_bytes(), s_cmp_lt_i32(s[2], 4).to_bytes(),
                     s_cbranch_scc1(simm16=-3).to_bytes(), s_endpgm().to_bytes()))
    buffers = [ctypes.create_string_buffer(code) for _ in range(2)]
    first, second = [call.lift(ctypes.addressof(buf), len(code), 1, 1, 1, 32, 1, 1, 0x19c, 0, "rdna3", 0) for buf in buffers]
    self.assertEqual(first.src[0].key, second.src[0].key)

  def test_decode_past_early_exit_stops_before_allocation_padding(self):
    from tinygrad.runtime.autogen.amd.cdna.ins import s_cbranch_scc1, s_endpgm, s_nop
    code = b"".join(inst.to_bytes() for inst in (s_cbranch_scc1(simm16=1), s_endpgm(), s_nop(), s_endpgm()))
    buf = ctypes.create_string_buffer(code + b"\xff"*4096)
    args = (ctypes.addressof(buf), len(code), 1, 1, 1, 1, 1, 1, 0, 0, "cdna", 0)
    prg = call.lift(*args)
    buf[len(code)] = b"\x00"
    self.assertIs(call.lift(*args), prg)


if __name__ == "__main__": unittest.main()
