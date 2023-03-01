from tinygrad.helpers import getenv, DEBUG
import hashlib
import ctypes
import llvmlite.binding as llvm  # type: ignore
from llvmlite import ir

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
llvm.initialize_native_asmparser()
target = llvm.Target.from_triple(llvm.get_process_triple())
optimizer = llvm.create_module_pass_manager()
target_machine = target.create_target_machine(opt=2)  # this opt actually can change things. ex: opt=3 means no FMA, opt=2 means FMA
target_machine.add_analysis_passes(optimizer)

# TODO: this makes compile times so much faster
if getenv("LLVMOPT"):
  llvm.set_option(str(), '-force-vector-interleave=4')  # this makes sum the same speed as torch, it also doubles the (slow) conv speed
  if DEBUG >= 4:
    llvm.set_option(str(), '--debug-only=loop-vectorize')
  #llvm.set_option(str(), '--debug')

  # does this do anything?
  builder = llvm.create_pass_manager_builder()
  builder.opt_level = 3
  builder.size_level = 0
  builder.loop_vectorize = True
  builder.slp_vectorize = True
  builder.populate(optimizer)

target_machine.set_asm_verbosity(True)
backing_mod = llvm.parse_assembly(str())
backing_mod.triple = llvm.get_process_triple()
engine = llvm.create_mcjit_compiler(backing_mod, target_machine)

def compile_llvm(module:ir.Module, buf_count:int):
  module.triple = llvm.get_process_triple()
  module.data_layout = engine.target_data
  llvm_ir = str(module)

  if DEBUG >= 2: print("LLVM IR:", llvm_ir)
  mod = llvm.parse_assembly(llvm_ir)
  mod.verify()
  optimizer.run(mod)
  if DEBUG >= 4: print("Optimized IR:", str(mod))
  mod.name = hashlib.sha1(llvm_ir.encode('utf-8')).hexdigest()
  if DEBUG >= 3: print("Assembly:", target_machine.emit_assembly(mod))
  engine.add_module(mod)
  engine.finalize_object()

  # call function (NOTE: if the types don't match, there's likely something wrong with the cache)
  #cfunc = CFUNCTYPE(ctypes.c_int, *[type(x._buf) for x in bufs])(LLVM.engine.get_function_address('exec'))
  # why is this needed without the types. fixed tests below
  # LLVM=1 OPT=2 python3 test/test_ops.py TestOps.test_cat TestOps.test_multicat
  cfunc = ctypes.CFUNCTYPE(ctypes.c_int, *[ctypes.POINTER(ctypes.c_float) for i in range(buf_count)])(engine.get_function_address('exec'))

  engine.remove_module(mod)
  return cfunc
