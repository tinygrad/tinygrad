from typing import ClassVar
from tinygrad.helpers import getenv, DEBUG
from tinygrad.ops import GlobalCounters
import hashlib
import time
import ctypes
from ctypes import CFUNCTYPE

import llvmlite.binding as llvm  # type: ignore
from llvmlite import ir  # type: ignore

class LLVM:
  target_machine : ClassVar[llvm.targets.TargetMachine] = None
  engine : ClassVar[llvm.executionengine.ExecutionEngine] = None
  optimizer : ClassVar[llvm.passmanagers.ModulePassManager] = None

  def __init__(self):
    if LLVM.engine is not None:
      return
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()
    target = llvm.Target.from_triple(llvm.get_process_triple())
    LLVM.optimizer = llvm.create_module_pass_manager()
    LLVM.target_machine = target.create_target_machine(opt=2)  # this opt actually can change things. ex: opt=3 means no FMA, opt=2 means FMA
    LLVM.target_machine.add_analysis_passes(LLVM.optimizer)

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
      builder.populate(LLVM.optimizer)

    LLVM.target_machine.set_asm_verbosity(True)
    backing_mod = llvm.parse_assembly(str())
    backing_mod.triple = llvm.get_process_triple()
    LLVM.engine = llvm.create_mcjit_compiler(backing_mod, LLVM.target_machine)

  # TODO: LLVMProgram
  def exec(self, module:ir.Module, bufs, op_estimate=0, mem_estimate=0):
    module.triple = llvm.get_process_triple()
    module.data_layout = self.engine.target_data
    llvm_ir = str(module)

    if DEBUG >= 2:
      print(llvm_ir)

    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    LLVM.optimizer.run(mod)
    if DEBUG >= 4:
      print("Optimized IR:")
      print(str(mod))
    mod.name = hashlib.sha1(llvm_ir.encode('utf-8')).hexdigest()
    if DEBUG >= 3:
      print(LLVM.target_machine.emit_assembly(mod))
    LLVM.engine.add_module(mod)
    LLVM.engine.finalize_object()

    # call function (NOTE: if the types don't match, there's likely something wrong with the cache)
    #cfunc = CFUNCTYPE(ctypes.c_int, *[type(x._buf) for x in bufs])(LLVM.engine.get_function_address('exec'))

    # why is this needed without the types. fixed tests below
    # LLVM=1 OPT=2 python3 test/test_ops.py TestOps.test_cat TestOps.test_multicat
    cfunc = CFUNCTYPE(ctypes.c_int, *[ctypes.POINTER(ctypes.c_float) for x in bufs])(LLVM.engine.get_function_address('exec'))

    st = time.monotonic()
    cfunc(*[x._buf for x in bufs])
    et = time.monotonic() - st
    if DEBUG >= 1:
      print(f"**LLVM** time {et*1000:7.2f} ms  OPs {op_estimate/1e6:7.2f}M -- {(op_estimate/1e9)/et:5.2f} GFLOPS -- {mem_estimate:10d} reads -- {(mem_estimate*4/1e9)/et:5.2f} GB/s")
    GlobalCounters.global_ops += op_estimate
    GlobalCounters.global_mem += mem_estimate

    # we are done
    LLVM.engine.remove_module(mod)
    return cfunc
