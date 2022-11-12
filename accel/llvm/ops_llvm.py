from __future__ import annotations
import hashlib
import math
import time
from typing import Tuple, Union, Dict, Any, List
from tinygrad.helpers import prod
from tinygrad.shapetracker import ShapeTracker, ZeroView
from tinygrad.ops import LazyOp, ASTKernel
import ctypes
import numpy as np
from ctypes import CFUNCTYPE
from tinygrad.ops import DEBUG, UnaryOps, BinaryOps, ReduceOps, ExplicitExecAST, GlobalCounters

from llvmlite import ir  # type: ignore
import llvmlite.binding as llvm  # type: ignore

def int_const(x): return ir.Constant(ir.IntType(64), x)

# this is only used on the crappy path
def idx_deref(builder, buf, ptr, idx):
  if DEBUG >= 1:
    print("viewcount:", len(buf.st.views), buf.st.expr(), ptr, "on", buf.shape)
  # TODO: unify this with expr in ShapeTracker
  valid = None
  for v in buf.st.views[0:-1][::-1]:
    if isinstance(v, ZeroView):
      if valid is None:
        valid = ir.Constant(ir.IntType(1), 1)
      acc = 1
      for s,(x,y) in list(zip(v.old_shape, v.arg))[::-1]:
        if x < 0 or y > s:
          lr = idx
          if acc != 1:
            lr = builder.sdiv(lr, int_const(acc))
          lr = builder.srem(lr, int_const(y-x))
          if x != 0:
            lr = builder.add(lr, int_const(x))
          if x < 0:
            valid = builder.and_(valid, builder.icmp_signed(">=", lr, int_const(0)))
          if y > s:
            valid = builder.and_(valid, builder.icmp_signed("<", lr, int_const(s)))
        acc *= y-x
    else:
      acc = 1
      ret = int_const(v.offset)
      if DEBUG >= 2:
        print(f"expanding index {v.shape_strides}")
      for i,(d,s) in enumerate(v.shape_strides[::-1]):
        if d != 1 and s != 0:
          # slow path
          lr = idx
          if acc != 1:
            lr = builder.sdiv(lr, int_const(acc))
          if acc*d != prod(buf.shape):
            lr = builder.srem(lr, int_const(d))
          if s != 1:
            lr = builder.mul(lr, int_const(s))
          ret = builder.add(ret, lr)
        acc *= d
      idx = ret
  if valid is not None:
    # this always does the load, so we have it load *0 if the arg won't be used
    # TODO: would control flow be faster?
    aug_idx = builder.select(valid, idx, int_const(0))
    return builder.select(valid, builder.load(builder.gep(ptr, [aug_idx], inbounds=True)), ir.Constant(ir.FloatType(), 0))
  else:
    return builder.load(builder.gep(ptr, [idx], inbounds=True))

class LLVM:
  target_machine = None
  engine = None
  optimizer = None

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

    llvm.set_option('', '-force-vector-interleave=4')  # this makes sum the same speed as torch, it also doubles the (slow) conv speed
    if DEBUG >= 4:
      llvm.set_option('', '--debug-only=loop-vectorize')
    #llvm.set_option('', '--debug')

    # does this do anything?
    builder = llvm.create_pass_manager_builder()
    builder.opt_level = 3
    builder.size_level = 0
    builder.loop_vectorize = True
    builder.slp_vectorize = True
    builder.populate(LLVM.optimizer)

    LLVM.target_machine.set_asm_verbosity(True)
    backing_mod = llvm.parse_assembly("")
    backing_mod.triple = llvm.get_process_triple()
    LLVM.engine = llvm.create_mcjit_compiler(backing_mod, LLVM.target_machine)

  def exec(self, module, bufs, op_estimate=0, mem_estimate=0):
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


# TODO: Refactor LLVMBuffer and GPUBuffer into ShapeTrackedBuffer
class LLVMBuffer(ExplicitExecAST):
  op_lookup = {
    UnaryOps.NOOP: lambda builder,x: x,
    UnaryOps.NEG: lambda builder,x: builder.fneg(x, flags=('fast',)),
    UnaryOps.RELU: lambda builder,x: builder.select(builder.fcmp_ordered("<=", ir.Constant(ir.FloatType(), 0), x, flags=('fast',)), x, ir.Constant(ir.FloatType(), 0)),
    UnaryOps.EXP: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.exp', [ir.FloatType()]), [x], fastmath=('fast',)),
    UnaryOps.LOG: lambda builder,x: builder.call(builder._block.module.declare_intrinsic('llvm.log', [ir.FloatType()]), [x], fastmath=('fast',)),
    UnaryOps.SIGN: lambda builder,x: builder.select(builder.fcmp_ordered("==", x, ir.Constant(ir.FloatType(), 0), flags=('fast',)), ir.Constant(ir.FloatType(), 0),
                                                    builder.select(builder.fcmp_ordered("<=", ir.Constant(ir.FloatType(), 0), x, flags=('fast',)), ir.Constant(ir.FloatType(), 1), ir.Constant(ir.FloatType(), -1))),
    UnaryOps.RECIPROCAL: lambda builder,x: builder.fdiv(ir.Constant(ir.FloatType(), 1), x, flags=('fast',)),
    BinaryOps.ADD: lambda builder,x,y: builder.fadd(x,y, flags=('fast',)),
    BinaryOps.SUB: lambda builder,x,y: builder.fsub(x,y, flags=('fast',)),
    BinaryOps.MUL: lambda builder,x,y: builder.fmul(x,y, flags=('fast',)),
    BinaryOps.DIV: lambda builder,x,y: builder.fdiv(x,y, flags=('fast',)),
    BinaryOps.POW: lambda builder,x,y: builder.call(builder._block.module.declare_intrinsic('llvm.pow', [ir.FloatType()]), [x,y], fastmath=('fast',)),
    BinaryOps.CMPEQ: lambda builder,x,y: builder.uitofp(builder.fcmp_ordered("==", x, y, flags=('fast',)), ir.FloatType())
  }
  start_for_op = {
    ReduceOps.SUM: ir.Constant(ir.FloatType(), 0),
    ReduceOps.MAX: ir.Constant(ir.FloatType(), -math.inf)
  }

  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf=None):
    super().__init__(shape, hostbuf)
    # TODO: force alignment?
    self._buf = (ctypes.c_float * (prod(self.shape)))() if hostbuf is None else hostbuf._buf
    #assert ctypes.addressof(self._buf) & 0x1F == 0

  def __repr__(self): return f"LLVMBuffer {str(self.st)}"

  @staticmethod
  def fromCPU(x):
    x = x.astype(np.float32)
    ret = LLVMBuffer(x.shape)
    ctypes.memmove(ret._buf, x.ctypes.data, prod(ret.shape)*4)
    return ret
  
  def toCPU(x): return np.ctypeslib.as_array(x.contiguous()._buf)[:prod(x.shape)].reshape(x.shape).copy()

  func_cache : Dict[str, Any] = {}
  @classmethod
  def exec_ast(cls, ast:LazyOp) -> LLVMBuffer:
    k = ASTKernel(ast)

    # cached kernel
    key = str(ast)  # TODO: does this uniquely determine the AST? No! The shapetracker can change. Do this better.
    if key in LLVMBuffer.func_cache:
      LLVMBuffer.func_cache[key](*[x._buf for x in k.bufs])
      return k.ret

    # cache miss, we have to process the kernel
    k.process()

    if DEBUG >= 2:
      print(ast)
      print("old:", k.shapes)
      print("old:", k.strides)
    
    # this stuff can't be hand coded
    kernel_output_axis : List[int] = []
    """
    CACHE_DIM = 32
    if len(k.shapes[0]) == 2:
      # cache tiling, makes permute fast
      k.reshape_and_permute(
        lambda shape: (shape[0]//CACHE_DIM, CACHE_DIM, shape[1]//CACHE_DIM, CACHE_DIM),
        (0,2,1,3))
    elif len(k.shapes[0]) == 3:
      if k.reduceop:
        if k.strides[1][-1] == 1 and k.strides[2][-1] == 1:
          DY, DX = 8, 8
        elif k.strides[1][-1] in [1,0] and k.strides[1][-2] in [1,0]:
          DY, DX = 4, 16
        else:
          DY, DX = 16, 4
        # matmul: YyXxK -> YXKyx
        k.reshape_and_permute(
          lambda shape: (shape[0]//DY, DY, shape[1]//DX, DX, shape[2]),
          (0,2,4,1,3))
        kernel_output_axis = [-2, -1]
      else:
        CACHE_L2_DIM = 256
        k.reshape_and_permute(
          lambda shape: (shape[0], shape[1]//CACHE_L2_DIM, CACHE_L2_DIM, shape[2]),
          (1,0,2,3))
        kernel_output_axis = [-1]
    elif len(k.shapes[0]) == 7:
      # conv: split chans and X
      DY, DX = 4, 16
      k.reshape_and_permute(
        lambda shape: (shape[0], shape[1]//DY, DY, shape[2], shape[3]//DX, DX, shape[4], shape[5], shape[6]),
        (0,1,3,4,6,7,8,2,5))
      kernel_output_axis = [-2, -1]
    
    if DEBUG >= 2:
      print("new:", k.shapes)
      print("new:", k.strides)
    """

    # the 4x4 need to go all the way at the end, even after reduce
    output_shape = k.shapes[0]
    full_shape = [x for x in k.shapes if x != output_shape]
    full_shape = output_shape if len(full_shape) == 0 else full_shape[0]

    full_shape = full_shape if not kernel_output_axis else full_shape[:-len(kernel_output_axis)]
    kernel_output_dim = prod([k.shapes[0][a] for a in kernel_output_axis])
    kernel_output_type = ir.FloatType() if kernel_output_dim == 1 else ir.VectorType(ir.FloatType(), kernel_output_dim)

    def get_idxs(builder, idx, buf_index):
      idx_offsets = [0]
      for axis in kernel_output_axis:
        new_idx_offsets = []
        for s in range(k.shapes[buf_index][axis]):
          for i in idx_offsets:
            new_idx_offsets.append(i + s * k.strides[buf_index][axis])
        idx_offsets = new_idx_offsets
      return [builder.add(idx, int_const(i)) for i in idx_offsets]
    
    # *** llvm specific below this line ***

    # create llvm function
    module = ir.Module(name=__file__)
    func = ir.Function(module, ir.FunctionType(ir.VoidType(), [ir.FloatType().as_pointer()]*(len(k.bufs))), name='exec')

    # force llvmlite to allow us to add function attribute then add the attribute
    func.attributes._known = func.attributes._known.union(frozenset(['"no-nans-fp-math"="true"']))
    func.attributes.add('"no-nans-fp-math"="true"')

    # construct the structure of the loops
    loop_entry = [ir.IRBuilder(func.append_basic_block(name="entry"))]
    loop_exit = []
    for i,_ in enumerate(full_shape): loop_entry.append(ir.IRBuilder(func.append_basic_block(name=f"loop_{i}")))
    for i,_ in enumerate(full_shape): loop_exit.append(ir.IRBuilder(func.append_basic_block(name=f"loopexit_{len(full_shape)-1-i}")))
    loop_exit.append(ir.IRBuilder(func.append_basic_block(name="exit")))
    loop_exit = loop_exit[::-1]

    # add the buffer indexing
    idx_level = [[int_const(o)] for o in k.offsets]
    for i in range(len(full_shape)):
      for j in range(len(k.bufs)):
        # stride
        si = loop_entry[i+1].phi(ir.IntType(64), name=f"idx_{j}_{i}")
        si.add_incoming(idx_level[j][-1], loop_entry[i]._block)
        si_ps = loop_exit[i+1].add(si, int_const(k.strides[j][i]))
        si.add_incoming(si_ps, loop_exit[i+1]._block)
        idx_level[j].append(si)

    # the ast parser
    def ast_parse(builder, x, level, reduce_result=None):
      if not isinstance(x, LazyOp):
        m = kernel_output_type(ir.Undefined)
        buf_index = k.bufs.index(x)
        for i, idx in enumerate(get_idxs(builder, idx_level[buf_index][level], buf_index)):
          if len(x.st.views) > 1:
            if DEBUG >= 1: print(f"WARNING: {x} has buffers with more than 1 view, can't optimize")
            element = idx_deref(builder, x, func.args[buf_index], idx)
          else:
            element = builder.load(builder.gep(func.args[buf_index], [idx], inbounds=True))
          m = element if kernel_output_dim == 1 else builder.insert_element(m, element, int_const(i))
        return m
      if isinstance(x.op, ReduceOps):
        if reduce_result is None:
          raise Exception("no reduce")
        return reduce_result
      values = [ast_parse(builder, v, level, reduce_result) for v in x.src]

      m = kernel_output_type(ir.Undefined)
      if kernel_output_dim == 1:
        return LLVMBuffer.op_lookup[x.op](builder, *values)
      else:
        # TODO: this only has to be done for certain ops
        for i in range(kernel_output_dim):
          value = [builder.extract_element(v, int_const(i)) for v in values]
          element = LLVMBuffer.op_lookup[x.op](builder, *value)
          m = builder.insert_element(m, element, int_const(i))
        return m

    # add the ast + final store
    store_loop = output_shape.index(1) if 1 in output_shape else -1

    # do the early ast
    reduce_result = None
    if k.reduceop:
      reduce_input = ast_parse(loop_exit[-1], k.reduceop.src[0], -1)
      phis = [LLVMBuffer.start_for_op[k.reduceop.op]]  # type: ignore
      if kernel_output_dim > 1:
        phis = [kernel_output_type(phis * kernel_output_dim)]
      for i in range(store_loop+1, len(loop_entry)):
        val = loop_entry[i].phi(kernel_output_type, f"reduce_phi_{i}")
        val.add_incoming(phis[-1], loop_entry[i-1]._block)
        phis.append(val)

      if k.reduceop.op == ReduceOps.SUM:
        reduce_result = loop_exit[-1].fadd(reduce_input, val, flags=('fast',))
      elif k.reduceop.op == ReduceOps.MAX:
        reduce_result = loop_exit[i].select(loop_exit[-1].fcmp_unordered(">", val, reduce_input, flags=('fast',)), val, reduce_input, flags=('fast',))

      for i,phi in enumerate(phis[1:]):
        phi.add_incoming(reduce_result, loop_exit[store_loop+1+i]._block)

    # do the late ast
    result = ast_parse(loop_exit[store_loop], ast, store_loop, reduce_result=reduce_result)

    # store result
    builder = loop_exit[store_loop]
    for i, idx in enumerate(get_idxs(builder, idx_level[0][store_loop], 0)):
      element = result if kernel_output_dim == 1 else builder.extract_element(result, int_const(i))
      builder.store(element, builder.gep(func.args[0], [idx], inbounds=True))
    
    # add the looping
    for i,s in enumerate(full_shape):
      loop_entry[i].branch(loop_entry[i+1]._block)
      idx = loop_entry[i+1].phi(ir.IntType(64), name=f"loopvar_{i}")
      idx.add_incoming(int_const(0), loop_entry[i]._block)
      idx_p1 = loop_exit[i+1].add(idx, int_const(1))
      idx.add_incoming(idx_p1, loop_exit[i+1]._block)
      loop_exit[i+1].cbranch(loop_exit[i+1].icmp_unsigned("==", idx_p1, int_const(s)), loop_exit[i]._block, loop_entry[i+1]._block)

    loop_entry[-1].branch(loop_exit[-1]._block)
    loop_exit[0].ret_void()
    LLVMBuffer.func_cache[key] = LLVM().exec(module, k.bufs, k.info.flops, sum(len(x._buf) for x in k.bufs))
    return k.ret
