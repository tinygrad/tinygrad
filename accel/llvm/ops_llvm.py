from __future__ import annotations
import os
import hashlib
import math
import time
from typing import Tuple, Union
from tinygrad.helpers import prod, all_same, dedup
from tinygrad.shapetracker import ShapeTracker, ZeroView, strides_for_shape
from tinygrad.ops import LazyOp
import ctypes
import numpy as np
from ctypes import CFUNCTYPE
from tinygrad.ops import DEBUG, UnaryOps, BinaryOps, ReduceOps, get_buffers, get_lazyops, ExplicitExecAST, get_lazyop_info
from llvmlite import ir  # type: ignore
import llvmlite.binding as llvm  # type: ignore

def int_const(x): return ir.Constant(ir.IntType(64), x)
def idx_deref(builder, buf, ptr, eidx):
  if eidx[2] == 1 and eidx[3] is None:
    idx = eidx[1]
  else:
    idx = builder.add(builder.mul(eidx[1], int_const(eidx[2])), eidx[3], name="idx")

  if DEBUG >= 1:
    print("viewcount:", len(buf.st.views), buf.st.expr(), ptr, "on", buf.shape)
  # TODO: unify this with expr in ShapeTracker
  valid = None
  for v in buf.st.views[::-1]:
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
          if acc%eidx[2] == 0 and len(buf.st.views) == 1:
            # the inner one doesn't matter
            lr = eidx[1]
            if acc//eidx[2] != 1:
              lr = builder.sdiv(lr, int_const(acc//eidx[2]))
            if (acc//eidx[2])*d != eidx[0]:
              lr = builder.srem(lr, int_const(d))
          elif acc*d <= eidx[2] and eidx[3] is not None and len(buf.st.views) == 1:
            # the outer one doesn't matter
            lr = eidx[3]
            if acc != 1:
              lr = builder.sdiv(lr, int_const(acc))
            if acc*d != eidx[2]:
              lr = builder.srem(lr, int_const(d))
          else:
            # slow path
            lr = idx
            if acc != 1:
              lr = builder.sdiv(lr, int_const(acc))
            if acc*d != (eidx[0]*eidx[2]):
              lr = builder.srem(lr, int_const(d))
          if s != 1:
            lr = builder.mul(lr, int_const(s))
          ret = builder.add(ret, lr)
        acc *= d
      idx = ret
  if valid is not None:
    return builder.select(valid, builder.load(builder.gep(ptr, [idx], inbounds=True)), ir.Constant(ir.FloatType(), 0))
  else:
    return builder.load(builder.gep(ptr, [idx], inbounds=True))

# https://blog.christianperone.com/2022/09/tutorial-on-using-llvm-to-jit-pytorch-fx-graphs-to-native-code-x86-arm-risc-v-wasm-part-i-scalars/
class LLVM:
  target_machine = None
  engine = None
  optimizer = None
  # if it can't vectorize
  # OPT=2 DEBUG=3 LLVM=1 FORWARD_ONLY=1 python3 test/test_ops.py TestOps.test_mul
  # if can't vectorize anything

  # looks like we have two options, either use clang or handle vectorization in tinygrad
  # for the sake of the GPU, we should probably do in tinygrad

  # ARM NEON is 128b wide, aka <4 x float> (similar to most GPUs)
  # Firestorm (big M1 core) can do up to 4 ops per cycle @ 3.2 GHz = 3.2*4*4*2 = 102.4 GFLOPS (fma)

  # There's also AMX https://github.com/corsix/amx/blob/main/README.md
  # It seems like torch CPU must be using this? I'm seeing ~150 GFLOPS with convs
  # Calling nnp_s4gemm_only_3x3__neon and nnp_owt8x8_3x3_with_bias__neon which don't seem like AMX
  # Could this be a winograd conv? Yes, nnp_owt8x8_3x3_with_bias__neon is in NNPACK 2d-winograd-8x8-3x3.c

  # 2048x2048 matmul in 9.88 ms (17.18 GOPS) = 1739 GFLOPS (so much! this has to be the AMX)
  # calling libBLAS.dylib`SGEMM
  #  0x1c3ac5070: 0x0020100d   .long  0x0020100d                ; AMX instruction 0 = ldx
  #  0x1c3ac5074: 0x0020102b   .long  0x0020102b                ; AMX instruction 1 = ldy (presumed typo in ldst.md)
  #  0x1c3ac5078: 0x0020119f   .long  0x0020119f                ; AMX instruction 12 = fma32
  #  0x1c3ac507c: 0x0020118e   .long  0x0020118e                ; AMX instruction 12 = fma32
  #  0x1c3ac5080: 0x9144410f   add    x15, x8, #0x110, lsl #12  ; =0x110000
  #  0x1c3ac5084: 0x00201188   .long  0x00201188                ; AMX instruction 12 = fma32
  #  0x1c3ac5088: 0x0020118f   .long  0x0020118f                ; AMX instruction 12 = fma32
  #  0x1c3ac508c: 0x8b0a016b   add    x11, x11, x10
  #  0x1c3ac5090: 0x8b0c01ad   add    x13, x13, x12
  #  0x1c3ac5094: 0xf1000529   subs   x9, x9, #0x1
  #  0x1c3ac5098: 0x54fffec1   b.ne   0x1c3ac5070               ; <+140>
  # z is 16x16 float32s. 1.64 TFLOPS is one dispatch per clock cycle. 3.2*16*16*2 = 1638.4

  # From HN: "On M1, for single-precision, one AMX P-unit is ~1.64 TFLOPs, one P-core is ~102 GFLOPS." which matches this

  def __init__(self):
    if LLVM.engine is not None:
      return
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()  # yes, even this one
    target = llvm.Target.from_triple(llvm.get_process_triple())
    LLVM.optimizer = llvm.create_module_pass_manager()
    LLVM.target_machine = target.create_target_machine(opt=3)  # this opt actually can change things
    LLVM.target_machine.add_analysis_passes(LLVM.optimizer)

    llvm.set_option('', '-ffp-contract=fast')   # does anything? is there no NEON FMA?
    llvm.set_option('', '-force-vector-interleave=4')  # this makes sum the same speed as torch, it also doubles the (slow) conv speed
    if DEBUG >= 4:
      llvm.set_option('', '--debug-only=loop-vectorize')

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

    # cache
    def notify_func(module, buffer):
      #print("notify", module.name)
      with open(f"/tmp/llvmcache/{module.name}", "wb") as f:
        f.write(buffer)
    def getbuffer_func(module):
      #print("getbuffer", module.name)
      try:
        with open(f"/tmp/llvmcache/{module.name}", "rb") as f:
          return f.read()
      except FileNotFoundError:
        return None
    # enable cache
    if int(os.getenv("LLVMCACHE", "0")):
      LLVM.engine.set_object_cache(notify_func, getbuffer_func)

  def exec(self, module, bufs, op_estimate=0):
    module.triple = llvm.get_process_triple()
    module.data_layout = self.engine.target_data
    llvm_ir = str(module)

    if DEBUG >= 2:
      print(llvm_ir)

    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    LLVM.optimizer.run(mod)
    mod.name = hashlib.sha1(llvm_ir.encode('utf-8')).hexdigest()
    if DEBUG >= 3:
      print(LLVM.target_machine.emit_assembly(mod))
    LLVM.engine.add_module(mod)
    LLVM.engine.finalize_object()

    # call function
    #cfunc = CFUNCTYPE(ctypes.c_int, *[type(x._buf) for x in bufs])(LLVM.engine.get_function_address('exec'))
    cfunc = CFUNCTYPE(ctypes.c_int, *[ctypes.POINTER(ctypes.c_float) for _ in bufs])(LLVM.engine.get_function_address('exec'))

    st = time.monotonic()
    cfunc(*[x._buf for x in bufs])
    et = time.monotonic() - st
    if DEBUG >= 1:
      print(f"**LLVM** time {et*1000:7.2f} ms  OPs {op_estimate/1e6:7.2f}M -- {(op_estimate/1e9)/et:5.2f} GFLOPS")

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
  start_for_op_4x4 = {
    ReduceOps.SUM: ir.VectorType(ir.FloatType(), 16)([ir.FloatType()(0.0)]*16),
    ReduceOps.MAX: ir.VectorType(ir.FloatType(), 16)([ir.FloatType()(-math.inf)]*16),
  }
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]], hostbuf=None):
    super().__init__(shape, hostbuf)
    self._buf = (ctypes.c_float * (prod(self.shape)))() if hostbuf is None else hostbuf._buf

  def __repr__(self): return f"LLVMBuffer {str(self.st)}"

  @staticmethod
  def fromCPU(x):
    x = x.astype(np.float32)
    ret = LLVMBuffer(x.shape)
    ctypes.memmove(ret._buf, x.ctypes.data, prod(ret.shape)*4)
    return ret
  
  def toCPU(x): return np.ctypeslib.as_array(x.contiguous_op()._buf)[:prod(x.shape)].reshape(x.shape).copy()

  # ast can contain one ReduceOp with arbitrary Binary/Unary ops
  func_cache = {}
  @classmethod
  def exec_ast(cls, ast:LazyOp) -> LLVMBuffer:
    key = str(ast)  # TODO: does this uniquely determine the AST? No! The shapetracker can change. Do this better.
    bufs = dedup(get_buffers(ast))
    info = get_lazyop_info(ast)
    ret = cls(info.shape)

    if key in LLVMBuffer.func_cache:
      LLVMBuffer.func_cache[key](ret._buf, *[x._buf for x in bufs])
      return ret

    # get the real buffers from the ast
    reduceops = [x for x in get_lazyops(ast) if isinstance(x.op, ReduceOps)]
    assert len(reduceops) <= 1, "max one reduce op in an ast"
    earlybufs = dedup(get_buffers(reduceops[0])) if len(reduceops) > 0 else []

    # check buffer sizes
    assert all_same([x.shape for x in earlybufs]), "all earlybufs must have the same shape"
    assert all_same([x.shape for x in bufs if x not in earlybufs]), "all latebufs must have the same shape"
    assert all_same([len(x.shape) for x in bufs]), "all bufs must have the same shape size"

    # create llvm function
    module = ir.Module(name=__file__)
    func = ir.Function(module, ir.FunctionType(ir.VoidType(), [ir.FloatType().as_pointer()]*(1+len(bufs))), name='exec')

    if not all(len(x.st.views) == 1 for x in bufs):
      print(f"WARNING: {ast} has buffers with more than 1 view, can't optimize")
    else:
      # include ret in the bufs
      bufs = [ret] + bufs
      shapes = [x.shape for x in bufs]
      strides = [x.st.views[0].strides for x in bufs]

      # remove places where the shape is all ones, this is cheap, easy, and correct
      all_ones = [all(s[i]==1 for s in shapes) for i in range(len(shapes[0]))]
      shapes = [[s[i] for i in range(len(s)) if not all_ones[i]] for s in shapes]
      strides = [[s[i] for i in range(len(s)) if not all_ones[i]] for s in strides]

      # find first mismatch, don't reduce this
      first_reduce = -1
      for i in range(len(shapes[0])):
        if not all_same([x[i] for x in shapes]):
          first_reduce = i
          break
      #print("first reduce", first_reduce)
      
      # merge dimensions if we can
      # TODO: does this always preserve the reduce dimension, NO
      # TODO: move this into shapetracker, with tests!
      rets = [[(shapes[j][0], strides[j][0])] for j in range(len(shapes))]
      for i in range(1, len(shapes[0])):
        can_merge = []
        for j in range(len(shapes)):
          # TODO: added the always mergability of 1s, is this right? if so, add to shapetracker in the 1 case
          can_merge.append((strides[j][i] != 0 and rets[j][-1][1] == shapes[j][i]*strides[j][i]) or (strides[j][i] == 0 and rets[j][-1][1] == 0))
        # more can merge than this
        can_merge = all(can_merge) and i != first_reduce
        for j in range(len(shapes)):
          if can_merge:
            rets[j][-1] = (rets[j][-1][0] * shapes[j][i], strides[j][i])
          else:
            rets[j].append((shapes[j][i], strides[j][i]))
      shapes, strides = [[y[0] for y in x] for x in rets], [[y[1] for y in x] for x in rets]

      #USE_4X4 = False
      USE_4X4 = True
      DY = 64
      DX = 1

      # TODO: change the order of the output_shape, and perhaps reshape everything
      # focus on the AMX instructions, that's the way to beat PyTorch on M1, since PyTorch can't use the convs
      # AMX can make quick work of a MUL->SUM AST block
      # This also splits the dimensions for cache chunking
      if len(shapes[0]) in [2,3]: # and len(reduceops) == 0:
        new_shapes, new_strides = [], []
        # there's 32 SIMD FP registers
        # track a 4x4 chunk of the matrix at once
        CACHE_DIM = 128
        for shape, stride in zip(shapes, strides):
          st = ShapeTracker(tuple(shape))
          st.strided(*zip(shape, stride))
          if len(shape) == 2:
            st.reshape(shape[0]//CACHE_DIM, min(shape[0], CACHE_DIM), shape[1]//CACHE_DIM, min(shape[1], CACHE_DIM))
            st.permute(0,2,1,3)
          else:
            if USE_4X4:
              # 0 1 2 - 3 4 5 - 6
              #st.reshape(shape[0]//CACHE_DIM, min(shape[0], CACHE_DIM//4), 4, shape[1]//CACHE_DIM, min(shape[1], CACHE_DIM//4), 4, shape[2])
              #st.permute(0,3,1,4,6,2,5)
              st.reshape(shape[0]//DY, DY, shape[1]//DX, DX, shape[2])
              st.permute(0,2,4,1,3)
            else:
              #st.reshape(shape[0]//CACHE_DIM, min(shape[0], CACHE_DIM), shape[1]//CACHE_DIM, min(shape[1], CACHE_DIM), shape[2])
              #st.permute(0,2,1,3,4)
              st.reshape(shape[0]//4, 4, shape[1]//4, 4, shape[2])
              st.permute(0,2,1,3,4)
              #st.permute(0,2,1,3,4)
              pass
          assert len(st.views) == 1
          new_shapes.append(st.shape)
          new_strides.append(st.strides)
        shapes, strides = new_shapes, new_strides

      # the 4x4 need to go all the way at the end, even after reduce
      output_shape = shapes[0]
      full_shape = [x for x in shapes if x != output_shape]
      full_shape = output_shape if len(full_shape) == 0 else full_shape[0]
      
      # remove the magic 4x4 at 2:4, since we run this as 4x4
      # TODO: gate this
      if USE_4X4:
        full_shape = full_shape[:-2]

      if DEBUG >= 2:
        print(ast)
        print(shapes)
        print(strides)
        print(full_shape, "->", output_shape)
      
      # construct the structure of the loops
      loop_entry = [ir.IRBuilder(func.append_basic_block(name="entry"))]
      loop_exit = []
      for i,_ in enumerate(full_shape):
        loop_entry.append(ir.IRBuilder(func.append_basic_block(name=f"loop_{i}")))
      for i,_ in enumerate(full_shape):
        loop_exit.append(ir.IRBuilder(func.append_basic_block(name=f"loopexit_{len(full_shape)-1-i}")))
      loop_exit.append(ir.IRBuilder(func.append_basic_block(name="exit")))
      loop_exit = loop_exit[::-1]

      # add the buffer indexing
      idx_level = [[int_const(0)] for _ in range(len(bufs))]
      for i in range(len(full_shape)):
        for j in range(len(bufs)):
          # stride
          si = loop_entry[i+1].phi(ir.IntType(64), name=f"idx_{j}_{i}")
          si.add_incoming(idx_level[j][-1], loop_entry[i]._block)
          si_ps = loop_exit[i+1].add(si, int_const(strides[j][i]))
          si.add_incoming(si_ps, loop_exit[i+1]._block)
          idx_level[j].append(si)


      if USE_4X4:
        val_type = ir.VectorType(ir.FloatType(), DY*DX)
      else:
        val_type = ir.FloatType()

      # the ast parser
      def ast_parse(builder, x, level, reduce_result=None):
        if not isinstance(x, LazyOp):
          buf_index = bufs.index(x)
          idx = idx_level[buf_index][level]
          if USE_4X4:
            # load 4x4
            idxs = []
            for y in range(DY):
              for x in range(DX):
                idxs.append(builder.add(idx, int_const(strides[buf_index][-2]*y + strides[buf_index][-1]*x)))

            # build <16 x float> vector
            m = ir.VectorType(ir.FloatType(), DY*DX)(ir.Undefined)
            for i, idx in enumerate(idxs):
              pts = builder.gep(func.args[buf_index], [idx], inbounds=True)
              element = builder.load(pts)
              m = builder.insert_element(m, element, int_const(i))
            return m
          else:
            # load 1x1
            return builder.load(builder.gep(func.args[buf_index], [idx], inbounds=True))
        if isinstance(x.op, ReduceOps):
          if reduce_result is None:
            raise Exception("no reduce")
          return reduce_result
        values = [ast_parse(builder, v, level, reduce_result) for v in x.src]
        return LLVMBuffer.op_lookup[x.op](builder, *values)

      # add the ast + final store
      store_loop = output_shape.index(1) if 1 in output_shape else -1

      # do the early ast
      reduce_result = None
      if len(reduceops) > 0:
        if reduceops[0].op == ReduceOps.SUM and reduceops[0].src[0].op == BinaryOps.MUL:
          reduce_input_0 = ast_parse(loop_exit[-1], reduceops[0].src[0].src[0], -1)
          reduce_input_1 = ast_parse(loop_exit[-1], reduceops[0].src[0].src[1], -1)
          fma = True
        else:
          reduce_input = ast_parse(loop_exit[-1], reduceops[0].src[0], -1)
          fma = False

        if USE_4X4:
          phis = [val_type([LLVMBuffer.start_for_op[ReduceOps.SUM]]*(DY*DX))]
        else:
          phis = [LLVMBuffer.start_for_op[ReduceOps.SUM]]
        for i in range(store_loop+1, len(loop_entry)):
          #val = loop_entry[i].phi(ir.FloatType(), f"reduce_phi_{i}")
          val = loop_entry[i].phi(val_type, f"reduce_phi_{i}")
          val.add_incoming(phis[-1], loop_entry[i-1]._block)
          phis.append(val)

        if reduceops[0].op == ReduceOps.SUM:
          if fma:
            reduce_result = loop_exit[-1].call(ir.Function(module, ir.FunctionType(val_type, [val_type, val_type, val_type]), name="llvm.fma"), [reduce_input_0, reduce_input_1, val], fastmath=('fast',))
          else:
            reduce_result = loop_exit[-1].fadd(reduce_input, val, flags=('fast',))
        elif reduceops[0].op == ReduceOps.MAX:
          # TODO: this doesn't respect the fast math flag
          reduce_result = loop_exit[-1].call(ir.Function(module, ir.FunctionType(ir.FloatType(), [ir.FloatType(), ir.FloatType()]), name="llvm.maximum"), [reduce_input, val], fastmath=('fast',))

        for i,phi in enumerate(phis[1:]):
          phi.add_incoming(reduce_result, loop_exit[store_loop+1+i]._block)

      # do the late ast
      result = ast_parse(loop_exit[store_loop], ast, store_loop, reduce_result=reduce_result)

      # store 4x4
      if USE_4X4:
        builder = loop_exit[store_loop]
        idxs = []
        for y in range(DY):
          for x in range(DX):
            idxs.append(builder.add(idx_level[0][store_loop], int_const(strides[0][-2]*y + strides[0][-1]*x)))

        for i, idx in enumerate(idxs):
          result_x = builder.extract_element(result, int_const(i))
          builder.store(result_x, builder.gep(func.args[0], [idx], inbounds=True))
      else:
        loop_exit[store_loop].store(result, loop_exit[store_loop].gep(func.args[0], [idx_level[0][store_loop]], inbounds=True))
      
      # add the looping
      for i,s in enumerate(full_shape):
        loop_entry[i].branch(loop_entry[i+1]._block)
        # index
        idx = loop_entry[i+1].phi(ir.IntType(64), name=f"loopvar_{i}")
        idx.add_incoming(int_const(0), loop_entry[i]._block)
        idx_p1 = loop_exit[i+1].add(idx, int_const(1))
        idx.add_incoming(idx_p1, loop_exit[i+1]._block)
        loop_exit[i+1].cbranch(loop_exit[i+1].icmp_unsigned("==", idx_p1, int_const(s)), loop_exit[i]._block, loop_entry[i+1]._block)

      loop_entry[-1].branch(loop_exit[-1]._block)
      loop_exit[0].ret_void()

      LLVMBuffer.func_cache[key] = LLVM().exec(module, bufs, info.flops)
      return ret

    def ast_parse(builder, x, idx, reduce_result=None, depth=0):
      if DEBUG >= 1:
        print("  "*depth+"ast:", reduce_result, x)
      if not isinstance(x, LazyOp):
        return idx_deref(builder, x, func.args[bufs.index(x)+1], idx)
      if isinstance(x.op, ReduceOps):
        if reduce_result is None:
          raise Exception("no reduce")
        return reduce_result
      values = [ast_parse(builder, v, idx, reduce_result, depth=depth+1) for v in x.src]
      return LLVMBuffer.op_lookup[x.op](builder, *values)

    # enter
    start_builder = ir.IRBuilder(func.append_basic_block(name="entry"))
    body_builder = ir.IRBuilder(func.append_basic_block(name="inner_loop"))
    start_builder.branch(body_builder._block)

    idx = body_builder.phi(ir.IntType(64))
    idx.add_incoming(int_const(0), start_builder._block)

    reduce_builder = ir.IRBuilder(func.append_basic_block(name="reduce_loop"))
    store_builder = ir.IRBuilder(func.append_basic_block(name="store_block"))

    if len(reduceops) > 0:
      assert len(earlybufs[0].shape) == len(reduceops[0].arg), "reduce only possible on matching shapes"
      if DEBUG >= 1:
        print(f"reduce {earlybufs[0].shape} -> {reduceops[0].arg}")
      red = prod([s for s,n in zip(earlybufs[0].shape, reduceops[0].arg) if n == 1])
      red_idx = reduce_builder.phi(ir.IntType(64))
      red_idx.add_incoming(int_const(0), body_builder._block)
      val = reduce_builder.phi(ir.FloatType())
      reduce_input = ast_parse(reduce_builder, reduceops[0].src[0], (prod(reduceops[0].arg), idx, red, red_idx))

      if reduceops[0].op == ReduceOps.SUM:
        val.add_incoming(ir.Constant(ir.FloatType(), 0), body_builder._block)
        reduce_result = reduce_builder.fadd(reduce_input, val)
      elif reduceops[0].op == ReduceOps.MAX:
        val.add_incoming(ir.Constant(ir.FloatType(), -math.inf), body_builder._block)
        reduce_result = reduce_builder.call(ir.Function(module, ir.FunctionType(ir.FloatType(), [ir.FloatType(), ir.FloatType()]), name="llvm.maxnum.f32"), [reduce_input, val])
      else:
        raise Exception(f"unknown ReduceOps {ast.op}")
      val.add_incoming(reduce_result, reduce_builder._block)

      red_idx_p1 = reduce_builder.add(red_idx, int_const(1))
      red_idx.add_incoming(red_idx_p1, reduce_builder._block)
      reduce_builder.cbranch(reduce_builder.icmp_unsigned("==", red_idx_p1, int_const(red)), store_builder._block, reduce_builder._block)
    else:
      reduce_result = None
      reduce_builder.branch(store_builder._block)

    body_builder.branch(reduce_builder._block)
    result = ast_parse(store_builder, ast, (prod(ret.shape), idx, 1, None), reduce_result)
    store_builder.store(result, store_builder.gep(func.args[0], [idx], inbounds=True))
    idx_p1 = store_builder.add(idx, int_const(1))
    idx.add_incoming(idx_p1, store_builder._block)

    exit_builder = ir.IRBuilder(func.append_basic_block(name="exit"))
    exit_builder.ret_void()

    store_builder.cbranch(store_builder.icmp_unsigned("==", idx_p1, int_const(prod(ret.shape))), exit_builder._block, body_builder._block)

    # **** llvm running ****
    LLVMBuffer.func_cache[key] = LLVM().exec(module, [ret] + bufs, info.flops)
    return ret
