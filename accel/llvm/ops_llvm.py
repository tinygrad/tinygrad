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

# https://github.com/corsix/amx/blob/main/Instructions.md
# 12 lines for AMX support
from functools import partial
class AMX():
  @staticmethod
  def nop_op_imm5(op, imm5, builder): builder.asm(ir.FunctionType(ir.VoidType(), []), f".word (0x201000 + ({op} << 5) + {imm5}); amx op {op} imm {imm5}", "", tuple(), True)
  @staticmethod
  def op_gpr(op, builder, gpr): builder.asm(ir.FunctionType(ir.VoidType(), [ir.IntType(64)]), f".word (0x201000 + ({op} << 5) + 0$0 - ((0$0 >> 4) * 6)); amx op {op}", "r", (gpr,), True)
  set, clr = partial(nop_op_imm5, 17, 0), partial(nop_op_imm5, 17, 1)
  ldx, ldy, stx, sty = partial(op_gpr, 0), partial(op_gpr, 1), partial(op_gpr, 2), partial(op_gpr, 3)
  ldz, stz, ldzi, stzi = partial(op_gpr, 4), partial(op_gpr, 5), partial(op_gpr, 6), partial(op_gpr, 7)
  extrx, extry = partial(op_gpr, 8), partial(op_gpr, 9)
  fma64, fms64, fma32, fms32 = partial(op_gpr, 10), partial(op_gpr, 11), partial(op_gpr, 12), partial(op_gpr, 13)
  mac16, fma16, fms16 = partial(op_gpr, 14), partial(op_gpr, 15), partial(op_gpr, 16)
  vecint, vecfp, matint, matfp, genlut = partial(op_gpr, 18), partial(op_gpr, 19), partial(op_gpr, 20), partial(op_gpr, 21), partial(op_gpr, 22)

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

    # call function (NOTE: if the types don't match, there's likely something wrong with the cache)
    #cfunc = CFUNCTYPE(ctypes.c_int, *[type(x._buf) for x in bufs])(LLVM.engine.get_function_address('exec'))

    # why is this needed without the types. fixed tests below
    # LLVM=1 OPT=2 python3 test/test_ops.py TestOps.test_cat TestOps.test_multicat
    cfunc = CFUNCTYPE(ctypes.c_int, *[ctypes.POINTER(ctypes.c_float) for x in bufs])(LLVM.engine.get_function_address('exec'))

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
    bufs = dedup(get_buffers(ast))
    info = get_lazyop_info(ast)
    ret = cls(info.shape)

    key = str(ast)  # TODO: does this uniquely determine the AST? No! The shapetracker can change. Do this better.
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

    # include ret in the bufs
    bufs = [ret] + bufs
    shapes = [x.shape for x in bufs]
    strides = [x.st.views[-1].strides for x in bufs]
    offsets = [x.st.views[-1].offset for x in bufs]

    # remove places where the shape is all ones, this is cheap, easy, and correct
    all_ones = [all(s[i]==1 for s in shapes) for i in range(len(shapes[0]))]
    # keep at least 1 one
    if all(all_ones):
      all_ones[-1] = False
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

    USE_AMX = True
    # TODO: change this independently?
    AMX_SZ_Y = 2
    AMX_SZ_X = 2
    USE_4X4 = False
    #DY, DX = 16, 4
    #DY, DX = 16, 16
    DY, DX = 16*AMX_SZ_Y, 16*AMX_SZ_X

    if len(shapes[0]) >= 3:
      USE_4X4 = True

    # TODO: change the order of the output_shape, and perhaps reshape everything
    # focus on the AMX instructions, that's the way to beat PyTorch on M1, since PyTorch can't use the convs
    # AMX can make quick work of a MUL->SUM AST block
    # This also splits the dimensions for cache chunking
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
      elif len(shape) == 7:
        # conv
        if USE_4X4:
          # split batch and X
          #st.reshape(shape[0]//DY, DY, shape[1], shape[2], shape[3]//DX, DX, shape[4], shape[5], shape[6])
          #st.permute(0,2,3,4,6,7,8,1,5)

          # split chans and X
          st.reshape(shape[0], shape[1]//DY, DY, shape[2], shape[3]//DX, DX, shape[4], shape[5], shape[6])
          st.permute(0,1,3,4,6,7,8,2,5)

          # split Y and X
          #st.reshape(shape[0], shape[1], shape[2]//DY, DY, shape[3]//DX, DX, shape[4], shape[5], shape[6])
          #st.permute(0,1,2,4,6,7,8,3,5)
      elif len(shape) == 3:
        # matmul
        if USE_4X4:
          st.reshape(shape[0]//DY, DY, shape[1]//DX, DX, shape[2])
          st.permute(0,2,4,1,3)   # YyXxK -> YXKyx

          #CACHE_DIM = 64
          #st.reshape(shape[0]//CACHE_DIM, CACHE_DIM//DY, DY, shape[1]//CACHE_DIM, CACHE_DIM//DX, DX, shape[2])
          #st.permute(0,3,1,4,6,2,5)

      assert len(st.views) == 1
      new_shapes.append(st.shape)
      new_strides.append(st.strides)
    shapes, strides = new_shapes, new_strides

    # the 4x4 need to go all the way at the end, even after reduce
    output_shape = shapes[0]
    full_shape = [x for x in shapes if x != output_shape]
    full_shape = output_shape if len(full_shape) == 0 else full_shape[0]
    
    # remove the magic 4x4 at 2:4, since we run this as 4x4
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
    idx_level = [[int_const(o)] for o in offsets]
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
    
    def get_idxs(builder, idx, buf_index):
      idxs = []
      for y in range(DY):
        for x in range(DX):
          idxs.append(builder.add(idx, int_const(strides[buf_index][-2]*y + strides[buf_index][-1]*x)))
      return idxs

    # the ast parser
    def ast_parse(builder, x, level, reduce_result=None):
      if not isinstance(x, LazyOp):
        buf_index = bufs.index(x)
        idx = idx_level[buf_index][level]
        if USE_AMX:
          fptr = builder.ptrtoint(func.args[buf_index], ir.IntType(64))
          if buf_index == 1:
            assert strides[buf_index][-2] == 1 and strides[buf_index][-1] == 0
            # TODO: this is broken for non multiples of 2
            for i in range(0, AMX_SZ_Y, 2):
              idx_n = builder.add(idx, int_const(i*16))
              AMX.ldy(builder, builder.add(fptr, builder.add(int_const(1 << 62 | i << 56), builder.mul(idx_n, int_const(4)))))
            return "AMX_Y"
          elif buf_index == 2:
            assert strides[buf_index][-2] == 0 and strides[buf_index][-1] == 1
            for i in range(0, AMX_SZ_X, 2):
              idx_n = builder.add(idx, int_const(i*16))
              AMX.ldx(builder, builder.add(fptr, builder.add(int_const(1 << 62 | i << 56), builder.mul(idx_n, int_const(4)))))
            return "AMX_X"
          else:
            assert "AMX only supports two buffers"
        elif USE_4X4:
          # build <16 x float> vector
          m = ir.VectorType(ir.FloatType(), DY*DX)(ir.Undefined)
          for i, idx in enumerate(get_idxs(builder, idx, buf_index)):
            element = builder.load(builder.gep(func.args[buf_index], [idx], inbounds=True))
            m = builder.insert_element(m, element, int_const(i))
          return m
        else:
          # load 1x1
          if len(x.st.views) > 1:
            if DEBUG >= 1:
              print(f"WARNING: {x} has buffers with more than 1 view, can't optimize")
            return idx_deref(builder, x, func.args[buf_index], idx)
          else:
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
      if USE_AMX and reduceops[0].op == ReduceOps.SUM and isinstance(reduceops[0].src[0], LazyOp) and reduceops[0].src[0].op == BinaryOps.MUL:
        reduce_input_1 = ast_parse(loop_exit[-1], reduceops[0].src[0].src[1], -1)
        reduce_input_0 = ast_parse(loop_exit[-1], reduceops[0].src[0].src[0], -1)
        assert reduce_input_0 == "AMX_Y" and reduce_input_1 == "AMX_X"
        fma = True
      else:
        reduce_input = ast_parse(loop_exit[-1], reduceops[0].src[0], -1)
        fma = False


      if USE_4X4:
        phis = [val_type([LLVMBuffer.start_for_op[reduceops[0].op]]*(DY*DX))]
      else:
        phis = [LLVMBuffer.start_for_op[reduceops[0].op]]
      if not USE_AMX:
        for i in range(store_loop+1, len(loop_entry)):
          val = loop_entry[i].phi(val_type, f"reduce_phi_{i}")
          val.add_incoming(phis[-1], loop_entry[i-1]._block)
          phis.append(val)

      if reduceops[0].op == ReduceOps.SUM:
        if fma:
          #reduce_result = loop_exit[-1].call(ir.Function(module, ir.FunctionType(val_type, [val_type, val_type, val_type]), name="llvm.fma"), [reduce_input_0, reduce_input_1, val], fastmath=('fast',))
          for j in range(AMX_SZ_Y):
            for i in range(AMX_SZ_X):
              z_row = j*AMX_SZ_X + i
              # NOTE: the x and y offsets are in <bytes> not <elements>!
              # <Z row> <X offset> <Y offset>
              AMX.fma32(loop_exit[-1], int_const(z_row<<20 | (i*16*4)<<10 | (j*16*4)))
          reduce_result = "AMX_Z"
        else:
          reduce_result = loop_exit[-1].fadd(reduce_input, val, flags=('fast',))
      elif reduceops[0].op == ReduceOps.MAX:
        # TODO: this doesn't respect the fast math flag. it won't vectorize, and i'm not sure if llvm supports it
        reduce_result = loop_exit[-1].call(ir.Function(module, ir.FunctionType(val_type, [val_type, val_type]), name="llvm.maximum"), [reduce_input, val], fastmath=('fast',))

      for i,phi in enumerate(phis[1:]):
        if reduce_result != "AMX_Z":
          phi.add_incoming(reduce_result, loop_exit[store_loop+1+i]._block)

    # do the late ast
    result = ast_parse(loop_exit[store_loop], ast, store_loop, reduce_result=reduce_result)

    # store 4x4
    if USE_4X4:
      builder = loop_exit[store_loop]
      if USE_AMX:
        zptr = builder.ptrtoint(func.args[0], ir.IntType(64))
        zptr = builder.add(zptr, builder.mul(idx_level[0][store_loop], int_const(4)))
        assert strides[0][-1] == 1
        # using all 64 Z registers (4 kB)
        for j in range(AMX_SZ_X):
          for k in range(16):
            # TODO: non mult
            for i in range(0,AMX_SZ_Y,2):
              z_row = j*AMX_SZ_X + i
              ptr = ((j*16)+k)*strides[0][-2] + i*16
              AMX.stz(builder, builder.add(zptr, int_const(1 << 62 | ((k*4+z_row) << 56) | ptr*4)))
      else:
        for i, idx in enumerate(get_idxs(builder, idx_level[0][store_loop], 0)):
          result_x = builder.extract_element(result, int_const(i))
          builder.store(result_x, builder.gep(func.args[0], [idx], inbounds=True))
    else:
      loop_exit[store_loop].store(result, loop_exit[store_loop].gep(func.args[0], [idx_level[0][store_loop]], inbounds=True))
    
    # add the looping
    set_clr = False
    for i,s in enumerate(full_shape):
      if USE_AMX and i == store_loop:
        AMX.set(loop_entry[store_loop])
      loop_entry[i].branch(loop_entry[i+1]._block)
      # index
      idx = loop_entry[i+1].phi(ir.IntType(64), name=f"loopvar_{i}")
      idx.add_incoming(int_const(0), loop_entry[i]._block)
      idx_p1 = loop_exit[i+1].add(idx, int_const(1))
      idx.add_incoming(idx_p1, loop_exit[i+1]._block)
      if USE_AMX and i+1 == store_loop:
        set_clr = True
        AMX.clr(loop_exit[store_loop])
      loop_exit[i+1].cbranch(loop_exit[i+1].icmp_unsigned("==", idx_p1, int_const(s)), loop_exit[i]._block, loop_entry[i+1]._block)

    loop_entry[-1].branch(loop_exit[-1]._block)
    if not set_clr:
      AMX.clr(loop_exit[0])
    loop_exit[0].ret_void()

    LLVMBuffer.func_cache[key] = LLVM().exec(module, bufs, info.flops)
    return ret
