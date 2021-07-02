#!/usr/bin/env python3

#import faulthandler
#faulthandler.enable()

# RISK architecture is going to change everything

# Arty A7-100T
#   256 MB of DDR3 with 2.6 GB/s of RAM bandwidth (vs 512 GB/s on S7t-VG6)
#   255K 19-bit elements

# S7t-VG6
#   16 GB of GDDR6
#   189 Mb embedded RAM, aka 9M 19-bit elements
#   2560 MLP blocks, 2 fp24 MULACC each

import os
import functools
import numpy as np
from collections import defaultdict

# 32x32 * 32x32 -> 32x32 matmul = 65536 FLOPS @ 1 GHz = 64 TOPS
# mulacc is 2048 FLOPS, 32x less
# 32x32 (aka 1024 element) ALU
# 1024 wide permute
# 1024 wide load/store (1 cycle to SRAM)
# all in elements, aka TF32 (19 bits)

# targets:
#   matmul input
#   matmul weights
#   ALU
#   permute

# 1024x1024x4x19 bits = 10MB
# fully strided
# load1024 <target>, <address>, <stride x (32)>, <stride y (32)>

# 4 slots
# <input> <weight> <output> <empty>
# <empty> <output> <input> <weight>
# <weight> <input> <empty> <output>

# TODO: delete slots, write malloc
SZ = 32
SLOTSIZE = 1024*1024*2   # 5MB, for 20MB total. 8M elements
sram = np.zeros((SLOTSIZE*4), dtype=np.float32)
regfile = {}
SLOT = lambda x: x*SLOTSIZE

from enum import Enum
class Reg(Enum):
  # can the ALU use the same registers?
  MATMUL_INPUT = 1
  MATMUL_WEIGHTS = 2
  MATMUL_OUTPUT = 3
  MATMUL_ACC = 4

# this should be a generic function with a LUT, similar to the ANE
class UnaryOps(Enum):
  RELU = 0
  EXP = 1
  LOG = 2
  GT0 = 3

class BinaryOps(Enum):
  ADD = 0
  SUB = 1
  MUL = 2
  DIV = 3
  MULACC = 4
  POW = 5

class ReduceOps(Enum):
  SUM = 0
  MAX = 1

for t in Reg:
  regfile[t] = np.zeros((SZ, SZ), dtype=np.float32)

# *** profiler ***

cnts = defaultdict(int)
tcnts = defaultdict(int)
utils = defaultdict(int)
maxdma = 0
def count(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    cnts[func.__name__] += 1
    tcnts[func.__name__] += 1
    return func(*args, **kwargs)
  return wrapper

import atexit
@atexit.register
def cherry_print_counts():
  print(cnts)
  print(tcnts)
  print(utils)
  util_n = sum([k[0]*k[1]*v for k,v in utils.items()])
  util_d = sum([SZ*SZ*v for k,v in utils.items()])
  print("%.2f GOPS %d maxdma" % ((tcnts['riski_matmul']*SZ*SZ*SZ*2 + tcnts['riski_mulacc']*SZ*SZ*2)*1e-9, maxdma))
  print("ran in %.2f us with util %.2f%% total %.2f us" % (sum(cnts.values())*1e-3, util_n*100/(util_d+1), sum(tcnts.values())*1e-3))

def cherry_reset_counts():
  global cnts, utils
  cnts = defaultdict(int)
  utils = defaultdict(int)

def cherry_regdump():
  print("\n***** regdump *****")
  print(regfile[Reg.MATMUL_INPUT])
  print(regfile[Reg.MATMUL_WEIGHTS])
  print(regfile[Reg.MATMUL_OUTPUT])

# *** instructions ***

@count
def riski_unop(op):
  if op == UnaryOps.RELU:
    regfile[Reg.MATMUL_OUTPUT] = np.maximum(regfile[Reg.MATMUL_INPUT], 0)
  elif op == UnaryOps.LOG:
    regfile[Reg.MATMUL_OUTPUT] = np.log(regfile[Reg.MATMUL_INPUT])
  elif op == UnaryOps.EXP:
    regfile[Reg.MATMUL_OUTPUT] = np.exp(regfile[Reg.MATMUL_INPUT])
  elif op == UnaryOps.GT0:
    regfile[Reg.MATMUL_OUTPUT] = (regfile[Reg.MATMUL_INPUT] >= 0)

@count
def riski_add():
  regfile[Reg.MATMUL_OUTPUT] = regfile[Reg.MATMUL_INPUT] + regfile[Reg.MATMUL_WEIGHTS]

@count
def riski_sub():
  regfile[Reg.MATMUL_OUTPUT] = regfile[Reg.MATMUL_INPUT] - regfile[Reg.MATMUL_WEIGHTS]

@count
def riski_mul(sel=None):
  if sel is not None:
    # for broadcasting too
    # TODO: put in other ops and work out transpose
    regfile[Reg.MATMUL_OUTPUT] = regfile[Reg.MATMUL_INPUT][sel:sel+1].T * regfile[Reg.MATMUL_WEIGHTS]
  else:
    regfile[Reg.MATMUL_OUTPUT] = regfile[Reg.MATMUL_INPUT] * regfile[Reg.MATMUL_WEIGHTS]

@count
def riski_div():
  regfile[Reg.MATMUL_OUTPUT] = regfile[Reg.MATMUL_INPUT] / regfile[Reg.MATMUL_WEIGHTS]

@count
def riski_mulacc():
  # riski_mul / MATMUL_ACC += Reg.MATMUL_OUTPUT
  regfile[Reg.MATMUL_ACC] += regfile[Reg.MATMUL_INPUT] * regfile[Reg.MATMUL_WEIGHTS]

@count
def riski_pow():
  regfile[Reg.MATMUL_OUTPUT] = regfile[Reg.MATMUL_INPUT] ** regfile[Reg.MATMUL_WEIGHTS]

@count
def riski_reduce_sum(cnt=SZ, tgt=0, acc=False):
  # TODO: use acc in reducer
  if acc:
    regfile[Reg.MATMUL_ACC][tgt] += regfile[Reg.MATMUL_OUTPUT][0:cnt].sum(axis=0)
  else:
    regfile[Reg.MATMUL_ACC][tgt] = regfile[Reg.MATMUL_OUTPUT][0:cnt].sum(axis=0)

@count
def riski_reduce_max(cnt=SZ, tgt=0, acc=False):
  if acc:
    regfile[Reg.MATMUL_ACC][tgt] = np.concatenate([regfile[Reg.MATMUL_OUTPUT][0:cnt], regfile[Reg.MATMUL_ACC][tgt]], axis=0).max(axis=0)
  else:
    regfile[Reg.MATMUL_ACC][tgt] = regfile[Reg.MATMUL_OUTPUT][0:cnt].max(axis=0)

# TODO: make accumulate a bit in the instruction available to all
binops = {BinaryOps.ADD: riski_add,
          BinaryOps.SUB: riski_sub,
          BinaryOps.MUL: riski_mul,
          BinaryOps.DIV: riski_div,
          BinaryOps.MULACC: riski_mulacc,
          BinaryOps.POW: riski_pow}

reduceops = {ReduceOps.SUM: riski_reduce_sum,
             ReduceOps.MAX: riski_reduce_max}

SLOW_MATMUL = os.getenv("SLOW_MATMUL", False)
@count
def riski_matmul(slow=SLOW_MATMUL):
  #print("LLL:\n",regfile[Reg.MATMUL_INPUT],"\n",regfile[Reg.MATMUL_WEIGHTS])
  if not slow:
    regfile[Reg.MATMUL_ACC] += \
      regfile[Reg.MATMUL_INPUT] @ \
      regfile[Reg.MATMUL_WEIGHTS]
  else:
    for l in range(0, SZ):
      riski_mul(l)
      riski_reduce_sum(tgt=l, acc=True)

@count
def riski_mov(tout, tin):
  ret = regfile[tin]
  regfile[tout] = np.copy(ret)

@count
def riski_zero(tout):
  regfile[tout][:, :] = 0

load_log = open("/tmp/risk_load_log", "w") if os.getenv("LOAD_LOG") else None

@count
def riski_load(target, address, stride_y=SZ, stride_x=1, len_y=SZ, len_x=SZ, zero=True, skip_first=False):
  global util_n, util_d
  if load_log is not None:
    load_log.write("%d %d %d %d %d\n" % (address, stride_y, stride_x, len_y, len_x))
  utils[(len_y, len_x)] += 1
  stride_y, stride_x = int(stride_y), int(stride_x)
  assert (address + stride_y*(len_y-1) + stride_x*(len_x-1)) < sram.shape[0]
  d = regfile[target]
  if zero:
    d[:] = 0
  if skip_first:
    d[1:(len_y+1), :len_x] = np.lib.stride_tricks.as_strided(sram[address:], (len_y, len_x), (stride_y*4, stride_x*4))
  else:
    d[:len_y, :len_x] = np.lib.stride_tricks.as_strided(sram[address:], (len_y, len_x), (stride_y*4, stride_x*4))
  """
  for y in range(0, len_y):
    for x in range(0, len_x):
      d[y, x] = sram[address + y*stride_y + x*stride_x]
  """

@count
def riski_store(target, address, stride_y=SZ, stride_x=1, len_y=SZ, len_x=SZ):
  stride_y, stride_x = int(stride_y), int(stride_x)
  assert (address + stride_y*(len_y-1) + stride_x*(len_x-1)) < sram.shape[0]
  d = regfile[target]
  np.lib.stride_tricks.as_strided(sram[address:], (len_y, len_x), (stride_y*4, stride_x*4))[:, :] = d[:len_y, :len_x]
  """
  for y in range(0, len_y):
    for x in range(0, len_x):
      sram[address + y*stride_y + x*stride_x] = d[y, x]
  """

# *** DMA engine ***

@count
def cherry_dmar(address, arr):
  global maxdma
  arr = arr.reshape(-1)
  assert(arr.shape[0] <= SLOTSIZE)
  maxdma = max(maxdma, arr.shape[0])
  #print("DMAR %d elements" % arr.shape[0])
  sram[address:address+arr.shape[0]] = arr

@count
def cherry_dmaw(address, shp):
  assert(np.prod(shp) <= SLOTSIZE)
  #print("DMAW %d elements" % np.prod(shp))
  return np.copy(sram[address:address+np.prod(shp)].reshape(shp))

# *** CHERRY code to be compiled ***

def cherry_reduceop(inp, op, axis, keepdims=False):
  dimlist, redlist = [], []
  if type(axis) == int:
    axis = [axis]
  if axis is None:
    # full reduce
    #osize = [1]*len(inp.shape)
    osize = [1]
    dimlist = [np.prod(inp.shape), 1]
    redlist = [True, False]
    #dimlist = [1, np.prod(inp.shape)]
    #redlist = [False, True]
  else:
    osize = np.array(inp.shape)
    osize[list(axis)] = 1

    # skip any early 1s
    sd = 0
    while sd < len(osize) and osize[sd] == 1 and inp.shape[sd] == 1:
      sd += 1

    # this would be good in the GPU
    for i in range(sd, len(inp.shape)):
      is_reduce_axis = osize[i] == 1 and inp.shape[i] != -1
      if len(dimlist) == 0:
        dimlist.append(inp.shape[i])
        redlist.append(is_reduce_axis)
      else:
        if redlist[-1] == is_reduce_axis:
          dimlist[-1] *= inp.shape[i]
        else:
          dimlist.append(inp.shape[i])
          redlist.append(is_reduce_axis)

    if not keepdims:
      nosize = []
      for i in range(osize.shape[0]):
        if i not in axis:
          nosize.append(osize[i])
      osize = nosize

  osize = tuple(osize)
  #print("reduce", op, inp.shape, axis, "->", osize, dimlist, redlist)
  inslot, outslot = 0, 2
  cherry_dmar(SLOT(inslot), inp)

  # dimlist is always the inshape
  # redlist is always [False, True, False, True, ...., True, False]

  # special case if redlist ends with True
  if len(redlist) > 0 and redlist[-1] == True:
    #print("special case redlist[-1] == True")
    outside = int(np.prod(dimlist[:-1]))
    for l in range(0, outside, SZ):
      reduce_size = min(SZ, outside-l)
      j = 0
      while j < dimlist[-1]:
        len_y = min(SZ if j == 0 else SZ-1, dimlist[-1]-j)
        riski_load(Reg.MATMUL_OUTPUT,
          SLOT(inslot) + l*dimlist[-1] + j,
          stride_y=1, stride_x=dimlist[-1],
          len_y=len_y,
          len_x=reduce_size,
          zero=j==0, skip_first=j!=0)
        reduceops[op](len_y+(j!=0))
        riski_mov(Reg.MATMUL_OUTPUT, Reg.MATMUL_ACC)  # move the first row
        j += SZ if j == 0 else SZ-1
      riski_store(Reg.MATMUL_ACC, SLOT(outslot) + l, len_y=1, len_x=reduce_size)
    # remove last dimension
    redlist = redlist[:-1]
    dimlist = dimlist[:-1]
    inslot, outslot = outslot, inslot

  # do the reduce merges one at a time
  while len(dimlist) >= 2:
    #print("proc", dimlist, redlist)

    for l in range(0, int(np.prod(dimlist[:-2]))):
      for k in range(0, dimlist[-1], SZ):
        reduce_size = min(SZ, dimlist[-1]-k)
        j = 0
        while j < dimlist[-2]:
          len_y = min(SZ if j == 0 else SZ-1, dimlist[-2]-j)
          riski_load(Reg.MATMUL_OUTPUT,
            SLOT(inslot) + l*dimlist[-2]*dimlist[-1] + j*dimlist[-1] + k,
            stride_y=dimlist[-1], stride_x=1,
            len_y=len_y,
            len_x=reduce_size,
            zero=j==0, skip_first=j!=0)
          #cherry_regdump()
          reduceops[op](len_y+(j!=0))
          riski_mov(Reg.MATMUL_OUTPUT, Reg.MATMUL_ACC)  # move the first row
          j += SZ if j == 0 else SZ-1
        riski_store(Reg.MATMUL_ACC, SLOT(outslot) + l*dimlist[-1] + k, len_y=1, len_x=reduce_size)
    inslot, outslot = outslot, inslot
    if len(dimlist) <= 2:
      break
    # merge [False, True] into [False]
    dimlist = dimlist[:-3] + [dimlist[-3]*dimlist[-1]]

  return cherry_dmaw(SLOT(inslot), osize)

def cherry_unop(x, op):
  cherry_dmar(SLOT(0), x)
  cnt = np.prod(x.shape)
  for i in range(0, np.prod(x.shape), SZ*SZ):
    riski_load(Reg.MATMUL_INPUT, SLOT(0)+i)
    riski_unop(op)
    riski_store(Reg.MATMUL_OUTPUT, SLOT(2)+i)
  return cherry_dmaw(SLOT(2), x.shape)

def cherry_binop(x, y, op):
  n_dims = max(len(x.shape), len(y.shape))
  shape_x, shape_y = np.ones(n_dims, dtype=np.int32), np.ones(n_dims, dtype=np.int32)
  shape_x[:len(x.shape)] = np.array(x.shape, dtype=np.int32)
  shape_y[:len(y.shape)] = np.array(y.shape, dtype=np.int32)
  if not np.all((shape_x == 1) | (shape_y == 1) | (shape_x == shape_y)):
    raise Exception(f"binary op unbroadcastable shape mismatch: {x.shape} vs {y.shape}")
  shape_ret = np.maximum(shape_x, shape_y)
  #print(shape_x, shape_y, shape_ret)

  dimlist, complist = [], [] # note: len(dimlist) may be less than n_dims
  def push(dim, comp):
    if len(complist) > 0 and complist[-1] == comp:
      dimlist[-1] *= dim
    elif comp != (False, False):
      dimlist.append(dim); complist.append(comp)
  for i in range(n_dims): # group together any adjacent dimensions that we can to simplify broadcasting
    push(max(shape_x[i], shape_y[i]), (shape_x[i] > 1, shape_y[i] > 1))

  #print(dimlist, complist)

  cherry_dmar(SLOT(0), x)
  cherry_dmar(SLOT(1), y)
  if len(dimlist) <= 1:
    if len(complist) == 0:
      complist = [(True, True)]
    for i in range(0, np.prod(shape_ret), SZ*SZ):
      if complist[0][0]:
        riski_load(Reg.MATMUL_INPUT, SLOT(0)+i)
      else:
        riski_load(Reg.MATMUL_INPUT, SLOT(0), stride_y=0, stride_x=0)
      if complist[0][1]:
        riski_load(Reg.MATMUL_WEIGHTS, SLOT(1)+i)
      else:
        riski_load(Reg.MATMUL_WEIGHTS, SLOT(1), stride_y=0, stride_x=0)
      binops[op]()
      riski_store(Reg.MATMUL_OUTPUT, SLOT(2)+i)
  else:
    # broadcasting on the inner 2 "real" dimensions sped up
    # NOTE: this can be made faster by supporting any dimensions
    def gd(idx, dims, comps):
      ret = 0
      mult = 1
      in_idx = idx
      for c,d in zip(comps[::-1], dims[::-1]):
        tt = idx % d
        idx = idx // d
        if c == False:
          continue
        ret += tt*mult
        mult *= d
      #print(ret, in_idx, dims, comps)
      return ret
    for i in range(0, int(np.prod(dimlist[:-2]))):
      off_0 = SLOT(0) + gd(i, dimlist[:-2], [x[0] for x in complist[:-2]])*\
       (dimlist[-2] if complist[-2][0] else 1)*(dimlist[-1] if complist[-1][0] else 1)
      off_1 = SLOT(1) + gd(i, dimlist[:-2], [x[1] for x in complist[:-2]])*\
        (dimlist[-2] if complist[-2][1] else 1)*(dimlist[-1] if complist[-1][1] else 1)
      off_2 = SLOT(2) + gd(i, dimlist[:-2], [True]*len(dimlist[:-2]))*dimlist[-2]*dimlist[-1]
      for j in range(0, dimlist[-2], SZ):
        for k in range(0, dimlist[-1], SZ):
          sy = complist[-2][0]*(dimlist[-1] if complist[-1][0] else 1)
          riski_load(Reg.MATMUL_INPUT,
            off_0 + j*sy + k*complist[-1][0],
            stride_y=sy, stride_x=complist[-1][0])
          sy = complist[-2][1]*(dimlist[-1] if complist[-1][1] else 1)
          riski_load(Reg.MATMUL_WEIGHTS,
            off_1 + j*sy + k*complist[-1][1],
            stride_y=sy, stride_x=complist[-1][1])
          binops[op]()
          # output is always "True"
          riski_store(Reg.MATMUL_OUTPUT, off_2 + j*dimlist[-1] + k,
            stride_y=dimlist[-1], stride_x=1,
            len_y=min(SZ, dimlist[-2]-j), len_x=min(SZ, dimlist[-1]-k))

  return cherry_dmaw(SLOT(2), shape_ret)

def cherry_matmul(x, w, transpose_x=False, transpose_w=False):
  # copy matrices into SRAM
  # x is M x K
  # w is K x N
  # out is M x N
  cherry_dmar(SLOT(0), x)
  cherry_dmar(SLOT(1), w)

  if transpose_x:
    K,M = x.shape[-2], x.shape[-1]
  else:
    M,K = x.shape[-2], x.shape[-1]
  if transpose_w:
    N = w.shape[-2]
    assert w.shape[-1] == K
  else:
    N = w.shape[-1]
    assert w.shape[-2] == K
  cnt = np.prod(x.shape[0:-2]) if len(x.shape) > 2 else 1

  # do matmul
  for c in range(cnt):
    for m in range(0, M, SZ):
      for n in range(0, N, SZ):
        riski_zero(Reg.MATMUL_ACC)
        for k in range(0, K, SZ):
          if transpose_x:
            riski_load(Reg.MATMUL_INPUT, SLOT(0)+c*M*K + k*M+m, 1, M, min(SZ, M-m), min(SZ, K-k))
          else:
            riski_load(Reg.MATMUL_INPUT, SLOT(0)+c*M*K + m*K+k, K, 1, min(SZ, M-m), min(SZ, K-k))
          if transpose_w:
            riski_load(Reg.MATMUL_WEIGHTS, SLOT(1)+c*K*N + n*K+k, 1, K, min(SZ, K-k), min(SZ, N-n))
          else:
            riski_load(Reg.MATMUL_WEIGHTS, SLOT(1)+c*K*N + k*N+n, N, 1, min(SZ, K-k), min(SZ, N-n))
          riski_matmul()
        riski_store(Reg.MATMUL_ACC, SLOT(2)+c*M*N + m*N+n, N, 1, min(SZ, M-m), min(SZ, N-n))

  # copy back from SRAM
  return cherry_dmaw(SLOT(2), (*x.shape[0:-2],M,N))

import unittest
class TestCherry(unittest.TestCase):
  def test_matmul_even(self):
    x = np.random.uniform(size=(SZ*8, SZ*8)).astype(np.float32)
    w = np.random.uniform(size=(SZ*8, SZ*8)).astype(np.float32)
    np.testing.assert_allclose(x @ w, cherry_matmul(x, w), rtol=1e-5)

  def test_matmul_small(self):
    x = np.array([[1,2,3],[4,5,6],[7,8,9]])
    w = np.array([[-1,-2,-3],[-4,-5,-6],[-7,-8,-9]])
    np.testing.assert_allclose(x @ w, cherry_matmul(x, w), rtol=1e-5)

  def test_matmul_uneven(self):
    x = np.random.uniform(size=(47, 79)).astype(np.float32)
    w = np.random.uniform(size=(79, 42)).astype(np.float32)
    np.testing.assert_allclose(x @ w, cherry_matmul(x, w), rtol=1e-5)

  def test_matmul_transpose(self):
    x = np.random.uniform(size=(33, 33)).astype(np.float32)
    w = np.random.uniform(size=(33, 33)).astype(np.float32)
    np.testing.assert_allclose(x @ w, cherry_matmul(x, w), rtol=1e-5)
    np.testing.assert_allclose(x.T @ w, cherry_matmul(x, w, True), rtol=1e-5)
    np.testing.assert_allclose(x @ w.T, cherry_matmul(x, w, False, True), rtol=1e-5)
    np.testing.assert_allclose(x.T @ w.T, cherry_matmul(x, w, True, True), rtol=1e-5)

  def test_matmul_transpose_uneven_w(self):
    x = np.random.uniform(size=(47, 79)).astype(np.float32)
    w = np.random.uniform(size=(42, 79)).astype(np.float32)
    np.testing.assert_allclose(x @ w.T, cherry_matmul(x, w, transpose_w=True), rtol=1e-5)

  def test_matmul_transpose_uneven_x(self):
    x = np.random.uniform(size=(79, 47)).astype(np.float32)
    w = np.random.uniform(size=(79, 42)).astype(np.float32)
    np.testing.assert_allclose(x.T @ w, cherry_matmul(x, w, transpose_x=True), rtol=1e-5)

  def test_mulacc_matmul(self):
    regfile[Reg.MATMUL_INPUT] = np.arange(1, SZ*SZ+1).reshape((SZ, SZ))
    regfile[Reg.MATMUL_WEIGHTS] = np.arange(1, SZ*SZ+1).reshape((SZ, SZ))*-1
    riski_zero(Reg.MATMUL_ACC)
    riski_matmul()
    tst1 = np.copy(regfile[Reg.MATMUL_ACC])
    riski_zero(Reg.MATMUL_ACC)
    riski_matmul(True)
    tst2 = np.copy(regfile[Reg.MATMUL_ACC])
    np.testing.assert_allclose(tst1, tst2)


if __name__ == "__main__":
  np.random.seed(1337)
  unittest.main(verbosity=2)

