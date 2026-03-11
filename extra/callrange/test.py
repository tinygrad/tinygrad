from tinygrad import UOp, dtypes, Device, Tensor

if __name__ == "__main__":
  B0 = UOp.new_buffer(Device.DEFAULT, 100, dtypes.float).reshape(10,10)
  B1 = UOp.new_buffer(Device.DEFAULT, 100, dtypes.float).reshape(10,10)


  b0 = UOp.param(0, dtypes.float, (10,10))
  b1 = UOp.param(1, dtypes.float, (10,10))
  r0 = UOp.range(10, axis_id=0)
  r1 = UOp.range(10, axis_id=1)

  fxn = (b0[r0, r1] + b1[r0, r1]).call(B0, B1)
  t = Tensor(fxn)
  t.realize()

# gemm (N,N)

# (N//k, k, N//k, k)


# what if call just implicitly ends all ranges and you don't need to connect them?
# you do have to connect them, and it does end the ranges

# if assign (store+after) is on call, we move the store into the call (indexed with the ranges) and replace the assign with an after


def gemm(A, B):
  N = 4096
  k = 128

  ia = UOp.param(0, dtypes.float, (k, k)).reshape(k, 1, k)
  ib = UOp.param(1, dtypes.float, (k, k)).reshape(1, k, k)
  gemm_fxn = (ia * ib).sum(2) # <-- rangeify this

  a = UOp.param(0, dtypes.float, (N, N))
  b = UOp.param(1, dtypes.float, (N, N))
  r0 = UOp.range(N//k, 0)
  r1 = UOp.range(N//k, 1)
  local_fxn = gemm_fxn.call(a.reshape(N//k, k, N//k, k)[r0, :, r1, :], b.reshape(N//k, k, N//k, k)[r0, :, r1, :], r0, r1).permute(0,2,1,3).reshape(N,N)

  fxn = local_fxn.call(A,B)



  return





  a = UOp.param(0, dtypes.float, (N//k, k, N//k, k))
  b = UOp.param(1, dtypes.float, (N//k, k, N//k, k))



  # inner kxk GEMM (are WMMAs calls?)
  ia = UOp.param(0, dtypes.float, (k,k)).reshape(k, 1, k)
  ib = UOp.param(1, dtypes.float, (k,k)).reshape(1, k, k)
  r0 = UOp.range(N//k, 0)
  r1 = UOp.range(N//k, 1)
  fxn = (ia * ib).sum(2).call(a[:, r0, :, r1], b[:, r0, :, r1])   # this call ends these ranges implicitly
  assert fxn.shape == (N//k, N//k, k, k)


  #.call(A, B, UOp.range(N//k), UOp.range(N//k))

  #r0 = UOp.param(2, dtypes.index, (), vmin_vmax=(0, N//k-1))
  #r1 = UOp.param(3, dtypes.index, (), vmin_vmax=(0, N//k-1))










# Q = [batch, seq_len, heads, dim]
# K = [batch, seq_len, head_kv, dim]
# V = [batch, seq_len, head_kv, dim]



