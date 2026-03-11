from tinygrad import UOp, dtypes, Device, Tensor

if __name__ == "__main__":
  B0 = UOp.new_buffer(Device.DEFAULT, 100, dtypes.float).reshape(10,10)
  B1 = UOp.new_buffer(Device.DEFAULT, 100, dtypes.float).reshape(10,10)

  R0 = UOp.range(10, axis_id=0)
  R1 = UOp.range(10, axis_id=1)

  b0 = UOp.param(0, dtypes.float, (10,10))
  b1 = UOp.param(1, dtypes.float, (10,10))
  r0 = UOp.param(2, dtypes.index, ())
  r1 = UOp.param(3, dtypes.index, ())

  fxn = (b0[r0, r1] + b1[r0, r1]).call(B0, B1, R0, R1)
  t = Tensor(fxn)
  t.realize()

