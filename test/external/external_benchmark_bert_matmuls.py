from tinygrad import Tensor

if __name__ == "__main__":
  # matmuls in bert layers
  BS=96
  shapes = [
    ((BS//6, 16, 512, 512), (BS//6, 16, 512, 64)),  # x48. 27 TFLOPS on 4090 with BEAM=4
    ((BS//6, 16, 512, 64), (BS//6, 16, 64, 512)),   # x48. 47 TFLOPS on 4090 with BEAM=4
  ]
  for s0, s1 in shapes:
    t0 = Tensor.empty(s0, dtype="half")
    t1 = Tensor.empty(s1, dtype="half")
    for _ in range(5):
      t0.matmul(t1, dtype="half").realize()
