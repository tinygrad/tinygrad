from tinygrad import Variable, Tensor

def sym_arange(N):
  v = Variable("v", 1, 10).bind(N)
  t = Tensor.ones(v, v)
  # pad left and pool
  t = t.pad((None, (v, 0))).flatten().pad(((0, v),)).reshape((v, v*2+1))
  # take first v and sum
  t = t.shrink(((0, v),)).sum(0)
  return t.realize().reshape(N)

if __name__ == "__main__":
  for N in [4, 5, 6]:
    print(sym_arange(N).numpy())
