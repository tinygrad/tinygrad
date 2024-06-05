from tinygrad import Variable, Tensor

def sym_arange(N):
  v = Variable("v", 1, 10).bind(N)
  t = Tensor.ones(v)
  # pad left and pool
  t = t.pad(((v-1, 0),)).expand(v, -1).flatten().pad(((0, v),)).reshape((v, v*2))
  # take first v and sum
  t = t.shrink(((0, v),)).sum(0) - 1
  return t.realize().reshape(N)

# this generates the same kernel
def sym_arange2(N):
  v = Variable("v", 1, 10).bind(N)
  t = Tensor.ones(v, v)
  # pad left and pool
  t = t.pad((None, (v-1, 0))).flatten().pad(((0, v),)).reshape((v, v*2))
  # take first v and sum
  t = t.shrink(((0, v),)).sum(0) - 1
  return t.realize().reshape(N)

if __name__ == "__main__":
  for N in [4, 5, 6]:
    print(sym_arange(N).numpy())
