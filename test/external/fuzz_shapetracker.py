import random
from test.unit.test_shapetracker import CheckingShapeTracker

def do_permute(st):
  perm = list(range(0, len(st.shape)))
  random.shuffle(perm)
  perm = tuple(perm)
  print("st.permute(", perm, ")")
  st.permute(perm)

def do_pad(st):
  c = random.randint(0, len(st.shape)-1)
  pad = tuple((random.randint(0,2), random.randint(0,2)) if i==c else (0,0) for i in range(len(st.shape)))
  print("st.pad(", pad, ")")
  st.pad(pad)

def do_reshape_add_1_at_end(st):
  shp = st.shape + (1,)
  print("st.reshape(", shp, ")")
  st.reshape(shp)

def do_reshape_combine_two(st):
  if len(st.shape) < 2: return
  c = random.randint(0, len(st.shape)-2)
  shp = st.shape[:c] + (st.shape[c] * st.shape[c+1], ) + st.shape[c+2:]
  print("st.reshape(", shp, ")")
  st.reshape(shp)

def do_shrink(st):
  while 1:
    shrink = tuple((random.randint(0, 2), random.randint(s-2,s)) for i,s in enumerate(st.shape))
    if all(x<y for (x,y) in shrink): break
  print("st.shrink(", shrink, ")")
  st.shrink(shrink)

def do_stride(st):
  c = random.randint(0, len(st.shape)-1)
  # TODO: negative strides have an issue
  stride = tuple(random.choice([-2,-1,2]) if i==c else 1 for i in range(len(st.shape)))
  #stride = tuple(random.choice([2]) if i==c else 1 for i in range(len(st.shape)))
  print("st.stride(", stride, ")")
  st.stride(stride)

def do_expand(st):
  c = [i for i,s in enumerate(st.shape) if s==1]
  if len(c) == 0: return
  c = random.choice(c)
  expand = tuple(random.choice([2,3,4]) if i==c else s for i,s in enumerate(st.shape))
  print("st.expand(", expand, ")")
  st.expand(expand)

if __name__ == "__main__":
  ops = [do_permute, do_pad, do_shrink, do_reshape_add_1_at_end, do_reshape_combine_two, do_stride, do_expand]
  while 1:
    st = CheckingShapeTracker((3, 3, 3))
    for i in range(3): random.choice(ops)(st)
    #st.simplify()
    st.assert_same()
