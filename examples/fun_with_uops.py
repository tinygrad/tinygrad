from tinygrad import UOp, dtypes
from tinygrad.schedule.indexing import BufferizeOpts
from tinygrad.schedule.rangeify import get_rangeify_map
#from tinygrad.engine.schedule import create_schedule_with_vars

# store a 10 by 10 grid of 0s
x,y = UOp.range(10, "x"), UOp.range(10, "y")
base = UOp.const(dtypes.float, 0, device="CPU").contiguous()
buf = UOp.bufferize(base, x, y, arg=BufferizeOpts("CPU"))
print(buf.pyrender())

# add 1
x,y = UOp.range(10, "x2"), UOp.range(10, "y2")
buf2 = UOp.bufferize(buf[x,y]+1, x, y, arg=BufferizeOpts("CPU"))

# TODO: pyrender should be print
print("\n\n")
print(buf2.pyrender())

print("\n\n")
sink = buf2.sink()
rr = get_rangeify_map(sink)[buf2]
print(rr.pyrender())

# schedule
#for si in create_schedule_with_vars(rr): print(si)

