from tinygrad.runtime.ops_gpu import CLProgram, CL, CLBuffer
import time

N = 1000000
a = CLBuffer(N*4)
b = CLBuffer(N*4)
c = CLBuffer(N*4)

prg = CLProgram("test", """__kernel void test(__global float *a, __global float *b, __global float *c) {
  int idx = get_global_id(0);
  a[idx] = b[idx] + c[idx];
}""")
prg.clprg(CL.cl_queue, [N,], None, a._cl, b._cl, c._cl)

t1 = time.monotonic_ns()
e1 = prg.clprg(CL.cl_queue, [N,], None, a._cl, b._cl, c._cl)
CL.cl_queue.finish()  # type: ignore
t2 = time.monotonic_ns()
time.sleep(3)
t3 = time.monotonic_ns()
e2 = prg.clprg(CL.cl_queue, [N,], None, a._cl, b._cl, c._cl)
CL.cl_queue.finish()  # type: ignore
t4 = time.monotonic_ns()

print(e1.profile.queued)
print(e1.profile.submit)
print(e1.profile.start)
print(e1.profile.end)

print(e1, e2)
print(t2-t1, e1.profile.end - e1.profile.start)
print(t4-t3, e2.profile.end - e2.profile.start)
print(t3-t2, e2.profile.queued-e1.profile.end)
print((t3-t2) / (e2.profile.start-e1.profile.end), "ratio")

print("ratio since boot", t1/e1.profile.start)

print(e1.profile.start)
print(e1.profile.end)
print(e2.profile.start)
print(e2.profile.end)
