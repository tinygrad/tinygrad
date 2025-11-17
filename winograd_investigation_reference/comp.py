# import time
# from statistics import median
# from tinygrad import Tensor
# from tinygrad.dtype import dtypes
# from tinygrad.helpers import Context, GlobalCounters

# def timed_conv(HW=256, Cin=128, Cout=128, reps=100, B=1):
#   x = Tensor.randn(B, Cin, HW, HW, dtype=dtypes.float32).realize()
#   w = Tensor.randn(Cout, Cin, 3, 3, dtype=dtypes.float32).realize()
#   def run(**kw):
#     with Context(**{k:int(bool(v)) for k,v in kw.items()}):
#       # warmup to compile
#       GlobalCounters.reset(); (x.conv2d(w, padding=1)).realize()
#       ts = []
#       for _ in range(reps):
#         t0 = time.perf_counter()
#         (x.conv2d(w, padding=1)).realize()
#         ts.append(time.perf_counter()-t0)
#       return median(ts)
#   t_base = run(WINO=0, WINO_OLD=0)
#   t_new  = run(WINO=1, WINO_OLD=0)
#   t_old  = run(WINO=0, WINO_OLD=1)
#   print(f"HW={HW}, Cin={Cin}, Cout={Cout}, B={B} | base={t_base*1e3:.1f}ms  new={t_new*1e3:.1f}ms  old={t_old*1e3:.1f}ms  "
#         f"speedup(new/base)={t_base/t_new:.2f}×")

# # Try a few that should favor Winograd
# for hw,cin,cout in [(128,64,64),(128,64,128),(224,128,128),(256,128,256),(384,256,256)]:
#   timed_conv(HW=hw, Cin=cin, Cout=cout, reps=7, B=1)


import time
from statistics import median
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context, GlobalCounters

def timed_conv(HW=256, Cin=128, Cout=128, reps=10, B=1):
  def run(label, **ctxflags):
    # Create fresh tensors for each benchmark run to avoid cache pollution
    x = Tensor.randn(B, Cin, HW, HW, dtype=dtypes.float32).realize()
    w = Tensor.randn(Cout, Cin, 3, 3, dtype=dtypes.float32).realize()
    
    with Context(**{k: int(bool(v)) for k, v in ctxflags.items()}):
      # compile time (first call)
      GlobalCounters.reset()
      t0 = time.perf_counter()
      (x.conv2d(w, padding=1)).realize()
      compile_time = time.perf_counter() - t0

      # steady-state time (subsequent runs)
      times = []
      for _ in range(reps):
        t0 = time.perf_counter()
        (x.conv2d(w, padding=1)).realize()
        times.append(time.perf_counter() - t0)
      run_time = median(times)

      print(f"{label:<10} | compile={compile_time*1e3:6.1f}ms  run={run_time*1e3:6.1f}ms")
      return compile_time, run_time

  print(f"\n=== HW={HW}, Cin={Cin}, Cout={Cout}, B={B} ===")
 #base_compile, base_run = run("Base", WINO=0, WINO_OLD=0)
  new_compile, new_run   = run("NewWinograd", WINO=1)
  old_compile, old_run   = run("OldWinograd", WINO=0, WINO_OLD=1)
  #print(f"OldWinograd compile={old_compile*1e3:6.1f}ms  run={old_run*1e3:6.1f}ms")
  # print(f"NewWinograd compile={new_compile*1e3:6.1f}ms  run={new_run*1e3:6.1f}ms")
  # print(f"OldWinograd compile={old_compile*1e3:6.1f}ms  run={old_run*1e3:6.1f}ms")
  # print(f"Speedup (new/base) compile={base_compile/new_compile:.2f}×  run={base_run/new_run:.2f}×")
  # print(f"Speedup (old/base) compile={base_compile/old_compile:.2f}×  run={base_run/old_run:.2f}×")

# try a few typical shapes
for hw, cin, cout in [(128,1,64), (128,64,128),(224,128,128),(256,128,256),(384,256,256)]: #,(128,64,128),(224,128,128),(256,128,256),(384,256,256)
  timed_conv(HW=hw, Cin=cin, Cout=cout, reps=5, B=10)