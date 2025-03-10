import time
import torch
from tinygrad.tensor import Tensor

N = 10000
torch_device = torch.device("cpu")
torch_dt = torch.float32

torch_a = (torch.rand(N, N, dtype=torch_dt) - 0.5).to(torch_device)
torch_b = (torch.rand(N, N, dtype=torch_dt) - 0.5).to(torch_device)

tiny_a = Tensor(torch_a.cpu().numpy())
tiny_b = Tensor(torch_b.cpu().numpy())

# test CAT with timing for torch
start = time.perf_counter()
torch_cat = torch.cat((torch_a, torch_b), dim=1)
end_torch = time.perf_counter()
torch_time = end_torch - start
print("Torch cat time:", torch_time)

# test CAT with timing for tinygrad
start = time.perf_counter()
tiny_cat = Tensor.cat(*[tiny_a, tiny_b], dim=1).realize()
end_tiny = time.perf_counter()
tiny_time = end_tiny - start
print("Tinygrad cat time:", tiny_time)
print(tiny_cat.lazydata)

# show how they compare
print(f"Tinygrad is {tiny_time / torch_time:.2f}x slower than torch")

from tinygrad.tensor import all_timings

# average time for each operation and print percentage of time taken
timings = {}
for t in all_timings:
    print("[cat profiling] Component timings:")
    for key, val in t.items():
        if key not in timings:
            timings[key] = 0
        timings[key] += val

print("[cat profiling] Total time taken:")
total_time = sum(timings.values())
for key, val in timings.items():
    print(f"{key}: {val/total_time:.2f}")

"""
  def timed_cat(self:Tensor, *args:Tensor, dim:int=0) -> Tensor:
      import os
      import time
      \"\"\"
      Optimized concatenation that uses a binary tree of tensor operations.
      All tensors must have the same shape except in the concatenating dimension.
      Profiles the time taken by each component if the environment variable PROFILE_CAT is set.
      \"\"\"
      do_profile = os.getenv("PROFILE_CAT") is not None
      timings = {}
  
      t0 = time.perf_counter() if do_profile else None
  
      dim = self._resolve_dim(dim)
      tensors = [self, *args]
  
      # Validate shapes
      for arg in args:
          assert arg.ndim == self.ndim and all(ti == ai for i, (ti, ai) in enumerate(zip(self.shape, arg.shape)) if i != dim)
      if do_profile:
          timings["validation"] = time.perf_counter() - t0
  
      # Calculate total size along dim
      t1 = time.perf_counter() if do_profile else None
      total = sum(t.shape[dim] for t in tensors)
      if do_profile:
          timings["total_calc"] = time.perf_counter() - t1
  
      # Cache offsets in the concatenating dimension.
      t2 = time.perf_counter() if do_profile else None
      def get_positions(ts):
          sizes = [t.shape[dim] for t in ts]
          positions = [0]
          for size in sizes[:-1]:
              positions.append(positions[-1] + size)
          return positions
      offsets = get_positions(tensors)
      if do_profile:
          timings["get_positions"] = time.perf_counter() - t2
  
      # Pre-pad each tensor once so that its data lands in its proper offset.
      t3 = time.perf_counter() if do_profile else None
      padded = []
      # Precompute pad factors for the concatenating dimension.
      pad_factors = [total - t.shape[dim] for t in tensors]
      padded = []
      for pos, t in zip(offsets, tensors):
          after_pad = total - (pos + t.shape[dim])
          pad = []
          for j in range(t.ndim):
              if j == dim:
                  pad.append((pos, after_pad))
              else:
                  pad.append(None)
          padded.append(t.pad(pad))
      if do_profile:
          timings["pre_pad"] = time.perf_counter() - t3
  
      # Iterative binary tree reduction (pairwise addition) to combine the padded tensors.
      t4 = time.perf_counter() if do_profile else None
      while len(padded) > 1:
          new_list = []
          for i in range(0, len(padded), 2):
              if i + 1 < len(padded):
                  new_list.append(padded[i] + padded[i + 1])
              else:
                  new_list.append(padded[i])
          padded = new_list
      if do_profile:
          timings["binary_tree_reduction"] = time.perf_counter() - t4
  
      if do_profile:
          # Print the timings dictionary.
          print("[cat profiling] Component timings:")
          for key, val in timings.items():
              print(f"  {key}: {val:.6f} sec")
          
          global all_timings
          all_timings.append(timings)
      return padded[0]
"""