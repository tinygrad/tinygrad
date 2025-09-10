import pickle
#import os
#os.environ["AMD_LLVM"] = "0"
from tinygrad.opt.search import beam_search

def main():
  #with open("/home/hooved/stable_diffusion/checkpoints/training_checkpoints/09041314/kernels_to_beam.pickle", "rb") as f:
  with open("/home/hooved/stable_diffusion/checkpoints/training_checkpoints/09050339/kernels_to_beam.pickle", "rb") as f:
    args_tuples = pickle.load(f)

  last_complete = -1

  for i, args in enumerate(args_tuples):
    if i <= last_complete: continue
    print(f"on kernel {i}")
    beam_search(*args[:-1], 1)

if __name__ == "__main__":
  main()