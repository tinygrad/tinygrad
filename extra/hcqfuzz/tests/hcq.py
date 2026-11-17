from extra.hcqfuzz.spec import TestSpec
import random

class HCQSignalFuzzer(TestSpec):
  def prepare(self, dev, seed):
    random.seed(seed)

    self.env = {
      "GPUS": random.choice([2, 3, 4, 5, 6]),
      "ITERS": random.randint(1000000, 10000000),
      "SEED": seed,
    }

    self.cmd = "python3 test/external/external_fuzz_hcq_signals.py"
    self.timeout = 30 * 60 # 30 minutes

  def get_exec_state(self): return self.env, self.cmd, self.timeout
