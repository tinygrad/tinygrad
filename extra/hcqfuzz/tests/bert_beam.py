from extra.hcqfuzz.spec import TestSpec
import random

class BertBeam(TestSpec):
  def prepare(self, dev, seed):
    random.seed(seed)

    self.env = {
      "DEFAULT_FLOAT": "HALF",
      "SUM_DTYPE": "HALF",
      "GPUS": 6,
      "BS": random.choice([96]),
      "EVAL_BS": random.choice([96]),
      "FUSE_ARANGE": 1,
      "FUSE_ARANGE_UINT": 0,
      "BEAM": 5,
      "BEAM_UOPS_MAX": 10000,
      "BEAM_UPCAST_MAX": 256,
      "BEAM_LOCAL_MAX": 1024,
      "BEAM_MIN_PROGRESS": 5,
      "IGNORE_JIT_FIRST_BEAM": 1,
      "BASEDIR": "/raid/datasets/wiki",
      "LOGMLPERF": 1,
      "SEED": seed,
      "RESET_STEP": 1,
      "BENCHMARK": 10,
      "BERT_LAYERS": 2,
    }

    self.cmd = "python3 examples/mlperf/model_train.py"
    self.timeout = 40

  def get_exec_state(self): return self.env, self.cmd, self.timeout
