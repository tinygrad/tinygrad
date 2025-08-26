import globvars as gv
from tinygrad import Tensor, Device, dtypes
import pickle
Device.DEFAULT="CPU"

# 6 clip scores generated from the first 6 prompts in the eval set, using mlperf ref implementation (small number as this requires my nvidia gpu)
real_clip = gv.clip
real_inception = gv.inception

# eval was run on 30,000 random samplings from the first 6 prompts in the eval set
# the same init. latent noise was used for a given prompt (different prompts have different init latents). Init latents were identical to those used in ref, above.
EVAL_CKPT_DIR="/home/hooved/stable_diffusion/checkpoints/training_checkpoints/08051713/run_eval_298669_rng"

# contains the random sampling indices
with open(f"{EVAL_CKPT_DIR}/rng.pickle", "rb") as f:
  rng = pickle.load(f)['rng']

name = "clip"
clip = Tensor.empty(30000, device=f"disk:{EVAL_CKPT_DIR}/{name}.bytes", dtype=dtypes.int if name in {"tokens", "end"} else dtypes.float).to("CPU").realize()
gv.md(real_clip['clip_score'].squeeze(1)[rng], clip)
# diff.abs().mean(): 0.0022213610354810953
# a.abs().mean(): 0.047515869140625
# diff.abs().max(): 0.00467275083065033

name = "inception"
inception = Tensor.empty(30000, 2048, device=f"disk:{EVAL_CKPT_DIR}/{name}.bytes", dtype=dtypes.int if name in {"tokens", "end"} else dtypes.float).to("CPU").realize()
gv.md(real_inception['inception_activation'][rng], inception)
# diff.abs().mean(): 0.002599575789645314
# a.abs().mean(): 0.369140625
# diff.abs().max(): 0.026995301246643066
