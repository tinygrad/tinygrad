from tinygrad import nn, Tensor, Device
from tinygrad.helpers import Timing

from extra.models.llama import Transformer
from examples.llama3 import MODEL_PARAMS

if __name__ == "__main__":
  Device.DEFAULT = "NULL"

  with Timing("***** create model in    "):
    # NOTE: max_context=None means no kv cache. kv cache has realize in the model
    model = Transformer(**MODEL_PARAMS["70B"]["args"], linear=nn.Linear, embedding=nn.Embedding, max_context=1024, jit=True, disable_kv_cache=True)

  with Timing("***** run model in       "):
    out = model(Tensor([[0]]), 0)

  with Timing("***** schedule in       "):
    si = out.schedule()

