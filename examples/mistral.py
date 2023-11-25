#!/usr/bin/env python3
import os, sys
sys.path.append(os.getcwd())

from tinygrad import Tensor, nn
from tinygrad.helpers import Timing
from examples.llama import Transformer, convert_from_huggingface
from sentencepiece import SentencePieceProcessor

if __name__ == "__main__":
  Tensor.no_grad = True

  spp = SentencePieceProcessor(model_file="weights/OpenHermes/tokenizer.model")

  # https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/blob/main/config.json
  with Timing("create model: "):
    model = Transformer(4096, 14336, n_heads=32, n_layers=32, norm_eps=1e-5, vocab_size=32002, n_kv_heads=8)

  # TODO: make loading bf16 fast
  """
  # TODO: add read only Tensors
  with Timing("load weights: "):
    part1 = nn.state.torch_load("weights/OpenHermes/pytorch_model-00001-of-00002.bin")
    part2 = nn.state.torch_load("weights/OpenHermes/pytorch_model-00002-of-00002.bin")

  with Timing("weights -> model: "):
    nn.state.load_state_dict(model, convert_from_huggingface(part1, model, 32, 8), strict=False)
    nn.state.load_state_dict(model, convert_from_huggingface(part2, model, 32, 8), strict=False)

  with Timing("saving float16 cache: "):
    nn.state.safe_save(nn.state.get_state_dict(model), "/tmp/cached_mistral")
  """

  with Timing("loading float16 cache: "):
    nn.state.load_state_dict(model, nn.state.safe_load("/tmp/cached_mistral"))

  outputted = ""
  def output(toks):
    global outputted
    cur = spp.decode(toks)[len(outputted):]
    sys.stdout.write(cur)
    sys.stdout.flush()
    outputted += cur

  user_prompt = "Do you like chicken?"
  toks = [spp.bos_id()] + spp.encode(user_prompt)

  temperature = 0.7
  max_length = 100

  start_pos = 0
  output(toks)
  for i in range(max_length):
    tok = model(Tensor([toks[start_pos:]]), start_pos, temperature).multinomial().item()
    start_pos = len(toks)
    toks.append(tok)

    if tok == spp.eos_id(): break
    output(toks)



