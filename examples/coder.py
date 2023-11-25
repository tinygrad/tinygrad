#!/usr/bin/env python3
import os, sys, json
from io import StringIO
sys.path.append(os.getcwd())

from tinygrad import Tensor, nn
from tinygrad.helpers import Timing, colored
from examples.llama import Transformer, convert_from_huggingface, MAX_CONTEXT
from contextlib import redirect_stdout

# https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/blob/main/tokenizer_config.json
#   "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",

IM_END = 32000
IM_START = 32001

def get_spp():
  from sentencepiece import SentencePieceProcessor
  import examples.sentencepiece_model_pb2 as spb2
  mp = spb2.ModelProto()
  with open("weights/OpenHermes/tokenizer.model", "rb") as f:
    mp.ParseFromString(f.read())
  mp.pieces.append(spb2.ModelProto.SentencePiece(piece="<|im_end|>", score=0))
  mp.pieces.append(spb2.ModelProto.SentencePiece(piece="<|im_start|>", score=0))
  with open("/tmp/tokenizer.model", "wb") as f:
    f.write(mp.SerializeToString())
  return SentencePieceProcessor(model_file="/tmp/tokenizer.model")

# TODO: make loading bf16 fast so we can remove this
def create_model_cache(model):
  # TODO: add read only Tensors
  with Timing("load weights: "):
    part1 = nn.state.torch_load("weights/OpenHermes/pytorch_model-00001-of-00002.bin")
    part2 = nn.state.torch_load("weights/OpenHermes/pytorch_model-00002-of-00002.bin")

  with Timing("weights -> model: "):
    nn.state.load_state_dict(model, convert_from_huggingface(part1, model, 32, 8), strict=False)
    nn.state.load_state_dict(model, convert_from_huggingface(part2, model, 32, 8), strict=False)

  with Timing("saving float16 cache: "):
    nn.state.safe_save(nn.state.get_state_dict(model), "/tmp/cached_mistral")

if __name__ == "__main__":
  Tensor.no_grad = True

  f = open("/Users/diane/fun/qstar/prm800k/prm800k/data/phase1_test.jsonl")
  first = json.loads(f.readline())
  #qq = first['question']['problem']
  #print(first)

  #qq = "A rocket costs 4 dollars. A pencil costs 7 dollars. A chicken costs 2 dollars. I spent 7 dollars and bought a rocket, what else did I buy?"
  #qq = "A street light is at the top of a $10ft$ tall pole. A woman $6ft$ tall walks away from the pole with a speed of $7\dfrac{ft}{sec}$ along a straight path. How fast is the tip of her shadow moving when she is $40ft$ from the base of the pole?"
  #exit(0)
  qq = "How are you?"

  # https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/blob/main/config.json
  with Timing("create model: "):
    model = Transformer(4096, 14336, n_heads=32, n_layers=32, norm_eps=1e-5, vocab_size=32002, n_kv_heads=8)

  #create_model_cache(model)
  with Timing("loading float16 cache: "):
    nn.state.load_state_dict(model, nn.state.safe_load("/tmp/cached_mistral"))

  spp = get_spp()
  outputted = ""
  def output(toks, color):
    global outputted
    cur = spp.decode(toks)[len(outputted):]
    sys.stdout.write(colored(cur, color))
    sys.stdout.flush()
    outputted += cur

  def encode_prompt(k, v):
    ret = []
    ret += [IM_START]+spp.encode(f"{k}\n{v}")+[IM_END]+spp.encode("\n")
    if k == "user": ret += [IM_START]+spp.encode("assistant\n")
    return ret

  toks = [spp.bos_id()] + encode_prompt("system", "You are Quentin. Quentin is a useful assistant who writes Python code to answer questions in a\n\n```python\n# insert code here\n```\n block")

  temperature = 0.7
  max_length = 1000

  start_pos = 0
  output(toks, "green")
  skip_user = False
  while 1:
    toks += encode_prompt("user", input("Q: "))
    old_output_len = len(outputted)
    while 1:
      assert len(toks) < MAX_CONTEXT, "context length exceeded"
      tok = model(Tensor([toks[start_pos:]]), start_pos, temperature).multinomial().item()
      start_pos = len(toks)
      toks.append(tok)
      output(toks, "blue")
      if tok == IM_END: break
      if tok == spp.eos_id(): break
      new_output = outputted[old_output_len:]

      if new_output.endswith("```") and '```python\n' in new_output:
        python_code = new_output.split('```python\n')[1].split("```")[0]
        # AI safety. Warning to user.
        # Do not press y if the AI is trying to do doing unsafe things.
        if input(colored(f" <-- PYTHON DETECTED, RUN IT? ", "red")).lower() == 'y':
          my_stdout = StringIO()
          try:
            with redirect_stdout(my_stdout): exec(python_code)
            result = my_stdout.getvalue()
          except Exception as e:
            result = str(e)
          toks += spp.encode(f"\nOutput:\n```\n{result}```")
          output(toks, "yellow")
    print("")
